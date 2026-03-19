"""
Incremental Structure from Motion (SfM).

Initialize from two views (essential matrix + triangulation), then add views
via PnP + triangulate new points. Output: sparse point cloud and camera poses.
"""
from __future__ import annotations

import logging
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger("opl.reconstruction.sfm")


def _triangulate_two_views(
    pts1: np.ndarray,
    pts2: np.ndarray,
    R1: np.ndarray,
    t1: np.ndarray,
    R2: np.ndarray,
    t2: np.ndarray,
    K: np.ndarray,
) -> np.ndarray:
    """Triangulate 3D points (world frame) from two views. pts1/pts2 Nx2, returns Nx3."""
    K = np.asarray(K, dtype=np.float64)
    P1 = K @ np.hstack((R1, t1.reshape(3, 1)))
    P2 = K @ np.hstack((R2, t2.reshape(3, 1)))
    pts1_t = np.ascontiguousarray(pts1.astype(np.float32).T)
    pts2_t = np.ascontiguousarray(pts2.astype(np.float32).T)
    pts4d = cv2.triangulatePoints(P1.astype(np.float32), P2.astype(np.float32), pts1_t, pts2_t)
    w = np.asarray(pts4d[3], dtype=np.float64).ravel()
    valid = np.abs(w) > 1e-6
    w_safe = np.where(valid, w, 1.0)
    pts3d = (pts4d[:3] / w_safe).T
    return pts3d, valid.ravel()


class IncrementalSfM:
    """
    Incremental SfM: two-view init, then add views by PnP + triangulation.
    All poses are world-to-camera (R, t): p_cam = R @ p_world + t.
    """

    def __init__(self, K: np.ndarray, min_matches: int = 50) -> None:
        self.K = np.asarray(K, dtype=np.float64)
        self.min_matches = min_matches
        self.points_3d: list[np.ndarray] = []  # list of (3,) in world
        self.keyframes: list[dict[str, Any]] = []  # R, t, kps, des, kp_to_point (list of point_idx or -1)
        self._orb = cv2.ORB_create(2000)
        self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

    def _extract(self, img: np.ndarray):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        kp, des = self._orb.detectAndCompute(gray, None)
        return kp, des

    def _match(self, des1, des2):
        if des1 is None or des2 is None:
            return []
        m = self._matcher.match(des1, des2)
        m = sorted(m, key=lambda x: x.distance)
        return m

    def add_first_two_views(self, img0: np.ndarray, img1: np.ndarray) -> bool:
        """Initialize from first two images. Returns True on success."""
        kp0, des0 = self._extract(img0)
        kp1, des1 = self._extract(img1)
        if des0 is None or des1 is None or len(kp0) < 8 or len(kp1) < 8:
            logger.warning("Insufficient features in first two views")
            return False
        matches = self._match(des0, des1)
        if len(matches) < self.min_matches:
            logger.warning("Too few matches between first two views: %d", len(matches))
            return False
        pts1 = np.array([kp0[m.queryIdx].pt for m in matches], dtype=np.float32)
        pts2 = np.array([kp1[m.trainIdx].pt for m in matches], dtype=np.float32)
        E, mask = cv2.findEssentialMat(pts1, pts2, self.K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        if E is None or mask is None:
            return False
        inliers = mask.ravel() != 0
        if np.count_nonzero(inliers) < 8:
            return False
        _, R, t, _ = cv2.recoverPose(E, pts1[inliers], pts2[inliers], self.K)
        R0, t0 = np.eye(3, dtype=np.float64), np.zeros((3, 1), dtype=np.float64)
        pts1_i = pts1[inliers]
        pts2_i = pts2[inliers]
        pts3d, valid = _triangulate_two_views(pts1_i, pts2_i, R0, t0, R, t, self.K)
        # Filter: front of both cameras, finite
        z0 = (R0 @ pts3d.T).T + t0.ravel()
        z1 = (R @ pts3d.T).T + t.ravel()
        ok = np.isfinite(pts3d).all(axis=1) & (z0[:, 2] > 0.1) & (z1[:, 2] > 0.1) & valid
        pts3d = pts3d[ok]
        if len(pts3d) < 10:
            return False
        self.points_3d = list(pts3d)
        inlier_indices = np.where(inliers)[0]
        kp_to_point_0 = [-1] * len(kp0)
        kp_to_point_1 = [-1] * len(kp1)
        for k in range(len(pts3d)):
            inlier_idx = np.where(ok)[0][k]
            match_idx = inlier_indices[inlier_idx]
            q, tr = matches[match_idx].queryIdx, matches[match_idx].trainIdx
            kp_to_point_0[q] = k
            kp_to_point_1[tr] = k
        self.keyframes = [
            {"R": R0, "t": t0, "kps": kp0, "des": des0, "kp_to_point": kp_to_point_0},
            {"R": R, "t": t, "kps": kp1, "des": des1, "kp_to_point": kp_to_point_1},
        ]
        logger.info("SfM init: 2 keyframes, %d points", len(self.points_3d))
        return True

    def add_view(self, img: np.ndarray) -> bool:
        """Add one more image; match to last keyframe, PnP + triangulate. Returns True on success."""
        if len(self.keyframes) == 0:
            return False
        kp, des = self._extract(img)
        if des is None or len(kp) < 8:
            return False
        prev = self.keyframes[-1]
        matches = self._match(prev["des"], des)
        if len(matches) < 8:
            return False
        pts_3d, pts_2d = [], []
        for m in matches:
            pid = prev["kp_to_point"][m.queryIdx]
            if pid < 0:
                continue
            pts_3d.append(self.points_3d[pid])
            pts_2d.append(kp[m.trainIdx].pt)
        if len(pts_3d) < 6:
            return False
        pts_3d = np.array(pts_3d, dtype=np.float64)
        pts_2d = np.array(pts_2d, dtype=np.float32).reshape(-1, 2)
        result = cv2.solvePnPRansac(
            pts_3d, pts_2d, self.K, None,
            reprojectionError=4.0, confidence=0.99,
        )
        success = bool(result[0])
        rvec = result[1]
        tvec = result[2]
        if not success or rvec is None or tvec is None:
            return False
        R, _ = cv2.Rodrigues(rvec)
        t = tvec.reshape(3, 1).astype(np.float64)
        # Triangulate new points from matches that don't have 3D yet
        R_prev = prev["R"]
        t_prev = prev["t"]
        new_pts1, new_pts2 = [], []
        new_kp_prev_idx, new_kp_cur_idx = [], []
        for m in matches:
            if prev["kp_to_point"][m.queryIdx] >= 0:
                continue
            new_pts1.append(prev["kps"][m.queryIdx].pt)
            new_pts2.append(kp[m.trainIdx].pt)
            new_kp_prev_idx.append(m.queryIdx)
            new_kp_cur_idx.append(m.trainIdx)
        kp_to_point_new = [-1] * len(kp)
        if len(new_pts1) >= 4:
            pt1 = np.array(new_pts1, dtype=np.float32)
            pt2 = np.array(new_pts2, dtype=np.float32)
            pts3d, valid = _triangulate_two_views(pt1, pt2, R_prev, t_prev, R, t, self.K)
            z_prev = (R_prev @ pts3d.T).T + t_prev.ravel()
            z_cur = (R @ pts3d.T).T + t.ravel()
            ok = np.isfinite(pts3d).all(axis=1) & (z_prev[:, 2] > 0.1) & (z_cur[:, 2] > 0.1) & valid.ravel()
            base = len(self.points_3d)
            count = 0
            for i in range(len(ok)):
                if not ok[i]:
                    continue
                self.points_3d.append(pts3d[i])
                kp_to_point_new[new_kp_cur_idx[i]] = base + count
                count += 1
        self.keyframes.append({"R": R, "t": t, "kps": kp, "des": des, "kp_to_point": kp_to_point_new})
        logger.info("SfM add view: %d keyframes, %d points", len(self.keyframes), len(self.points_3d))
        return True

    def get_points_array(self) -> np.ndarray:
        """Return Nx3 array of 3D points."""
        if not self.points_3d:
            return np.zeros((0, 3), dtype=np.float64)
        return np.array(self.points_3d, dtype=np.float64)

    def get_poses(self) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Return (list of R, list of t) world-to-camera for each keyframe."""
        R_list = [kf["R"] for kf in self.keyframes]
        t_list = [kf["t"] for kf in self.keyframes]
        return R_list, t_list
