import cv2
import numpy as np

from backend.optimizer import BundleAdjuster

class Tracker:

    def __init__(self, K, map_obj=None):
        self.K = np.asarray(K, dtype=np.float64)
        self.prev_frame = None
        self.map = map_obj
        self.bundle_adjuster = BundleAdjuster(self.K)

    def process(self, frame):

        if self.prev_frame is None:
            self.prev_frame = frame
            return frame

        # 匹配
        if self.prev_frame.descriptors is None or frame.descriptors is None:
            # No features detected in one of the frames
            frame.pose_R = self.prev_frame.pose_R
            frame.pose_t = self.prev_frame.pose_t
            self.prev_frame = frame
            return frame

        bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        matches = bf.match(
            self.prev_frame.descriptors,
            frame.descriptors
        )
        if matches is None or len(matches) < 8:
            # Essential matrix (5-point) needs enough correspondences; keep previous pose
            frame.pose_R = self.prev_frame.pose_R
            frame.pose_t = self.prev_frame.pose_t
            self.prev_frame = frame
            return frame

        pts1 = np.array([
            self.prev_frame.keypoints[m.queryIdx].pt
            for m in matches
        ], dtype=np.float32).reshape(-1, 2)

        pts2 = np.array([
            frame.keypoints[m.trainIdx].pt
            for m in matches
        ], dtype=np.float32).reshape(-1, 2)

        if pts1.shape[0] < 8 or pts2.shape[0] != pts1.shape[0]:
            frame.pose_R = self.prev_frame.pose_R
            frame.pose_t = self.prev_frame.pose_t
            self.prev_frame = frame
            return frame

        # 估计位姿
        E, mask = cv2.findEssentialMat(
            pts1, pts2, self.K, method=cv2.RANSAC, prob=0.999, threshold=1.0
        )
        if E is None or mask is None:
            frame.pose_R = self.prev_frame.pose_R
            frame.pose_t = self.prev_frame.pose_t
            self.prev_frame = frame
            return frame

        inliers = mask.ravel() != 0
        if np.count_nonzero(inliers) < 5:
            frame.pose_R = self.prev_frame.pose_R
            frame.pose_t = self.prev_frame.pose_t
            self.prev_frame = frame
            return frame

        _, R, t, _ = cv2.recoverPose(E, pts1[inliers], pts2[inliers], self.K)

        # 三角化生成3D点
        K32 = self.K.astype(np.float32)
        R32 = R.astype(np.float32)
        t32 = t.astype(np.float32)

        P1 = K32 @ np.hstack((np.eye(3, dtype=np.float32), np.zeros((3, 1), dtype=np.float32)))
        P2 = K32 @ np.hstack((R32, t32))

        pts1_t = np.ascontiguousarray(pts1[inliers].T, dtype=np.float32)
        pts2_t = np.ascontiguousarray(pts2[inliers].T, dtype=np.float32)

        points_4d = cv2.triangulatePoints(P1, P2, pts1_t, pts2_t)
        w = points_4d[3]
        valid = np.abs(w) > 1e-6
        points_3d = (points_4d[:3, valid] / w[valid]).T

        # Basic sanity filter: remove NaN/Inf and unreasonable depths
        finite = np.isfinite(points_3d).all(axis=1)
        points_3d = points_3d[finite]
        if points_3d.size:
            # keep points in front of camera and within a loose range
            points_3d = points_3d[(points_3d[:, 2] > 0.0) & (np.abs(points_3d[:, :3]) < 1e4).all(axis=1)]

        # ------ BA优化引入 -------
        # 如果三角化后的点和观测足够，执行一次BA优化
        if points_3d.shape[0] >= 8:
            # 1. 准备BA优化初值（rvec, tvec）
            # 从R/t转换到rvec, tvec (上一帧为世界，当前帧相对位姿)
            # prev_R, prev_t 不变，当前 frame 的 R, t 是增量更新
            curr_R = R @ self.prev_frame.pose_R
            curr_t = self.prev_frame.pose_t + t
            # 旋转矩阵转为旋转向量
            rvec, _ = cv2.Rodrigues(curr_R)
            tvec = curr_t.ravel()
            # 用于优化的3D点和2D观测点
            points_2d = pts2[inliers][valid][:, :]
            points_3d_valid = points_3d

            # 若对应关系不足则不做ba
            if points_2d.shape[0] == points_3d_valid.shape[0] and points_3d_valid.shape[0] >= 8:
                # BA优化当前帧的 rvec, tvec
                rvec_opt, tvec_opt = self.bundle_adjuster.optimize_pose(
                    points_3d_valid,
                    points_2d,
                    rvec,
                    tvec
                )
                # 优化后更新pose
                # 再转为R/t
                R_opt, _ = cv2.Rodrigues(rvec_opt)
                frame.pose_R = R_opt
                frame.pose_t = tvec_opt.reshape(3, 1)
            else:
                # 如果数据太少，不做优化，常规累加
                frame.pose_R = curr_R
                frame.pose_t = curr_t
        else:
            # BA数据不够，常规累加
            frame.pose_R = R @ self.prev_frame.pose_R
            frame.pose_t = self.prev_frame.pose_t + t

        # 加入地图
        if self.map is not None and len(points_3d) > 0:
            self.map.add_points(points_3d)

        self.prev_frame = frame

        # 判断是否关键帧（简单策略）
        if frame.id % 10 == 0:
            frame.is_keyframe = True
            if self.map is not None:
                self.map.add_keyframe(frame)

        return frame