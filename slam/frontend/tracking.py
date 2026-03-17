import cv2
import numpy as np


class Tracker:

    def __init__(self, K, map_obj=None):
        self.K = np.asarray(K, dtype=np.float64)
        self.prev_frame = None
        self.map = map_obj

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
        points_3d = (points_4d[:3] / points_4d[3]).T

        # 加入地图
        if self.map is not None:
            self.map.add_points(points_3d)
        # 更新位姿（关键！）
        frame.pose_R = R @ self.prev_frame.pose_R
        frame.pose_t = self.prev_frame.pose_t + t

        self.prev_frame = frame

        # 判断是否关键帧（简单策略）
        if frame.id % 10 == 0:
            frame.is_keyframe = True
            if self.map is not None:
                self.map.add_keyframe(frame)

        return frame