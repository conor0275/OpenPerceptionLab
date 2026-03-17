import numpy as np
import cv2


class Triangulator:

    def triangulate(self, pts1, pts2, K):

        K = np.asarray(K, dtype=np.float64)

        pts1 = np.asarray(pts1, dtype=np.float64)
        pts2 = np.asarray(pts2, dtype=np.float64)

        # 构造投影矩阵
        P1 = K @ np.hstack((np.eye(3, dtype=np.float64), np.zeros((3, 1), dtype=np.float64)))
        P2 = K @ np.hstack((np.eye(3, dtype=np.float64), np.array([[1.0], [0.0], [0.0]], dtype=np.float64)))

        # 转置
        pts1 = pts1.T
        pts2 = pts2.T

        # 三角化
        points_4d = cv2.triangulatePoints(P1, P2, pts1, pts2)

        # 转换为3D
        points_3d = points_4d[:3] / points_4d[3]

        return points_3d.T