import numpy as np


class CameraModel:

    def __init__(self, fx, fy, cx, cy):    # 相机内参矩阵

        self.K = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ])

    def project(self, point_3d, R, t):

        # 转换为相机坐标
        point_cam = R @ point_3d + t

        # 投影
        point_img = self.K @ point_cam

        # 归一化
        u = point_img[0] / point_img[2]
        v = point_img[1] / point_img[2]

        return np.array([u, v])