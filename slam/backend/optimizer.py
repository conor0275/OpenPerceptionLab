import numpy as np
import cv2
from scipy.optimize import least_squares


class BundleAdjuster:

    def __init__(self, K):
        self.K = K

    def project(self, points_3d, rvec, tvec):

        pts_proj, _ = cv2.projectPoints(
            points_3d,
            rvec,
            tvec,
            self.K,
            None
        )

        return pts_proj.reshape(-1, 2)

    def residuals(self, params, points_3d, points_2d):

        rvec = params[:3]
        tvec = params[3:6]

        proj = self.project(points_3d, rvec, tvec)

        return (proj - points_2d).ravel()

    def optimize_pose(self, points_3d, points_2d, rvec, tvec):

        x0 = np.hstack((rvec.ravel(), tvec.ravel()))

        res = least_squares(
            self.residuals,
            x0,
            args=(points_3d, points_2d),
            method='lm'
        )

        rvec_opt = res.x[:3]
        tvec_opt = res.x[3:6]

        return rvec_opt, tvec_opt