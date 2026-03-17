import cv2
import numpy as np


class PoseEstimator:

    def estimate_from_essential(self, pts1, pts2, K):

        E, mask = cv2.findEssentialMat(
            pts1,
            pts2,
            K,
            method=cv2.RANSAC,
            prob=0.999,
            threshold=1.0
        )

        _, R, t, mask = cv2.recoverPose(
            E,
            pts1,
            pts2,
            K
        )

        return R, t, mask


    def estimate_pnp(self, pts3d, pts2d, K):

        success, rvec, tvec = cv2.solvePnP(
            pts3d,
            pts2d,
            K,
            None
        )

        R, _ = cv2.Rodrigues(rvec)

        return R, tvec