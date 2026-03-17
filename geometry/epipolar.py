import cv2
import numpy as np


class EpipolarGeometry:
    def compute_fundamental(self, kp1, kp2, matches):
        """
        Compute fundamental matrix F from matched keypoints.

        Returns:
            F: (3,3) fundamental matrix
            pts1: (N,2) int32 points in image1
            pts2: (N,2) int32 points in image2
            mask: (N,1) inlier mask from RANSAC
        """
        pts1 = []
        pts2 = []

        for m in matches:
            pts1.append(kp1[m.queryIdx].pt)
            pts2.append(kp2[m.trainIdx].pt)

        pts1 = np.int32(pts1)
        pts2 = np.int32(pts2)

        F, mask = cv2.findFundamentalMat(
            pts1,
            pts2,
            cv2.FM_RANSAC
        )

        return F, pts1, pts2, mask

def draw_epilines(img1, img2, lines, pts1, pts2):
    """
    Draw epipolar lines on img1 and corresponding points on both images.

    Args:
        img1, img2: grayscale images
        lines: epilines corresponding to pts2 (shape (N,3))
        pts1, pts2: point arrays (N,2)
    """
    r, c = img1.shape

    img1_color = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    img2_color = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)

    for r_line, pt1, pt2 in zip(lines, pts1, pts2):
        color = tuple(np.random.randint(0, 255, 3).tolist())

        # line: ax + by + c = 0
        if abs(float(r_line[1])) < 1e-12:
            continue

        x0, y0 = map(int, [0, -r_line[2] / r_line[1]])
        x1, y1 = map(int, [c, -(r_line[2] + r_line[0] * c) / r_line[1]])

        img1_color = cv2.line(img1_color, (x0, y0), (x1, y1), color, 1)
        img1_color = cv2.circle(img1_color, tuple(map(int, pt1)), 5, color, -1)
        img2_color = cv2.circle(img2_color, tuple(map(int, pt2)), 5, color, -1)

    return img1_color, img2_color
