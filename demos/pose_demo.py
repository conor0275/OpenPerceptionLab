import os
import cv2
import numpy as np

from geometry.feature_matching import FeatureMatcher
from geometry.epipolar import EpipolarGeometry
from geometry.triangulation import Triangulator
from geometry.pose_estimation import PoseEstimator

# 读取图像（可通过 opl demo pose --image1 a.jpg --image2 b.jpg 指定）
path1 = os.environ.get("OPL_DEMO_IMAGE1") or "image1.jpg"
path2 = os.environ.get("OPL_DEMO_IMAGE2") or "image2.jpg"
img1 = cv2.imread(path1, 0)
img2 = cv2.imread(path2, 0)
if img1 is None or img2 is None:
    raise FileNotFoundError(f"Images not found: {path1!r}, {path2!r}")

# 相机内参（假设）
K = np.array([
    [800, 0, 320],
    [0, 800, 240],
    [0, 0, 1]
])

# 特征匹配
matcher = FeatureMatcher()
kp1, kp2, matches = matcher.match(img1, img2)

# 转点
pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

# ========== Step1：恢复位姿（2D-2D） ==========
pose = PoseEstimator()
R, t, mask = pose.estimate_from_essential(pts1, pts2, K)

print("R:\n", R)
print("t:\n", t)

# 只保留内点
inlier_mask = mask.ravel() != 0
pts1 = pts1[inlier_mask]
pts2 = pts2[inlier_mask]

if pts1.shape[0] < 4:
    raise RuntimeError(f"Not enough inliers after recoverPose: {pts1.shape[0]}")

# ========== Step2：三角化 ==========
tri = Triangulator()

K32 = np.asarray(K, dtype=np.float32)
R32 = np.asarray(R, dtype=np.float32)
t32 = np.asarray(t, dtype=np.float32)

P1 = K32 @ np.hstack((np.eye(3, dtype=np.float32), np.zeros((3, 1), dtype=np.float32)))
P2 = K32 @ np.hstack((R32, t32))
P1 = np.ascontiguousarray(P1, dtype=np.float32)
P2 = np.ascontiguousarray(P2, dtype=np.float32)

pts1_t = np.ascontiguousarray(pts1.T, dtype=np.float32)
pts2_t = np.ascontiguousarray(pts2.T, dtype=np.float32)

points_4d = cv2.triangulatePoints(P1, P2, pts1_t, pts2_t)
points_3d = (points_4d[:3] / points_4d[3]).T

# ========== Step3：PnP（验证） ==========
R_pnp, t_pnp = pose.estimate_pnp(points_3d, pts2, K)

print("\nPnP R:\n", R_pnp)
print("PnP t:\n", t_pnp)