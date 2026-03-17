import cv2
import numpy as np
import matplotlib.pyplot as plt

from geometry.feature_matching import FeatureMatcher
from geometry.epipolar import EpipolarGeometry
from geometry.triangulation import Triangulator

# 读取图像
img1 = cv2.imread("image1.jpg", 0)
img2 = cv2.imread("image2.jpg", 0)

# 特征匹配
matcher = FeatureMatcher()
kp1, kp2, matches = matcher.match(img1, img2)

# 计算F
epi = EpipolarGeometry()
F, pts1, pts2, mask = epi.compute_fundamental(kp1, kp2, matches)

# 取内点
pts1 = pts1[mask.ravel()==1]
pts2 = pts2[mask.ravel()==1]

# 相机内参（假设）
K = np.array([
    [800, 0, 320],
    [0, 800, 240],
    [0, 0, 1]
])

# 三角化
tri = Triangulator()
points_3d = tri.triangulate(pts1, pts2, K)

print("3D points shape:", points_3d.shape)

# 可视化点云（2D投影）
plt.scatter(points_3d[:,0], points_3d[:,2], s=2)
plt.title("Simple Point Cloud")
plt.xlabel("X")
plt.ylabel("Z")
plt.show()