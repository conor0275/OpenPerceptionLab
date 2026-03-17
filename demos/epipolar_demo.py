import cv2
import matplotlib.pyplot as plt

from geometry.feature_matching import FeatureMatcher
from geometry.epipolar import EpipolarGeometry, draw_epilines

# 读取图像
img1 = cv2.imread("image1.jpg", 0)
img2 = cv2.imread("image2.jpg", 0)

# 特征匹配
matcher = FeatureMatcher()
kp1, kp2, matches = matcher.match(img1, img2)

# 计算F矩阵
epi = EpipolarGeometry()
F, pts1, pts2, mask = epi.compute_fundamental(kp1, kp2, matches)

# 选内点
pts1 = pts1[mask.ravel()==1]
pts2 = pts2[mask.ravel()==1]

# 计算对极线
lines1 = cv2.computeCorrespondEpilines(
    pts2.reshape(-1,1,2), 2, F
)
lines1 = lines1.reshape(-1,3)

# 绘制
img1_epi, img2_epi = draw_epilines(
    img1, img2,
    lines1,
    pts1, pts2
)

plt.figure(figsize=(10,5))

plt.subplot(1,2,1)
plt.imshow(img1_epi)
plt.title("Epipolar Lines (Image1)")

plt.subplot(1,2,2)
plt.imshow(img2_epi)
plt.title("Matched Points (Image2)")

plt.show()