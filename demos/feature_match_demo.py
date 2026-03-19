import os
import cv2
import matplotlib.pyplot as plt

from geometry.feature_matching import FeatureMatcher

matcher = FeatureMatcher()

# 读取图像（可通过 opl demo feature_match --image1 a.jpg --image2 b.jpg 指定）
path1 = os.environ.get("OPL_DEMO_IMAGE1") or "image1.jpg"
path2 = os.environ.get("OPL_DEMO_IMAGE2") or "image2.jpg"
img1 = cv2.imread(path1, 0)
img2 = cv2.imread(path2, 0)
if img1 is None or img2 is None:
    raise FileNotFoundError(f"Images not found: {path1!r}, {path2!r}")

kp1, kp2, matches = matcher.match(img1, img2)

match_img = cv2.drawMatches(
    img1, kp1,
    img2, kp2,
    matches[:50],
    None,
    flags=2
)

plt.figure(figsize=(12,6))
plt.imshow(match_img)
plt.title("Feature Matches")
plt.axis("off")
plt.show()