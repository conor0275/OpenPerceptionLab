import cv2
import matplotlib.pyplot as plt

from geometry.feature_matching import FeatureMatcher

matcher = FeatureMatcher()

img1 = cv2.imread("image1.jpg", 0)
img2 = cv2.imread("image2.jpg", 0)

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