import os
import cv2
import matplotlib.pyplot as plt
from perception.depth.midas_depth import MiDaSDepth

# 初始化模型
depth_model = MiDaSDepth()

# 读取图像（可通过 opl demo depth --image path 指定）
image_path = os.environ.get("OPL_DEMO_IMAGE") or "test.jpg"
image = cv2.imread(image_path)
if image is None:
    raise FileNotFoundError(f"Image not found: {image_path}")

# 预测深度
depth = depth_model.predict(image)

# 显示
plt.subplot(1,2,1)
plt.title("Image")
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

plt.subplot(1,2,2)
plt.title("Depth Map")
plt.imshow(depth, cmap="inferno")

plt.show()