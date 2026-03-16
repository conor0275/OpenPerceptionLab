import cv2
import matplotlib.pyplot as plt

from perception.perception_system import PerceptionSystem

# 初始化系统
system = PerceptionSystem()

# 读取图像
image = cv2.imread("test.jpg")

# 运行系统
det, seg, depth = system.run(image)

# detection可视化
det_img = det[0].plot()

# segmentation显示
seg_img = seg

# depth显示
depth_img = depth

# 可视化
plt.figure(figsize=(12,4))

plt.subplot(1,3,1)
plt.title("Detection")
plt.imshow(cv2.cvtColor(det_img, cv2.COLOR_BGR2RGB))

plt.subplot(1,3,2)
plt.title("Segmentation")
plt.imshow(seg_img)

plt.subplot(1,3,3)
plt.title("Depth")
plt.imshow(depth_img, cmap="inferno")

plt.show()