import cv2
import matplotlib.pyplot as plt
from perception.segmentation.deeplab_segment import DeepLabSegmenter

# 初始化模型
segmenter = DeepLabSegmenter()

# 读取图像
image = cv2.imread("test.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 预测
mask = segmenter.predict(image)

# 显示
plt.subplot(1,2,1)
plt.title("Image")
plt.imshow(image)

plt.subplot(1,2,2)
plt.title("Segmentation")
plt.imshow(mask)

plt.show()