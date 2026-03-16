from ultralytics import YOLO
import cv2

# 加载模型
model = YOLO("yolov8n.pt")

# 读取图片
img = cv2.imread("test.jpg")

# 推理
results = model(img)

# 可视化
annotated = results[0].plot()

cv2.imshow("result", annotated)
cv2.waitKey(0)