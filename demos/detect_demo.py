import os
from ultralytics import YOLO
import cv2

# 加载模型
model = YOLO("yolov8n.pt")

# 读取图片（可通过 opl demo detect --image path 指定）
image_path = os.environ.get("OPL_DEMO_IMAGE") or "test.jpg"
img = cv2.imread(image_path)
if img is None:
    raise FileNotFoundError(f"Image not found: {image_path}")

# 推理
results = model(img)

# 可视化
annotated = results[0].plot()

cv2.imshow("result", annotated)
cv2.waitKey(0)