import numpy as np
from geometry.camera_model import CameraModel

# 相机参数
fx = 800
fy = 800
cx = 320
cy = 240

camera = CameraModel(fx, fy, cx, cy)

# 3D点
point_3d = np.array([1, 1, 5])

# 相机位姿
R = np.eye(3)
t = np.zeros(3)

# 投影
pixel = camera.project(point_3d, R, t)

print("pixel coordinate:", pixel)