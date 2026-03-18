import cv2
import numpy as np
import os

from core.frame import Frame
from frontend.feature import FeatureExtractor
from frontend.tracking import Tracker
from visualization.viewer import TrajectoryViewer
from backend.map import Map
from visualization.viewer import PointCloudViewer

# 相机内参
K = np.array([
    [800, 0, 320],
    [0, 800, 240],
    [0, 0, 1]
])

camera_index = int(os.environ.get("OPL_CAMERA_INDEX", "0"))
cap = cv2.VideoCapture(camera_index)

extractor = FeatureExtractor()

frame_id = 0

trajectory = []
viewer = TrajectoryViewer()

map = Map()
tracker = Tracker(K, map)

pc_viewer = PointCloudViewer()

while True:

    ret, img = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    frame = Frame(gray, K, frame_id)

    kp, des = extractor.extract(gray)

    frame.keypoints = kp
    frame.descriptors = des

    frame = tracker.process(frame)

    trajectory.append(frame.pose_t.copy())
    viewer.update(frame.pose_t)

    pc_viewer.update(map.points)
    
    # 可视化
    cv2.imshow("frame", img)

    if cv2.waitKey(1) == 27:
        break

    frame_id += 1

    print("Map points:", len(map.points))
    print("Keyframes:", len(map.keyframes))

cap.release()
cv2.destroyAllWindows()