from __future__ import annotations

import os
import logging

import cv2
import numpy as np

# 存储点云 + 关键帧
from slam.backend.map import Map
# 图像+特征+位姿
from slam.core.frame import Frame
# 提取 ORB 特征
from slam.frontend.feature import FeatureExtractor
# 估计相机运动
from slam.frontend.tracking import Tracker
# 可视化轨迹/点云
from slam.visualization.viewer import PointCloudViewer, TrajectoryViewer
from openperceptionlab.config import AppConfig, load_config


logger = logging.getLogger("opl.slam")


def main(
    camera_index: int | None = None,
    config: AppConfig | None = None,
    config_path: str | None = None,
    load_map_path: str | None = None,
    save_map_path: str | None = None,
) -> int:
    if config is None:
        config = load_config(config_path)

    K = config.intrinsics.as_K()

    if camera_index is None:
        camera_index = int(os.environ.get("OPL_CAMERA_INDEX", str(config.camera.index)))
    config.camera.index = camera_index

    cap = cv2.VideoCapture(camera_index)
    if config.camera.width is not None:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(config.camera.width))
    if config.camera.height is not None:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(config.camera.height))
    if config.camera.fps is not None:
        cap.set(cv2.CAP_PROP_FPS, int(config.camera.fps))

    extractor = FeatureExtractor()
    if load_map_path:
        map_ = Map.load(load_map_path)
        logger.info("Loaded map from %s: %d points, %d keyframes", load_map_path, len(map_.points), len(map_.keyframes))
    else:
        map_ = Map()
    tracker = Tracker(K, map_, keyframe_interval=config.slam.keyframe_interval)

    frame_id = 0
    traj_viewer = TrajectoryViewer() if config.viewer.show_trajectory else None
    pc_viewer = (
        PointCloudViewer(max_points=config.viewer.max_points, point_size=config.viewer.point_size)
        if config.viewer.show_pointcloud
        else None
    )

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

        if traj_viewer is not None:
            traj_viewer.update(frame.pose_t)
        if pc_viewer is not None:
            pc_viewer.update(map_.points)

        if config.viewer.show_frame:
            cv2.imshow("frame", img)
            key = cv2.waitKey(1)
        else:
            key = cv2.waitKey(1)

        if key == 27:
            break

        frame_id += 1
        if config.slam.log_every_n_frames > 0 and (frame_id % config.slam.log_every_n_frames == 0):
            logger.info("frame=%d map_points=%d keyframes=%d", frame_id, len(map_.points), len(map_.keyframes))

    cap.release()
    cv2.destroyAllWindows()
    if save_map_path:
        map_.save(save_map_path)
        logger.info("Saved map to %s: %d points, %d keyframes", save_map_path, len(map_.points), len(map_.keyframes))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())