"""
Load trajectories for fusion: VO from visual SLAM map, LiDAR from saved .npz.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from fusion.poses import pose_4x4_from_Rt


def load_vo_trajectory_from_map(path: str | Path) -> list[np.ndarray]:
    """
    Load keyframe poses from visual SLAM map (.npz).
    Map stores world-to-camera R, t; we convert to body-to-world 4x4.
    Returns list of 4x4 poses (one per keyframe).
    """
    path = Path(path)
    data = np.load(path, allow_pickle=False)
    kf_R = data["keyframe_R"]
    kf_t = data["keyframe_t"]
    out = []
    for i in range(len(kf_R)):
        R = kf_R[i]
        t = kf_t[i].reshape(3, 1)
        T = pose_4x4_from_Rt(R, t)
        out.append(T)
    return out


def load_lidar_trajectory(path: str | Path) -> list[np.ndarray]:
    """
    Load LiDAR trajectory from .npz (saved by lidar-slam --save-trajectory).
    Expects key 'poses' with shape (N, 4, 4).
    """
    path = Path(path)
    data = np.load(path, allow_pickle=False)
    poses = data["poses"]
    if poses.ndim == 3:
        return [poses[i] for i in range(poses.shape[0])]
    return []


def save_trajectory_2d(poses_2d: list[tuple[float, float, float]], path: str | Path) -> None:
    """Save fused 2D trajectory to .npz (x, y, theta)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.array(poses_2d, dtype=np.float64)
    np.savez_compressed(path, x=arr[:, 0], y=arr[:, 1], theta=arr[:, 2])
