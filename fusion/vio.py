"""
Visual-Inertial Odometry (VIO) for Stage 4.

Uses VO pose; when IMU buffer is provided, propagates with IMU and optionally
fuses with VO (simple average of propagated pose and VO for now).
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger("opl.fusion.vio")


def _try_imu_propagate(pose_4x4: np.ndarray, velocity: np.ndarray, imu_buffer: list, dt: float):
    """If fusion.imu_propagation available, propagate with last IMU sample."""
    try:
        from fusion.imu_propagation import propagate_pose
    except ImportError:
        return pose_4x4, velocity
    if not imu_buffer:
        return pose_4x4, velocity
    # Use last sample as constant for dt
    sample = imu_buffer[-1]
    gyro = sample.get("gyro", np.zeros(3))
    acc = sample.get("acc", np.zeros(3))
    pose_4x4, velocity = propagate_pose(pose_4x4, velocity, gyro, acc, dt)
    return pose_4x4, velocity


class VIOEstimator:
    """
    VIO: visual-only or visual + IMU propagation. When imu_buffer is given,
    propagates pose with IMU then uses VO (or simple fusion).
    """

    def __init__(self, use_imu_propagation: bool = True, **kwargs: Any) -> None:
        self._kwargs = kwargs
        self._pose: np.ndarray = np.eye(4)
        self._velocity: np.ndarray = np.zeros(3)
        self._use_imu = bool(use_imu_propagation)

    def process_frame(self, pose_vo: np.ndarray, imu_buffer: list | None = None, dt: float = 0.033) -> np.ndarray:
        """
        Process one frame: VO pose and optional IMU. If IMU and use_imu_propagation,
        propagate previous pose by dt then blend with VO (average position, keep VO rotation).
        """
        R_vo = np.asarray(pose_vo[:3, :3])
        t_vo = np.asarray(pose_vo[:3, 3])
        if imu_buffer and self._use_imu:
            self._pose, self._velocity = _try_imu_propagate(
                self._pose.copy(), self._velocity.copy(), imu_buffer, dt
            )
            # Blend: use VO rotation (more reliable), average position with propagated
            t_prop = self._pose[:3, 3]
            t_fused = 0.7 * t_vo + 0.3 * t_prop
            self._pose = np.eye(4)
            self._pose[:3, :3] = R_vo
            self._pose[:3, 3] = t_fused
        else:
            self._pose = np.eye(4)
            self._pose[:3, :3] = R_vo
            self._pose[:3, 3] = t_vo.ravel()
        return self._pose.copy()

    @property
    def pose(self) -> np.ndarray:
        """Current pose (4x4 body-to-world)."""
        return self._pose.copy()
