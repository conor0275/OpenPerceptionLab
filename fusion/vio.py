"""
Visual-Inertial Odometry (VIO) placeholder for Stage 4.

This module defines the VIO interface: consume images (and optionally IMU),
output pose. Current implementation is visual-only (reuses VO pose); IMU
preintegration and joint optimization are the intended next steps for
full VIO. Keeping this stub makes the fusion pipeline ready for when
IMU is added (e.g. GTSAM IMU factors).
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger("opl.fusion.vio")


class VIOEstimator:
    """
    VIO pose estimator interface.
    Today: visual-only, returns the same pose as VO (no IMU).
    Later: fuse visual pose with IMU preintegration and optional joint optimization.
    """

    def __init__(self, **kwargs: Any) -> None:
        self._kwargs = kwargs
        self._pose: np.ndarray = np.eye(4)

    def process_frame(self, pose_vo: np.ndarray, imu_buffer: list | None = None) -> np.ndarray:
        """
        Process one frame: take visual odometry pose and optional IMU data.
        Returns fused pose (currently just VO).
        """
        # Placeholder: no IMU fusion yet
        if imu_buffer:
            logger.debug("IMU buffer length %d (not used yet)", len(imu_buffer))
        R = np.asarray(pose_vo[:3, :3])
        t = np.asarray(pose_vo[:3, 3])
        self._pose = np.eye(4)
        self._pose[:3, :3] = R
        self._pose[:3, 3] = t.ravel()
        return self._pose.copy()

    @property
    def pose(self) -> np.ndarray:
        """Current pose (4x4 body-to-world)."""
        return self._pose.copy()
