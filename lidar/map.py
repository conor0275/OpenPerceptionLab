"""Global point cloud map: accumulate scans with poses, voxel downsample."""
from __future__ import annotations

import numpy as np
import open3d as o3d


class PointCloudMap:
    """Accumulate point clouds in world frame with optional voxel downsampling."""

    def __init__(self, voxel_size: float = 0.05) -> None:
        self.voxel_size = float(voxel_size)
        self._global_pcd = o3d.geometry.PointCloud()
        self._trajectory: list[np.ndarray] = []  # list of 4x4 poses

    def add_scan(self, pcd: o3d.geometry.PointCloud, pose: np.ndarray) -> None:
        """Transform scan to world frame and merge into global map."""
        pcd_world = pcd.transform(pose)
        self._global_pcd += pcd_world
        self._trajectory.append(pose.copy())

    def get_global_pcd(self, downsample: bool = True) -> o3d.geometry.PointCloud:
        """Return global point cloud, optionally voxel-downsampled."""
        if downsample and self.voxel_size > 0 and len(self._global_pcd.points) > 0:
            return self._global_pcd.voxel_down_sample(self.voxel_size)
        return o3d.geometry.PointCloud(self._global_pcd)

    @property
    def trajectory(self) -> list[np.ndarray]:
        return self._trajectory
