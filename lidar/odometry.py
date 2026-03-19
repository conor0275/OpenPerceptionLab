"""LiDAR odometry: frame-to-frame ICP, optionally frame-to-model."""
from __future__ import annotations

import logging
import numpy as np
import open3d as o3d

from lidar.registration import icp_point_to_point

logger = logging.getLogger("opl.lidar.odometry")


class LiDAROdometry:
    """Frame-to-frame ICP odometry. Optionally use point-to-plane for better accuracy."""

    def __init__(
        self,
        max_correspondence_distance: float = 0.5,
        max_iteration: int = 50,
        use_point_to_plane: bool = False,
    ) -> None:
        self.max_correspondence_distance = max_correspondence_distance
        self.max_iteration = max_iteration
        self.use_point_to_plane = use_point_to_plane
        self._prev_pcd: o3d.geometry.PointCloud | None = None
        self._pose_world: np.ndarray = np.eye(4)  # current pose in world

    def process(self, pcd: o3d.geometry.PointCloud) -> np.ndarray:
        """
        Register current scan to previous; update and return world pose (4x4).
        First frame: identity. Later: prev_pose @ icp(prev, curr).
        """
        if self._prev_pcd is None:
            self._prev_pcd = pcd
            return self._pose_world.copy()

        if self.use_point_to_plane:
            from lidar.registration import icp_point_to_plane
            T_delta, _ = icp_point_to_plane(
                pcd,
                self._prev_pcd,
                init_pose=np.eye(4),
                max_correspondence_distance=self.max_correspondence_distance,
                max_iteration=self.max_iteration,
            )
        else:
            T_delta, _ = icp_point_to_point(
                pcd,
                self._prev_pcd,
                init_pose=np.eye(4),
                max_correspondence_distance=self.max_correspondence_distance,
                max_iteration=self.max_iteration,
            )

        # T_world_curr = T_world_prev @ T_prev_curr  (prev in world, curr relative to prev)
        # T_prev_curr from ICP: target=prev, source=curr -> transforms source to target, so curr -> prev
        # So T_world_curr = T_world_prev @ inv(T_delta)? No. ICP gives T such that source * T = target (in same frame).
        # So curr * T_delta = prev  =>  curr = prev * inv(T_delta). In world: curr_world = prev_world * inv(T_delta)?
        # Convention: we want pose such that p_world = pose @ p_curr. So pose = T_world_curr.
        # prev_world = self._pose_world (pose of prev frame). curr in prev frame: p_prev = T_delta @ p_curr, so p_curr = inv(T_delta) @ p_prev.
        # p_world = pose_prev @ p_prev = pose_prev @ T_delta @ p_curr  =>  pose_curr = pose_prev @ T_delta. So we use T_delta as "prev to curr" in local frame.
        # Actually Open3D ICP: result.transformation is T that aligns source to target, i.e. source_transform = T @ source, and we want source_transform ≈ target.
        # So T transforms source (curr) into target (prev) frame. So T is T_prev_curr (curr in prev frame). So p_prev = T @ p_curr.
        # p_world = pose_prev @ p_prev = pose_prev @ T @ p_curr  =>  pose_curr = pose_prev @ T. So:
        self._pose_world = self._pose_world @ T_delta
        self._prev_pcd = pcd
        return self._pose_world.copy()
