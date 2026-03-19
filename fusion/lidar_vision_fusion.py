"""
LiDAR-Vision fusion via 2D pose graph.

Takes two trajectories (e.g. from visual SLAM keyframes and LiDAR odometry),
converts to 2D (x, y, yaw), builds a pose graph with both as between-factor
sources, optimizes, and returns fused trajectory. This demonstrates the
multi-sensor fusion architecture; for 3D or IMU, extend with GTSAM/Ceres.
"""
from __future__ import annotations

import logging
import numpy as np

from fusion.pose_graph_2d import PoseGraph2D
from fusion.poses import pose_4x4_to_2d, relative_pose_2d

logger = logging.getLogger("opl.fusion.lidar_vision")


def trajectories_to_2d(poses_4x4: list[np.ndarray]) -> list[tuple[float, float, float]]:
    """Convert list of 4x4 body-to-world poses to 2D (x, y, theta)."""
    out = []
    for T in poses_4x4:
        out.append(pose_4x4_to_2d(np.asarray(T)))
    return out


def fuse_trajectories_2d(
    vo_poses_2d: list[tuple[float, float, float]],
    lidar_poses_2d: list[tuple[float, float, float]],
    *,
    prior_use: str = "vo",
    max_iter: int = 20,
) -> list[tuple[float, float, float]]:
    """
    Fuse VO and LiDAR 2D trajectories in a pose graph.

    - prior_use: "vo" or "lidar" — which trajectory to use for prior on first node.
    - Adds between factors for VO chain and LiDAR chain; same node indices
      (trajectories must have same length; if not, we use the shorter length).
    """
    n = min(len(vo_poses_2d), len(lidar_poses_2d))
    if n == 0:
        return []
    graph = PoseGraph2D()
    for i in range(n):
        # Initialize node with average of VO and LiDAR at this index
        xv, yv, thv = vo_poses_2d[i]
        xl, yl, thl = lidar_poses_2d[i]
        graph.add_node(0.5 * (xv + xl), 0.5 * (yv + yl), 0.5 * (thv + thl))
    if prior_use == "lidar":
        x0, y0, th0 = lidar_poses_2d[0]
    else:
        x0, y0, th0 = vo_poses_2d[0]
    graph.set_prior(x0, y0, th0)
    # VO between factors
    for i in range(n - 1):
        dx, dy, dth = relative_pose_2d(
            vo_poses_2d[i][0], vo_poses_2d[i][1], vo_poses_2d[i][2],
            vo_poses_2d[i + 1][0], vo_poses_2d[i + 1][1], vo_poses_2d[i + 1][2],
        )
        graph.add_between(i, i + 1, dx, dy, dth)
    # LiDAR between factors (same nodes, different measurements)
    for i in range(n - 1):
        dx, dy, dth = relative_pose_2d(
            lidar_poses_2d[i][0], lidar_poses_2d[i][1], lidar_poses_2d[i][2],
            lidar_poses_2d[i + 1][0], lidar_poses_2d[i + 1][1], lidar_poses_2d[i + 1][2],
        )
        graph.add_between(i, i + 1, dx, dy, dth)
    return graph.optimize(max_iter=max_iter)


def fuse_vo_lidar_trajectories(
    vo_poses_4x4: list[np.ndarray],
    lidar_poses_4x4: list[np.ndarray],
    *,
    prior_use: str = "vo",
    max_iter: int = 20,
) -> list[tuple[float, float, float]]:
    """
    Fuse VO and LiDAR trajectories given as 4x4 body-to-world poses.
    Converts to 2D, builds pose graph with both chains as factors, optimizes.
    """
    vo_2d = trajectories_to_2d(vo_poses_4x4)
    lidar_2d = trajectories_to_2d(lidar_poses_4x4)
    return fuse_trajectories_2d(vo_2d, lidar_2d, prior_use=prior_use, max_iter=max_iter)
