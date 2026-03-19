"""Point cloud registration (ICP) via Open3D."""
from __future__ import annotations

import numpy as np
import open3d as o3d


def icp_point_to_point(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    init_pose: np.ndarray | None = None,
    max_correspondence_distance: float = 0.5,
    max_iteration: int = 50,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run point-to-point ICP. Returns (4x4 transformation, 6D fitness rmse).
    """
    if init_pose is None:
        init_pose = np.eye(4)
    result = o3d.pipelines.registration.registration_icp(
        source,
        target,
        max_correspondence_distance,
        init_pose,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iteration),
    )
    return result.transformation, np.array([result.fitness, result.inlier_rmse])


def icp_point_to_plane(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    init_pose: np.ndarray | None = None,
    max_correspondence_distance: float = 0.5,
    max_iteration: int = 50,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run point-to-plane ICP (requires normals). Returns (4x4 transformation, 6D fitness rmse).
    """
    if init_pose is None:
        init_pose = np.eye(4)
    if not source.has_normals():
        source.estimate_normals()
    if not target.has_normals():
        target.estimate_normals()
    result = o3d.pipelines.registration.registration_icp(
        source,
        target,
        max_correspondence_distance,
        init_pose,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iteration),
    )
    return result.transformation, np.array([result.fitness, result.inlier_rmse])
