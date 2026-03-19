"""
Pose conversion utilities for fusion.

Unifies representations: 4x4 (T_body_to_world), (R, t) world-to-camera,
and 2D (x, y, theta) for pose graph in the horizontal plane.
"""
from __future__ import annotations

import numpy as np


def pose_4x4_from_Rt(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Build 4x4 body-to-world pose from world-to-camera (R, t).
    Convention: p_cam = R @ p_world + t  =>  p_world = R.T @ (p_cam - t) = R.T @ p_cam - R.T @ t.
    So body (camera) to world: R_b2w = R.T, t_b2w = -R.T @ t.
    Returns T such that p_world = T @ p_body (homogeneous).
    """
    R = np.asarray(R, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64).ravel()[:3]
    R_b2w = R.T
    t_b2w = -R_b2w @ t
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R_b2w
    T[:3, 3] = t_b2w
    return T


def pose_4x4_to_2d(T: np.ndarray) -> tuple[float, float, float]:
    """
    Extract 2D pose (x, y, theta) from 4x4 body-to-world.
    Uses x = T[0,3], y = T[2,3] (forward is Z), theta = atan2(R[2,0], R[0,0]) (yaw).
    """
    T = np.asarray(T, dtype=np.float64)
    x = float(T[0, 3])
    y = float(T[2, 3])
    R = T[:3, :3]
    # Yaw from R: we use R[0,0]=cos(th), R[2,0]=-sin(th) in pose_2d_to_4x4, so th = atan2(-R[2,0], R[0,0])
    theta = float(np.arctan2(-R[2, 0], R[0, 0]))
    return x, y, theta


def pose_2d_to_4x4(x: float, y: float, theta: float) -> np.ndarray:
    """Build 4x4 body-to-world from 2D (x, y, yaw). Yaw around Y axis; x,z in world."""
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([
        [c, 0, s],
        [0, 1, 0],
        [-s, 0, c],
    ], dtype=np.float64)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[0, 3] = x
    T[1, 3] = 0.0
    T[2, 3] = y
    return T


def relative_pose_2d(x_i: float, y_i: float, th_i: float,
                     x_j: float, y_j: float, th_j: float) -> tuple[float, float, float]:
    """
    Relative pose from frame i to frame j in 2D: (dx_ij, dy_ij, dth_ij)
    such that in frame i, j is at (dx_ij, dy_ij) with orientation th_i + dth_ij.
    """
    ci, si = np.cos(th_i), np.sin(th_i)
    dx = ci * (x_j - x_i) + si * (y_j - y_i)
    dy = -si * (x_j - x_i) + ci * (y_j - y_i)
    dth = th_j - th_i
    return dx, dy, dth


def compose_pose_2d(x_i: float, y_i: float, th_i: float,
                     dx: float, dy: float, dth: float) -> tuple[float, float, float]:
    """Compose: (x_j, y_j, th_j) = (x_i, y_i, th_i) + (dx, dy, dth) in frame i."""
    ci, si = np.cos(th_i), np.sin(th_i)
    x_j = x_i + ci * dx - si * dy
    y_j = y_i + si * dx + ci * dy
    th_j = th_i + dth
    return x_j, y_j, th_j
