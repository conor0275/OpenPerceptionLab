"""
Simple IMU propagation (integrate gyro + acc) to mitigate Stage 3/4: no IMU.

Euler integration: orientation from gyro, position/velocity from acc in world frame.
For full VIO use preintegration (e.g. GTSAM) later.
"""
from __future__ import annotations

import logging
import numpy as np

logger = logging.getLogger("opl.fusion.imu")


def propagate_pose(
    pose_4x4: np.ndarray,
    velocity: np.ndarray,
    gyro: np.ndarray,
    acc: np.ndarray,
    dt: float,
    gravity: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Propagate pose and velocity using IMU. All in world frame.
    pose_4x4: 4x4 body-to-world. velocity: 3, world. gyro/acc: 3, body (rad/s, m/s^2).
    Returns (new_pose_4x4, new_velocity_3).
    """
    if gravity is None:
        gravity = np.array([0, -9.81, 0], dtype=np.float64)
    R = np.asarray(pose_4x4[:3, :3], dtype=np.float64)
    t = np.asarray(pose_4x4[:3, 3], dtype=np.float64)
    # Rotation: R_new = R @ exp(omega * dt). First-order: exp(w*dt) ≈ I + skew(w)*dt
    w = np.asarray(gyro, dtype=np.float64).ravel()[:3]
    wx, wy, wz = w[0], w[1], w[2]
    skew = np.array([[0, -wz, wy], [wz, 0, -wx], [-wy, wx, 0]], dtype=np.float64) * dt
    R_delta = np.eye(3) + skew
    R_new = R @ R_delta
    R_new = R_new / np.linalg.norm(R_new, axis=0, keepdims=True)
    # Acc in world: a_world = R @ acc + gravity. v_new = v + a_world * dt, t_new = t + v*dt + 0.5*a*dt^2
    a_world = R @ np.asarray(acc, dtype=np.float64).ravel()[:3] + gravity
    v_new = velocity + a_world * dt
    t_new = t + velocity * dt + 0.5 * a_world * (dt ** 2)
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = R_new
    out[:3, 3] = t_new
    return out, v_new
