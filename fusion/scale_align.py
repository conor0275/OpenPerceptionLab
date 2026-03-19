"""
Scale alignment: align monocular VO trajectory to LiDAR scale (mitigate Stage 2 scale ambiguity).

Given two trajectories (e.g. VO and LiDAR) with same number of frames, compute scale s
so that scaled VO displacements best match LiDAR displacements (least squares).
"""
from __future__ import annotations

import logging
import numpy as np

logger = logging.getLogger("opl.fusion.scale_align")


def align_scale_2d(
    vo_poses_2d: list[tuple[float, float, float]],
    lidar_poses_2d: list[tuple[float, float, float]],
) -> float:
    """
    Compute scale s such that |s * vo_displacement| approximates |lidar_displacement|
    in least-squares sense over consecutive frames. Returns s (positive).
    """
    n = min(len(vo_poses_2d), len(lidar_poses_2d))
    if n < 2:
        return 1.0
    vo = np.array(vo_poses_2d[:n], dtype=np.float64)
    li = np.array(lidar_poses_2d[:n], dtype=np.float64)
    d_vo = np.diff(vo[:, :2], axis=0)  # (n-1, 2)
    d_li = np.diff(li[:, :2], axis=0)
    n_vo = np.linalg.norm(d_vo, axis=1, keepdims=True)
    n_li = np.linalg.norm(d_li, axis=1, keepdims=True)
    n_vo = np.where(n_vo < 1e-9, 1e-9, n_vo)
    n_li = np.where(n_li < 1e-9, 1e-9, n_li)
    # s * n_vo ~ n_li  =>  s = (n_li / n_vo). Mean over segments.
    s = (n_li / n_vo).ravel()
    s = s[np.isfinite(s) & (s > 0)]
    if len(s) == 0:
        return 1.0
    return float(np.median(s))


def scaled_vo_trajectory_2d(
    vo_poses_2d: list[tuple[float, float, float]],
    scale: float,
    origin: tuple[float, float, float] | None = None,
) -> list[tuple[float, float, float]]:
    """Return VO trajectory with positions scaled and optionally shifted to origin (first pose)."""
    out = []
    ox, oy, ot = origin or (vo_poses_2d[0][0], vo_poses_2d[0][1], vo_poses_2d[0][2])
    for x, y, th in vo_poses_2d:
        out.append(((x - ox) * scale + ox, (y - oy) * scale + oy, th))
    return out
