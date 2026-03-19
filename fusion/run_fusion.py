"""
Multi-sensor fusion pipeline (Stage 4): load or generate trajectories, fuse, save/plot.

Fuses visual and LiDAR trajectories in a 2D pose graph to demonstrate the
multi-sensor fusion architecture. For production, extend with 3D pose graph
and GTSAM/Ceres.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from fusion.io import load_lidar_trajectory, load_vo_trajectory_from_map, save_trajectory_2d
from fusion.lidar_vision_fusion import fuse_vo_lidar_trajectories
from fusion.poses import pose_2d_to_4x4

logger = logging.getLogger("opl.fusion")


def generate_synthetic_trajectories(
    n_frames: int = 10,
    vo_noise: float = 0.02,
    lidar_noise: float = 0.01,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """
    Generate two synthetic 2D trajectories (same path, different drift/noise)
    as 4x4 poses. Used for fusion demo without real data.
    """
    # Ground truth: simple path in x-z plane (y=0)
    gt_2d = [(0.0, 0.0, 0.0)]
    for k in range(1, n_frames):
        x = 0.1 * k + 0.02 * np.sin(k * 0.5)
        y = 0.05 * k
        th = 0.1 * np.sin(k * 0.3)
        gt_2d.append((x, y, th))
    vo_poses = []
    lidar_poses = []
    for k, (x, y, th) in enumerate(gt_2d):
        # VO: add noise/drift
        xv = x + vo_noise * (k + 1) + 0.01 * np.random.randn()
        yv = y + vo_noise * 0.5 * k + 0.01 * np.random.randn()
        thv = th + 0.01 * np.random.randn()
        vo_poses.append(pose_2d_to_4x4(xv, yv, thv))
        # LiDAR: different noise
        xl = x + lidar_noise * k + 0.005 * np.random.randn()
        yl = y + lidar_noise * 0.3 * k + 0.005 * np.random.randn()
        thl = th + 0.005 * np.random.randn()
        lidar_poses.append(pose_2d_to_4x4(xl, yl, thl))
    return vo_poses, lidar_poses


def run_fusion_demo(
    output_path: str | Path | None = None,
    n_frames: int = 15,
    max_iter: int = 20,
) -> int:
    """
    Run fusion on synthetic trajectories and optionally save fused 2D trajectory.
    """
    vo_poses, lidar_poses = generate_synthetic_trajectories(n_frames=n_frames)
    fused_2d = fuse_vo_lidar_trajectories(
        vo_poses, lidar_poses, prior_use="vo", max_iter=max_iter
    )
    logger.info("Fused trajectory: %d poses", len(fused_2d))
    if output_path is not None:
        save_trajectory_2d(fused_2d, output_path)
        logger.info("Saved fused trajectory to %s", output_path)
    return 0


def main(
    vo_map_path: str | Path | None = None,
    lidar_trajectory_path: str | Path | None = None,
    output_path: str | Path | None = None,
    demo: bool = False,
    demo_frames: int = 15,
    prior_use: str = "vo",
    max_iter: int = 20,
) -> int:
    """
    Fuse VO and LiDAR trajectories.
    Either provide vo_map_path + lidar_trajectory_path, or use demo=True for synthetic data.
    """
    if demo:
        return run_fusion_demo(output_path=output_path, n_frames=demo_frames, max_iter=max_iter)
    if vo_map_path is None or lidar_trajectory_path is None:
        logger.error("Provide both --vo-map and --lidar-trajectory, or use --demo")
        return 1
    vo_map_path = Path(vo_map_path)
    lidar_trajectory_path = Path(lidar_trajectory_path)
    if not vo_map_path.exists():
        logger.error("VO map not found: %s", vo_map_path)
        return 1
    if not lidar_trajectory_path.exists():
        logger.error("LiDAR trajectory not found: %s", lidar_trajectory_path)
        return 1
    vo_poses = load_vo_trajectory_from_map(vo_map_path)
    lidar_poses = load_lidar_trajectory(lidar_trajectory_path)
    if len(vo_poses) == 0:
        logger.error("No keyframes in VO map")
        return 1
    if len(lidar_poses) == 0:
        logger.error("No poses in LiDAR trajectory")
        return 1
    fused_2d = fuse_vo_lidar_trajectories(
        vo_poses, lidar_poses, prior_use=prior_use, max_iter=max_iter
    )
    logger.info("Fused trajectory: %d poses", len(fused_2d))
    if output_path is not None:
        save_trajectory_2d(fused_2d, output_path)
        logger.info("Saved fused trajectory to %s", output_path)
    return 0


if __name__ == "__main__":
    import argparse
    from openperceptionlab.logging_utils import setup_logging

    setup_logging("INFO")
    p = argparse.ArgumentParser(description="LiDAR-Vision fusion (Stage 4)")
    p.add_argument("--vo-map", type=str, default=None, help="Visual SLAM map .npz (keyframe poses)")
    p.add_argument("--lidar-trajectory", type=str, default=None, help="LiDAR trajectory .npz")
    p.add_argument("--output", "-o", type=str, default=None, help="Output fused 2D trajectory .npz")
    p.add_argument("--demo", action="store_true", help="Run on synthetic trajectories")
    p.add_argument("--demo-frames", type=int, default=15, help="Number of frames for demo")
    p.add_argument("--prior", type=str, default="vo", choices=["vo", "lidar"], help="Which trajectory to fix as prior")
    p.add_argument("--max-iter", type=int, default=20, help="Pose graph max iterations")
    args = p.parse_args()
    raise SystemExit(
        main(
            vo_map_path=args.vo_map,
            lidar_trajectory_path=args.lidar_trajectory,
            output_path=args.output,
            demo=args.demo,
            demo_frames=args.demo_frames,
            prior_use=args.prior,
            max_iter=args.max_iter,
        )
    )
