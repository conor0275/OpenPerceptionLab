"""LiDAR SLAM pipeline: PCD sequence → ICP odometry → global map → save."""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import open3d as o3d

from lidar.io import load_pcd, load_pcd_sequence, save_pcd
from lidar.map import PointCloudMap
from lidar.odometry import LiDAROdometry

logger = logging.getLogger("opl.lidar")


def main(
    sequence_dir: str | Path,
    output_map: str | Path,
    voxel_size: float = 0.05,
    max_correspondence_distance: float = 0.5,
    show_live: bool = False,
    pattern: str = "*.pcd",
    save_trajectory_path: str | Path | None = None,
) -> int:
    """
    Run LiDAR SLAM on a directory of PCD files. Saves global map to output_map.
    If save_trajectory_path is set, saves trajectory as .npz (poses Nx4x4) for fusion.
    """
    sequence_dir = Path(sequence_dir)
    output_map = Path(output_map)
    paths = load_pcd_sequence(sequence_dir, pattern=pattern)
    if len(paths) == 0:
        logger.error("No PCD files found in %s with pattern %s", sequence_dir, pattern)
        return 1

    odom = LiDAROdometry(max_correspondence_distance=max_correspondence_distance)
    map_ = PointCloudMap(voxel_size=voxel_size)
    vis = None
    if show_live:
        vis = o3d.visualization.Visualizer()
        vis.create_window("LiDAR Map")

    for i, p in enumerate(paths):
        pcd = load_pcd(p)
        if len(pcd.points) == 0:
            logger.warning("Empty point cloud: %s", p)
            continue
        # Optional: voxel downsample for speed
        if voxel_size > 0:
            pcd = pcd.voxel_down_sample(voxel_size)
        pose = odom.process(pcd)
        map_.add_scan(pcd, pose)
        logger.info("Frame %d/%d %s pose updated", i + 1, len(paths), p.name)
        if vis is not None:
            global_pcd = map_.get_global_pcd(downsample=True)
            vis.clear_geometries()
            vis.add_geometry(global_pcd)
            vis.poll_events()
            vis.update_renderer()

    output_map.parent.mkdir(parents=True, exist_ok=True)
    final_pcd = map_.get_global_pcd(downsample=True)
    save_pcd(final_pcd, output_map)
    logger.info("Saved map to %s (%d points)", output_map, len(final_pcd.points))
    if save_trajectory_path is not None:
        traj = map_.trajectory
        if len(traj) > 0:
            poses_arr = np.array(traj, dtype=np.float64)
            np.savez_compressed(save_trajectory_path, poses=poses_arr)
            logger.info("Saved trajectory to %s (%d poses)", save_trajectory_path, len(traj))
    if vis is not None:
        vis.destroy_window()
    return 0


if __name__ == "__main__":
    import argparse
    from openperceptionlab.logging_utils import setup_logging

    setup_logging("INFO")
    p = argparse.ArgumentParser(description="LiDAR SLAM from PCD sequence")
    p.add_argument("sequence_dir", type=str, help="Directory containing PCD files")
    p.add_argument("--output-map", "-o", type=str, default="lidar_map.pcd", help="Output map path")
    p.add_argument("--voxel-size", type=float, default=0.05, help="Voxel size for map and downsampling")
    p.add_argument("--max-correspondence-distance", type=float, default=0.5, help="ICP max correspondence distance")
    p.add_argument("--show", action="store_true", help="Show live map window")
    p.add_argument("--pattern", type=str, default="*.pcd", help="Glob pattern for PCD files")
    args = p.parse_args()
    raise SystemExit(main(args.sequence_dir, args.output_map, args.voxel_size, args.max_correspondence_distance, args.show, args.pattern))
