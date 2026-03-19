"""Generate a small synthetic PCD sequence for testing LiDAR SLAM."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import open3d as o3d

from lidar.io import save_pcd


def generate_sample_sequence(output_dir: str | Path, num_frames: int = 5) -> None:
    """
    Write num_frames synthetic PCDs with slight motion (so ICP has something to register).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    for i in range(num_frames):
        # Simple cube of points, slightly translated each frame
        x = rng.uniform(0, 2, 500)
        y = rng.uniform(0, 2, 500)
        z = rng.uniform(0, 2, 500)
        pts = np.column_stack([x, y, z]).astype(np.float64)
        # Move frame slightly so ICP can register
        pts[:, 0] += i * 0.3
        pts[:, 1] += i * 0.1
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        save_pcd(pcd, output_dir / f"frame_{i:04d}.pcd")
    print(f"Wrote {num_frames} PCDs to {output_dir}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("output_dir", type=str, default="sample_pcds", nargs="?")
    p.add_argument("-n", "--num-frames", type=int, default=5)
    args = p.parse_args()
    generate_sample_sequence(args.output_dir, args.num_frames)
