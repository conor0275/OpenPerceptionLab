"""Point cloud I/O (PCD/PLY) via Open3D."""
from __future__ import annotations

from pathlib import Path

import open3d as o3d


def load_pcd(path: str | Path) -> o3d.geometry.PointCloud:
    """Load point cloud from PCD or PLY file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Point cloud not found: {path}")
    pcd = o3d.io.read_point_cloud(str(path))
    return pcd


def save_pcd(pcd: o3d.geometry.PointCloud, path: str | Path) -> None:
    """Save point cloud to PCD file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(path), pcd, write_ascii=False)


def load_pcd_sequence(
    directory: str | Path,
    pattern: str = "*.pcd",
    sort: bool = True,
) -> list[Path]:
    """List PCD files in a directory, optionally sorted by name."""
    directory = Path(directory)
    if not directory.is_dir():
        raise NotADirectoryError(f"Not a directory: {directory}")
    paths = list(directory.glob(pattern))
    if sort:
        paths.sort(key=lambda p: p.name)
    return paths
