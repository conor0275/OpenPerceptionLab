"""
I/O for 3D reconstruction: load images, save point clouds (PLY) and camera poses.
"""
from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger("opl.reconstruction.io")


def load_images_from_dir(
    dir_path: str | Path,
    pattern: str = "*",
    sort: bool = True,
) -> list[tuple[Path, np.ndarray]]:
    """Load images from directory. Returns list of (path, BGR image)."""
    dir_path = Path(dir_path)
    if not dir_path.is_dir():
        raise FileNotFoundError(f"Not a directory: {dir_path}")
    paths = list(dir_path.glob(pattern))
    paths = [p for p in paths if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")]
    paths = [p for p in paths if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")]
    if sort:
        paths.sort(key=lambda p: p.name)
    out = []
    for p in paths:
        img = cv2.imread(str(p))
        if img is None:
            logger.warning("Skip unreadable image: %s", p)
            continue
        out.append((p, img))
    return out


def default_intrinsics_from_image(img: np.ndarray) -> np.ndarray:
    """Default 3x3 K: fx=fy=width, cx=width/2, cy=height/2."""
    h, w = img.shape[:2]
    fx = fy = float(w)
    cx, cy = w / 2.0, h / 2.0
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)


def save_ply(
    path: str | Path,
    points: np.ndarray,
    colors: np.ndarray | None = None,
) -> None:
    """Save point cloud to PLY (ASCII). points: Nx3, colors optional Nx3 uint8 RGB."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = len(points)
    has_color = colors is not None and len(colors) == n
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write("element vertex %d\n" % n)
        f.write("property float x\nproperty float y\nproperty float z\n")
        if has_color:
            f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for i in range(n):
            x, y, z = points[i]
            if has_color:
                r, g, b = colors[i].clip(0, 255).astype(int)
                f.write("%f %f %f %d %d %d\n" % (x, y, z, r, g, b))
            else:
                f.write("%f %f %f\n" % (x, y, z))
    logger.info("Saved PLY: %s (%d points)", path, n)


def save_poses_npz(path: str | Path, R_list: list[np.ndarray], t_list: list[np.ndarray]) -> None:
    """Save camera poses (world-to-camera R, t) to .npz."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    R_arr = np.array(R_list, dtype=np.float64)
    t_arr = np.array([t.ravel() for t in t_list], dtype=np.float64)
    np.savez_compressed(path, R=R_arr, t=t_arr)
    logger.info("Saved poses: %s (%d cameras)", path, len(R_list))
