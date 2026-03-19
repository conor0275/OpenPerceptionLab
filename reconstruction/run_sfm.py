"""
SfM pipeline: load images, run incremental SfM, save point cloud and poses (Stage 5).
"""
from __future__ import annotations

import logging
from pathlib import Path

from reconstruction.io import (
    load_images_from_dir,
    default_intrinsics_from_image,
    save_ply,
    save_poses_npz,
)
from reconstruction.sfm import IncrementalSfM

logger = logging.getLogger("opl.reconstruction")


def main(
    images_dir: str | Path,
    output_ply: str | Path | None = None,
    output_poses: str | Path | None = None,
    min_matches: int = 50,
    image_pattern: str = "*",
) -> int:
    """
    Run incremental SfM on a directory of images. Saves sparse point cloud (PLY) and camera poses (npz).
    """
    images_dir = Path(images_dir)
    image_list = load_images_from_dir(images_dir, pattern=image_pattern)
    if len(image_list) < 2:
        logger.error("Need at least 2 images in %s", images_dir)
        return 1
    _, img0 = image_list[0]
    K = default_intrinsics_from_image(img0)
    sfm = IncrementalSfM(K, min_matches=min_matches)
    if not sfm.add_first_two_views(img0, image_list[1][1]):
        logger.error("SfM init failed (first two views)")
        return 1
    for path, img in image_list[2:]:
        sfm.add_view(img)
    points = sfm.get_points_array()
    if len(points) == 0:
        logger.error("No 3D points reconstructed")
        return 1
    if output_ply:
        save_ply(Path(output_ply), points)
    R_list, t_list = sfm.get_poses()
    if output_poses:
        save_poses_npz(Path(output_poses), R_list, t_list)
    logger.info("SfM done: %d points, %d cameras", len(points), len(R_list))
    return 0


if __name__ == "__main__":
    import argparse
    from openperceptionlab.logging_utils import setup_logging

    setup_logging("INFO")
    p = argparse.ArgumentParser(description="Incremental SfM (Stage 5)")
    p.add_argument("images_dir", type=str, help="Directory of images")
    p.add_argument("--output", "-o", type=str, default="sfm_pointcloud.ply", help="Output PLY path")
    p.add_argument("--poses", type=str, default=None, help="Output poses .npz path")
    p.add_argument("--min-matches", type=int, default=50, help="Min matches for two-view init")
    p.add_argument("--pattern", type=str, default="*.jpg", help="Image glob pattern")
    args = p.parse_args()
    raise SystemExit(
        main(
            args.images_dir,
            output_ply=args.output,
            output_poses=args.poses,
            min_matches=args.min_matches,
            image_pattern=args.pattern,
        )
    )
