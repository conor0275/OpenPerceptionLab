from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path

from openperceptionlab.logging_utils import setup_logging


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="openperceptionlab",
        description="OpenPerceptionLab unified entrypoint (camera-first).",
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (DEBUG/INFO/WARNING/ERROR). Default: INFO",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    slam = sub.add_parser("slam", help="Run real-time monocular SLAM (camera).")
    slam.add_argument("--camera", type=int, default=0, help="Camera index (default: 0).")
    slam.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config YAML/JSON (optional).",
    )
    slam.add_argument("--load-map", type=str, default=None, help="Load map from .npz file before running.")
    slam.add_argument("--save-map", type=str, default=None, help="Save map to .npz file on exit.")

    lidar_slam = sub.add_parser("lidar-slam", help="LiDAR SLAM from PCD sequence (Stage 3).")
    lidar_slam.add_argument("sequence_dir", type=str, nargs="?", default=None, help="Directory of PCD files (or use --sample).")
    lidar_slam.add_argument("--output-map", "-o", type=str, default="lidar_map.pcd", help="Output map path.")
    lidar_slam.add_argument("--voxel-size", type=float, default=0.05, help="Voxel size for map.")
    lidar_slam.add_argument("--max-correspondence-distance", type=float, default=0.5, help="ICP max correspondence distance.")
    lidar_slam.add_argument("--show", action="store_true", help="Show live map window.")
    lidar_slam.add_argument("--sample", action="store_true", help="Generate sample PCDs in ./sample_pcds and run on them.")
    lidar_slam.add_argument("--pattern", type=str, default="*.pcd", help="Glob pattern for PCD files.")
    lidar_slam.add_argument("--save-trajectory", type=str, default=None, help="Save LiDAR trajectory to .npz for fusion (Stage 4).")

    fusion_cmd = sub.add_parser("fusion", help="LiDAR-Vision pose graph fusion (Stage 4).")
    fusion_cmd.add_argument("--vo-map", type=str, default=None, help="Visual SLAM map .npz (keyframe poses).")
    fusion_cmd.add_argument("--lidar-trajectory", type=str, default=None, help="LiDAR trajectory .npz from lidar-slam --save-trajectory.")
    fusion_cmd.add_argument("--output", "-o", type=str, default=None, help="Output fused 2D trajectory .npz.")
    fusion_cmd.add_argument("--demo", action="store_true", help="Run on synthetic trajectories (no data files).")
    fusion_cmd.add_argument("--demo-frames", type=int, default=15, help="Number of frames for --demo.")
    fusion_cmd.add_argument("--prior", type=str, default="vo", choices=["vo", "lidar"], help="Which trajectory to fix as prior.")
    fusion_cmd.add_argument("--max-iter", type=int, default=20, help="Pose graph max iterations.")

    demos = sub.add_parser("demo", help="Run demos (vision/geometry).")
    demos.add_argument("--image", type=str, default=None, help="Input image path (single-image demos).")
    demos.add_argument("--camera", type=int, default=None, help="Camera device index (demos that support live input).")
    demos.add_argument("--image1", type=str, default=None, help="First image path (two-image demos).")
    demos.add_argument("--image2", type=str, default=None, help="Second image path (two-image demos).")
    demos_sub = demos.add_subparsers(dest="demo", required=True)

    demos_sub.add_parser("detect", help="Run detection demo.").set_defaults(
        _script="demos/detect_demo.py"
    )
    demos_sub.add_parser("segment", help="Run segmentation demo.").set_defaults(
        _script="demos/segmentation_demo.py"
    )
    demos_sub.add_parser("depth", help="Run depth demo.").set_defaults(
        _script="demos/depth_demo.py"
    )
    demos_sub.add_parser("perception", help="Run perception pipeline demo.").set_defaults(
        _script="demos/perception_demo.py"
    )
    demos_sub.add_parser("camera", help="Run camera projection demo.").set_defaults(
        _script="demos/camera_projection_demo.py"
    )
    demos_sub.add_parser("epipolar", help="Run epipolar geometry demo.").set_defaults(
        _script="demos/epipolar_demo.py"
    )
    demos_sub.add_parser("triangulation", help="Run triangulation demo.").set_defaults(
        _script="demos/triangulation_demo.py"
    )
    demos_sub.add_parser("pose", help="Run pose estimation demo.").set_defaults(
        _script="demos/pose_demo.py"
    )
    demos_sub.add_parser("feature_match", help="Run feature matching demo.").set_defaults(
        _script="demos/feature_match_demo.py"
    )

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(args.log_level)

    if args.cmd == "slam":
        from slam.run_slam import main as slam_main

        return int(
            slam_main(
                camera_index=args.camera,
                config_path=args.config,
                load_map_path=args.load_map,
                save_map_path=args.save_map,
            )
        )

    if args.cmd == "lidar-slam":
        from lidar.run_lidar_slam import main as lidar_main

        if getattr(args, "sample", False):
            from pathlib import Path
            from lidar.sample_data import generate_sample_sequence
            sample_dir = Path("sample_pcds")
            generate_sample_sequence(sample_dir)
            seq_dir = sample_dir
        else:
            seq_dir = getattr(args, "sequence_dir", None)
            if not seq_dir:
                print("Provide sequence_dir or use --sample to generate sample PCDs.", file=sys.stderr)
                return 1
        return int(
            lidar_main(
                sequence_dir=seq_dir,
                output_map=args.output_map,
                voxel_size=args.voxel_size,
                max_correspondence_distance=args.max_correspondence_distance,
                show_live=args.show,
                pattern=args.pattern,
                save_trajectory_path=getattr(args, "save_trajectory", None),
            )
        )

    if args.cmd == "fusion":
        from fusion.run_fusion import main as fusion_main

        return int(
            fusion_main(
                vo_map_path=getattr(args, "vo_map", None),
                lidar_trajectory_path=getattr(args, "lidar_trajectory", None),
                output_path=getattr(args, "output", None),
                demo=getattr(args, "demo", False),
                demo_frames=getattr(args, "demo_frames", 15),
                prior_use=getattr(args, "prior", "vo"),
                max_iter=getattr(args, "max_iter", 20),
            )
        )

    if args.cmd == "demo":
        import os

        if getattr(args, "image", None) is not None:
            os.environ["OPL_DEMO_IMAGE"] = args.image
        if getattr(args, "camera", None) is not None:
            os.environ["OPL_DEMO_CAMERA"] = str(args.camera)
        if getattr(args, "image1", None) is not None:
            os.environ["OPL_DEMO_IMAGE1"] = args.image1
        if getattr(args, "image2", None) is not None:
            os.environ["OPL_DEMO_IMAGE2"] = args.image2
        script_path = (Path(__file__).resolve().parents[1] / args._script).resolve()
        sys.argv = [str(script_path)]
        runpy.run_path(str(script_path), run_name="__main__")
        return 0

    raise SystemExit("Unknown command")


if __name__ == "__main__":
    raise SystemExit(main())

