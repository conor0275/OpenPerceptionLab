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

