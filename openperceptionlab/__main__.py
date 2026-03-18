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

    demos = sub.add_parser("demo", help="Run demos (vision/geometry).")
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

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(args.log_level)

    if args.cmd == "slam":
        from slam.run_slam import main as slam_main

        return int(slam_main(camera_index=args.camera, config_path=args.config))

    if args.cmd == "demo":
        # Demos are still script-like for now; keep the behavior stable.
        script_path = (Path(__file__).resolve().parents[1] / args._script).resolve()
        sys.argv = [str(script_path)]
        runpy.run_path(str(script_path), run_name="__main__")
        return 0

    raise SystemExit("Unknown command")


if __name__ == "__main__":
    raise SystemExit(main())

