from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _run_script(rel_path: str, argv: list[str]) -> int:
    """
    Run a repo script (e.g. slam/run_slam.py) as if executed directly.
    """
    script_path = (REPO_ROOT / rel_path).resolve()
    if not script_path.exists():
        raise SystemExit(f"Script not found: {script_path}")

    sys.argv = [str(script_path), *argv]
    runpy.run_path(str(script_path), run_name="__main__")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="openperceptionlab",
        description="OpenPerceptionLab unified entrypoint (camera-first).",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    slam = sub.add_parser("slam", help="Run real-time monocular SLAM (camera).")
    slam.add_argument("--camera", type=int, default=0, help="Camera index (default: 0).")

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

    if args.cmd == "slam":
        # Pass camera index through env var for now (minimal change to existing script)
        import os

        os.environ["OPL_CAMERA_INDEX"] = str(args.camera)
        return _run_script("slam/run_slam.py", [])

    if args.cmd == "demo":
        return _run_script(args._script, [])

    raise SystemExit("Unknown command")


if __name__ == "__main__":
    raise SystemExit(main())

