from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(slots=True)
class CameraConfig:
    index: int = 0
    width: int | None = None
    height: int | None = None
    fps: int | None = None


@dataclass(slots=True)
class IntrinsicsConfig:
    fx: float = 800.0
    fy: float = 800.0
    cx: float = 320.0
    cy: float = 240.0

    def as_K(self) -> np.ndarray:
        return np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )


@dataclass(slots=True)
class ViewerConfig:
    show_frame: bool = True
    show_trajectory: bool = True
    show_pointcloud: bool = True
    max_points: int = 200_000
    point_size: float = 2.0


@dataclass(slots=True)
class SlamConfig:
    keyframe_interval: int = 10
    log_every_n_frames: int = 30


@dataclass(slots=True)
class AppConfig:
    camera: CameraConfig = field(default_factory=CameraConfig)
    intrinsics: IntrinsicsConfig = field(default_factory=IntrinsicsConfig)
    viewer: ViewerConfig = field(default_factory=ViewerConfig)
    slam: SlamConfig = field(default_factory=SlamConfig)


def _deep_update(dst: dict[str, Any], src: dict[str, Any]) -> dict[str, Any]:
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_update(dst[k], v)
        else:
            dst[k] = v
    return dst


def _as_dict(cfg: AppConfig) -> dict[str, Any]:
    return {
        "camera": {
            "index": cfg.camera.index,
            "width": cfg.camera.width,
            "height": cfg.camera.height,
            "fps": cfg.camera.fps,
        },
        "intrinsics": {
            "fx": cfg.intrinsics.fx,
            "fy": cfg.intrinsics.fy,
            "cx": cfg.intrinsics.cx,
            "cy": cfg.intrinsics.cy,
        },
        "viewer": {
            "show_frame": cfg.viewer.show_frame,
            "show_trajectory": cfg.viewer.show_trajectory,
            "show_pointcloud": cfg.viewer.show_pointcloud,
            "max_points": cfg.viewer.max_points,
            "point_size": cfg.viewer.point_size,
        },
        "slam": {
            "keyframe_interval": cfg.slam.keyframe_interval,
            "log_every_n_frames": cfg.slam.log_every_n_frames,
        },
    }


def _from_dict(d: dict[str, Any]) -> AppConfig:
    cam = d.get("camera", {})
    intr = d.get("intrinsics", {})
    view = d.get("viewer", {})
    slam = d.get("slam", {})

    return AppConfig(
        camera=CameraConfig(
            index=int(cam.get("index", 0)),
            width=cam.get("width", None),
            height=cam.get("height", None),
            fps=cam.get("fps", None),
        ),
        intrinsics=IntrinsicsConfig(
            fx=float(intr.get("fx", 800.0)),
            fy=float(intr.get("fy", 800.0)),
            cx=float(intr.get("cx", 320.0)),
            cy=float(intr.get("cy", 240.0)),
        ),
        viewer=ViewerConfig(
            show_frame=bool(view.get("show_frame", True)),
            show_trajectory=bool(view.get("show_trajectory", True)),
            show_pointcloud=bool(view.get("show_pointcloud", True)),
            max_points=int(view.get("max_points", 200_000)),
            point_size=float(view.get("point_size", 2.0)),
        ),
        slam=SlamConfig(
            keyframe_interval=int(slam.get("keyframe_interval", 10)),
            log_every_n_frames=int(slam.get("log_every_n_frames", 30)),
        ),
    )


def load_config(path: str | Path | None) -> AppConfig:
    """
    Load config from YAML/JSON, overriding defaults. If path is None, returns defaults.
    """
    cfg = AppConfig()
    if path is None:
        return cfg

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {p}")

    if p.suffix.lower() in {".yml", ".yaml"}:
        import yaml

        raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    elif p.suffix.lower() == ".json":
        import json

        raw = json.loads(p.read_text(encoding="utf-8"))
    else:
        raise ValueError("Config must be .yaml/.yml or .json")

    merged = _as_dict(cfg)
    _deep_update(merged, raw)
    return _from_dict(merged)

