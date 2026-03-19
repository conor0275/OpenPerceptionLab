from __future__ import annotations

import numpy as np
from pathlib import Path


class Map:
    def __init__(self) -> None:
        self.keyframes: list = []
        self.points: list = []

    def add_keyframe(self, frame) -> None:
        self.keyframes.append(frame)

    def add_points(self, points_3d: np.ndarray) -> None:
        for p in points_3d:
            self.points.append(p)

    def save(self, path: str | Path) -> None:
        """Save map points and keyframe poses to a .npz file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if len(self.points) == 0:
            points_arr = np.zeros((0, 3), dtype=np.float64)
        else:
            points_arr = np.asarray(self.points, dtype=np.float64).reshape(-1, 3)
        if len(self.keyframes) == 0:
            kf_ids = np.array([], dtype=np.int64)
            kf_R = np.zeros((0, 3, 3), dtype=np.float64)
            kf_t = np.zeros((0, 3), dtype=np.float64)
        else:
            kf_ids = np.array([f.id for f in self.keyframes], dtype=np.int64)
            kf_R = np.array([np.asarray(f.pose_R, dtype=np.float64) for f in self.keyframes])
            kf_t = np.array([np.asarray(f.pose_t, dtype=np.float64).ravel() for f in self.keyframes])
        np.savez_compressed(
            path,
            points=points_arr,
            keyframe_ids=kf_ids,
            keyframe_R=kf_R,
            keyframe_t=kf_t,
        )

    @classmethod
    def load(cls, path: str | Path) -> "Map":
        """Load map from a .npz file. Keyframes are restored with id, pose_R, pose_t only."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Map file not found: {path}")
        data = np.load(path, allow_pickle=False)
        m = cls()
        points = data["points"]
        if points.size > 0:
            m.points = list(points.reshape(-1, 3))
        kf_ids = data["keyframe_ids"]
        kf_R = data["keyframe_R"]
        kf_t = data["keyframe_t"]
        for i in range(len(kf_ids)):
            stub = type("KeyframeStub", (), {})()
            stub.id = int(kf_ids[i])
            stub.pose_R = kf_R[i].copy()
            stub.pose_t = kf_t[i].reshape(3, 1)
            m.keyframes.append(stub)
        return m