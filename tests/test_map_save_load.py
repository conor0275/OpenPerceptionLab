"""Tests for slam Map save/load."""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from slam.backend.map import Map


def test_map_save_load_empty() -> None:
    m = Map()
    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
        path = f.name
    try:
        m.save(path)
        loaded = Map.load(path)
        assert len(loaded.points) == 0
        assert len(loaded.keyframes) == 0
    finally:
        Path(path).unlink(missing_ok=True)


def test_map_save_load_with_points_and_keyframes() -> None:
    m = Map()
    m.add_points(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
    stub = type("Frame", (), {})()
    stub.id = 0
    stub.pose_R = np.eye(3)
    stub.pose_t = np.array([[0.0], [0.0], [0.0]])
    m.add_keyframe(stub)
    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
        path = f.name
    try:
        m.save(path)
        loaded = Map.load(path)
        assert len(loaded.points) == 2
        np.testing.assert_allclose(loaded.points[0], [1.0, 2.0, 3.0])
        assert len(loaded.keyframes) == 1
        assert loaded.keyframes[0].id == 0
        np.testing.assert_allclose(loaded.keyframes[0].pose_R, np.eye(3))
    finally:
        Path(path).unlink(missing_ok=True)


def test_map_load_missing_raises() -> None:
    with pytest.raises(FileNotFoundError, match="Map file not found"):
        Map.load("/nonexistent/map.npz")
