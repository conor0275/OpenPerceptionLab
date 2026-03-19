"""Tests for openperceptionlab config loading and defaults."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from openperceptionlab.config import AppConfig, load_config


def test_load_config_none_returns_defaults() -> None:
    cfg = load_config(None)
    assert isinstance(cfg, AppConfig)
    assert cfg.camera.index == 0
    assert cfg.intrinsics.fx == 800.0
    assert cfg.viewer.show_frame is True
    assert cfg.slam.keyframe_interval == 10


def test_intrinsics_as_K() -> None:
    cfg = load_config(None)
    K = cfg.intrinsics.as_K()
    assert K.shape == (3, 3)
    assert K.dtype == np.float64
    np.testing.assert_allclose(K[0, 0], cfg.intrinsics.fx)
    np.testing.assert_allclose(K[1, 1], cfg.intrinsics.fy)
    np.testing.assert_allclose(K[0, 2], cfg.intrinsics.cx)
    np.testing.assert_allclose(K[1, 2], cfg.intrinsics.cy)
    assert K[2, 2] == 1.0


def test_load_config_yaml_override() -> None:
    with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
        Path(f.name).write_text(
            "camera:\n  index: 1\nintrinsics:\n  fx: 1000.0\n",
            encoding="utf-8",
        )
    try:
        cfg = load_config(f.name)
        assert cfg.camera.index == 1
        assert cfg.intrinsics.fx == 1000.0
    finally:
        Path(f.name).unlink(missing_ok=True)


def test_load_config_json_override() -> None:
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        Path(f.name).write_text(
            json.dumps({"slam": {"keyframe_interval": 5}}),
            encoding="utf-8",
        )
    try:
        cfg = load_config(f.name)
        assert cfg.slam.keyframe_interval == 5
    finally:
        Path(f.name).unlink(missing_ok=True)


def test_load_config_missing_file_raises() -> None:
    with pytest.raises(FileNotFoundError, match="Config not found"):
        load_config("/nonexistent/config.yaml")


def test_load_config_invalid_extension_raises() -> None:
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
        Path(f.name).write_text("x: 1", encoding="utf-8")
    try:
        with pytest.raises(ValueError, match=".yaml/.yml or .json"):
            load_config(f.name)
    finally:
        Path(f.name).unlink(missing_ok=True)
