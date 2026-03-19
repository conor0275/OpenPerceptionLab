"""Smoke tests: ensure slam and related modules import without error."""
from __future__ import annotations


def test_import_slam_run_slam() -> None:
    from slam.run_slam import main

    assert callable(main)


def test_import_slam_submodules() -> None:
    from slam.backend.map import Map
    from slam.backend.optimizer import BundleAdjuster
    from slam.core.frame import Frame
    from slam.frontend.feature import FeatureExtractor
    from slam.frontend.tracking import Tracker

    assert Map is not None
    assert BundleAdjuster is not None
    assert Frame is not None
    assert FeatureExtractor is not None
    assert Tracker is not None


def test_import_openperceptionlab_entry() -> None:
    from openperceptionlab.__main__ import main as opl_main

    assert callable(opl_main)
