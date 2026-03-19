"""Smoke tests: lidar package and LiDAR SLAM entry."""
from __future__ import annotations


def test_import_lidar_io() -> None:
    from lidar.io import load_pcd_sequence, save_pcd

    assert save_pcd is not None
    assert load_pcd_sequence is not None


def test_import_lidar_registration() -> None:
    from lidar.registration import icp_point_to_point

    assert icp_point_to_point is not None


def test_import_lidar_map() -> None:
    from lidar.map import PointCloudMap

    m = PointCloudMap(voxel_size=0.05)
    assert m.voxel_size == 0.05


def test_import_lidar_odometry() -> None:
    from lidar.odometry import LiDAROdometry

    o = LiDAROdometry()
    assert o is not None


def test_import_lidar_run_slam() -> None:
    from lidar.run_lidar_slam import main

    assert callable(main)
