"""Smoke tests for fusion package (Stage 4)."""
import numpy as np

from fusion.pose_graph_2d import PoseGraph2D
from fusion.poses import pose_4x4_to_2d, pose_2d_to_4x4, pose_4x4_from_Rt, relative_pose_2d
from fusion.lidar_vision_fusion import fuse_trajectories_2d, fuse_vo_lidar_trajectories
from fusion.vio import VIOEstimator
from fusion.run_fusion import main as fusion_main, generate_synthetic_trajectories


def test_fusion_imports():
    """All fusion submodules are importable."""
    assert PoseGraph2D is not None
    assert fuse_trajectories_2d is not None
    assert VIOEstimator is not None


def test_pose_2d_roundtrip():
    """2D pose to 4x4 and back is consistent."""
    x, y, th = 1.0, 2.0, 0.5
    T = pose_2d_to_4x4(x, y, th)
    x2, y2, th2 = pose_4x4_to_2d(T)
    np.testing.assert_allclose([x2, y2, th2], [x, y, th])


def test_pose_graph_optimize():
    """Pose graph with prior and one between factor optimizes."""
    g = PoseGraph2D()
    g.add_node(0.0, 0.0, 0.0)
    g.add_node(1.0, 0.0, 0.0)
    g.set_prior(0.0, 0.0, 0.0)
    g.add_between(0, 1, 1.0, 0.0, 0.0)
    out = g.optimize(max_iter=10)
    assert len(out) == 2
    np.testing.assert_allclose(out[0], (0.0, 0.0, 0.0))
    np.testing.assert_allclose(out[1], (1.0, 0.0, 0.0), atol=1e-5)


def test_fuse_trajectories_2d():
    """Fusing two identical 2D trajectories returns same length."""
    traj = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.1, 0.0)]
    fused = fuse_trajectories_2d(traj, traj, prior_use="vo", max_iter=5)
    assert len(fused) == 3


def test_fuse_vo_lidar_4x4():
    """Fuse two 4x4 pose lists returns 2D trajectory."""
    T0 = pose_2d_to_4x4(0.0, 0.0, 0.0)
    T1 = pose_2d_to_4x4(1.0, 0.0, 0.0)
    fused = fuse_vo_lidar_trajectories([T0, T1], [T0, T1], max_iter=5)
    assert len(fused) == 2


def test_vio_estimator():
    """VIOEstimator returns pose from VO input."""
    est = VIOEstimator()
    pose_vo = np.eye(4)
    pose_vo[0, 3] = 1.0
    out = est.process_frame(pose_vo)
    assert out.shape == (4, 4)
    assert out[0, 3] == 1.0


def test_fusion_demo_exit_code():
    """Fusion demo runs and returns 0."""
    code = fusion_main(demo=True, demo_frames=5)
    assert code == 0
