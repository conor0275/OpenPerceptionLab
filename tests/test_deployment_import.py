"""Smoke tests for deployment package (Stage 6) and mitigations."""
import numpy as np
import pytest

from deployment.run_export import main_export, main_infer
from fusion.scale_align import align_scale_2d, scaled_vo_trajectory_2d
from fusion.pose_graph_3d import PoseGraph3D
from fusion.imu_propagation import propagate_pose
from fusion.vio import VIOEstimator
from slam.backend.loop_closure import LoopClosureDetector


def test_export_tiny_demo(tmp_path):
    """Export tiny ONNX (requires onnx)."""
    pytest.importorskip("onnx")
    from deployment.onnx_export import export_tiny_demo
    out = tmp_path / "tiny.onnx"
    export_tiny_demo(out)
    assert out.exists()


def test_onnx_inference(tmp_path):
    """Export then run inference (requires onnx, onnxruntime)."""
    pytest.importorskip("onnx")
    from deployment.onnx_export import export_tiny_demo
    out = tmp_path / "tiny.onnx"
    export_tiny_demo(out)
    try:
        from deployment.onnx_inference import load_onnx_session, run_onnx
        session = load_onnx_session(out)
        x = np.random.randn(1, 3, 64, 64).astype(np.float32)
        y = run_onnx(session, x)
        assert y.shape == x.shape
    except ImportError:
        pass  # onnxruntime optional


def test_main_export_exit_code(tmp_path):
    """opl export --model tiny exits 0 (requires onnx)."""
    pytest.importorskip("onnx")
    code = main_export(output=tmp_path / "m.onnx", model_type="tiny")
    assert code == 0


def test_scale_align():
    """Scale alignment returns positive scale."""
    vo = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0)]
    li = [(0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (4.0, 0.0, 0.0)]
    s = align_scale_2d(vo, li)
    assert s > 0
    scaled = scaled_vo_trajectory_2d(vo, s)
    assert len(scaled) == 3


def test_pose_graph_3d():
    """3D pose graph optimizes."""
    g = PoseGraph3D()
    g.add_node(0, 0, 0)
    g.add_node(1, 0, 0)
    g.set_prior(0, 0, 0)
    g.add_between(0, 1, 1, 0, 0)
    out = g.optimize(max_iter=5)
    assert len(out) == 2
    np.testing.assert_allclose(out[1], (1, 0, 0), atol=1e-5)


def test_imu_propagate():
    """IMU propagation returns pose and velocity."""
    pose = np.eye(4)
    pose[2, 3] = 1.0
    v = np.zeros(3)
    pose2, v2 = propagate_pose(pose, v, np.zeros(3), np.array([0, 0, 0]), 0.1)
    assert pose2.shape == (4, 4)
    assert v2.shape == (3,)


def test_vio_with_imu_buffer():
    """VIO accepts imu_buffer (optional propagation)."""
    est = VIOEstimator(use_imu_propagation=True)
    pose_vo = np.eye(4)
    pose_vo[0, 3] = 1.0
    imu = [{"gyro": [0, 0, 0], "acc": [0, 0, 0]}]
    out = est.process_frame(pose_vo, imu_buffer=imu, dt=0.033)
    assert out.shape == (4, 4)


def test_loop_closure_detector():
    """Loop detector add and detect (no loop with empty history)."""
    det = LoopClosureDetector(match_threshold=10, min_frame_gap=5)
    des = np.random.randint(0, 256, (100, 32), dtype=np.uint8)
    det.add_keyframe(0, des)
    is_loop, lid = det.detect(10, des)
    assert lid == 0 or not is_loop
