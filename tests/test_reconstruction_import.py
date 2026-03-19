"""Smoke tests for reconstruction package (Stage 5)."""
import numpy as np

from reconstruction.io import load_images_from_dir, default_intrinsics_from_image, save_ply, save_poses_npz
from reconstruction.sfm import IncrementalSfM, _triangulate_two_views
from reconstruction.run_sfm import main as sfm_main
from reconstruction.nerf_3dgs import run_nerf_placeholder, run_3dgs_placeholder


def test_reconstruction_imports():
    """All reconstruction submodules are importable."""
    assert IncrementalSfM is not None
    assert save_ply is not None
    assert run_nerf_placeholder is not None


def test_triangulate_two_views():
    """_triangulate_two_views returns points and valid mask."""
    K = np.eye(3)
    K[0, 0] = K[1, 1] = 500
    K[0, 2], K[1, 2] = 320, 240
    R1, t1 = np.eye(3), np.zeros((3, 1))
    R2 = np.eye(3)
    t2 = np.array([[0.1], [0], [0]])
    pts1 = np.array([[320, 240], [100, 100]], dtype=np.float32)
    pts2 = pts1.copy()
    pts3d, valid = _triangulate_two_views(pts1, pts2, R1, t1, R2, t2, K)
    assert pts3d.shape[0] == 2
    assert valid.shape[0] == 2


def test_sfm_init_two_views(tmp_path):
    """IncrementalSfM can init from two synthetic images."""
    import cv2
    from reconstruction.sample_images import generate_sample_images
    generate_sample_images(tmp_path, n=2)
    images = load_images_from_dir(tmp_path)
    assert len(images) >= 2
    K = default_intrinsics_from_image(images[0][1])
    sfm = IncrementalSfM(K, min_matches=30)
    ok = sfm.add_first_two_views(images[0][1], images[1][1])
    assert ok
    assert len(sfm.points_3d) >= 10
    assert len(sfm.keyframes) == 2


def test_save_ply(tmp_path):
    """save_ply writes a valid file."""
    pts = np.random.randn(5, 3).astype(np.float64)
    out = tmp_path / "out.ply"
    save_ply(out, pts)
    assert out.exists()
    text = out.read_text()
    assert "element vertex 5" in text
    assert "property float x" in text


def test_nerf_placeholder():
    """NeRF/3DGS placeholders return 0."""
    assert run_nerf_placeholder() == 0
    assert run_3dgs_placeholder() == 0


def test_sfm_main_sample_exit_code(tmp_path):
    """SfM run with sample images exits 0 and produces PLY."""
    from pathlib import Path
    from reconstruction.sample_images import generate_sample_images
    sample_dir = tmp_path / "sample_sfm"
    generate_sample_images(sample_dir, n=4)
    out_ply = tmp_path / "sfm_out.ply"
    code = sfm_main(images_dir=sample_dir, output_ply=out_ply, min_matches=30)
    assert code == 0
    assert out_ply.exists()
