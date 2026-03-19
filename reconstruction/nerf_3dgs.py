"""
NeRF / 3D Gaussian Splatting placeholder (Stage 5).

Full NeRF or 3DGS training typically requires PyTorch, CUDA, and large dependencies.
This module documents the intent; integrate external tools (e.g. nerfstudio, gsplat)
or add a minimal training script in a later iteration.
"""
from __future__ import annotations

import logging

logger = logging.getLogger("opl.reconstruction.nerf_3dgs")


def run_nerf_placeholder(
    images_dir: str | None = None,
    **kwargs,
) -> int:
    """
    Placeholder for NeRF training. Use external tools (e.g. nerfstudio) or
    add a minimal PyTorch NeRF script later.
    """
    logger.info(
        "NeRF/3DGS: use external tools (e.g. nerfstudio, gsplat) or add training script later. "
        "Input: images_dir=%s", images_dir
    )
    return 0


def run_3dgs_placeholder(
    images_dir: str | None = None,
    **kwargs,
) -> int:
    """
    Placeholder for 3D Gaussian Splatting. Use external tools or add later.
    """
    logger.info(
        "3D Gaussian Splatting: use external tools or add later. Input: images_dir=%s",
        images_dir,
    )
    return 0
