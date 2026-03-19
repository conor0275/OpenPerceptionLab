"""CLI for export and inference (Stage 6)."""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from deployment.onnx_export import export_tiny_demo

logger = logging.getLogger("opl.deployment")


def main_export(
    output: str | Path = "model_tiny.onnx",
    model_type: str = "tiny",
) -> int:
    """Export model to ONNX. model_type: tiny (no torch) or depth (requires torch)."""
    output = Path(output)
    if model_type == "tiny":
        export_tiny_demo(output)
        return 0
    if model_type == "depth":
        try:
            from deployment.onnx_export import export_depth_onnx
            export_depth_onnx(output)
            return 0
        except ImportError as e:
            logger.error("Depth export needs torch and onnx: %s", e)
            return 1
    logger.error("Unknown model_type: %s (use tiny or depth)", model_type)
    return 1


def main_infer(onnx_path: str | Path, image_path: str | Path | None = None) -> int:
    """Run ONNX inference. If image_path given, load and run (demo)."""
    from deployment.onnx_inference import load_onnx_session, run_onnx
    try:
        session = load_onnx_session(onnx_path)
    except ImportError as e:
        logger.error("Install onnxruntime: pip install onnxruntime. %s", e)
        return 1
    dummy = np.random.randn(1, 3, 64, 64).astype(np.float32)
    out = run_onnx(session, dummy)
    logger.info("Inference output shape: %s", out.shape)
    return 0
