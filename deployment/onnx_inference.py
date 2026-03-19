"""
Run ONNX model inference (Stage 6). Requires: pip install onnxruntime (or onnxruntime-gpu).
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger("opl.deployment.inference")


def load_onnx_session(model_path: str | Path):
    """Load ONNX model; returns inference session. Needs onnxruntime."""
    try:
        import onnxruntime as ort
    except ImportError as e:
        raise ImportError("Install onnxruntime: pip install onnxruntime") from e
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"ONNX model not found: {path}")
    return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])


def run_onnx(session, input_array: np.ndarray, input_name: str | None = None) -> np.ndarray:
    """Run one forward pass. input_array: NCHW float32. Returns first output."""
    if input_name is None:
        input_name = session.get_inputs()[0].name
    out = session.run(None, {input_name: input_array.astype(np.float32)})
    return out[0]
