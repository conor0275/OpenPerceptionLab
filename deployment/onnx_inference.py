"""
Run ONNX model inference (Stage 6).

- Default: ``onnxruntime`` CPU (CI-friendly).
- GPU branch: ``pip install onnxruntime-gpu`` and use ``load_onnx_session(..., prefer_gpu=True)``.
- TensorRT via ORT: ``prefer_tensorrt=True`` (requires ORT build with TensorRT + TRT libs).
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger("opl.deployment.inference")


def load_onnx_session(
    model_path: str | Path,
    *,
    prefer_gpu: bool = False,
    prefer_tensorrt: bool = False,
    device_id: int = 0,
):
    """
    Load ONNX model. Returns ``onnxruntime.InferenceSession``.

    If ``prefer_tensorrt`` is True, tries TensorRT EP first (see ``deployment/ort_gpu.py``).
    If only ``prefer_gpu``, uses CUDA then CPU.
    """
    try:
        import onnxruntime as ort
    except ImportError as e:
        raise ImportError("Install onnxruntime: pip install onnxruntime") from e

    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"ONNX model not found: {path}")

    from deployment.ort_gpu import build_provider_list, make_session_options

    providers, desc = build_provider_list(
        prefer_tensorrt=prefer_tensorrt,
        prefer_gpu=prefer_gpu,
        device_id=device_id,
    )
    so = make_session_options()
    logger.info("ORT session providers: %s", desc)
    return ort.InferenceSession(str(path), sess_options=so, providers=providers)


def run_onnx(session, input_array: np.ndarray, input_name: str | None = None) -> np.ndarray:
    """Run one forward pass. input_array: NCHW float32. Returns first output."""
    if input_name is None:
        input_name = session.get_inputs()[0].name
    out = session.run(None, {input_name: input_array.astype(np.float32)})
    return out[0]
