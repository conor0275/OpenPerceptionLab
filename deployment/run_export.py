"""CLI for export and inference (Stage 6). GPU / TensorRT via flags."""
from __future__ import annotations

import logging
import time
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


def main_infer(
    onnx_path: str | Path,
    image_path: str | Path | None = None,
    *,
    prefer_gpu: bool = False,
    prefer_tensorrt: bool = False,
    device_id: int = 0,
    warmup: int = 3,
    runs: int = 1,
    list_providers: bool = False,
) -> int:
    """
    Run ONNX inference. Uses random dummy input unless image_path is set (resize to model input).

    ``warmup`` / ``runs``: for latency measurement on GPU (runs > 1 prints mean ms).
    """
    from deployment.onnx_inference import load_onnx_session, run_onnx

    try:
        import onnxruntime as ort

        if list_providers:
            print("onnxruntime providers available:", ort.get_available_providers())
    except ImportError:
        pass

    try:
        session = load_onnx_session(
            onnx_path,
            prefer_gpu=prefer_gpu,
            prefer_tensorrt=prefer_tensorrt,
            device_id=device_id,
        )
    except ImportError as e:
        logger.error("Install onnxruntime (or onnxruntime-gpu for CUDA): %s", e)
        return 1

    inp = session.get_inputs()[0]
    shape = inp.shape
    # Resolve dynamic dims to concrete size for dummy/image path
    def _dim(d, default: int) -> int:
        if isinstance(d, str) or d is None:
            return default
        if isinstance(d, int) and d > 0:
            return d
        return default

    n = _dim(shape[0] if len(shape) > 0 else 1, 1)
    c = _dim(shape[1] if len(shape) > 1 else 3, 3)
    h = _dim(shape[2] if len(shape) > 2 else 64, 64)
    w = _dim(shape[3] if len(shape) > 3 else 64, 64)

    if image_path is not None:
        try:
            import cv2
        except ImportError:
            logger.error("OpenCV required for --image: pip install opencv-python")
            return 1
        img = cv2.imread(str(image_path))
        if img is None:
            logger.error("Failed to read image: %s", image_path)
            return 1
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (w, h)).astype(np.float32) / 255.0
        x = np.transpose(img, (2, 0, 1))[np.newaxis, ...]
    else:
        x = np.random.randn(n, c, h, w).astype(np.float32)

    for _ in range(max(0, warmup)):
        run_onnx(session, x)

    times: list[float] = []
    out = None
    for _ in range(max(1, runs)):
        t0 = time.perf_counter()
        out = run_onnx(session, x)
        times.append((time.perf_counter() - t0) * 1000.0)

    logger.info("Inference output shape: %s", out.shape if out is not None else None)
    if runs > 1:
        logger.info("Latency ms (mean of %d): %.3f", runs, float(np.mean(times)))
    return 0
