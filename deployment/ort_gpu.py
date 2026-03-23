"""
ONNX Runtime session setup for GPU / TensorRT execution providers (hardcore deploy path).

Requires: ``pip install onnxruntime-gpu`` (version must match your CUDA/cuDNN).
Optional TensorRT EP: ORT wheel built with TensorRT, plus TensorRT libraries on PATH / LD_LIBRARY_PATH.

See docs/GPU_DEPLOY.md for environment setup.
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("opl.deployment.ort_gpu")


def _tensorrt_provider_options(device_id: int = 0) -> dict[str, Any]:
    """Options for TensorrtExecutionProvider (ORT integrates TensorRT for subgraphs)."""
    return {
        "device_id": device_id,
        "trt_max_workspace_size": 2 * 1024 * 1024 * 1024,  # 2 GB
        "trt_engine_cache_enable": True,
        "trt_engine_cache_path": ".ort_trt_cache",
        "trt_fp16_enable": True,
    }


def build_provider_list(
    *,
    prefer_tensorrt: bool = False,
    prefer_gpu: bool = False,
    device_id: int = 0,
) -> tuple[list[Any], str]:
    """
    Build (providers, description) for InferenceSession.

    Order:
    - tensorrt + gpu: TensorrtExecutionProvider -> CUDAExecutionProvider -> CPU
    - gpu only: CUDA -> CPU
    - neither: CPU only
    """
    try:
        import onnxruntime as ort
    except ImportError as e:
        raise ImportError("Install onnxruntime or onnxruntime-gpu") from e

    available = set(ort.get_available_providers())
    logger.debug("ORT available providers: %s", available)

    trt_opts = _tensorrt_provider_options(device_id)
    cuda = "CUDAExecutionProvider"
    trt = "TensorrtExecutionProvider"
    cpu = "CPUExecutionProvider"

    trt_ok = prefer_tensorrt and trt in available
    if prefer_tensorrt and not trt_ok:
        logger.warning(
            "TensorrtExecutionProvider not in ORT build (have: %s). "
            "Falling back to CUDA if available.",
            available,
        )
    # 仅 --tensorrt 时也应尝试 CUDA，而不是直接落 CPU
    want_cuda = prefer_gpu or prefer_tensorrt

    if trt_ok:
        providers: list[Any] = [
            (trt, trt_opts),
            cuda,
            cpu,
        ]
        return providers, "TensorRT EP -> CUDA -> CPU"

    if want_cuda and cuda in available:
        providers = [
            (cuda, {"device_id": device_id}),
            cpu,
        ]
        return providers, "CUDA -> CPU"

    if want_cuda and cuda not in available:
        logger.warning(
            "CUDAExecutionProvider not available (have: %s). Using CPU.",
            available,
        )

    return [cpu], "CPU only"


def make_session_options() -> Any:
    """Graph optimizations enabled (all levels)."""
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return so
