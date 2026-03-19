"""
TensorRT placeholder (Stage 6). For production, convert ONNX to TensorRT engine
on target GPU and use TensorRT Python API or C++ runtime.
"""
from __future__ import annotations

import logging

logger = logging.getLogger("opl.deployment.tensorrt")


def build_engine_placeholder(onnx_path: str, output_engine_path: str) -> bool:
    """
    Placeholder for TensorRT engine build. In production use:
    - trtexec --onnx=model.onnx --saveEngine=model.engine
    - Or tensorrt Python API: builder, network, parser, engine
    """
    logger.info(
        "TensorRT: convert ONNX to engine on target machine. "
        "Example: trtexec --onnx=%s --saveEngine=%s",
        onnx_path,
        output_engine_path,
    )
    return False
