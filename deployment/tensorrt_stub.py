"""
TensorRT-related entry points (Stage 6).

**Hardcore / GPU branch** (see ``docs/GPU_DEPLOY.md``):

- **Runtime (推荐)**：ONNX Runtime ``TensorrtExecutionProvider`` —— 配置见 ``deployment/ort_gpu.py``，
  CLI：``opl infer model.onnx --tensorrt --gpu``.
- **离线 engine**：``deployment.trtexec_helper.build_engine_trtexec`` 调用 ``trtexec`` 生成 ``.engine``.

本模块保留占位函数，避免旧代码 import 报错；新逻辑请用上述模块。
"""
from __future__ import annotations

import logging

logger = logging.getLogger("opl.deployment.tensorrt")


def build_engine_placeholder(onnx_path: str, output_engine_path: str) -> bool:
    """
    占位：仅打日志。真实构建请使用：

    - ``from deployment.trtexec_helper import build_engine_trtexec``
    - 或命令行：``trtexec --onnx=... --saveEngine=...``
    """
    logger.info(
        "TensorRT: use trtexec or deployment.trtexec_helper. Example: trtexec --onnx=%s --saveEngine=%s",
        onnx_path,
        output_engine_path,
    )
    return False
