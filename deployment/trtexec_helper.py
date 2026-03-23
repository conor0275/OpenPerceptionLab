"""
Optional: build a TensorRT engine from ONNX using NVIDIA ``trtexec`` (command-line).

This does **not** require TensorRT Python bindings; ``trtexec`` ships with TensorRT SDK.
Use when you want a standalone ``.engine`` file for C++ deployment or tooling.

Example (adjust paths for your TensorRT install):

.. code-block:: bash

    trtexec --onnx=depth.onnx --saveEngine=depth_fp16.engine --fp16 --workspace=4096
"""
from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

logger = logging.getLogger("opl.deployment.trtexec")


def trtexec_on_path() -> str | None:
    return shutil.which("trtexec")


def build_engine_trtexec(
    onnx_path: str | Path,
    engine_path: str | Path,
    *,
    fp16: bool = True,
    workspace_mb: int = 4096,
    extra_args: list[str] | None = None,
) -> bool:
    """
    Run ``trtexec`` to produce ``engine_path`` from ``onnx_path``.

    Returns True on success. False if trtexec missing or subprocess fails.
    """
    exe = trtexec_on_path()
    if not exe:
        logger.error("trtexec not found on PATH. Install TensorRT SDK and add its bin directory.")
        return False
    onnx_path = Path(onnx_path)
    engine_path = Path(engine_path)
    if not onnx_path.is_file():
        logger.error("ONNX not found: %s", onnx_path)
        return False
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        exe,
        f"--onnx={onnx_path.resolve()}",
        f"--saveEngine={engine_path.resolve()}",
        f"--workspace={workspace_mb}",
    ]
    if fp16:
        cmd.append("--fp16")
    if extra_args:
        cmd.extend(extra_args)
    logger.info("Running: %s", " ".join(cmd))
    try:
        r = subprocess.run(cmd, check=True, capture_output=True, text=True)
        if r.stdout:
            logger.debug("%s", r.stdout[-2000:])
        return True
    except subprocess.CalledProcessError as e:
        logger.error("trtexec failed: %s", e.stderr or e)
        return False
