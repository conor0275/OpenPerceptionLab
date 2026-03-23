#!/usr/bin/env python3
"""
Benchmark ONNX Runtime inference across available providers (CPU vs CUDA vs TensorRT EP).

Usage (from repo root, PYTHONPATH=.):
  python scripts/benchmark_onnx_providers.py path/to/model.onnx --warmup 10 --runs 100

Requires: onnx + onnxruntime or onnxruntime-gpu (see docs/GPU_DEPLOY.md).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

# Repo root as path
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("onnx_path", type=Path)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--runs", type=int, default=50)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--c", type=int, default=3)
    p.add_argument("--h", type=int, default=64)
    p.add_argument("--w", type=int, default=64)
    args = p.parse_args()

    try:
        import onnxruntime as ort
    except ImportError:
        print("pip install onnxruntime or onnxruntime-gpu", file=sys.stderr)
        return 1

    print("Available providers:", ort.get_available_providers())

    from deployment.onnx_inference import load_onnx_session, run_onnx

    modes = [
        ("CPU", dict(prefer_gpu=False, prefer_tensorrt=False)),
        ("CUDA", dict(prefer_gpu=True, prefer_tensorrt=False)),
        ("TensorRT_EP", dict(prefer_gpu=True, prefer_tensorrt=True)),
    ]

    x = np.random.randn(args.batch, args.c, args.h, args.w).astype(np.float32)

    for label, kwargs in modes:
        try:
            sess = load_onnx_session(args.onnx_path, **kwargs)
        except Exception as e:
            print(f"{label}: skip ({e})")
            continue
        for _ in range(args.warmup):
            run_onnx(sess, x)
        t0 = time.perf_counter()
        for _ in range(args.runs):
            run_onnx(sess, x)
        ms = (time.perf_counter() - t0) / args.runs * 1000.0
        print(f"{label}: mean {ms:.3f} ms/run (runs={args.runs})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
