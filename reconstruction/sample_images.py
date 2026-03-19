"""
Generate a minimal set of synthetic images for SfM demo (no real data).
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def generate_sample_images(dir_path: str | Path, n: int = 5, size: tuple[int, int] = (640, 480)) -> list[Path]:
    """
    Create n images with overlapping content (shifted + noise) so ORB can match.
    Returns list of saved paths.
    """
    dir_path = Path(dir_path)
    dir_path.mkdir(parents=True, exist_ok=True)
    w, h = size
    # Base image: grid + circles to get stable keypoints
    base = np.uint8(np.clip(128 + np.random.randn(h, w) * 30, 0, 255))
    for i in range(20):
        cx, cy = np.random.randint(50, w - 50), np.random.randint(50, h - 50)
        cv2.circle(base, (cx, cy), 15, 200, 2)
    paths = []
    for i in range(n):
        dx, dy = int(20 * np.sin(i) * 2), int(15 * np.cos(i * 0.7) * 2)
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        img = cv2.warpAffine(base, M, (w, h))
        img = np.clip(img.astype(np.float32) + np.random.randn(h, w) * 5, 0, 255).astype(np.uint8)
        p = dir_path / f"frame_{i:04d}.jpg"
        cv2.imwrite(str(p), img)
        paths.append(p)
    return paths
