"""
Loop closure detection stub (mitigate Stage 2: no loop closure).

Stores keyframe descriptors; detects potential loop by matching current to past keyframes.
Does not correct the map; backend can use loop constraints for pose graph optimization later.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger("opl.slam.loop_closure")


class LoopClosureDetector:
    """
    Detect loop: compare current keyframe descriptors to stored keyframes.
    If match count above threshold and frame gap large enough, report loop.
    """

    def __init__(self, match_threshold: int = 30, min_frame_gap: int = 20) -> None:
        self.match_threshold = match_threshold
        self.min_frame_gap = min_frame_gap
        self._keyframes: list[dict[str, Any]] = []  # id, descriptors

    def add_keyframe(self, frame_id: int, descriptors: np.ndarray | None) -> None:
        if descriptors is None or len(descriptors) == 0:
            return
        self._keyframes.append({"id": frame_id, "descriptors": np.asarray(descriptors)})

    def detect(self, frame_id: int, descriptors: np.ndarray | None) -> tuple[bool, int]:
        """
        Check if current frame closes loop with any past keyframe.
        Returns (is_loop, loop_keyframe_id). loop_keyframe_id is -1 if no loop.
        """
        if descriptors is None or len(descriptors) == 0:
            return False, -1
        try:
            import cv2
        except ImportError:
            return False, -1
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        best_id = -1
        best_matches = 0
        for kf in self._keyframes:
            if frame_id - kf["id"] < self.min_frame_gap:
                continue
            m = matcher.match(descriptors, kf["descriptors"])
            m = [x for x in m if x.distance < 50]
            if len(m) > best_matches:
                best_matches = len(m)
                best_id = kf["id"]
        if best_matches >= self.match_threshold:
            logger.info("Loop detected: frame %d -> keyframe %d (%d matches)", frame_id, best_id, best_matches)
            return True, best_id
        return False, -1
