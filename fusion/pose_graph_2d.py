"""
2D pose graph optimizer for multi-sensor fusion.

Minimizes prior (fix first pose) and between factors (relative pose constraints).
Uses Gauss-Newton; no GTSAM dependency so it runs everywhere. For production
scale, swap in GTSAM or Ceres.
"""
from __future__ import annotations

import logging
import numpy as np

from fusion.poses import compose_pose_2d

logger = logging.getLogger("opl.fusion.pose_graph")


def _angle_diff(a: float, b: float) -> float:
    """Normalize angle difference to [-pi, pi]."""
    d = a - b
    while d > np.pi:
        d -= 2 * np.pi
    while d < -np.pi:
        d += 2 * np.pi
    return d


class PoseGraph2D:
    """
    2D pose graph: nodes (x, y, theta), prior on node 0, between factors.
    Optimize with Gauss-Newton to get consistent trajectory.
    """

    def __init__(self) -> None:
        self._nodes: list[tuple[float, float, float]] = []
        self._prior: tuple[float, float, float] | None = None  # (x0, y0, th0)
        self._edges: list[tuple[int, int, float, float, float]] = []  # (i, j, dx, dy, dth)

    def add_node(self, x: float, y: float, theta: float) -> int:
        """Add a node; returns node index."""
        idx = len(self._nodes)
        self._nodes.append((float(x), float(y), float(theta)))
        return idx

    def set_prior(self, x: float, y: float, theta: float) -> None:
        """Fix the first node to (x, y, theta). Call after adding at least one node."""
        self._prior = (float(x), float(y), float(theta))

    def add_between(self, i: int, j: int, dx: float, dy: float, dth: float) -> None:
        """Add relative constraint from node i to j: in frame i, j is at (dx, dy) with orientation diff dth."""
        self._edges.append((int(i), int(j), float(dx), float(dy), float(dth)))

    def _state_to_nodes(self, state: np.ndarray) -> list[tuple[float, float, float]]:
        n = len(self._nodes)
        out = []
        for k in range(n):
            out.append((state[3 * k], state[3 * k + 1], state[3 * k + 2]))
        return out

    def _nodes_to_state(self) -> np.ndarray:
        state = np.zeros(3 * len(self._nodes), dtype=np.float64)
        for k, (x, y, th) in enumerate(self._nodes):
            state[3 * k] = x
            state[3 * k + 1] = y
            state[3 * k + 2] = th
        return state

    def optimize(self, max_iter: int = 20, tol: float = 1e-6) -> list[tuple[float, float, float]]:
        """
        Run Gauss-Newton. Returns list of optimized (x, y, theta) per node.
        """
        n = len(self._nodes)
        if n == 0:
            return []
        state = self._nodes_to_state()
        for it in range(max_iter):
            residuals = []
            jac_rows = []
            # Prior on node 0
            if self._prior is not None:
                x0, y0, th0 = self._prior
                residuals.extend([
                    state[0] - x0,
                    state[1] - y0,
                    _angle_diff(state[2], th0),
                ])
                row = np.zeros(3 * n)
                row[0] = 1
                jac_rows.append(row)
                row = np.zeros(3 * n)
                row[1] = 1
                jac_rows.append(row)
                row = np.zeros(3 * n)
                row[2] = 1
                jac_rows.append(row)
            # Between factors
            for (i, j, dx, dy, dth) in self._edges:
                xi, yi, thi = state[3 * i], state[3 * i + 1], state[3 * i + 2]
                xj, yj, thj = state[3 * j], state[3 * j + 1], state[3 * j + 2]
                xj_p, yj_p, thj_p = compose_pose_2d(xi, yi, thi, dx, dy, dth)
                res_x = xj_p - xj
                res_y = yj_p - yj
                res_th = _angle_diff(thj_p, thj)
                residuals.extend([res_x, res_y, res_th])
                ci, si = np.cos(thi), np.sin(thi)
                # d(res_x)/d(state)
                row = np.zeros(3 * n)
                row[3 * i] = 1
                row[3 * i + 2] = -si * dx - ci * dy
                row[3 * j] = -1
                jac_rows.append(row)
                row = np.zeros(3 * n)
                row[3 * i + 1] = 1
                row[3 * i + 2] = ci * dx - si * dy
                row[3 * j + 1] = -1
                jac_rows.append(row)
                row = np.zeros(3 * n)
                row[3 * i + 2] = 1
                row[3 * j + 2] = -1
                jac_rows.append(row)
            r = np.array(residuals, dtype=np.float64)
            J = np.array(jac_rows, dtype=np.float64)
            # GN step: (J^T J) delta = -J^T r
            try:
                delta = np.linalg.solve(J.T @ J, -J.T @ r)
            except np.linalg.LinAlgError:
                logger.warning("Pose graph: singular system at iter %d", it)
                break
            state = state + delta
            # Normalize angles
            for k in range(n):
                th = state[3 * k + 2]
                while th > np.pi:
                    th -= 2 * np.pi
                while th < -np.pi:
                    th += 2 * np.pi
                state[3 * k + 2] = th
            if np.linalg.norm(delta) < tol:
                logger.debug("Pose graph converged at iter %d", it)
                break
        return self._state_to_nodes(state)

    def get_nodes(self) -> list[tuple[float, float, float]]:
        """Return current node poses (before or after optimize)."""
        return list(self._nodes)
