"""
3D pose graph (translation-only optimization) to mitigate Stage 4 limitation (2D -> 3D).

Optimizes node positions (x,y,z) with between-factor constraints on relative translation.
Rotations are kept from measurements; for full 6-DOF use GTSAM/Ceres later.
"""
from __future__ import annotations

import logging
import numpy as np

logger = logging.getLogger("opl.fusion.pose_graph_3d")


class PoseGraph3D:
    """
    3D pose graph: nodes (x, y, z), prior on first node, between factors (relative translation).
    Rotation from each node is fixed (from input); we only optimize translation.
    """

    def __init__(self) -> None:
        self._positions: list[tuple[float, float, float]] = []
        self._rotations: list[np.ndarray] = []  # 3x3 per node (for relative transform)
        self._prior: tuple[float, float, float] | None = None
        self._edges: list[tuple[int, int, float, float, float]] = []  # i, j, dx, dy, dz (relative in frame i)

    def add_node(self, x: float, y: float, z: float, R: np.ndarray | None = None) -> int:
        idx = len(self._positions)
        self._positions.append((float(x), float(y), float(z)))
        self._rotations.append(np.eye(3) if R is None else np.asarray(R, dtype=np.float64)[:3, :3].copy())
        return idx

    def set_prior(self, x: float, y: float, z: float) -> None:
        self._prior = (float(x), float(y), float(z))

    def add_between(self, i: int, j: int, dx: float, dy: float, dz: float) -> None:
        self._edges.append((int(i), int(j), float(dx), float(dy), float(dz)))

    def optimize(self, max_iter: int = 20, tol: float = 1e-6) -> list[tuple[float, float, float]]:
        n = len(self._positions)
        if n == 0:
            return []
        state = np.array(self._positions, dtype=np.float64).ravel()
        R_list = self._rotations
        for it in range(max_iter):
            residuals = []
            jac_rows = []
            if self._prior is not None:
                residuals.extend([state[0] - self._prior[0], state[1] - self._prior[1], state[2] - self._prior[2]])
                row = np.zeros(3 * n)
                row[0] = row[1] = row[2] = 1
                jac_rows.append(row)
                row = np.zeros(3 * n)
                row[1] = 1
                jac_rows.append(row)
                row = np.zeros(3 * n)
                row[2] = 1
                jac_rows.append(row)
            for (i, j, dx, dy, dz) in self._edges:
                ti = state[3 * i : 3 * i + 3]
                tj = state[3 * j : 3 * j + 3]
                Ri = R_list[i]
                pred_tj = ti + Ri @ np.array([dx, dy, dz])
                res = pred_tj - tj
                residuals.extend(res)
                for d in range(3):
                    row = np.zeros(3 * n)
                    row[3 * i + d] = 1
                    row[3 * j + d] = -1
                    jac_rows.append(row)
            r = np.array(residuals, dtype=np.float64)
            J = np.array(jac_rows, dtype=np.float64)
            try:
                delta = np.linalg.solve(J.T @ J, -J.T @ r)
            except np.linalg.LinAlgError:
                break
            state = state + delta
            if np.linalg.norm(delta) < tol:
                break
        return [(state[3 * k], state[3 * k + 1], state[3 * k + 2]) for k in range(n)]
