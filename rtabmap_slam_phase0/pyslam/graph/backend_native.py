"""
Native pose-graph backend. No external optimizer dependency: always
available, fully inspectable, and used as the GTSAM cross-check.

Parameterization: each unknown pose is represented as base_pose @
exp(xi), xi in R^6, and the whole graph is optimised in one
scipy.optimize.least_squares call over the concatenated tangent-space
perturbation vector. This is the standard manifold-NLLS approach
(equivalent in spirit to what g2o/GTSAM/Ceres do per linearisation);
scipy's internal Levenberg-Marquardt iterations handle the nonlinearity,
and se3_exp is a valid global retraction so large perturbations are not
a correctness problem, only a conditioning one for extreme cases.
"""
from __future__ import annotations
from typing import Optional
import numpy as np
from scipy.optimize import least_squares
from scipy.linalg import cholesky

from pyslam.core.types import Link
from pyslam.core import lie


class NativeBackend:
    def __init__(self, huber_delta: float = 1.0):
        self.huber_delta = huber_delta
        self._poses: dict[int, np.ndarray] = {}
        self._links: list[Link] = []

    def add_node(self, id: int, pose: np.ndarray) -> None:
        self._poses[id] = pose.copy()

    def add_link(self, link: Link) -> None:
        self._links.append(link)

    def _factor_residual(self, T_a: np.ndarray, T_b: np.ndarray, link: Link) -> np.ndarray:
        T_ab_pred = lie.se3_inverse(T_a) @ T_b
        err = lie.se3_log(lie.se3_inverse(link.T_ab) @ T_ab_pred)  # (6,)
        L = cholesky(link.info, lower=False)  # L^T L = info
        w = L @ err
        if link.kind == "loop":
            n = np.linalg.norm(w)
            if n > self.huber_delta and n > 1e-12:
                w = w * np.sqrt(self.huber_delta / n)
        return w

    def optimize(self, fixed: list[int]) -> dict[int, np.ndarray]:
        node_ids = list(self._poses.keys())
        unknown_ids = [i for i in node_ids if i not in fixed]
        if len(unknown_ids) == 0:
            return {i: self._poses[i].copy() for i in node_ids}
        idx_of = {nid: k for k, nid in enumerate(unknown_ids)}
        base = {i: self._poses[i].copy() for i in node_ids}

        def unpack(x: np.ndarray) -> dict[int, np.ndarray]:
            poses = {}
            for i in node_ids:
                if i in idx_of:
                    xi = x[6 * idx_of[i]: 6 * idx_of[i] + 6]
                    poses[i] = base[i] @ lie.se3_exp(xi)
                else:
                    poses[i] = base[i]
            return poses

        def resfun(x: np.ndarray) -> np.ndarray:
            poses = unpack(x)
            out = []
            for link in self._links:
                if link.a not in poses or link.b not in poses:
                    continue
                out.append(self._factor_residual(poses[link.a], poses[link.b], link))
            if not out:
                return np.zeros(1)
            return np.concatenate(out)

        x0 = np.zeros(6 * len(unknown_ids))
        # method='lm' (MINPACK) requires #residuals >= #variables, which
        # fails early in a run (e.g. right after the very first loop
        # closure, when few links exist yet relative to free poses) --
        # hit exactly this on the corridor-loop fixture. 'trf' has no such
        # restriction and handles the rank-deficient/underdetermined case
        # gracefully (falls back toward the initial guess for unconstrained
        # directions), so it's the safer default here.
        result = least_squares(resfun, x0, method="trf", xtol=1e-12, ftol=1e-12,
                                gtol=1e-12, max_nfev=3000)
        final = unpack(result.x)
        self._poses = final
        return {i: final[i].copy() for i in node_ids}

    def get_poses(self) -> dict[int, np.ndarray]:
        return {i: p.copy() for i, p in self._poses.items()}
