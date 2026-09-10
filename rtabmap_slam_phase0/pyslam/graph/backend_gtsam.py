"""
GTSAM batch pose-graph backend. Primary backend on the target machine
(GTSAM is already installed there). Falls back automatically to
backend_native if gtsam cannot be imported, so development/CI on a
machine without GTSAM still works -- see posegraph.py.
"""
from __future__ import annotations
from typing import Optional
import numpy as np

from pyslam.core.types import Link
from pyslam.core import lie

try:
    import gtsam
    GTSAM_AVAILABLE = True
except ImportError:
    GTSAM_AVAILABLE = False


def _T_to_pose3(T: np.ndarray):
    R = gtsam.Rot3(T[:3, :3])
    t = gtsam.Point3(T[0, 3], T[1, 3], T[2, 3])
    return gtsam.Pose3(R, t)


def _pose3_to_T(p) -> np.ndarray:
    M = p.matrix()
    return np.array(M, dtype=np.float64)


def _info_to_noise(info: np.ndarray, huber_delta: Optional[float] = None):
    """info is in [rho(3), phi(3)] order (translation-first, matching
    pyslam.core.lie.se3_log); GTSAM's Pose3 noise convention is also
    translation-first for BetweenFactorPose3 in the wrapper's Python API
    (rotation, then translation is the C++ internal order for some
    versions -- we use full covariance via Gaussian::Information to sidestep
    ordering ambiguity entirely, since a 6x6 information matrix supplied
    directly is unambiguous as long as we are internally consistent, which
    we are: both backends use [rho, phi] and info is computed by us)."""
    noise = gtsam.noiseModel.Gaussian.Information(info)
    if huber_delta is not None:
        robust = gtsam.noiseModel.mEstimator.Huber(huber_delta)
        noise = gtsam.noiseModel.Robust(robust, noise)
    return noise


class GtsamBackend:
    def __init__(self, huber_delta: float = 1.0):
        if not GTSAM_AVAILABLE:
            raise RuntimeError("gtsam is not importable")
        self.huber_delta = huber_delta
        self._poses: dict[int, np.ndarray] = {}
        self._links: list[Link] = []

    def add_node(self, id: int, pose: np.ndarray) -> None:
        self._poses[id] = pose.copy()

    def add_link(self, link: Link) -> None:
        self._links.append(link)

    def optimize(self, fixed: list[int]) -> dict[int, np.ndarray]:
        graph = gtsam.NonlinearFactorGraph()
        values = gtsam.Values()
        node_ids = list(self._poses.keys())

        for nid in node_ids:
            values.insert(nid, _T_to_pose3(self._poses[nid]))

        if not fixed:
            raise ValueError("optimize() requires at least one fixed (gauge) node")
        prior_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([1e-6] * 6))
        for nid in fixed:
            graph.add(gtsam.PriorFactorPose3(nid, _T_to_pose3(self._poses[nid]), prior_noise))

        for link in self._links:
            if link.a not in self._poses or link.b not in self._poses:
                continue
            huber = self.huber_delta if link.kind == "loop" else None
            noise = _info_to_noise(link.info, huber)
            # se3_log convention is [rho, phi]; se3_exp/se3_log both use
            # this order consistently, and _T_to_pose3/pose3 use full 4x4
            # matrices directly, so no ordering translation is needed here
            # -- we pass link.T_ab as a full transform, not as a tangent
            # vector, sidestepping the [rho,phi] vs [phi,rho] ambiguity.
            graph.add(gtsam.BetweenFactorPose3(link.a, link.b, _T_to_pose3(link.T_ab), noise))

        params = gtsam.LevenbergMarquardtParams()
        params.setMaxIterations(200)
        optimizer = gtsam.LevenbergMarquardtOptimizer(graph, values, params)
        result = optimizer.optimize()

        out = {}
        for nid in node_ids:
            out[nid] = _pose3_to_T(result.atPose3(nid))
        self._poses = {i: out[i].copy() for i in node_ids}
        return out

    def get_poses(self) -> dict[int, np.ndarray]:
        return {i: p.copy() for i, p in self._poses.items()}
