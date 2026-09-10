"""
Turns a loop-closure hypothesis into a trustworthy 6-DoF constraint.

Phase-0 scope: PnP+RANSAC in both directions (a's 3D vs b's 2D, and vice
versa), requiring the two independent estimates to agree before
accepting. This bidirectional check is cheap insurance against the
single most damaging SLAM failure mode (a confidently wrong loop
closure) and is deliberately present from day one rather than deferred.

ICP depth-refinement and Hessian-based covariance are later-phase
upgrades (Phase 4/7 in the architecture brief); Phase 0 uses an
inlier/residual-driven heuristic information matrix, which is enough to
make the optimiser's weighting meaningful without being falsely precise.
"""
from __future__ import annotations
from typing import Optional
import numpy as np
import cv2

from pyslam.core.types import Node, Link, Intrinsics
from pyslam.core.config import Config
from pyslam.core import lie


def _match(desc_a: np.ndarray, desc_b: np.ndarray, ratio: float = 0.8):
    if desc_a.shape[0] == 0 or desc_b.shape[0] == 0:
        return []
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    knn = matcher.knnMatch(desc_a, desc_b, k=2)
    good = []
    for pair in knn:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < ratio * n.distance:
            good.append((m.queryIdx, m.trainIdx))
    return good


def _solve_pnp(obj_pts: np.ndarray, img_pts: np.ndarray, K: np.ndarray, reproj_px: float,
                min_inliers: int):
    if obj_pts.shape[0] < 6:
        return None
    try:
        ok, rvec, tvec, inliers = cv2.solvePnPRansac(
            obj_pts.astype(np.float64), img_pts.astype(np.float64), K, None,
            reprojectionError=reproj_px, confidence=0.999, iterationsCount=500,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
    except cv2.error:
        return None
    if not ok or inliers is None or len(inliers) < min_inliers:
        return None
    R, _ = cv2.Rodrigues(rvec)
    T_cam_obj = lie.make_T(R, tvec.reshape(3))
    inlier_idx = inliers.reshape(-1)
    # residual RMS over inliers
    proj, _ = cv2.projectPoints(obj_pts[inlier_idx], rvec, tvec, K, None)
    proj = proj.reshape(-1, 2)
    residual = np.linalg.norm(proj - img_pts[inlier_idx], axis=1)
    rms = float(np.sqrt(np.mean(residual ** 2))) if len(residual) else 0.0
    return T_cam_obj, inlier_idx, rms


class GeometricVerifier:
    def __init__(self, cfg: Config, intr: Intrinsics):
        self.cfg = cfg
        self.K = intr.K()

    def verify(self, a: Node, b: Node) -> Optional[Link]:
        sig_a, sig_b = a.sig, b.sig
        matches_ab = _match(sig_a.desc, sig_b.desc)   # (idx_a, idx_b)
        if len(matches_ab) < self.cfg.verify_min_inliers:
            return None

        idx_a = np.array([m[0] for m in matches_ab])
        idx_b = np.array([m[1] for m in matches_ab])

        # --- direction 1: a's 3D points -> b's image (gives T_b_a: b<-a) ---
        va = sig_a.valid[idx_a]
        r1 = None
        if va.sum() >= self.cfg.verify_min_inliers:
            obj = sig_a.kp3d[idx_a[va]]
            img = sig_b.kp[idx_b[va]]
            r1 = _solve_pnp(obj, img, self.K, self.cfg.verify_reproj_px, self.cfg.verify_min_inliers)

        # --- direction 2: b's 3D points -> a's image (gives T_a_b: a<-b) ---
        vb = sig_b.valid[idx_b]
        r2 = None
        if vb.sum() >= self.cfg.verify_min_inliers:
            obj = sig_b.kp3d[idx_b[vb]]
            img = sig_a.kp[idx_a[vb]]
            r2 = _solve_pnp(obj, img, self.K, self.cfg.verify_reproj_px, self.cfg.verify_min_inliers)

        if r1 is None and r2 is None:
            return None

        T_ab_from_1 = lie.se3_inverse(r1[0]) if r1 is not None else None  # a<-b
        T_ab_from_2 = r2[0] if r2 is not None else None                   # a<-b directly

        if T_ab_from_1 is not None and T_ab_from_2 is not None:
            xi_diff = lie.se3_log(lie.se3_inverse(T_ab_from_1) @ T_ab_from_2)
            trans_diff = np.linalg.norm(xi_diff[:3])
            rot_diff = np.degrees(np.linalg.norm(xi_diff[3:]))
            if (trans_diff > self.cfg.verify_bidirectional_trans_m or
                    rot_diff > self.cfg.verify_bidirectional_rot_deg):
                return None  # the two independent estimates disagree -> reject
            n1, n2 = len(r1[1]), len(r2[1])
            T_ab = T_ab_from_1 if n1 >= n2 else T_ab_from_2
            n_inliers = max(n1, n2)
            rms = min(r1[2], r2[2])
        elif T_ab_from_1 is not None:
            T_ab, n_inliers, rms = T_ab_from_1, len(r1[1]), r1[2]
        else:
            T_ab, n_inliers, rms = T_ab_from_2, len(r2[1]), r2[2]

        inlier_ratio = n_inliers / max(len(matches_ab), 1)
        if inlier_ratio < self.cfg.verify_min_inlier_ratio:
            return None

        if np.linalg.norm(T_ab[:3, 3]) > self.cfg.verify_max_translation_m:
            return None  # see verify_max_translation_m docstring in config.py

        # heuristic information matrix: more inliers + lower residual -> tighter
        conf = (n_inliers / max(self.cfg.verify_min_inliers, 1)) * (1.0 / (1.0 + rms))
        conf = float(np.clip(conf, 0.05, 50.0))
        info = np.eye(6) * conf

        return Link(a=a.id, b=b.id, T_ab=T_ab, info=info, kind="loop",
                    n_inliers=n_inliers, inlier_ratio=inlier_ratio, residual_rms=rms)
