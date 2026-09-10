"""
Phase-0 odometry: frame-to-KEYFRAME PnP tracking (not full frame-to-map;
that upgrade is Phase 1). Every incoming frame is matched against the most
recent keyframe's 3D points via PnP+RANSAC. When the tracked motion or
match quality crosses a threshold, the current frame is promoted to a new
keyframe and an odometry Link is emitted between consecutive keyframes.

This is deliberately the simplest thing that (a) doesn't drift
catastrophically over short sequences and (b) has an honest failure mode
(LOST) rather than silently producing garbage.
"""
from __future__ import annotations
from typing import Optional
import numpy as np
import cv2

from pyslam.core.types import Signature, Frame, OdomResult
from pyslam.core.config import Config
from pyslam.core import lie


class VisualOdometry:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        self.ref_sig: Optional[Signature] = None
        self.ref_pose: np.ndarray = np.eye(4)   # world<-cam pose of the reference keyframe
        self.cur_pose: np.ndarray = np.eye(4)   # world<-cam pose of the latest tracked frame

        self.last_keyframe_created: bool = False
        self.last_keyframe_signature: Optional[Signature] = None
        self.last_keyframe_link_T: Optional[np.ndarray] = None   # T_prevkf_newkf
        self.last_keyframe_link_info: Optional[np.ndarray] = None
        self.last_keyframe_link_inliers: int = 0

        self._lost_streak = 0
        self.status = "OK"
        self._initialised = False

    # ------------------------------------------------------------------
    def _match(self, sig_query: Signature, sig_train: Signature):
        if sig_query.desc.shape[0] == 0 or sig_train.desc.shape[0] == 0:
            return []
        train_valid_idx = np.where(sig_train.valid)[0]
        if train_valid_idx.size == 0:
            return []
        train_desc = sig_train.desc[train_valid_idx]
        knn = self.matcher.knnMatch(sig_query.desc, train_desc, k=2)
        good = []
        for pair in knn:
            if len(pair) < 2:
                continue
            m, n = pair
            if m.distance < 0.8 * n.distance:
                good.append((m.queryIdx, train_valid_idx[m.trainIdx]))
        return good

    def _pnp(self, obj_pts: np.ndarray, img_pts: np.ndarray, K: np.ndarray, reproj_px: float):
        if obj_pts.shape[0] < 6:
            return None
        try:
            flag = cv2.USAC_MAGSAC
        except AttributeError:
            flag = cv2.SOLVEPNP_ITERATIVE
        try:
            ok, rvec, tvec, inliers = cv2.solvePnPRansac(
                obj_pts.astype(np.float64), img_pts.astype(np.float64), K, None,
                reprojectionError=reproj_px, confidence=0.999, iterationsCount=300,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
        except cv2.error:
            return None
        if not ok or inliers is None or len(inliers) < self.cfg.odom_min_inliers:
            return None
        R, _ = cv2.Rodrigues(rvec)
        T_train_query = lie.make_T(R, tvec.reshape(3))  # maps ref-frame pts into query(cam) frame
        return T_train_query, inliers.reshape(-1)

    # ------------------------------------------------------------------
    def update(self, sig: Signature, frame: Frame) -> OdomResult:
        self.last_keyframe_created = False
        K = frame.intr.K()

        if not self._initialised:
            self.ref_sig = sig
            self.ref_pose = np.eye(4)
            self.cur_pose = np.eye(4)
            self._initialised = True
            self.last_keyframe_created = True
            self.last_keyframe_signature = sig
            self.last_keyframe_link_T = None
            self.status = "OK"
            self._lost_streak = 0
            return OdomResult(pose=self.cur_pose.copy(), T_rel=np.eye(4),
                               info=np.eye(6) * 1e6, n_inliers=0, status="OK")

        matches = self._match(sig, self.ref_sig)
        if len(matches) < self.cfg.odom_min_inliers:
            return self._handle_lost()

        q_idx = np.array([m[0] for m in matches])
        t_idx = np.array([m[1] for m in matches])
        obj_pts = self.ref_sig.kp3d[t_idx]          # 3D points in reference keyframe's camera frame
        img_pts = sig.kp[q_idx]

        result = self._pnp(obj_pts, img_pts, K, self.cfg.odom_reproj_px)
        if result is None:
            return self._handle_lost()
        T_ref_cur, inlier_idx = result             # p_ref = T_ref_cur @ p_cur  (ref<-cur)
        n_inliers = len(inlier_idx)

        self._lost_streak = 0
        self.status = "OK"

        T_cur_ref = lie.se3_inverse(T_ref_cur)      # cur<-ref, i.e. maps ref-frame pts to cur cam
        # cur_pose (world<-cur) = ref_pose (world<-ref) @ T_ref_cur? careful:
        # T_ref_cur maps points FROM cur frame INTO ref frame (p_ref = T_ref_cur @ p_cur).
        # We want world<-cur = world<-ref @ ref<-cur = self.ref_pose @ T_ref_cur.
        self.cur_pose = self.ref_pose @ T_ref_cur
        T_rel = T_ref_cur

        info = np.eye(6) * (n_inliers / max(len(matches), 1)) * 100.0

        # -------- keyframe decision --------
        xi = lie.se3_log(T_ref_cur)
        trans = np.linalg.norm(xi[:3])
        rot_deg = np.degrees(np.linalg.norm(xi[3:]))
        inlier_ratio = n_inliers / max(len(matches), 1)

        need_new_kf = (
            trans > self.cfg.keyframe_trans_m or
            rot_deg > self.cfg.keyframe_rot_deg or
            n_inliers < self.cfg.keyframe_min_inliers
        )
        if need_new_kf:
            self.last_keyframe_created = True
            self.last_keyframe_signature = sig
            self.last_keyframe_link_T = T_ref_cur.copy()
            self.last_keyframe_link_info = info.copy()
            self.last_keyframe_link_inliers = n_inliers
            self.ref_sig = sig
            self.ref_pose = self.cur_pose.copy()

        return OdomResult(pose=self.cur_pose.copy(), T_rel=T_rel, info=info,
                           n_inliers=n_inliers, status="OK")

    def _handle_lost(self) -> OdomResult:
        self._lost_streak += 1
        if self._lost_streak >= self.cfg.lost_consecutive_frames:
            self.status = "LOST"
        return OdomResult(pose=self.cur_pose.copy(), T_rel=np.eye(4),
                           info=np.eye(6) * 1e-6, n_inliers=0, status=self.status)

    def force_new_keyframe(self, sig: Signature) -> None:
        """Used to bootstrap tracking again after LOST (Phase 0: simply
        restarts local tracking from the current frame; global relocalisation
        against WM is a later-phase upgrade)."""
        self.ref_sig = sig
        self.ref_pose = self.cur_pose.copy()
        self._lost_streak = 0
        self.status = "OK"
        self.last_keyframe_created = True
        self.last_keyframe_signature = sig
        self.last_keyframe_link_T = None
