"""
Single frozen config object. Hashed into every run directory so every
output is traceable to the exact settings that produced it.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict, field
import hashlib
import json


@dataclass(frozen=True)
class Config:
    # --- features ---
    n_features: int = 1000
    grid_cols: int = 8
    grid_rows: int = 6
    orb_scale_factor: float = 1.2
    orb_n_levels: int = 8

    # --- depth association ---
    depth_min_m: float = 0.3
    depth_max_m: float = 4.0
    depth_patch: int = 3          # (2k+1)x(2k+1) median patch
    depth_grad_max_m: float = 0.15  # reject if local depth gradient exceeds this

    # --- odometry ---
    keyframe_trans_m: float = 0.15
    keyframe_rot_deg: float = 15.0
    keyframe_min_inliers: int = 60
    odom_reproj_px: float = 3.0
    odom_min_inliers: int = 15
    lost_consecutive_frames: int = 5

    # --- memory ---
    stm_size: int = 10

    # --- vocabulary / retrieval ---
    vocab_size: int = 1000
    bow_top_k: int = 20

    # --- bayes filter ---
    bayes_neighbor_spread: float = 0.8   # mass kept on self+neighbours vs diffused
    bayes_new_place_prior: float = 0.6   # prior weight for "new place" transition
    hypothesis_threshold: float = 0.15
    hypothesis_hysteresis: int = 2

    # --- verification ---
    verify_min_inliers: int = 20
    verify_min_inlier_ratio: float = 0.3
    verify_reproj_px: float = 3.0
    verify_bidirectional_trans_m: float = 0.05
    verify_bidirectional_rot_deg: float = 2.0
    verify_max_translation_m: float = 3.0    # sanity bound: a WM candidate is
        # already appearance-matched (i.e. plausibly nearby); a "verified"
        # link proposing a huge jump is much more likely to be a PnP
        # bas-relief/degenerate solution (common on near-planar scenes,
        # e.g. corridor walls) than a real distant-but-textured-alike loop.
        # Full ICP-based refinement (Phase 4) removes the need for this
        # blunt cap; until then it is cheap, safe insurance.

    # --- graph ---
    prior_sigma_pos: float = 1e-6
    prior_sigma_rot: float = 1e-6
    odom_sigma_pos: float = 0.02
    odom_sigma_rot_deg: float = 1.0
    loop_huber_delta: float = 1.0

    # --- misc ---
    seed: int = 12345
    log_level: str = "INFO"

    def to_dict(self) -> dict:
        return asdict(self)

    def hash(self) -> str:
        blob = json.dumps(self.to_dict(), sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:12]

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump({"config": self.to_dict(), "hash": self.hash()}, f, indent=2)

    @staticmethod
    def load(path: str) -> "Config":
        with open(path) as f:
            d = json.load(f)
        return Config(**d["config"])
