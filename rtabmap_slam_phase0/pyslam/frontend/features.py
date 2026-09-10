"""
Phase-0 Sensory Memory: ORB features, grid-bucketed for spatial spread,
associated with depth (median-of-patch, range + gradient rejection),
back-projected into the camera frame.

Depth uncertainty model (used later by odometry/verify weighting):
  sigma_z ~= z^2 * sigma_d / (f * b)      (stereo depth camera)
"""
from __future__ import annotations
import numpy as np
import cv2

from pyslam.core.types import Frame, Signature, Intrinsics
from pyslam.core.config import Config

_next_id = [0]


def _reset_id_counter(start: int = 0) -> None:
    _next_id[0] = start


def _alloc_id() -> int:
    i = _next_id[0]
    _next_id[0] += 1
    return i


def make_orb(cfg: Config):
    return cv2.ORB_create(
        nfeatures=cfg.n_features * 3,   # over-detect, then grid-bucket down
        scaleFactor=cfg.orb_scale_factor,
        nlevels=cfg.orb_n_levels,
    )


def grid_bucket(kps, descs, cfg: Config, width: int, height: int):
    """Keep at most n_features total, spread across a grid_cols x grid_rows
    grid, ranked by response within each cell. Prevents all keypoints
    clustering on one high-texture patch."""
    if len(kps) == 0:
        return [], np.zeros((0, 32), dtype=np.uint8)
    cell_w = width / cfg.grid_cols
    cell_h = height / cfg.grid_rows
    per_cell = max(1, cfg.n_features // (cfg.grid_cols * cfg.grid_rows))

    buckets: dict[tuple[int, int], list[int]] = {}
    for idx, kp in enumerate(kps):
        cx = min(cfg.grid_cols - 1, int(kp.pt[0] // cell_w))
        cy = min(cfg.grid_rows - 1, int(kp.pt[1] // cell_h))
        buckets.setdefault((cx, cy), []).append(idx)

    keep = []
    for key, idxs in buckets.items():
        idxs.sort(key=lambda i: kps[i].response, reverse=True)
        keep.extend(idxs[:per_cell])

    keep = keep[: cfg.n_features] if len(keep) > cfg.n_features else keep
    out_kps = [kps[i] for i in keep]
    out_descs = descs[keep] if len(keep) > 0 else np.zeros((0, 32), dtype=np.uint8)
    return out_kps, out_descs


def associate_depth(kp_xy: np.ndarray, depth_raw: np.ndarray, intr: Intrinsics, cfg: Config):
    """For each 2D keypoint, sample a median depth patch, reject on range /
    local-gradient grounds, and back-project into the camera optical frame.
    Returns (kp3d (N,3) float32, valid (N,) bool).

    Vectorised across all keypoints at once (patch extraction via a single
    fancy-index gather + masked-array median/range) rather than a
    per-keypoint Python loop -- the latter measured as the dominant
    per-frame cost (~450ms/frame at ~900 keypoints), which is well outside
    even offline-iteration budget, let alone the >=10Hz live-camera target.
    """
    n = kp_xy.shape[0]
    kp3d = np.full((n, 3), np.nan, dtype=np.float32)
    valid = np.zeros(n, dtype=bool)
    if n == 0:
        return kp3d, valid

    h, w = depth_raw.shape
    depth_m = depth_raw.astype(np.float64) * intr.depth_scale
    k = cfg.depth_patch

    padded = np.pad(depth_m, k, mode="constant", constant_values=0.0)
    ix = np.clip(np.round(kp_xy[:, 0]).astype(np.int64), 0, w - 1)
    iy = np.clip(np.round(kp_xy[:, 1]).astype(np.int64), 0, h - 1)
    cx, cy = ix + k, iy + k  # coords in the padded array

    offsets = np.arange(-k, k + 1)
    doff, joff = np.meshgrid(offsets, offsets, indexing="ij")
    doff, joff = doff.ravel(), joff.ravel()  # each (2k+1)^2,

    patch_y = cy[:, None] + doff[None, :]     # (N, P)
    patch_x = cx[:, None] + joff[None, :]     # (N, P)
    patches = padded[patch_y, patch_x]        # (N, P)

    valid_mask = patches > 0
    counts = valid_mask.sum(axis=1)
    enough = counts >= 5

    masked = np.ma.array(patches, mask=~valid_mask)
    z_med = np.ma.median(masked, axis=1).filled(0.0)
    z_max = np.ma.max(masked, axis=1).filled(0.0)
    z_min = np.ma.min(masked, axis=1).filled(0.0)
    spread = z_max - z_min

    ok = (enough & (z_med >= cfg.depth_min_m) & (z_med <= cfg.depth_max_m) &
          (spread <= cfg.depth_grad_max_m))

    px, py = kp_xy[:, 0], kp_xy[:, 1]
    x = (px - intr.cx) / intr.fx * z_med
    y = (py - intr.cy) / intr.fy * z_med
    kp3d[ok] = np.stack([x[ok], y[ok], z_med[ok]], axis=1).astype(np.float32)
    valid[ok] = True
    return kp3d, valid


def extract_signature(frame: Frame, cfg: Config, orb=None) -> Signature:
    if orb is None:
        orb = make_orb(cfg)
    gray = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2GRAY)
    kps, descs = orb.detectAndCompute(gray, None)
    if descs is None:
        descs = np.zeros((0, 32), dtype=np.uint8)
    kps, descs = grid_bucket(kps, descs, cfg, frame.intr.width, frame.intr.height)

    kp_xy = np.array([kp.pt for kp in kps], dtype=np.float32).reshape(-1, 2)
    kp3d, valid = associate_depth(kp_xy, frame.depth, frame.intr, cfg)

    return Signature(
        id=_alloc_id(),
        t=frame.t,
        kp=kp_xy,
        kp3d=kp3d,
        desc=descs,
        valid=valid,
        rgb=frame.rgb,
        depth=frame.depth,
    )
