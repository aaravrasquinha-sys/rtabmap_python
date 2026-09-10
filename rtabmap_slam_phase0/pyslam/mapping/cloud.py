"""
Global cloud is always a VIEW: each node's per-node cloud (in its own
camera frame, computed once from its Signature) is re-transformed by the
node's current pose_map every time this is called. Nothing is
accumulated irreversibly, so a graph re-optimisation just means calling
this again.
"""
from __future__ import annotations
import numpy as np

from pyslam.core.types import Node
from pyslam.core import lie


def voxel_downsample(pts: np.ndarray, colors: np.ndarray, voxel: float) -> tuple[np.ndarray, np.ndarray]:
    if pts.shape[0] == 0:
        return pts, colors
    keys = np.floor(pts / voxel).astype(np.int64)
    _, unique_idx = np.unique(keys, axis=0, return_index=True)
    return pts[unique_idx], colors[unique_idx]


def assemble_cloud(nodes: list[Node], K: np.ndarray, depth_scale: float,
                    stride: int = 4, max_depth_m: float = 4.0,
                    voxel: float = 0.03) -> tuple[np.ndarray, np.ndarray]:
    """Regenerate the full world-frame cloud from scratch using each
    node's current pose_map. Deterministic given the node set + poses."""
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    all_pts = []
    all_colors = []
    for node in nodes:
        if node.sig.depth is None or node.sig.rgb is None:
            continue
        depth = node.sig.depth[::stride, ::stride].astype(np.float64) * depth_scale
        rgb = node.sig.rgb[::stride, ::stride]
        hh, ww = depth.shape
        ys, xs = np.meshgrid(np.arange(hh), np.arange(ww), indexing="ij")
        px = xs * stride
        py = ys * stride
        z = depth
        valid = (z > 0) & (z < max_depth_m)
        if not np.any(valid):
            continue
        x = (px[valid] - cx) / fx * z[valid]
        y = (py[valid] - cy) / fy * z[valid]
        pts_cam = np.stack([x, y, z[valid]], axis=1).astype(np.float64)
        colors = rgb[valid]

        pts_world = lie.transform_points(node.pose_map, pts_cam)
        all_pts.append(pts_world)
        all_colors.append(colors)

    if not all_pts:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint8)

    pts = np.concatenate(all_pts, axis=0).astype(np.float32)
    colors = np.concatenate(all_colors, axis=0).astype(np.uint8)
    if voxel > 0:
        pts, colors = voxel_downsample(pts, colors, voxel)
    return pts, colors
