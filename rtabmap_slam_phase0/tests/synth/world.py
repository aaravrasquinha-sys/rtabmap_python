"""
Synthetic world generator for gate testing.

Produces analytic scenes (textured planar walls) so that ground-truth
projection and back-projection can be checked to near machine precision
when noise is disabled. This is deliberately simple geometry -- the goal
is a trustworthy oracle, not a photorealistic renderer.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple
import numpy as np

from pyslam.core.types import Intrinsics, Frame
from pyslam.core import lie


# ---------------------------------------------------------------- geometry

@dataclass
class Wall:
    """An axis-aligned rectangular wall patch with a precomputed random
    (non-periodic) texture, so that no two walls -- and no two patches
    within a wall -- look alike by construction. An earlier version used a
    deterministic sin/checker formula, which is periodic: local ORB
    patches from genuinely different physical locations could match each
    other by coincidence (verified on the corridor-loop fixture -- it
    caused both weak BoW discriminability and an outright false
    mid-corridor loop-closure hypothesis). Per-wall seeded noise has no
    such structure: two walls are statistically independent draws."""
    origin: np.ndarray      # (3,) one corner, world frame
    u_axis: np.ndarray      # (3,) unit vector, in-plane
    v_axis: np.ndarray      # (3,) unit vector, in-plane, orthogonal to u_axis
    normal: np.ndarray      # (3,) unit vector, outward
    u_len: float
    v_len: float
    texture_seed: int = 0
    texels_per_m: float = 24.0    # fine-detail resolution (corner richness for ORB)
    coarse_cells_per_wall: int = 10  # regional-distinctiveness resolution (for BoW)

    def __post_init__(self):
        rng = np.random.default_rng(self.texture_seed)
        fine_nu = max(4, int(self.u_len * self.texels_per_m))
        fine_nv = max(4, int(self.v_len * self.texels_per_m))
        coarse_nu = max(2, self.coarse_cells_per_wall)
        coarse_nv = max(2, self.coarse_cells_per_wall)

        # three independent RGB channels, each a mix of a coarse random
        # field (regional distinctiveness) and a fine random field (dense
        # corners for ORB); nearest-neighbour sampled, no interpolation
        # needed -- block noise gives sharp, corner-rich boundaries.
        self._fine = rng.integers(0, 256, size=(fine_nv, fine_nu, 3), dtype=np.uint16)
        self._coarse = rng.integers(0, 256, size=(coarse_nv, coarse_nu, 3), dtype=np.uint16)

    def sample_color(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """u, v pre-normalised to [0,1] (i.e. already divided by u_len/v_len
        by the caller)."""
        fv, fu = self._fine.shape[0], self._fine.shape[1]
        cv_, cu = self._coarse.shape[0], self._coarse.shape[1]
        fi = np.clip((v * fv).astype(np.int64), 0, fv - 1)
        fj = np.clip((u * fu).astype(np.int64), 0, fu - 1)
        ci = np.clip((v * cv_).astype(np.int64), 0, cv_ - 1)
        cj = np.clip((u * cu).astype(np.int64), 0, cu - 1)
        fine_col = self._fine[fi, fj].astype(np.float64)
        coarse_col = self._coarse[ci, cj].astype(np.float64)
        mixed = 0.45 * fine_col + 0.55 * coarse_col
        return np.clip(mixed, 0, 255).astype(np.uint8)


@dataclass
class World:
    walls: List[Wall] = field(default_factory=list)

    def add_box_room(self, center: np.ndarray, size: np.ndarray, seed0: int = 0):
        """Interior-facing walls of an axis-aligned box (a simple room)."""
        cx, cy, cz = center
        sx, sy, sz = size / 2.0
        specs = [
            # origin, u_axis, v_axis, normal, u_len, v_len
            (np.array([cx - sx, cy - sy, cz - sz]), np.array([0, 1, 0]), np.array([0, 0, 1]),
             np.array([1, 0, 0]), 2 * sy, 2 * sz),   # -X wall, normal +x
            (np.array([cx + sx, cy - sy, cz - sz]), np.array([0, 1, 0]), np.array([0, 0, 1]),
             np.array([-1, 0, 0]), 2 * sy, 2 * sz),  # +X wall, normal -x
            (np.array([cx - sx, cy - sy, cz - sz]), np.array([1, 0, 0]), np.array([0, 0, 1]),
             np.array([0, 1, 0]), 2 * sx, 2 * sz),   # -Y wall
            (np.array([cx - sx, cy + sy, cz - sz]), np.array([1, 0, 0]), np.array([0, 0, 1]),
             np.array([0, -1, 0]), 2 * sx, 2 * sz),  # +Y wall
            (np.array([cx - sx, cy - sy, cz - sz]), np.array([1, 0, 0]), np.array([0, 1, 0]),
             np.array([0, 0, 1]), 2 * sx, 2 * sy),   # floor
            (np.array([cx - sx, cy - sy, cz + sz]), np.array([1, 0, 0]), np.array([0, 1, 0]),
             np.array([0, 0, -1]), 2 * sx, 2 * sy),  # ceiling
        ]
        for i, (o, u, v, n, ul, vl) in enumerate(specs):
            self.walls.append(Wall(o.astype(float), u.astype(float), v.astype(float),
                                    n.astype(float), ul, vl, texture_seed=seed0 + i))

    def add_corridor_loop(self, seed0: int = 100):
        """A rectangular corridor (hollow box ring) guaranteeing a revisit."""
        # Four straight segments forming a loop, each modelled as a pair of
        # facing walls + floor/ceiling. Kept simple: reuse box rooms chained.
        segs = [
            (np.array([0, 0, 0]), np.array([6, 2, 2.4])),
            (np.array([6, 3, 0]), np.array([2, 4, 2.4])),
            (np.array([0, 6, 0]), np.array([6, 2, 2.4])),
            (np.array([-3, 3, 0]), np.array([2, 4, 2.4])),
        ]
        for i, (c, s) in enumerate(segs):
            self.add_box_room(c, s, seed0=seed0 + i * 10)

    def add_two_rooms(self, seed_a: int = 200, seed_b: int = 200):
        """Two rooms with IDENTICAL texture seeds -> deliberate perceptual
        aliasing, for the false-loop-closure gate."""
        self.add_box_room(np.array([0, 0, 0]), np.array([4, 4, 2.4]), seed0=seed_a)
        self.add_box_room(np.array([20, 0, 0]), np.array([4, 4, 2.4]), seed0=seed_b)


# ---------------------------------------------------------------- renderer

class Renderer:
    """Analytic ray-plane intersection renderer: pinhole projection with
    z-buffering across the wall set. Exact (no interpolation error) so it
    can serve as a ground-truth oracle."""

    def __init__(self, world: World, intr: Intrinsics):
        self.world = world
        self.intr = intr

    def render(self, T_world_body: np.ndarray, T_body_cam: np.ndarray = None,
               rgb_noise_std: float = 0.0, depth_noise_model=None,
               rng: np.random.Generator = None) -> Tuple[np.ndarray, np.ndarray]:
        """Returns (rgb HxWx3 uint8, depth HxW uint16 in mm)."""
        if T_body_cam is None:
            T_body_cam = np.eye(4)
        T_world_cam = T_world_body @ T_body_cam
        T_cam_world = lie.se3_inverse(T_world_cam)

        W, H = self.intr.width, self.intr.height
        fx, fy, cx, cy = self.intr.fx, self.intr.fy, self.intr.cx, self.intr.cy

        xs, ys = np.meshgrid(np.arange(W), np.arange(H))
        dirs_cam = np.stack([(xs - cx) / fx, (ys - cy) / fy, np.ones_like(xs, dtype=np.float64)],
                             axis=-1).reshape(-1, 3)
        R_wc = T_world_cam[:3, :3]
        origin_w = T_world_cam[:3, 3]
        # IMPORTANT: do NOT normalise dirs_w. dirs_cam has z==1 for every
        # pixel by construction ([(x-cx)/fx, (y-cy)/fy, 1]); a depth camera
        # reports Z-depth (camera-frame forward distance), not Euclidean
        # range along the ray. Solving the plane intersection with this
        # un-normalised, rotation-only-transformed direction makes the
        # resulting `t` exactly equal to camera-frame Z, matching how a
        # real RGB-D sensor (and our own back-projection: p = Kinv@[u,v,1]*Z)
        # defines depth. Normalising here was the bug: it silently returned
        # Euclidean range, which only coincides with Z-depth at the
        # principal point and diverges toward the image periphery.
        dirs_w = dirs_cam @ R_wc.T

        best_t = np.full(dirs_w.shape[0], np.inf)
        best_color = np.zeros((dirs_w.shape[0], 3), dtype=np.float64)
        hit_mask = np.zeros(dirs_w.shape[0], dtype=bool)

        for wall in self.world.walls:
            n = wall.normal
            denom = dirs_w @ n
            valid = np.abs(denom) > 1e-9
            t = np.full(dirs_w.shape[0], np.inf)
            t[valid] = ((wall.origin - origin_w) @ n) / denom[valid]
            pts = origin_w[None, :] + dirs_w * t[:, None]
            rel = pts - wall.origin[None, :]
            u = rel @ wall.u_axis
            v = rel @ wall.v_axis
            inb = valid & (t > 1e-4) & (u >= 0) & (u <= wall.u_len) & (v >= 0) & (v <= wall.v_len)
            closer = inb & (t < best_t)
            if not np.any(closer):
                continue
            colors = wall.sample_color(u[closer] / max(wall.u_len, 1e-6),
                                        v[closer] / max(wall.v_len, 1e-6))
            best_t[closer] = t[closer]
            best_color[closer] = colors
            hit_mask[closer] = True

        rgb = best_color.reshape(H, W, 3)
        if rgb_noise_std > 0 and rng is not None:
            rgb = rgb + rng.normal(scale=rgb_noise_std, size=rgb.shape)
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)

        depth_m = best_t.reshape(H, W).copy()
        depth_m[~hit_mask.reshape(H, W)] = 0.0
        depth_m[depth_m > self.intr.depth_scale * 65535] = 0.0

        if depth_noise_model is not None and rng is not None:
            depth_m = depth_noise_model(depth_m, hit_mask.reshape(H, W), self.intr, rng)

        depth_raw = np.round(depth_m / self.intr.depth_scale).astype(np.uint16)
        return rgb, depth_raw


def stereo_depth_noise(depth_m: np.ndarray, valid: np.ndarray, intr: Intrinsics,
                        rng: np.random.Generator, sigma_d_px: float = 0.15) -> np.ndarray:
    """sigma_z = z^2 * sigma_d / (f * b) -- the stereo-depth noise model."""
    out = depth_m.copy()
    z = depth_m
    with np.errstate(divide='ignore', invalid='ignore'):
        sigma_z = (z ** 2) * sigma_d_px / (intr.fx * intr.baseline)
    noise = rng.normal(scale=np.where(valid, sigma_z, 0.0))
    out = np.where(valid, z + noise, 0.0)
    out[out < 0] = 0.0
    return out


# ---------------------------------------------------------------- trajectory

def spline_trajectory(waypoints: np.ndarray, n_samples: int, loop: bool = False) -> np.ndarray:
    """Catmull-Rom spline through waypoints (N,3) -> (n_samples,3), C1 continuous."""
    pts = waypoints
    n = len(pts)
    if loop:
        pts = np.vstack([pts[-1:], pts, pts[:2]])
    else:
        pts = np.vstack([pts[:1], pts, pts[-1:]])

    out = []
    n_segs = n if loop else n - 1
    per_seg = max(1, n_samples // max(n_segs, 1))
    for i in range(n_segs):
        p0, p1, p2, p3 = pts[i], pts[i + 1], pts[i + 2], pts[i + 3]
        for s in range(per_seg):
            t = s / per_seg
            t2, t3 = t * t, t * t * t
            pt = 0.5 * ((2 * p1) + (-p0 + p2) * t +
                        (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2 +
                        (-p0 + 3 * p1 - 3 * p2 + p3) * t3)
            out.append(pt)
    return np.array(out)


def poses_from_path(path_xyz: np.ndarray, up=np.array([0, 0, 1.0]),
                     look_ahead: int = 3) -> List[np.ndarray]:
    """Build world<-body poses that face along the direction of travel,
    body convention: x forward, y left, z up (then a fixed body->camera
    extrinsic converts to the optical frame)."""
    n = len(path_xyz)
    poses = []
    for i in range(n):
        j = min(i + look_ahead, n - 1)
        fwd = path_xyz[j] - path_xyz[i]
        if np.linalg.norm(fwd) < 1e-6:
            fwd = np.array([1.0, 0, 0])
        fwd = fwd / np.linalg.norm(fwd)
        right = np.cross(fwd, up)
        if np.linalg.norm(right) < 1e-6:
            right = np.array([0, 1.0, 0])
        right /= np.linalg.norm(right)
        true_up = np.cross(right, fwd)
        R = np.stack([fwd, right, true_up], axis=1)  # columns = body axes in world
        T = lie.make_T(R, path_xyz[i])
        poses.append(T)
    return poses


# body (x-fwd, y-left, z-up) -> camera optical (x-right, y-down, z-fwd).
# T_body_cam maps points in the camera frame into the body frame, so its
# rotation COLUMNS are the camera axes expressed in the body frame:
#   x_cam (right)   in body = -y_body   -> column 0 = [0,-1,0]
#   y_cam (down)    in body = -z_body   -> column 1 = [0, 0,-1]
#   z_cam (forward) in body =  x_body   -> column 2 = [1, 0, 0]
_R_BODY_CAM = np.column_stack([
    np.array([0, -1, 0.0]),
    np.array([0, 0, -1.0]),
    np.array([1, 0, 0.0]),
])
T_BODY_CAM = lie.make_T(_R_BODY_CAM, np.zeros(3))
