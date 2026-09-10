from __future__ import annotations
from typing import Optional
import numpy as np

from pyslam.core.types import Link
from pyslam.core.config import Config
from pyslam.core.log import get_logger
from pyslam.graph.backend_native import NativeBackend
from pyslam.graph import backend_gtsam

log = get_logger("graph.posegraph")


def make_backend(cfg: Config, prefer: str = "auto"):
    """prefer: 'auto' | 'gtsam' | 'native'."""
    if prefer in ("auto", "gtsam"):
        if backend_gtsam.GTSAM_AVAILABLE:
            try:
                log.info("Using GTSAM pose-graph backend.")
                return backend_gtsam.GtsamBackend(huber_delta=cfg.loop_huber_delta)
            except Exception as e:
                log.warning(f"GTSAM backend failed to initialise ({e}); falling back to native.")
        elif prefer == "gtsam":
            raise RuntimeError("gtsam requested but not importable")
    log.info("Using native pose-graph backend.")
    return NativeBackend(huber_delta=cfg.loop_huber_delta)


class PoseGraph:
    def __init__(self, cfg: Config, prefer_backend: str = "auto"):
        self.cfg = cfg
        self.backend = make_backend(cfg, prefer_backend)
        self.node_ids: list[int] = []
        self.links: list[Link] = []

    def add_node(self, node_id: int, pose: np.ndarray) -> None:
        self.backend.add_node(node_id, pose)
        self.node_ids.append(node_id)

    def add_link(self, link: Link) -> None:
        self.backend.add_link(link)
        self.links.append(link)

    def optimize(self, fixed: Optional[list[int]] = None) -> dict[int, np.ndarray]:
        if fixed is None:
            fixed = [self.node_ids[0]] if self.node_ids else []
        return self.backend.optimize(fixed)

    def get_poses(self) -> dict[int, np.ndarray]:
        return self.backend.get_poses()
