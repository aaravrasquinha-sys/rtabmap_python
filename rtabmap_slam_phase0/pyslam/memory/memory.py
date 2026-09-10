"""
Phase-0 memory management.

Implements STM (fixed-size deque with rehearsal merge) and an unbounded
in-RAM WM. LTM / transfer-under-budget is Phase 2 -- the `Memory`
protocol already has the right shape (`enforce_budget`) so that upgrade
is a body swap, not an interface change.
"""
from __future__ import annotations
from collections import deque
from typing import Optional
import numpy as np

from pyslam.core.types import Node
from pyslam.core.config import Config


def bow_cosine_similarity(a_ids: np.ndarray, b_ids: np.ndarray) -> float:
    """Cheap rehearsal similarity: Jaccard-like overlap of assigned word ids.
    Used only for STM merge decisions, not for real place recognition."""
    if a_ids is None or b_ids is None or len(a_ids) == 0 or len(b_ids) == 0:
        return 0.0
    sa, sb = set(a_ids.tolist()), set(b_ids.tolist())
    inter = len(sa & sb)
    union = len(sa | sb)
    return inter / union if union > 0 else 0.0


class Memory:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.stm: deque[int] = deque()
        self.wm: dict[int, Node] = {}
        self.ltm: dict[int, Node] = {}   # unused in Phase 0, present for interface stability
        self._all: dict[int, Node] = {}
        self.rehearsal_threshold = 0.6

    # ------------------------------------------------------------ core API
    def add(self, node: Node) -> None:
        self._all[node.id] = node
        merged = False
        if node.sig.word_ids is not None:
            for other_id in list(self.stm):
                other = self._all[other_id]
                sim = bow_cosine_similarity(node.sig.word_ids, other.sig.word_ids)
                if sim >= self.rehearsal_threshold:
                    other.weight += node.weight + 1
                    merged = True
                    break
        if not merged:
            self.stm.append(node.id)
        while len(self.stm) > self.cfg.stm_size:
            old_id = self.stm.popleft()
            self.wm[old_id] = self._all[old_id]

    def working_set(self) -> list[int]:
        return list(self.wm.keys())

    def get(self, node_id: int) -> Node:
        return self._all[node_id]

    def on_loop(self, node_id: int) -> None:
        """Retrieval hook. No-op in Phase 0 (no LTM to retrieve from)."""
        return None

    def enforce_budget(self, last_update_ms: float) -> None:
        """Transfer-under-budget hook. No-op in Phase 0."""
        return None

    def all_node_ids(self) -> list[int]:
        return list(self._all.keys())

    def stm_ids(self) -> list[int]:
        return list(self.stm)
