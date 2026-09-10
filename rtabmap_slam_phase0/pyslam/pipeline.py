"""
The Phase-0 pipeline: a single-threaded sequential loop wiring every
subsystem together. Deliberately no threads (see architecture brief,
section 7 rule #1) -- this eliminates the hardest bug class while
correctness of the algorithm itself is established.

    capture -> features -> odometry -> memory -> retrieval -> bayes
             -> verify -> graph -> map

This file should stay short. If it grows past ~250 lines, logic has
leaked in that belongs in one of the subsystem modules.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import time
import numpy as np

from pyslam.core.types import Frame, Node, Link, SensorSource, Intrinsics
from pyslam.core.config import Config
from pyslam.core.log import get_logger

from pyslam.frontend.features import extract_signature, make_orb, _reset_id_counter
from pyslam.frontend.odometry import VisualOdometry
from pyslam.memory.memory import Memory
from pyslam.vpr.vocab import load_vocabulary, assign_words
from pyslam.vpr.bow import BowIndex
from pyslam.vpr.raw_match import score_candidates
from pyslam.vpr.likelihood import normalize_likelihood
from pyslam.loop.bayes import BayesFilter
from pyslam.loop.verify import GeometricVerifier
from pyslam.graph.posegraph import PoseGraph

log = get_logger("pipeline")


@dataclass
class PipelineResult:
    memory: Memory
    graph: PoseGraph
    node_gt: dict = field(default_factory=dict)          # node_id -> gt 4x4, synthetic only
    loop_events: list = field(default_factory=list)        # (new_id, old_id, Link)
    telemetry: list = field(default_factory=list)           # list of dict
    odom_trajectory: list = field(default_factory=list)     # (t, 4x4) every processed frame
    status_log: list = field(default_factory=list)
    n_frames: int = 0
    n_keyframes: int = 0


class Pipeline:
    def __init__(self, cfg: Config, vocab_path: Optional[str] = None,
                 backend_prefer: str = "auto"):
        self.cfg = cfg
        _reset_id_counter(0)
        self.orb = None  # created lazily once we know we're extracting features
        self.vocab = load_vocabulary(vocab_path) if vocab_path else None
        self.memory = Memory(cfg)
        self.bow_index = BowIndex(cfg.vocab_size) if self.vocab is not None else None
        self.bayes = BayesFilter(cfg)
        self.graph = PoseGraph(cfg, prefer_backend=backend_prefer)
        self.odometry: Optional[VisualOdometry] = None
        self.verifier: Optional[GeometricVerifier] = None
        self._prev_kf_id: Optional[int] = None
        self._first_node_id: Optional[int] = None

    def run(self, source: SensorSource, max_frames: Optional[int] = None,
            verbose: bool = True) -> PipelineResult:
        intr = source.intrinsics()
        self.odometry = VisualOdometry(self.cfg)
        self.verifier = GeometricVerifier(self.cfg, intr)
        self.orb = make_orb(self.cfg)

        result = PipelineResult(memory=self.memory, graph=self.graph)

        n = 0
        for frame in source:
            t0 = time.perf_counter()
            sig = extract_signature(frame, self.cfg, self.orb)
            if self.vocab is not None:
                sig.word_ids = assign_words(sig.desc, self.vocab)

            odom_res = self.odometry.update(sig, frame)
            result.odom_trajectory.append((frame.t, odom_res.pose.copy()))

            if self.odometry.status == "LOST":
                result.status_log.append(f"[t={frame.t:.2f}] LOST, attempting recovery")
                self.odometry.force_new_keyframe(sig)

            if self.odometry.last_keyframe_created:
                node = Node(id=sig.id, sig=sig,
                            pose_odom=self.odometry.cur_pose.copy(),
                            pose_map=self.odometry.cur_pose.copy(),
                            weight=1)
                self.memory.add(node)
                if self.bow_index is not None:
                    self.bow_index.add(node.id, sig.word_ids)
                self.graph.add_node(node.id, node.pose_odom)
                if self._first_node_id is None:
                    self._first_node_id = node.id

                if frame.gt_pose is not None:
                    result.node_gt[node.id] = frame.gt_pose.copy()

                if self._prev_kf_id is not None and self.odometry.last_keyframe_link_T is not None:
                    odom_link = Link(
                        a=self._prev_kf_id, b=node.id,
                        T_ab=self.odometry.last_keyframe_link_T,
                        info=self.odometry.last_keyframe_link_info,
                        kind="odom",
                        n_inliers=self.odometry.last_keyframe_link_inliers,
                    )
                    self.graph.add_link(odom_link)
                self._prev_kf_id = node.id
                result.n_keyframes += 1

                self._try_loop_closure(node, result)

            n += 1
            result.n_frames = n
            result.telemetry.append({
                "t": frame.t, "frame_id": frame.frame_id,
                "duration_ms": (time.perf_counter() - t0) * 1000.0,
                "odom_status": self.odometry.status,
                "n_inliers": odom_res.n_inliers,
                "wm_size": len(self.memory.working_set()),
                "keyframe": self.odometry.last_keyframe_created,
            })
            if verbose and n % 30 == 0:
                log.info(f"frame {n}: t={frame.t:.2f}s status={self.odometry.status} "
                         f"kf={result.n_keyframes} wm={len(self.memory.working_set())}")

            if max_frames is not None and n >= max_frames:
                break

        return result

    def _try_loop_closure(self, node: Node, result: PipelineResult) -> None:
        candidate_ids = self.memory.working_set()
        if not candidate_ids:
            return
        candidate_nodes = [self.memory.get(cid) for cid in candidate_ids]

        # See vpr/raw_match.py docstring for why Phase 0 scores retrieval
        # by direct descriptor matching rather than through the fixed
        # BoW vocabulary (bow_index is still populated above for future
        # phases, just not used to drive this decision yet).
        raw_scores = score_candidates(node.sig, candidate_nodes)

        likelihood = normalize_likelihood(raw_scores)
        hyp = self.bayes.update(likelihood, candidate_ids)
        if hyp is None:
            return

        old_node = self.memory.get(hyp.node_id)
        link = self.verifier.verify(old_node, node)
        if link is None:
            return

        self.graph.add_link(link)
        self.memory.on_loop(hyp.node_id)
        result.loop_events.append((node.id, hyp.node_id, link))
        log.info(f"LOOP CLOSURE: node {node.id} <-> node {hyp.node_id} "
                 f"(posterior={hyp.posterior:.3f}, inliers={link.n_inliers})")

        fixed = [self._first_node_id] if self._first_node_id is not None else []
        new_poses = self.graph.optimize(fixed)
        for nid, T in new_poses.items():
            try:
                self.memory.get(nid).pose_map = T
            except KeyError:
                pass
