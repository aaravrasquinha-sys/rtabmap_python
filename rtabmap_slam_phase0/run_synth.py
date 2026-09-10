"""
Run the pipeline against a synthetic scene -- no hardware, no recorded
bag needed. Useful as a quick sanity check that the code runs at all
after an edit, and as a demo of the flagship square-loop scenario used
in the gate suite.

    python3 run_synth.py
    python3 run_synth.py --scenario corridor   # harder, known not to close the loop yet (see Phase Evaluation doc)
"""
from __future__ import annotations
import argparse
import json
import os
import time
import sys
import numpy as np

sys.path.insert(0, ".")

from pyslam.core.config import Config
from pyslam.core.log import get_logger
from pyslam.core.types import Intrinsics
from pyslam.pipeline import Pipeline
from pyslam.sensors.synthetic import SyntheticSource
from pyslam.tools.evaluate import ate_rmse
from pyslam.mapping.cloud import assemble_cloud
from pyslam.mapping.export import write_ply
from tests.synth.world import World, poses_from_path, spline_trajectory

log = get_logger("run_synth")

# the D435i rig intrinsics this whole project targets
RIG_INTRINSICS = Intrinsics(fx=606.75, fy=606.57, cx=320.19, cy=237.06,
                             width=640, height=480,
                             depth_scale=0.0010000000474974513, baseline=0.0499)


def square_loop_scenario():
    w = World()
    w.add_box_room(np.array([0, 0, 0]), np.array([3, 3, 2.4]), seed0=1)
    waypoints = np.array([
        [0.0, 0.0, 0.0], [0.7, 0.0, 0.0], [0.7, 0.7, 0.0], [0.0, 0.7, 0.0],
        [0.0, 0.0, 0.0], [0.1, 0.02, 0.0],
    ])
    path = spline_trajectory(waypoints, n_samples=120, loop=False)
    poses = poses_from_path(path, look_ahead=4)
    return w, poses


def corridor_scenario():
    """Known-hard scenario: larger rectangular corridor loop. Odometry
    drift and retrieval are both harder here; see the Phase Evaluation
    doc's 'known limitations' section. Kept in the codebase as the
    Phase-1+ target, not a Phase-0 pass/fail gate."""
    w = World()
    w.add_corridor_loop(seed0=100)
    waypoints = np.array([
        [0.0, 1.0, 0.0], [5.0, 1.0, 0.0], [6.5, 1.0, 0.0], [6.5, 5.0, 0.0],
        [6.5, 6.5, 0.0], [1.5, 6.5, 0.0], [0.0, 6.5, 0.0], [-1.5, 5.0, 0.0],
        [-1.5, 1.0, 0.0], [0.0, 1.0, 0.0],
    ])
    path = spline_trajectory(waypoints, n_samples=180, loop=False)
    poses = poses_from_path(path, look_ahead=4)
    return w, poses


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", choices=["square", "corridor"], default="square")
    ap.add_argument("--backend", choices=["auto", "gtsam", "native"], default="auto")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    run_dir = os.path.join("runs", f"run_{int(time.time())}_synth_{args.scenario}")
    os.makedirs(run_dir, exist_ok=True)

    world, poses = square_loop_scenario() if args.scenario == "square" else corridor_scenario()
    source = SyntheticSource(world, RIG_INTRINSICS, poses, dt=1 / 10.0, add_noise=True, seed=args.seed)

    cfg = Config()
    cfg.save(os.path.join(run_dir, "config.json"))
    pipeline = Pipeline(cfg, vocab_path=None, backend_prefer=args.backend)
    result = pipeline.run(source, verbose=True)

    node_ids = sorted(result.node_gt.keys())
    est_odom = np.array([pipeline.memory.get(i).pose_odom[:3, 3] for i in node_ids])
    est_map = np.array([pipeline.memory.get(i).pose_map[:3, 3] for i in node_ids])
    gt = np.array([result.node_gt[i][:3, 3] for i in node_ids])
    ate_odom = ate_rmse(est_odom, gt)
    ate_map = ate_rmse(est_map, gt)

    summary = {
        "scenario": args.scenario,
        "n_frames": result.n_frames,
        "n_keyframes": result.n_keyframes,
        "n_loop_closures": len(result.loop_events),
        "ate_odom_cm": ate_odom * 100,
        "ate_graph_cm": ate_map * 100,
    }
    with open(os.path.join(run_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    log.info(f"Scenario: {args.scenario}")
    log.info(f"Frames: {result.n_frames}  Keyframes: {result.n_keyframes}  "
             f"Loop closures: {len(result.loop_events)}")
    log.info(f"ATE odom-only: {ate_odom*100:.2f} cm | ATE after graph: {ate_map*100:.2f} cm")

    nodes = [pipeline.memory.get(i) for i in pipeline.memory.all_node_ids()]
    pts, colors = assemble_cloud(nodes, RIG_INTRINSICS.K(), RIG_INTRINSICS.depth_scale)
    ply_path = os.path.join(run_dir, "map.ply")
    write_ply(ply_path, pts, colors)
    log.info(f"Map: {ply_path} ({pts.shape[0]} points)")
    log.info(f"Run directory: {run_dir}")


if __name__ == "__main__":
    main()
