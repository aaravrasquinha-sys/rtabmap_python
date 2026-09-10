"""
Replay a recorded .bag through the pipeline. This is the primary
iteration loop after Phase 0's hardware bring-up: record once on the
D435i, then develop/debug entirely against the replay.

    python3 run_bag.py data/bags/room_loop.bag
    python3 run_bag.py data/bags/room_loop.bag --max-frames 200
"""
from __future__ import annotations
import argparse
import json
import os
import time
import sys

sys.path.insert(0, ".")

from pyslam.core.config import Config
from pyslam.core.log import get_logger
from pyslam.pipeline import Pipeline
from pyslam.sensors.bagfile import BagReader
from pyslam.mapping.cloud import assemble_cloud
from pyslam.mapping.export import write_ply

log = get_logger("run_bag")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bag_path", type=str)
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--backend", choices=["auto", "gtsam", "native"], default="auto")
    args = ap.parse_args()

    run_dir = os.path.join("runs", f"run_{int(time.time())}_bag")
    os.makedirs(run_dir, exist_ok=True)
    log.info(f"Run directory: {run_dir}")

    source = BagReader(args.bag_path)
    log.info(f"Loaded {source.n_frames} frames from {args.bag_path}")

    cfg = Config()
    cfg.save(os.path.join(run_dir, "config.json"))

    pipeline = Pipeline(cfg, vocab_path=None, backend_prefer=args.backend)
    result = pipeline.run(source, max_frames=args.max_frames, verbose=True)
    source.close()

    with open(os.path.join(run_dir, "telemetry.json"), "w") as f:
        json.dump(result.telemetry, f, indent=2)
    with open(os.path.join(run_dir, "summary.json"), "w") as f:
        json.dump({
            "bag": args.bag_path,
            "n_frames": result.n_frames,
            "n_keyframes": result.n_keyframes,
            "n_loop_closures": len(result.loop_events),
            "status_log": result.status_log,
        }, f, indent=2)

    nodes = [pipeline.memory.get(i) for i in pipeline.memory.all_node_ids()]
    intr = source.intrinsics()
    pts, colors = assemble_cloud(nodes, intr.K(), intr.depth_scale)
    ply_path = os.path.join(run_dir, "map.ply")
    write_ply(ply_path, pts, colors)

    log.info(f"Done. frames={result.n_frames} keyframes={result.n_keyframes} "
             f"loop_closures={len(result.loop_events)}")
    log.info(f"Map: {ply_path} ({pts.shape[0]} points)")
    log.info(f"Run directory: {run_dir}")


if __name__ == "__main__":
    main()
