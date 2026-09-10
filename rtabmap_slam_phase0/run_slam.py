"""
Run the pipeline live on a connected D435i.

    python3 run_slam.py --realsense --imu
    python3 run_slam.py --realsense --imu --record data/bags/room_loop.bag
    python3 run_slam.py --realsense --imu --max-frames 300

Every run writes a run_<timestamp>/ directory under runs/ with the exact
config used, telemetry, the trajectory, and (on request) the recorded
bag -- see tools/bundle.py to package one of these for sharing.
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
from pyslam.mapping.cloud import assemble_cloud
from pyslam.mapping.export import write_ply

log = get_logger("run_slam")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--realsense", action="store_true", required=True)
    ap.add_argument("--imu", action="store_true", help="enable IMU streams (recorded, not yet fused)")
    ap.add_argument("--record", type=str, default=None, help="also write a .bag of this run")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--backend", choices=["auto", "gtsam", "native"], default="auto")
    args = ap.parse_args()

    run_dir = os.path.join("runs", f"run_{int(time.time())}")
    os.makedirs(run_dir, exist_ok=True)
    log.info(f"Run directory: {run_dir}")

    from pyslam.sensors.realsense import RealSenseSource
    source = RealSenseSource(enable_imu=args.imu)

    if args.record:
        from pyslam.sensors.bagfile import BagWriter
        writer = BagWriter(args.record, source.intrinsics())
        orig_iter = source.__iter__

        def recording_iter():
            for frame in orig_iter():
                writer.write_frame(frame)
                yield frame
            writer.close()
        source.__iter__ = recording_iter

    cfg = Config()
    cfg.save(os.path.join(run_dir, "config.json"))

    pipeline = Pipeline(cfg, vocab_path=None, backend_prefer=args.backend)
    result = pipeline.run(source, max_frames=args.max_frames, verbose=True)
    source.close()

    with open(os.path.join(run_dir, "telemetry.json"), "w") as f:
        json.dump(result.telemetry, f, indent=2)
    with open(os.path.join(run_dir, "summary.json"), "w") as f:
        json.dump({
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
