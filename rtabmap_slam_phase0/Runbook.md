# Phase 0 Runbook

## What this is

A complete, working, single-threaded RTAB-Map-style SLAM pipeline:
capture -> ORB features -> frame-to-keyframe PnP odometry -> STM/WM
memory -> retrieval -> Bayes filter -> geometric verification -> pose
graph (GTSAM/native) -> point-cloud map. Deliberately the simplest
correct version of every component (see architecture doc) so later
phases are swaps behind fixed interfaces, not rewrites.

## Dependencies

```
pip install numpy scipy scikit-learn opencv-contrib-python pyrealsense2
```

GTSAM: already installed on the target machine per your setup — used
automatically as the primary graph backend. If it's ever missing or
its `numpy<2.0` pin conflicts with the rest of the environment, the
pipeline **automatically falls back to a native backend** (no crash,
just a log line: `Using native pose-graph backend.`).

## First thing to run, always

```
python3 -m pyslam.selftest
```

~50s, no hardware needed. Runs the full synthetic gate suite (SE3
identities, renderer geometric oracle, Bayes filter spike-rejection and
sustained-match-fires checks, verifier false-accept check, and a full
end-to-end synthetic loop closure with an ATE check). **If this is
red, don't touch the camera** — go straight to `tools/env_probe.py`
instead.

Expected output: `6/6 passed`.

## Environment check (before first hardware run)

```
python3 -m pyslam.tools.env_probe --realsense
```

Confirms OpenCV/GTSAM/pyrealsense2 versions and symbol availability,
queries the connected D435i for firmware/serial/USB mode, and compares
its measured intrinsics against the expected rig values (fx=606.75,
fy=606.57, cx=320.19, cy=237.06, 640x480, depth_scale≈0.001,
baseline≈0.0499). A mismatch is a warning, not a crash — the pipeline
always uses whatever the device actually reports.

## Running on synthetic data (no camera)

```
python3 run_synth.py                       # flagship square-loop scenario (validated, passes)
python3 run_synth.py --scenario corridor   # harder scenario, known not to close the loop yet
```

## Running live on the D435i

```
python3 run_slam.py --realsense --imu
python3 run_slam.py --realsense --imu --record data/bags/room_loop.bag
python3 run_slam.py --realsense --imu --max-frames 300
```

IMU is captured and stored on every Frame from frame 0, but **not yet
fused into odometry** — that's a Phase-5 upgrade. Recording it now
means bags captured today are already IMU-ready for later.

**Record the 4 reference bags now, while the camera is out**, per the
architecture doc's workflow — after this, hardware mostly leaves the
iteration loop and everything below runs against replays:

| Bag | What to capture |
|---|---|
| `desk_static.bag` | 60s, camera stationary on a desk |
| `room_loop.bag` | 60-90s, walk a loop around a room, return to start |
| `corridor_out_back.bag` | 90s, down a corridor and back |
| `hard_case.bag` | blank wall / fast turn / low light |

```
mkdir -p data/bags
python3 run_slam.py --realsense --imu --record data/bags/room_loop.bag
```

## Replaying a recorded bag

```
python3 run_bag.py data/bags/room_loop.bag
python3 run_bag.py data/bags/room_loop.bag --max-frames 200
```

## Every run writes a `runs/run_<timestamp>/` directory

Contains `config.json` (exact settings used), `telemetry.json`
(per-frame timing/status), `summary.json` (keyframes, loop closures,
ATE if ground truth is available), and `map.ply` (open in MeshLab /
CloudCompare / any PLY viewer). If something looks wrong, zip and send
the whole directory:

```
python3 -m pyslam.tools.bundle runs/run_1234567890
```

## Known limitations in this phase (see Phase Evaluation doc for detail)

- **Performance is not real-time yet**: ~300ms/frame in this
  environment (offline, not the target machine — numbers will differ
  on yours; report what you see). Correctness was the Phase-0
  priority; a profiling/vectorization pass is the natural Phase-1
  companion to the odometry upgrade.
- **The corridor/open-area loop-closure scenario doesn't close the
  loop yet** — only the tighter square-loop scenario is validated.
  This is a real, understood limitation (weaker retrieval signal over
  longer time gaps + no graph-neighbor-aware belief diffusion yet),
  not a crash or a silent wrong-answer. Full detail in the Phase
  Evaluation doc.
- No threading (single-threaded by design, see architecture doc §7
  rule 1). No LTM (WM only, unbounded). Fixed vocabulary code exists
  but isn't used to drive loop-closure decisions in this phase (see
  `vpr/raw_match.py` docstring for why).
