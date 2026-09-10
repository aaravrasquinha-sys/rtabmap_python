"""
Phase-0 gate suite (G0). Run via `python -m pyslam.selftest`.

These are the checks that must pass before touching hardware, and
before any Phase-1+ change is considered safe to build on. See the
Phase Evaluation doc for the hardware-side checks (G0-HW) that
complement this.
"""
from __future__ import annotations
import sys
import numpy as np

sys.path.insert(0, ".")  # allow running from repo root without install

from pyslam.core import lie
from pyslam.core.types import Intrinsics, Frame
from pyslam.core.config import Config


def _fail(msg: str) -> None:
    raise AssertionError(msg)


def test_se3_identities():
    rng = np.random.default_rng(0)
    max_err = 0.0
    for _ in range(5000):
        w = rng.normal(scale=1.0, size=3)
        if rng.random() < 0.2:
            w = w / (np.linalg.norm(w) + 1e-12) * rng.uniform(np.pi - 1e-3, np.pi)
        R = lie.so3_exp(w)
        R2 = lie.so3_exp(lie.so3_log(R))
        max_err = max(max_err, np.linalg.norm(R - R2))
    assert max_err < 1e-9, f"so3 exp/log roundtrip err {max_err}"

    max_err = 0.0
    for _ in range(5000):
        xi = rng.normal(scale=0.5, size=6)
        T = lie.se3_exp(xi)
        T2 = lie.se3_exp(lie.se3_log(T))
        max_err = max(max_err, np.linalg.norm(T - T2))
    assert max_err < 1e-9, f"se3 exp/log roundtrip err {max_err}"

    max_err = 0.0
    for _ in range(2000):
        T = lie.se3_exp(rng.normal(scale=0.3, size=6))
        xi = rng.normal(scale=0.1, size=6)
        lhs = lie.se3_adjoint(T) @ xi
        rhs = lie.se3_log(T @ lie.se3_exp(xi) @ lie.se3_inverse(T))
        max_err = max(max_err, np.linalg.norm(lhs - rhs))
    assert max_err < 1e-9, f"adjoint identity err {max_err}"
    print("  [ok] SE3/SO3 identities")


def test_renderer_oracle():
    from tests.synth.world import World, Renderer, T_BODY_CAM

    intr = Intrinsics(fx=606.75, fy=606.57, cx=320.19, cy=237.06, width=640, height=480,
                       depth_scale=0.001, baseline=0.0499)
    w = World()
    w.add_box_room(np.array([0, 0, 0]), np.array([3, 3, 2.4]), seed0=1)
    r = Renderer(w, intr)
    T_wb = lie.make_T(np.eye(3), np.array([0.1, -0.1, 0.2]))
    rgb, depth = r.render(T_wb, T_BODY_CAM)  # noise-free
    K = intr.K()
    Kinv = np.linalg.inv(K)
    T_wc = T_wb @ T_BODY_CAM

    rng = np.random.default_rng(1)
    max_err = 0.0
    for _ in range(3000):
        px = rng.integers(0, intr.width)
        py = rng.integers(0, intr.height)
        d = depth[py, px]
        if d == 0:
            continue
        z = d * intr.depth_scale
        p_cam = Kinv @ np.array([px, py, 1.0]) * z
        p_world = (T_wc[:3, :3] @ p_cam) + T_wc[:3, 3]
        best = min(abs((p_world - wl.origin) @ wl.normal) for wl in w.walls)
        max_err = max(max_err, best)
    assert max_err < 1e-6, f"renderer oracle max plane-distance err {max_err}"
    print("  [ok] Renderer geometric oracle")


def test_bayes_filter_single_spike_no_fire():
    from pyslam.loop.bayes import BayesFilter

    cfg = Config()
    bf = BayesFilter(cfg)
    candidates = [1, 2, 3, 4, 5]
    # warm up with neutral evidence
    for _ in range(5):
        bf.update(np.ones(len(candidates) + 1), candidates)
    # single-frame spike for candidate 3
    L = np.ones(len(candidates) + 1)
    L[2] = 8.0
    hyp = bf.update(L, candidates)
    assert hyp is None, "single-frame spike must not fire on its own update"
    # immediately followed by neutral evidence again -- must not fire either
    fired_after = False
    for _ in range(3):
        h = bf.update(np.ones(len(candidates) + 1), candidates)
        fired_after = fired_after or (h is not None)
    assert not fired_after, "single-frame spike must not cause a delayed false fire"
    print("  [ok] Bayes filter rejects single-frame likelihood spikes")


def test_bayes_filter_sustained_match_fires():
    from pyslam.loop.bayes import BayesFilter

    cfg = Config()
    bf = BayesFilter(cfg)
    candidates = [1, 2, 3, 4, 5]
    fired = False
    for _ in range(6):
        L = np.ones(len(candidates) + 1)
        L[2] = 3.0
        hyp = bf.update(L, candidates)
        if hyp is not None:
            fired = True
            assert hyp.node_id == 3
            break
    assert fired, "sustained moderate match should eventually fire"
    print("  [ok] Bayes filter fires on sustained evidence")


def test_verify_rejects_unrelated_pairs():
    from tests.synth.world import World, Renderer, T_BODY_CAM
    from pyslam.frontend.features import extract_signature, make_orb
    from pyslam.loop.verify import GeometricVerifier
    from pyslam.core.types import Node

    intr = Intrinsics(fx=606.75, fy=606.57, cx=320.19, cy=237.06, width=640, height=480,
                       depth_scale=0.001, baseline=0.0499)
    w = World()
    w.add_box_room(np.array([0, 0, 0]), np.array([3, 3, 2.4]), seed0=1)
    w.add_box_room(np.array([20, 0, 0]), np.array([3, 3, 2.4]), seed0=50)
    r = Renderer(w, intr)
    cfg = Config()
    orb = make_orb(cfg)

    def make_node(pos, nid):
        T_wb = lie.make_T(np.eye(3), np.array(pos))
        rgb, depth = r.render(T_wb, T_BODY_CAM)
        frame = Frame(t=0.0, rgb=rgb, depth=depth, intr=intr)
        sig = extract_signature(frame, cfg, orb)
        sig.id = nid
        return Node(id=nid, sig=sig, pose_odom=T_wb, pose_map=T_wb.copy())

    verifier = GeometricVerifier(cfg, intr)
    false_accepts = 0
    n_trials = 15
    rng = np.random.default_rng(3)
    for i in range(n_trials):
        a = make_node([rng.uniform(-0.5, 0.5), rng.uniform(-0.5, 0.5), 0], i * 2)
        b = make_node([20 + rng.uniform(-0.5, 0.5), rng.uniform(-0.5, 0.5), 0], i * 2 + 1)
        link = verifier.verify(a, b)
        if link is not None:
            false_accepts += 1
    assert false_accepts == 0, f"{false_accepts}/{n_trials} false accepts on unrelated rooms"
    print(f"  [ok] Verifier: 0/{n_trials} false accepts on unrelated locations")


def test_end_to_end_loop_closure():
    from tests.synth.world import World, poses_from_path, spline_trajectory
    from pyslam.sensors.synthetic import SyntheticSource
    from pyslam.pipeline import Pipeline
    from pyslam.tools.evaluate import ate_rmse

    intr = Intrinsics(fx=606.75, fy=606.57, cx=320.19, cy=237.06, width=640, height=480,
                       depth_scale=0.001, baseline=0.0499)
    w = World()
    w.add_box_room(np.array([0, 0, 0]), np.array([3, 3, 2.4]), seed0=1)
    waypoints = np.array([
        [0.0, 0.0, 0.0], [0.7, 0.0, 0.0], [0.7, 0.7, 0.0], [0.0, 0.7, 0.0],
        [0.0, 0.0, 0.0], [0.1, 0.02, 0.0],
    ])
    path = spline_trajectory(waypoints, n_samples=120, loop=False)
    poses = poses_from_path(path, look_ahead=4)
    source = SyntheticSource(w, intr, poses, dt=1 / 10.0, add_noise=True, seed=1)

    cfg = Config()
    pipe = Pipeline(cfg, vocab_path=None, backend_prefer="native")
    result = pipe.run(source, verbose=False)

    assert len(result.loop_events) >= 1, "expected at least one loop closure on the square-loop fixture"
    for new_id, old_id, link in result.loop_events:
        gt_new = result.node_gt[new_id][:3, 3]
        gt_old = result.node_gt[old_id][:3, 3]
        true_dist = np.linalg.norm(gt_new - gt_old)
        assert true_dist < 0.20, f"loop closure {new_id}<->{old_id} true distance {true_dist}m is not a real match"

    node_ids = sorted(result.node_gt.keys())
    est_odom = np.array([pipe.memory.get(i).pose_odom[:3, 3] for i in node_ids])
    est_map = np.array([pipe.memory.get(i).pose_map[:3, 3] for i in node_ids])
    gt = np.array([result.node_gt[i][:3, 3] for i in node_ids])
    ate_odom = ate_rmse(est_odom, gt)
    ate_map = ate_rmse(est_map, gt)
    assert ate_map <= ate_odom * 1.01, "graph optimisation should not make ATE worse"
    print(f"  [ok] End-to-end loop closure: {len(result.loop_events)} events, "
          f"ATE odom={ate_odom*100:.2f}cm -> graph={ate_map*100:.2f}cm")


ALL_TESTS = [
    test_se3_identities,
    test_renderer_oracle,
    test_bayes_filter_single_spike_no_fire,
    test_bayes_filter_sustained_match_fires,
    test_verify_rejects_unrelated_pairs,
    test_end_to_end_loop_closure,
]
