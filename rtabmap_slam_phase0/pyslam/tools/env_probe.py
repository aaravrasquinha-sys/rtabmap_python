"""
Run BEFORE touching the camera:  python -m pyslam.tools.env_probe [--realsense]

Checks the actual installed versions and symbol availability against
what the codebase assumes, and (with --realsense) queries the D435i
device itself for FW, intrinsics, and IMU presence.
"""
from __future__ import annotations
import sys
import json
import argparse

sys.path.insert(0, ".")


def probe_python_env() -> dict:
    import numpy, cv2, scipy, sklearn
    out = {
        "python": sys.version.split()[0],
        "numpy": numpy.__version__,
        "opencv": cv2.__version__,
        "scipy": scipy.__version__,
        "sklearn": sklearn.__version__,
    }
    # symbols this codebase relies on
    symbols = ["ORB_create", "BFMatcher", "solvePnPRansac", "Rodrigues", "projectPoints"]
    out["opencv_symbols"] = {s: hasattr(cv2, s) for s in symbols}
    out["opencv_usac_magsac"] = hasattr(cv2, "USAC_MAGSAC")
    return out


def probe_gtsam() -> dict:
    try:
        import gtsam
        # minimal functional check, not just import
        g = gtsam.NonlinearFactorGraph()
        v = gtsam.Values()
        v.insert(0, gtsam.Pose3())
        return {"available": True, "functional_check": "ok"}
    except ImportError as e:
        return {"available": False, "reason": str(e)}
    except Exception as e:
        return {"available": True, "functional_check": f"FAILED: {e}"}


def probe_pyrealsense2() -> dict:
    try:
        import pyrealsense2 as rs
    except ImportError as e:
        return {"available": False, "reason": str(e)}
    return {"available": True, "version": getattr(rs, "__version__", "unknown")}


def probe_realsense_device() -> dict:
    try:
        import pyrealsense2 as rs
    except ImportError:
        return {"error": "pyrealsense2 not importable"}

    ctx = rs.context()
    devices = ctx.query_devices()
    if len(devices) == 0:
        return {"error": "no RealSense device found -- check USB connection/permissions"}

    dev = devices[0]
    info = {
        "name": dev.get_info(rs.camera_info.name),
        "serial_number": dev.get_info(rs.camera_info.serial_number),
        "firmware_version": dev.get_info(rs.camera_info.firmware_version),
        "usb_type": dev.get_info(rs.camera_info.usb_type_descriptor) if dev.supports(rs.camera_info.usb_type_descriptor) else "unknown",
    }

    # IMU presence + calibration check
    imu_streams = []
    for s in dev.query_sensors():
        for p in s.get_stream_profiles():
            if p.stream_type() in (rs.stream.accel, rs.stream.gyro):
                imu_streams.append(str(p.stream_type()))
    info["imu_streams_available"] = sorted(set(imu_streams))
    info["imu_calibration_warning"] = (
        "D435i ships with IMU intrinsics UNCALIBRATED from factory. "
        "If accel/gyro readings look biased or roll/pitch drift quickly, "
        "run Intel's IMU calibration tool before enabling any IMU-based "
        "fusion (not used by the Phase-0 pipeline, but recorded)."
    )

    # try a short pipeline start to confirm streaming actually works
    try:
        pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        profile = pipeline.start(cfg)
        color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
        rs_intr = color_stream.get_intrinsics()
        depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
        info["measured_intrinsics"] = {
            "fx": rs_intr.fx, "fy": rs_intr.fy, "cx": rs_intr.ppx, "cy": rs_intr.ppy,
            "width": rs_intr.width, "height": rs_intr.height, "depth_scale": depth_scale,
        }
        expected = dict(fx=606.75, fy=606.57, cx=320.19, cy=237.06, width=640, height=480,
                         depth_scale=0.0010000000474974513)
        mismatches = []
        for k, v in expected.items():
            got = info["measured_intrinsics"].get(k)
            if got is None:
                continue
            if isinstance(v, float):
                if abs(got - v) > 0.5:
                    mismatches.append(f"{k}: expected~{v}, got {got}")
            elif got != v:
                mismatches.append(f"{k}: expected {v}, got {got}")
        info["intrinsics_match_expected"] = (len(mismatches) == 0)
        if mismatches:
            info["intrinsics_mismatches"] = mismatches
        pipeline.stop()
    except Exception as e:
        info["stream_test_error"] = str(e)

    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--realsense", action="store_true", help="also query the connected D435i")
    args = ap.parse_args()

    report = {
        "python_env": probe_python_env(),
        "gtsam": probe_gtsam(),
        "pyrealsense2": probe_pyrealsense2(),
    }
    if args.realsense:
        report["realsense_device"] = probe_realsense_device()

    print(json.dumps(report, indent=2))

    # human-readable summary + hard warnings
    print("\n--- summary ---")
    print(f"OpenCV: {report['python_env']['opencv']}  "
          f"(USAC_MAGSAC {'available' if report['python_env']['opencv_usac_magsac'] else 'MISSING -> falls back to SOLVEPNP_ITERATIVE'})")
    if report["gtsam"]["available"]:
        print(f"GTSAM: available, functional check: {report['gtsam']['functional_check']}")
    else:
        print(f"GTSAM: NOT available ({report['gtsam']['reason']}) -> pipeline will use the native backend")
    if args.realsense:
        dev = report.get("realsense_device", {})
        if "error" in dev:
            print(f"RealSense: ERROR - {dev['error']}")
        else:
            print(f"RealSense: {dev.get('name')} SN={dev.get('serial_number')} "
                  f"FW={dev.get('firmware_version')} USB={dev.get('usb_type')}")
            if not dev.get("intrinsics_match_expected", True):
                print(f"  WARNING: measured intrinsics differ from the expected rig values: "
                      f"{dev.get('intrinsics_mismatches')}")


if __name__ == "__main__":
    main()
