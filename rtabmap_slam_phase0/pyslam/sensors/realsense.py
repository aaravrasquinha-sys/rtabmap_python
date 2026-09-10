"""
Intel RealSense D435i capture.

Design notes:
  - Depth is aligned to the color stream in hardware/software via
    rs.align, so Frame.rgb and Frame.depth always share one intrinsics
    matrix and one pixel grid. This removes an entire class of bug.
  - Intrinsics are read FROM THE DEVICE, not hardcoded, but the expected
    values for this rig (see run_slam.py --realsense) are:
      fx=606.75 fy=606.57 cx=320.19 cy=237.06 640x480
      baseline=0.0499 depth_scale=0.0010000000474974513
    env_probe.py should be run first to confirm these match; a mismatch
    is a hard warning, not a silent override.
  - Accel (~250 Hz) and gyro (~200 Hz) arrive as independent streams at
    different rates. We buffer both and, for each color frame, take the
    IMU samples since the previous color frame and pair them onto a
    common per-frame array via nearest-gyro-timestamp interpolation of
    accel (gyro is the better-conditioned, evenly-spaced stream). This
    mirrors Phase-0's simplified IMU handling: IMU is RECORDED from
    frame 0 but not yet fused into odometry (that is Phase 5 / "P5").
  - All timestamps use frame.get_timestamp() (device/hardware clock,
    milliseconds) converted to seconds, per the core.types "one
    monotonic float64 timeline" rule. Host time is not used for fusion.
"""
from __future__ import annotations
from typing import Iterator, Optional
import numpy as np

from pyslam.core.types import Frame, Intrinsics
from pyslam.core.log import get_logger

log = get_logger("sensors.realsense")


class RealSenseSource:
    def __init__(self, width: int = 640, height: int = 480, fps: int = 30,
                 enable_imu: bool = True, serial: Optional[str] = None):
        try:
            import pyrealsense2 as rs
        except ImportError as e:
            raise RuntimeError(
                "pyrealsense2 is not importable. Install with `pip install pyrealsense2` "
                "or run env_probe.py to see what's missing."
            ) from e
        self._rs = rs
        self.width, self.height, self.fps = width, height, fps
        self.enable_imu = enable_imu

        self.pipeline = rs.pipeline()
        cfg = rs.config()
        if serial:
            cfg.enable_device(serial)
        cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        cfg.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        if enable_imu:
            cfg.enable_stream(rs.stream.accel, rs.format.motion_xyz32f, 250)
            cfg.enable_stream(rs.stream.gyro, rs.format.motion_xyz32f, 200)

        profile = self.pipeline.start(cfg)
        self.align = rs.align(rs.stream.color)

        color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
        rs_intr = color_stream.get_intrinsics()
        depth_sensor = profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()

        # baseline: distance between the depth (left IR) and RGB module,
        # read from extrinsics if available, else fall back to the D435i
        # nominal value.
        try:
            depth_stream = profile.get_stream(rs.stream.depth)
            ex = depth_stream.get_extrinsics_to(color_stream)
            baseline = float(np.linalg.norm(ex.translation))
            if baseline < 1e-4:
                baseline = 0.0499
        except Exception:
            baseline = 0.0499

        self.intr = Intrinsics(
            fx=rs_intr.fx, fy=rs_intr.fy, cx=rs_intr.ppx, cy=rs_intr.ppy,
            width=rs_intr.width, height=rs_intr.height,
            depth_scale=self.depth_scale, baseline=baseline,
        )
        log.info(f"RealSense started: {self.intr}")

        self._imu_buffer: list[tuple[float, np.ndarray]] = []  # (t, [gx,gy,gz] or accel)
        self._accel_buf: list[tuple[float, np.ndarray]] = []
        self._gyro_buf: list[tuple[float, np.ndarray]] = []
        self._prev_t: Optional[float] = None
        self._frame_id = 0

    def intrinsics(self) -> Intrinsics:
        return self.intr

    def _drain_motion(self, frames) -> None:
        rs = self._rs
        for f in frames:
            if f.is_motion_frame():
                mf = f.as_motion_frame()
                t = mf.get_timestamp() / 1000.0
                data = mf.get_motion_data()
                v = np.array([data.x, data.y, data.z], dtype=np.float64)
                if mf.get_profile().stream_type() == rs.stream.accel:
                    self._accel_buf.append((t, v))
                elif mf.get_profile().stream_type() == rs.stream.gyro:
                    self._gyro_buf.append((t, v))

    def _pair_imu_since(self, t_prev: float, t_cur: float) -> np.ndarray:
        """Build the (N,7) [t,gx,gy,gz,ax,ay,az] array for gyro samples in
        (t_prev, t_cur], with accel linearly interpolated onto gyro times."""
        gyro = [(t, v) for (t, v) in self._gyro_buf if t_prev < t <= t_cur]
        if not gyro or len(self._accel_buf) < 2:
            self._gyro_buf = [(t, v) for (t, v) in self._gyro_buf if t > t_cur]
            return np.zeros((0, 7), dtype=np.float64)
        accel_t = np.array([t for t, _ in self._accel_buf])
        accel_v = np.array([v for _, v in self._accel_buf])
        rows = []
        for t, w in gyro:
            a = np.array([np.interp(t, accel_t, accel_v[:, k]) for k in range(3)])
            rows.append([t, *w, *a])
        # keep buffers bounded
        self._gyro_buf = [(t, v) for (t, v) in self._gyro_buf if t > t_cur]
        self._accel_buf = [(t, v) for (t, v) in self._accel_buf if t > t_prev - 0.05]
        return np.array(rows, dtype=np.float64)

    def __iter__(self) -> Iterator[Frame]:
        rs = self._rs
        while True:
            frames = self.pipeline.wait_for_frames()
            if self.enable_imu:
                self._drain_motion(frames)
            aligned = self.align.process(frames)
            color_frame = aligned.get_color_frame()
            depth_frame = aligned.get_depth_frame()
            if not color_frame or not depth_frame:
                continue
            t = color_frame.get_timestamp() / 1000.0

            bgr = np.asanyarray(color_frame.get_data())
            rgb = bgr[:, :, ::-1].copy()
            depth = np.asanyarray(depth_frame.get_data()).copy()  # uint16, raw units

            imu = None
            if self.enable_imu and self._prev_t is not None:
                imu = self._pair_imu_since(self._prev_t, t)
            self._prev_t = t

            yield Frame(t=t, rgb=rgb, depth=depth, intr=self.intr, imu=imu,
                        frame_id=self._frame_id)
            self._frame_id += 1

    def close(self) -> None:
        try:
            self.pipeline.stop()
        except Exception:
            pass
