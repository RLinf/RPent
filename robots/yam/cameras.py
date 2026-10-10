# Copyright 2026 The RPent Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""RGBD camera snapshots for the RPent YAM facade."""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any

import numpy as np

from robots.yam.contracts import YAM_CAMERA_NAMES
from robots.yam.geometry import YamCalibration, load_calibration


@dataclass
class YamRgbdFrame:
    rgb: np.ndarray
    depth: np.ndarray
    camera_meta: dict[str, Any]


class YamRgbdCameraRig:
    """Three RealSense RGBD streams with host and sensor timestamps.

    Construction is side-effect free. ``open()`` starts RealSense pipelines and
    should be called before the CAN runtime is connected.
    """

    def __init__(
        self,
        config: dict[str, Any],
        *,
        calibration: YamCalibration | None = None,
    ) -> None:
        self.config = dict(config)
        self.calibration = calibration or load_calibration(config)
        self.camera_timeout_s = float(self.config.get("camera_frame_timeout_s", 1.0))
        self.warmup_frames = int(self.config.get("camera_warmup_frames", 15))
        self._rs: Any | None = None
        self._pipelines: dict[str, Any] = {}
        self._aligners: dict[str, Any] = {}
        self._profiles: dict[str, Any] = {}
        self._depth_scales: dict[str, float] = {}
        self._last_frames: dict[str, YamRgbdFrame] = {}
        self._frame_lock = threading.RLock()
        self._frame_threads: dict[str, threading.Thread] = {}
        self._thread_errors: dict[str, Exception] = {}
        self._opened = False
        self._camera_specs = self._normalize_camera_specs(self.config)

    @staticmethod
    def _normalize_camera_specs(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
        raw = config.get("cameras")
        if raw is None:
            raise ValueError("YAM config requires cameras for top/left/right RGBD")
        specs: dict[str, dict[str, Any]] = {}
        if isinstance(raw, dict):
            items = raw.items()
        else:
            items = ((entry.get("name"), entry) for entry in raw)
        for name, spec in items:
            if name not in YAM_CAMERA_NAMES:
                raise ValueError(
                    f"unknown YAM camera {name!r}; expected {list(YAM_CAMERA_NAMES)}"
                )
            values = dict(spec)
            serial = str(values.get("serial", values.get("serial_number", ""))).strip()
            if not serial:
                raise ValueError(f"YAM camera {name!r} requires a serial")
            width = int(values.get("width", values.get("resolution", [640, 480])[0]))
            height = int(values.get("height", values.get("resolution", [640, 480])[1]))
            fps = int(values.get("fps", 30))
            if width <= 0 or height <= 0 or fps <= 0:
                raise ValueError(f"invalid YAM camera geometry for {name!r}")
            if (width, height, fps) != (640, 480, 30):
                raise ValueError(
                    f"YAM camera {name!r} must run at 640x480@30, "
                    f"got {width}x{height}@{fps}"
                )
            values.update(
                {
                    "name": name,
                    "serial": serial,
                    "width": width,
                    "height": height,
                    "fps": fps,
                }
            )
            specs[str(name)] = values
        missing = sorted(set(YAM_CAMERA_NAMES) - set(specs))
        if missing:
            raise ValueError(f"YAM camera config is missing {missing}")
        return specs

    def open(self) -> None:
        if self._opened:
            return
        try:
            import pyrealsense2 as rs
        except ModuleNotFoundError as error:
            raise RuntimeError(
                "missing dependency pyrealsense2 for YAM RGBD cameras; install "
                "the RealSense runtime on the robot host"
            ) from error
        self._rs = rs
        try:
            for name in YAM_CAMERA_NAMES:
                spec = self._camera_specs[name]
                pipeline = rs.pipeline()
                cfg = rs.config()
                cfg.enable_device(spec["serial"])
                cfg.enable_stream(
                    rs.stream.color,
                    spec["width"],
                    spec["height"],
                    rs.format.bgr8,
                    spec["fps"],
                )
                cfg.enable_stream(
                    rs.stream.depth,
                    spec["width"],
                    spec["height"],
                    rs.format.z16,
                    spec["fps"],
                )
                profile = pipeline.start(cfg)
                self._pipelines[name] = pipeline
                self._profiles[name] = profile
                self._aligners[name] = rs.align(rs.stream.color)
                depth_sensor = profile.get_device().first_depth_sensor()
                self._depth_scales[name] = float(depth_sensor.get_depth_scale())
            for _ in range(self.warmup_frames):
                for name in YAM_CAMERA_NAMES:
                    frame = self._capture_one_direct(name)
                    with self._frame_lock:
                        self._last_frames[name] = frame
            with self._frame_lock:
                self._thread_errors.clear()
            self._opened = True
            for name in YAM_CAMERA_NAMES:
                thread = threading.Thread(
                    target=self._capture_loop,
                    args=(name,),
                    name=f"yam-camera-{name}",
                    daemon=True,
                )
                self._frame_threads[name] = thread
                thread.start()
        except Exception:
            self.close()
            raise

    def close(self) -> None:
        errors: list[Exception] = []
        self._opened = False
        for name, pipeline in list(reversed(self._pipelines.items())):
            try:
                pipeline.stop()
            except Exception as error:
                errors.append(error)
            finally:
                self._pipelines.pop(name, None)
                self._aligners.pop(name, None)
                self._profiles.pop(name, None)
                self._depth_scales.pop(name, None)
        for name, thread in list(self._frame_threads.items()):
            if thread is not threading.current_thread() and thread.is_alive():
                thread.join(timeout=2.0)
            if thread.is_alive():
                errors.append(RuntimeError(f"YAM camera thread {name!r} did not stop"))
            else:
                self._frame_threads.pop(name, None)
        self._opened = False
        if errors:
            raise RuntimeError(
                "failed to close one or more YAM cameras: "
                + "; ".join(str(error) for error in errors)
            ) from errors[0]

    def snapshot(
        self, *, not_before_monotonic_s: float | None = None
    ) -> dict[str, Any]:
        self.open()
        snapshot_id = uuid.uuid4().hex
        deadline = time.monotonic() + self.camera_timeout_s
        stale: list[str] = []
        old: list[str] = []
        while True:
            now = time.monotonic()
            with self._frame_lock:
                if self._thread_errors:
                    details = "; ".join(
                        f"{name}: {error}"
                        for name, error in sorted(self._thread_errors.items())
                    )
                    raise RuntimeError(f"YAM camera capture thread error: {details}")
                if set(YAM_CAMERA_NAMES).issubset(self._last_frames):
                    stale = self._stale_camera_names_locked(now)
                    old = self._old_camera_names_locked(not_before_monotonic_s)
                    if not stale and not old:
                        views = {
                            name: self._copy_frame(self._last_frames[name])
                            for name in YAM_CAMERA_NAMES
                        }
                        break
            if now >= deadline:
                missing = sorted(set(YAM_CAMERA_NAMES) - set(self._last_frames))
                if missing:
                    raise RuntimeError(
                        f"YAM cameras have no cached frames yet: {missing}"
                    )
                if stale:
                    raise RuntimeError(
                        "YAM cached camera frames are stale: "
                        + ", ".join(stale)
                        + f"; timeout_s={self.camera_timeout_s}"
                    )
                raise RuntimeError(
                    "YAM cameras did not produce frames after requested host time: "
                    + ", ".join(old)
                )
            time.sleep(0.005)
        return {
            "snapshot_id": snapshot_id,
            "views": views,
        }

    def _capture_loop(self, camera_name: str) -> None:
        while self._opened:
            try:
                frame = self._capture_one_direct(camera_name)
                with self._frame_lock:
                    self._last_frames[camera_name] = frame
                    self._thread_errors.pop(camera_name, None)
            except Exception as error:
                with self._frame_lock:
                    self._thread_errors[camera_name] = error
                time.sleep(0.01)

    def _capture_one_direct(self, camera_name: str) -> YamRgbdFrame:
        name = self._normalize_name(camera_name)
        self._require_rs()
        host_before_s = time.time()
        host_before_monotonic_s = time.monotonic()
        try:
            raw_frames = self._pipelines[name].wait_for_frames(
                int(self.camera_timeout_s * 1000)
            )
            aligned = self._aligners[name].process(raw_frames)
            color_frame = aligned.get_color_frame()
            depth_frame = aligned.get_depth_frame()
            if not color_frame or not depth_frame:
                raise RuntimeError(f"YAM camera {name!r} returned incomplete RGBD")
            bgr = np.asanyarray(color_frame.get_data())
            rgb = np.ascontiguousarray(bgr[..., ::-1], dtype=np.uint8)
            depth_m = (
                np.asanyarray(depth_frame.get_data()).astype(np.float32)
                * self._depth_scales[name]
            )
            meta = self._camera_meta(
                name,
                color_frame,
                depth_frame,
                host_before_s=host_before_s,
                host_before_monotonic_s=host_before_monotonic_s,
                host_after_s=time.time(),
                host_after_monotonic_s=time.monotonic(),
            )
            return YamRgbdFrame(rgb=rgb, depth=depth_m, camera_meta=meta)
        except Exception as error:
            if name in self._last_frames:
                raise RuntimeError(
                    f"YAM camera {name!r} failed to produce a fresh RGBD frame"
                ) from error
            raise

    def _camera_meta(
        self,
        name: str,
        color_frame: Any,
        depth_frame: Any,
        *,
        host_before_s: float | None = None,
        host_before_monotonic_s: float | None = None,
        host_after_s: float | None = None,
        host_after_monotonic_s: float | None = None,
    ) -> dict[str, Any]:
        spec = self._camera_specs[name]
        intrinsic_k = None
        distortion_model = None
        distortion_coeffs: list[float] | None = None
        if color_frame is not None:
            profile = color_frame.profile.as_video_stream_profile()
            intr = profile.get_intrinsics()
            intrinsic_k = np.array(
                [[intr.fx, 0.0, intr.ppx], [0.0, intr.fy, intr.ppy], [0.0, 0.0, 1.0]],
                dtype=np.float64,
            )
            distortion_model = str(getattr(intr, "model", ""))
            distortion_coeffs = [float(value) for value in getattr(intr, "coeffs", [])]
        if intrinsic_k is None:
            intrinsic_k = np.full((3, 3), np.nan, dtype=np.float64)
        timestamps = {
            "host_before_time_s": host_before_s,
            "host_after_time_s": host_after_s,
            "host_before_monotonic_s": host_before_monotonic_s,
            "host_after_monotonic_s": host_after_monotonic_s,
            "color_sensor_timestamp_ms": self._frame_timestamp_ms(color_frame),
            "depth_sensor_timestamp_ms": self._frame_timestamp_ms(depth_frame),
        }
        return {
            "name": name,
            "serial": spec["serial"],
            "intrinsic_K": np.asarray(intrinsic_k, dtype=np.float64),
            "distortion_model": distortion_model,
            "distortion_coeffs": distortion_coeffs,
            "cam2world_cv": (
                self.calibration.cam2world_cv(name)
                if name in self.calibration.camera_to_world
                else np.full((4, 4), np.nan, dtype=np.float64)
            ),
            "width": int(spec["width"]),
            "height": int(spec["height"]),
            "timestamps": timestamps,
            "world_frame": self.calibration.world_frame,
            "sync_mode": "sequential_host_timestamped_rgbd",
        }

    def _stale_camera_names_locked(self, now_monotonic_s: float) -> list[str]:
        stale: list[str] = []
        for name in YAM_CAMERA_NAMES:
            frame = self._last_frames.get(name)
            if frame is None:
                continue
            timestamps = frame.camera_meta.get("timestamps", {})
            host_after = timestamps.get("host_after_monotonic_s")
            if host_after is None:
                stale.append(name)
                continue
            age_s = float(now_monotonic_s - float(host_after))
            frame.camera_meta["frame_age_s"] = age_s
            if age_s > self.camera_timeout_s:
                stale.append(name)
        return stale

    def _old_camera_names_locked(
        self, not_before_monotonic_s: float | None
    ) -> list[str]:
        if not_before_monotonic_s is None:
            return []
        old: list[str] = []
        for name in YAM_CAMERA_NAMES:
            frame = self._last_frames.get(name)
            if frame is None:
                continue
            timestamps = frame.camera_meta.get("timestamps", {})
            host_before = timestamps.get("host_before_monotonic_s")
            if host_before is None or float(host_before) < float(
                not_before_monotonic_s
            ):
                old.append(name)
        return old

    @staticmethod
    def _frame_timestamp_ms(frame: Any) -> float | None:
        if frame is None:
            return None
        try:
            return float(frame.get_timestamp())
        except Exception:
            return None

    @staticmethod
    def _normalize_name(camera_name: str) -> str:
        name = str(camera_name)
        if name not in YAM_CAMERA_NAMES:
            raise ValueError(
                f"unknown YAM camera {camera_name!r}; expected {list(YAM_CAMERA_NAMES)}"
            )
        return name

    def _require_rs(self) -> Any:
        if self._rs is None:
            raise RuntimeError("YAM RealSense module has not been initialized")
        return self._rs

    @staticmethod
    def _copy_frame(frame: YamRgbdFrame) -> YamRgbdFrame:
        return YamRgbdFrame(
            rgb=frame.rgb.copy(),
            depth=frame.depth.copy(),
            camera_meta={
                **frame.camera_meta,
                "timestamps": dict(frame.camera_meta.get("timestamps", {})),
            },
        )


__all__ = ["YamRgbdCameraRig", "YamRgbdFrame"]
