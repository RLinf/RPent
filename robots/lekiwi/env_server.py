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

"""LeKiwi RPC service owning RLinf robot parts directly, without Gym or Ray."""

from __future__ import annotations

import argparse
import threading
import time
from typing import Any

import numpy as np

from robots.lekiwi.motion import (
    ANGLE_TOLERANCE_DEG,
    FEEDBACK_TIMEOUT_S,
    MAX_DURATION_S,
    MOTION_TIMEOUT_S,
    STOP_TIMEOUT_S,
    Motion,
    stopped,
)
from robots.lekiwi.runtime_config import load_runtime_config
from rpent.robots.components.env_facade_base import BaseEnvFacade


class LeKiwiEnvFacade(BaseEnvFacade):
    """Expose observations and blocking motion calls over the existing RPC."""

    _CONTROL_METHODS = {"reset", "drive", "rotate", "stop"}

    def __init__(self, backend: Any, *, camera_only: bool = False) -> None:
        self._backend = backend
        self._camera_only = camera_only
        super().__init__()

    def _register_rpc(self) -> None:
        methods = {"get_env_meta", "get_observation", "get_camera_meta"}
        if not self._camera_only:
            methods.add("get_robot_state")
        self._readonly_methods.update(f"env.{name}" for name in methods)
        if not self._camera_only:
            methods.update(self._CONTROL_METHODS)
        for name in methods:
            self._rpc[f"env.{name}"] = getattr(self._backend, name)

    def _dispatch(self, method, args, kwargs, *, session_id=None):
        # The backend serializes motor operations itself. A waiting camera must
        # not delay motion or stop requests.
        if not self._camera_only and method in {
            f"env.{name}" for name in self._CONTROL_METHODS
        }:
            return self._rpc[method](*args, **kwargs)
        return super()._dispatch(method, args, kwargs, session_id=session_id)


def _camera_infos(cfg: Any) -> dict[str, Any]:
    from rlinf.robotics.parts.cameras import CameraInfo

    return {
        name: CameraInfo(
            name=name,
            serial_number=str(device.serial),
            camera_type=device.get("type", "uvc"),
        )
        for name, device in cfg.cameras.items()
    }


def _camera_metadata(cameras: dict[str, Any], main: str) -> dict[str, Any]:
    return {
        "source": "rlinf_camera",
        "color_order": "RGB",
        "cameras": {
            name: {
                "serial_number": camera.camera_info.serial_number,
                "camera_type": camera.camera_info.camera_type,
                "resolution": list(camera.camera_info.resolution),
            }
            for name, camera in cameras.items()
        },
        "observation_camera_map": {
            "main": main,
            **{
                f"extra_{i}": name
                for i, name in enumerate(sorted(set(cameras) - {main}))
            },
        },
    }


class LeKiwiCameraBackend:
    """Serve only the main RLinf camera, at its delivered resolution."""

    def __init__(self, cfg: Any) -> None:
        from rlinf.robotics.parts.cameras import Camera

        self._main = cfg.main_camera
        self._camera = Camera.of(_camera_infos(cfg)[self._main])
        self._lock = threading.Lock()
        try:
            self._camera.connect()
            if not self._camera.is_ready(timeout=3.0):
                raise RuntimeError(f"Camera {self._main} delivered no frame within 3s")
        except BaseException:
            self._camera.disconnect()
            raise

    def get_env_meta(self) -> dict[str, Any]:
        return {"ok": True, "camera_only": True, "action_dim": 0}

    def get_observation(self) -> dict[str, Any]:
        with self._lock:
            return {"main_images": self._camera.get_frame(attempts=2)[..., ::-1]}

    def get_camera_meta(self) -> dict[str, Any]:
        return _camera_metadata({self._main: self._camera}, self._main)

    def close_env(self) -> None:
        with self._lock:
            self._camera.disconnect()


class LeKiwiEnvBackend:
    """Own one RLinf robot and execute one bounded motion call at a time."""

    def __init__(self, cfg: Any) -> None:
        from rlinf.robotics.robots.lekiwi import LeKiwiRobot

        self.cfg = cfg
        self.step_period_s = 1.0 / cfg.drive.step_frequency
        self.velocity_limits = np.array(
            [
                cfg.drive.max_linear_velocity,
                cfg.drive.max_linear_velocity,
                cfg.drive.max_angular_velocity,
            ]
        )
        self._motion_lock = threading.Lock()
        self._cancel = threading.Event()
        self._bus_lock = threading.Lock()
        self._camera_lock = threading.Lock()
        self._closed = False
        self._resetting = False
        self._steps = 0
        self._control_error: str | None = None
        self.robot = LeKiwiRobot.build(
            port=cfg.robot.serial_port,
            node_rank=None,
            wheel_radius=cfg.robot.wheel_radius,
            base_radius=cfg.robot.base_radius,
            max_raw=cfg.robot.max_raw,
            cameras=_camera_infos(cfg),
        )
        try:
            self.robot.connect()
            self.base = self.robot.child("base")
            self.cameras = {name: self.robot.child(name) for name in cfg.cameras}
            for name, camera in self.cameras.items():
                if not camera.is_ready(timeout=3.0):
                    raise RuntimeError(f"Camera {name} delivered no frame within 3s")
            self._stop_and_confirm()
        except BaseException:
            self.robot.disconnect()
            raise

    def get_env_meta(self) -> dict[str, Any]:
        return {
            "ok": True,
            "action_dim": 3,
            "velocity_limits": self.velocity_limits.tolist(),
            "step_frequency": self.cfg.drive.step_frequency,
        }

    def _read_velocity(self) -> np.ndarray:
        before = time.monotonic()
        velocity = np.asarray(self.base.get_observation()["velocity"], dtype=float)
        if time.monotonic() - before > FEEDBACK_TIMEOUT_S:
            raise RuntimeError("wheel feedback read timed out")
        if velocity.shape != (3,) or not np.isfinite(velocity).all():
            raise ValueError("wheel feedback must contain three finite numbers")
        return velocity

    def get_robot_state(self) -> dict[str, Any]:
        with self._bus_lock:
            velocity = self._read_velocity()
            return {
                "velocity": velocity.tolist(),
                "action_dim": 3,
                "velocity_limits": self.velocity_limits.tolist(),
                "step_frequency": self.cfg.drive.step_frequency,
                "steps": self._steps,
                "control_error": self._control_error,
            }

    def get_observation(self) -> dict[str, Any]:
        """Read the base briefly, then capture images without holding its lock."""
        import cv2

        with self._bus_lock:
            velocity = self._read_velocity()
        frames = {}
        width, height = self.cfg.drive.frame_size
        with self._camera_lock:
            for name, camera in self.cameras.items():
                frame = camera.get_frame(
                    max_age=self.cfg.drive.camera_max_age, attempts=2
                )
                h, w = frame.shape[:2]
                scale = min(w / width, h / height)
                cw, ch = int(width * scale), int(height * scale)
                left, top = (w - cw) // 2, (h - ch) // 2
                frames[name] = cv2.resize(
                    frame[top : top + ch, left : left + cw, ::-1], (width, height)
                )
        result = {
            "states": velocity,
            "main_images": frames[self.cfg.main_camera],
            "steps": self._steps,
        }
        extras = [
            frames[name] for name in sorted(frames) if name != self.cfg.main_camera
        ]
        if extras:
            result["extra_view_images"] = np.stack(extras)
        return result

    def get_camera_meta(self) -> dict[str, Any]:
        metadata = _camera_metadata(self.cameras, self.cfg.main_camera)
        metadata["observation_frame_size"] = list(self.cfg.drive.frame_size)
        return metadata

    def _clip_velocity(self, velocity: Any) -> np.ndarray:
        requested = np.asarray(velocity, dtype=float)
        if requested.shape != (3,) or not np.isfinite(requested).all():
            raise ValueError("velocity must contain three finite numbers")
        return np.clip(requested, -self.velocity_limits, self.velocity_limits)

    def _check_available(self) -> None:
        if self._closed or self._resetting or self._control_error is not None:
            raise RuntimeError(
                "LeKiwi control unavailable; resolve the fault and reset"
            )

    def drive(self, velocity: Any, duration_s: float = 1.0) -> dict[str, Any]:
        """Hold a body velocity for a bounded duration, stop, and return."""
        action = self._clip_velocity(velocity)
        if not np.isfinite(duration_s) or not 0 < duration_s <= MAX_DURATION_S:
            raise ValueError(
                f"duration_s must be positive and at most {MAX_DURATION_S}"
            )
        return self._execute(action, duration_s=float(duration_s))

    def rotate(self, angle_deg: float) -> dict[str, Any]:
        """Turn by a signed wheel-estimated angle, stop, and return."""
        if not np.isfinite(angle_deg) or abs(angle_deg) > 360:
            raise ValueError("angle_deg must be finite and within [-360, 360]")
        return self._execute(
            np.array([0.0, 0.0, min(30.0, self.velocity_limits[2])]),
            angle_deg=float(angle_deg),
        )

    def stop(self) -> dict[str, Any]:
        """Interrupt the active call and immediately send zero wheel velocity."""
        with self._bus_lock:
            self._cancel.set()
            try:
                self.base.reset()
            except Exception as exc:
                self._control_error = f"stop failed: {exc}"
                raise
        return {"ok": True}

    def _stop_and_confirm(self, motion: Motion | None = None) -> None:
        """Allow bounded deceleration before deciding whether stopping failed."""
        self.base.reset()
        deadline = time.monotonic() + STOP_TIMEOUT_S
        while True:
            velocity = self._read_velocity()
            now = time.monotonic()
            if motion is not None:
                motion.sample(velocity, now)
            if stopped(velocity):
                return
            if now >= deadline:
                raise RuntimeError("zero wheel feedback not confirmed")
            time.sleep(min(self.step_period_s, deadline - now))

    def _execute(self, velocity: np.ndarray, **target: Any) -> dict[str, Any]:
        # Reject concurrent actions instead of queuing motion behind another call.
        if not self._motion_lock.acquire(blocking=False):
            raise RuntimeError("motion already active; stop it before another call")
        try:
            with self._bus_lock:
                self._check_available()
                self._cancel.clear()
            return self._run_motion(Motion(velocity, **target))
        finally:
            self._motion_lock.release()

    def _run_motion(self, motion: Motion) -> dict[str, Any]:
        status, error, confirmed = "failed", None, False
        deadline = time.monotonic() + MOTION_TIMEOUT_S
        try:
            while True:
                with self._bus_lock:
                    if self._cancel.is_set():
                        status, error = "cancelled", "motion cancelled"
                        break
                    if time.monotonic() >= deadline:
                        error = "motion execution timed out"
                        break
                    motion.sample(self._read_velocity(), time.monotonic())
                    action, complete = motion.command(time.monotonic())
                    if complete:
                        status = "succeeded"
                        break
                    self.base.send_action({"velocity": action})
                    self._steps += 1
                wait = self.step_period_s
                if motion.duration_s is not None:
                    wait = min(
                        wait,
                        max(
                            0.0,
                            motion.started_at + motion.duration_s - time.monotonic(),
                        ),
                    )
                self._cancel.wait(wait)
        except Exception as exc:
            status, error = "failed", f"{type(exc).__name__}: {exc}"
            motion.feedback_valid = False
            with self._bus_lock:
                self._control_error = error
        finally:
            with self._bus_lock:
                try:
                    self._stop_and_confirm(motion if motion.feedback_valid else None)
                    confirmed = True
                except Exception as exc:
                    status = "failed"
                    error = f"{error + '; ' if error else ''}stop failed: {exc}"
                    self._control_error = error
        if (
            status == "succeeded"
            and motion.angle_deg is not None
            and abs(motion.angle_deg - motion.estimated_angle_deg) > ANGLE_TOLERANCE_DEG
        ):
            status, error = (
                "failed",
                "rotation drifted outside tolerance while stopping",
            )
        return {
            **motion.result(time.monotonic()),
            "status": status,
            "error": error,
            "stop_confirmed": confirmed,
        }

    def reset(self) -> dict[str, Any]:
        """Stop and verify feedback before clearing a control fault."""
        with self._bus_lock:
            if self._closed or self._resetting:
                raise RuntimeError("LeKiwi environment is closed or resetting")
            self._resetting = True
        try:
            self.stop()
            with self._motion_lock:
                with self._bus_lock:
                    if self._closed:
                        raise RuntimeError("LeKiwi environment is closed")
                    self._stop_and_confirm()
                    self._control_error = None
                return {"ok": True, **self.get_observation()}
        finally:
            with self._bus_lock:
                self._resetting = False

    def close_env(self) -> None:
        """Stop and drain the active call before disconnecting robot and cameras."""
        with self._bus_lock:
            if self._closed:
                return
            self._closed = True
        try:
            self.stop()
        finally:
            with self._motion_lock, self._bus_lock, self._camera_lock:
                self.robot.disconnect()


def main(
    *,
    backend_class=LeKiwiEnvBackend,
    load_runtime_config=load_runtime_config,
    facade_class=LeKiwiEnvFacade,
) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--transport", choices=["http", "socket"], default="http")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--robot-config", default=None)
    parser.add_argument("--parent-watch", action="store_true")
    parser.add_argument(
        "--camera-only",
        action="store_true",
        help="Serve the main camera without opening the motor bus.",
    )
    args = parser.parse_args()
    cfg = load_runtime_config(args.robot_config)
    backend = LeKiwiCameraBackend(cfg) if args.camera_only else backend_class(cfg)
    try:
        facade = (
            facade_class(backend, camera_only=True)
            if args.camera_only
            else facade_class(backend)
        )
        facade.serve(
            transport=args.transport,
            host=args.host,
            port=args.port,
            parent_watch=args.parent_watch,
        )
    finally:
        backend.close_env()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
