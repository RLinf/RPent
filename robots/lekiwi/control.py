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

"""Exclusive wheel control and the small HTTP adapter used by ROS workers."""

from __future__ import annotations

import math
import threading
import time
import uuid
from typing import Any

from rpent.utils.logging import get_logger

logger = get_logger("lekiwi_control")


def vector(value: Any) -> list[float]:
    """Validate one planar body velocity."""
    if len(value) != 3:
        raise ValueError("velocity must contain exactly three values")
    result = [float(item) for item in value]
    if not all(math.isfinite(item) for item in result):
        raise ValueError("velocity must contain only finite values")
    return result


class EnvControl:
    """Run manual and ROS commands through one RLinf step loop.

    The runtime changes modes. A generation identifies each navigation run,
    so an exiting ROS child cannot submit commands to the next run.
    """

    def __init__(
        self, env, raw, *, max_linear, max_angular, command_timeout, simulated=False
    ):
        self.env, self.raw = env, raw
        self.max_linear, self.max_angular = max_linear, max_angular
        self.command_timeout = command_timeout
        self.simulated = simulated
        self.lock = threading.RLock()
        self.step_lock = threading.RLock()
        self._closing = threading.Event()
        self._ready = threading.Event()
        self._thread = None
        self.mode = "manual"
        self.generation = uuid.uuid4().hex
        self._target = [0.0, 0.0, 0.0]
        self._deadline = 0.0
        self.stop_revision = 0
        self._navigation_armed = False
        self._navigation_min_expiry = 0.0
        self._sample = {}
        self._sample_monotonic = 0.0
        self._fault = None
        self.observation = None
        self.observed_at = 0.0
        self.steps = 0

    def start(self):
        self._thread = threading.Thread(
            target=self._run, name="lekiwi-env-step", daemon=True
        )
        self._thread.start()
        if not self._ready.wait(10):
            self.close()
            raise TimeoutError("Env initial observation timed out")
        if self._fault:
            raise RuntimeError(self._fault)

    def stop(self):
        with self.step_lock, self.lock:
            self._target = [0.0, 0.0, 0.0]
            self._deadline = 0.0
            self._navigation_armed = False
            # Commands sent before this stop expire within command_timeout.
            self._navigation_min_expiry = time.time() + self.command_timeout
            self.stop_revision += 1
            self.raw.stop()

    def arm_navigation(self, stop_revision: int) -> None:
        """Allow a new goal only if no stop occurred during its submission."""
        with self.lock:
            if self.mode != "navigation" or self.stop_revision != stop_revision:
                raise RuntimeError("Navigation submission interrupted by stop")
            self._navigation_armed = True

    def switch_mode(self, mode):
        if mode not in {"manual", "navigation", "stopping"}:
            raise ValueError("Unknown control mode")
        with self.step_lock, self.lock:
            self.stop()
            self.mode = mode
            self.generation = uuid.uuid4().hex

    def navigation_info(self):
        with self.lock:
            if self.mode != "navigation":
                raise RuntimeError("Navigation is not active")
            return {
                "generation": self.generation,
                "command_timeout_s": self.command_timeout,
                "simulated": self.simulated,
            }

    def submit(self, velocity, *, mode, expires_at, generation=None):
        x, y, yaw = vector(velocity)
        with self.lock:
            if self._closing.is_set() or self._fault:
                raise RuntimeError("Env control is unavailable")
            if mode != self.mode or (
                mode == "navigation" and generation != self.generation
            ):
                raise RuntimeError("Command belongs to an inactive mode")
            remaining = expires_at - time.time()
            if not math.isfinite(remaining) or remaining <= 0:
                raise ValueError("Command is expired")
            if mode == "navigation" and (
                not self._navigation_armed or expires_at <= self._navigation_min_expiry
            ):
                return {"accepted": False, "target": [0.0, 0.0, 0.0]}
            if mode == "navigation":
                remaining = min(remaining, self.command_timeout)
            scale = max(1.0, math.hypot(x, y) / self.max_linear)
            self._target = [
                x / scale,
                y / scale,
                max(-self.max_angular, min(self.max_angular, yaw)),
            ]
            self._deadline = time.monotonic() + remaining
            target = self._target[:]
        return {"accepted": True, "target": target}

    def set_velocity(self, generation, velocity, expires_at):
        return self.submit(
            velocity, mode="navigation", generation=generation, expires_at=expires_at
        )

    def state(self):
        with self.lock:
            return {
                **self._sample,
                "sample_age_s": time.monotonic() - self._sample_monotonic
                if self._sample
                else None,
                "mode": self.mode,
                "fault": self._fault,
                "closed": self._closing.is_set(),
                "simulated": self.simulated,
            }

    def _update_sample(self):
        sample = self.raw.get_velocity_sample()
        if not sample or sample["sampled_monotonic"] <= self._sample_monotonic:
            return
        measured = vector(sample["velocity"])
        measured[2] = math.radians(measured[2])
        with self.lock:
            self._sample = {
                "sampled_at": sample["sampled_at"],
                "measured_velocity": measured,
            }
            self._sample_monotonic = sample["sampled_monotonic"]

    def _run(self):
        try:
            self.env.reset()
            self._update_sample()
            while not self._closing.is_set():
                with self.step_lock:
                    with self.lock:
                        active = time.monotonic() < self._deadline
                        target = self._target[:] if active else [0.0, 0.0, 0.0]
                    import numpy as np

                    action = np.array(
                        [target[0], target[1], math.degrees(target[2])],
                        dtype=np.float32,
                    )
                    observation, *_ = self.env.step(action[None, :])
                    self._update_sample()
                    with self.lock:
                        self.observation = observation
                        self.observed_at = time.monotonic()
                        self.steps += 1
                self._ready.set()
                time.sleep(0)  # Let stop and reset acquire the step lock.
        except Exception as exc:
            with self.lock:
                self._fault = f"{type(exc).__name__}: {exc}"
            logger.exception("LeKiwi Env step failed")
        finally:
            try:
                self.stop()
            finally:
                self._ready.set()

    def close(self):
        self._closing.set()
        self.stop()
        if self._thread is not None:
            self._thread.join(timeout=5)
            if self._thread.is_alive():
                raise TimeoutError("Env step has not returned")


class MockBase:
    """Ideal velocity feedback for connection checks; never opens a device."""

    def __init__(self) -> None:
        self.velocity = [0.0, 0.0, 0.0]

    def connect(self) -> None:
        self.reset()

    def disconnect(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.velocity = [0.0, 0.0, 0.0]

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        self.velocity = list(action["velocity"])
        return {"velocity": self.velocity[:]}

    def get_observation(self) -> dict[str, Any]:
        return {"velocity": self.velocity[:]}
