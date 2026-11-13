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

"""Timing and wheel-feedback integration for bounded motion calls."""

from __future__ import annotations

from typing import Any

import numpy as np

MAX_DURATION_S = 10.0
MOTION_TIMEOUT_S = 60.0
FEEDBACK_TIMEOUT_S = 0.25
STOP_TIMEOUT_S = 1.0
ANGLE_TOLERANCE_DEG = 2.0


def stopped(velocity: np.ndarray) -> bool:
    """Whether measured translation and rotation are near zero."""
    return bool(np.linalg.norm(velocity[:2]) <= 0.01 and abs(velocity[2]) <= 1.0)


class Motion:
    """Local execution data for a timed drive or relative rotation.

    The service owns locking and motor I/O. Only rotation integrates feedback;
    a drive completes by elapsed command time, without a distance guarantee.
    """

    def __init__(
        self,
        velocity: np.ndarray,
        *,
        duration_s: float | None = None,
        angle_deg: float | None = None,
    ) -> None:
        self.velocity = velocity.copy()
        self.duration_s = duration_s
        self.angle_deg = angle_deg
        self.started_at: float | None = None
        self.estimated_angle_deg = 0.0
        self.sampled_at: float | None = None
        self.measured = np.zeros(3)
        self.feedback_valid = False

    def sample(self, measured: np.ndarray, now: float) -> None:
        """Track fresh feedback, integrating yaw only for a rotation."""
        if self.sampled_at is not None:
            dt = now - self.sampled_at
            if dt <= 0:
                self.feedback_valid = False
                raise RuntimeError("wheel feedback is out of order")
            if self.angle_deg is not None:
                self.estimated_angle_deg += (self.measured[2] + measured[2]) * dt / 2
        self.measured = measured
        self.sampled_at = now
        self.feedback_valid = True

    def command(self, now: float) -> tuple[np.ndarray, bool]:
        """Return the next velocity and whether to finish by stopping."""
        if self.started_at is None:
            self.started_at = now
        if self.duration_s is not None:
            return self.velocity, now - self.started_at >= self.duration_s
        error = self.angle_deg - self.estimated_angle_deg
        action = np.zeros(3)
        action[2] = np.clip(error, -self.velocity[2], self.velocity[2])
        return action, abs(error) <= ANGLE_TOLERANCE_DEG

    def result(self, now: float) -> dict[str, Any]:
        """Return measured execution details after stopping."""
        result = {
            "commanded_velocity": self.velocity.tolist(),
            "elapsed_s": now - self.started_at if self.started_at is not None else 0.0,
            "feedback_valid": self.feedback_valid,
        }
        if self.duration_s is not None:
            result["duration_s"] = self.duration_s
        if self.angle_deg is not None:
            result.update(
                angle_deg=self.angle_deg,
                estimated_angle_deg=self.estimated_angle_deg,
                remaining_angle_deg=self.angle_deg - self.estimated_angle_deg,
            )
        return result
