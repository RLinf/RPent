# Copyright 2026 The RPent Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at https://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The two YAM followers, opened through RLinf's dual YAM robot.

``YamFollowerPair`` is RPent's only writer to the followers. ``YamAgentEnv``
owns the motion guards; the pair refuses stale or non-finite data, clips the
grippers to ``[0, 1]``, and holds both arms after any failed read or write.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from rpent.utils.logging import get_logger

_LOGGER = get_logger(__name__)
_SIDES = ("left", "right")
_ARM_JOINTS = np.array([*range(6), *range(7, 13)])


class YamMoveCancelled(RuntimeError):
    """A move was cancelled; the followers stay connected and held."""


@dataclass(frozen=True)
class YamFollowerState:
    """One follower reading: six joints in radians, then the gripper opening."""

    qpos: np.ndarray
    timestamp_s: float


@dataclass(frozen=True)
class YamPairState:
    """Left and right follower readings."""

    left: YamFollowerState
    right: YamFollowerState

    def as_vector(self) -> np.ndarray:
        """Return the 14-D left-then-right qpos."""
        return np.concatenate([self.left.qpos, self.right.qpos])


@dataclass(frozen=True)
class YamCommandResult:
    """The 14-D target requested, the target sent, and why it was refused."""

    requested: np.ndarray
    accepted: np.ndarray
    clipped: bool = False
    rejection_reason: str | None = None


def _qpos14(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (14,):
        raise ValueError(f"YAM {name} must be 14-D, got shape {array.shape}")
    return array.copy()


def _arm_settings(device: Any) -> dict[str, Any]:
    for name in ("grav_comp_kd", "coulomb_friction"):
        if getattr(device, name, None) is not None:
            raise ValueError(
                f"The pinned i2rt cannot set YAM {name!r} per follower; "
                f"leave {device.channel} {name} null"
            )
    return {
        "ee_mass": device.ee_mass,
        "gripper_limits": device.gripper_limits,
        "gravity_comp_factor": device.gravity_comp_factor,
        "use_coulomb_friction": device.use_coulomb_friction,
        "enable_auto_recovery": device.enable_auto_recovery,
    }


def build_follower_pair(
    left: Any,
    right: Any,
    *,
    joint_lower: Any,
    joint_upper: Any,
    step_frequency: float,
    feedback_timeout_s: float,
) -> YamFollowerPair:
    """Declare both followers on RLinf's dual YAM robot without opening them.

    Args:
        left: Left follower settings: ``channel``, ``arm_type``,
            ``gripper_type``, and the i2rt values in ``_arm_settings``.
        right: Right follower settings, as for ``left``.
        joint_lower: ``(2, 6)`` lower joint limits in radians, left first.
        joint_upper: ``(2, 6)`` upper joint limits in radians, left first.
        step_frequency: Rate :meth:`YamFollowerPair.move_to_qpos` commands at,
            in hertz.
        feedback_timeout_s: Joint feedback older than this fails a read.
    """
    from rlinf.robotics.parts.arms.yam import YamArm
    from rlinf.robotics.robots.dual_yam import DualYamRobot

    models = {(device.arm_type, device.gripper_type) for device in (left, right)}
    if len(models) != 1:
        raise ValueError(
            "Both YAM followers must use one arm_type and gripper_type, "
            f"but got {sorted(models)}"
        )
    ((arm_type, gripper_type),) = models
    robot = DualYamRobot.build(
        left_can_channel=left.channel,
        right_can_channel=right.channel,
        arm_type=arm_type,
        gripper_type=gripper_type,
        left_arm=_arm_settings(left),
        right_arm=_arm_settings(right),
        feedback_timeout_s=feedback_timeout_s,
    )
    return YamFollowerPair(
        *(robot.child(side).child("arm", YamArm) for side in _SIDES),
        joint_lower=joint_lower,
        joint_upper=joint_upper,
        step_frequency=step_frequency,
    )


class YamFollowerPair:
    """Left and right YAM followers.

    Args:
        left: Unopened left arm part.
        right: Unopened right arm part.
        joint_lower: ``(2, 6)`` lower joint limits in radians, left first.
        joint_upper: ``(2, 6)`` upper joint limits in radians, left first.
        step_frequency: Rate :meth:`move_to_qpos` commands at, in hertz.
    """

    def __init__(
        self,
        left: Any,
        right: Any,
        *,
        joint_lower: Any,
        joint_upper: Any,
        step_frequency: float,
    ) -> None:
        self._arms = (left, right)
        self._lower = np.asarray(joint_lower, dtype=np.float64).reshape(2, 6)
        self._upper = np.asarray(joint_upper, dtype=np.float64).reshape(2, 6)
        self._step_frequency = float(step_frequency)
        #: The connected arm parts, left first; ``None`` while closed.
        self.followers: tuple[Any, Any] | None = None
        self._closed = False
        self._release_error: BaseException | None = None

    def connect_followers(self) -> None:
        """Open the left follower, then the right, each held at its measured pose.

        Startup fails on joint limits wider than i2rt allows and on absent or
        stale feedback, closing whatever opened. A failed arm connection leaves
        its release state unknown and prevents reconnection.
        """
        if self._closed:
            raise RuntimeError("cannot reconnect closed YAM followers")
        if self.followers is not None:
            return
        opened = []
        try:
            for arm in self._arms:
                try:
                    arm.connect()
                except BaseException as error:
                    # RLinf can discard the handle after a failed startup release.
                    self._release_error = self._release_error or error
                    self._closed = True
                    raise
                opened.append(arm)
            self.followers = self._arms
            self._check_joint_limits()
            self.read_state()
        except BaseException:
            self.followers = None
            self._release(opened)
            raise

    def read_state(self) -> YamPairState:
        """Read both followers, refusing stale or non-finite feedback."""
        states = []
        try:
            for arm in self._require_followers():
                state = arm.get_state()
                states.append(
                    YamFollowerState(
                        qpos=np.concatenate(
                            [state.arm_joint_position, state.gripper_position]
                        ),
                        timestamp_s=float(state.timestamp),
                    )
                )
        except Exception:
            self.emergency_hold()
            raise
        return YamPairState(*states)

    def command(self, action: Any) -> YamCommandResult:
        """Send one 14-D target to both followers, left first.

        A non-finite target is not sent; both arms hold instead. The joints
        pass unchanged and i2rt clips them to its travel.
        """
        requested = _qpos14(action, "action")
        followers = self._require_followers()
        if not np.all(np.isfinite(requested)):
            return YamCommandResult(
                requested=requested,
                accepted=self.hold(),
                rejection_reason="non_finite_action",
            )
        try:
            self.read_state()
        except Exception:
            self.emergency_hold()
            raise
        accepted = requested.copy()
        accepted[[6, 13]] = np.clip(accepted[[6, 13]], 0.0, 1.0)
        try:
            for arm, target in zip(followers, accepted.reshape(2, 7), strict=True):
                arm.move_joints(target[:6])
                arm.move_gripper(target[6:])
        except Exception:
            self.emergency_hold()
            raise
        return YamCommandResult(
            requested=requested,
            accepted=accepted,
            clipped=not np.array_equal(requested, accepted),
        )

    def move_to_qpos(
        self,
        target: Any,
        *,
        duration_s: float,
        max_joint_delta: float,
        tolerance: float,
        timeout_s: float,
        cancelled: Callable[[], bool] | None = None,
    ) -> np.ndarray:
        """Move smoothly to an absolute 14-D target, then hold it.

        Consecutive setpoints differ by at most ``max_joint_delta`` per joint.
        After the interpolation the target is re-sent until every value is
        within ``tolerance``. When ``cancelled`` returns true, both arms hold
        and :class:`YamMoveCancelled` is raised.

        Returns:
            The measured 14-D qpos the followers hold.
        """
        target = _qpos14(target, "move target")
        if not np.all(np.isfinite(target)):
            raise ValueError("YAM move target must be finite")
        joints = target.reshape(2, 7)[:, :6]
        if np.any(joints < self._lower) or np.any(joints > self._upper):
            raise ValueError("YAM move target is outside the joint limits")
        if np.any((target[[6, 13]] < 0.0) | (target[[6, 13]] > 1.0)):
            raise ValueError("YAM move gripper targets must be within [0, 1]")
        duration_s, max_joint_delta, tolerance, timeout_s = (
            float(value)
            for value in (duration_s, max_joint_delta, tolerance, timeout_s)
        )
        if not np.all(np.isfinite([duration_s, max_joint_delta, tolerance, timeout_s])):
            raise ValueError("YAM move timing and tolerance must be finite")
        if duration_s < 0 or max_joint_delta <= 0 or tolerance <= 0:
            raise ValueError("invalid YAM move timing or tolerance")
        if timeout_s < duration_s:
            raise ValueError("YAM move timeout must be at least its duration")

        start = self.read_state().as_vector()
        demand = np.abs(target[_ARM_JOINTS] - start[_ARM_JOINTS]) / max_joint_delta
        # smoothstep's steepest slope is 1.5 times the mean step.
        steps = max(
            1,
            int(np.ceil(duration_s * self._step_frequency)),
            int(np.ceil(1.5 * np.max(demand))),
        )
        period_s = 1.0 / self._step_frequency
        deadline_s = time.monotonic() + timeout_s
        _LOGGER.info(
            "Moving YAM followers over %d steps (%.2fs nominal)",
            steps,
            steps * period_s,
        )

        def check_cancelled() -> None:
            if cancelled is not None and cancelled():
                raise YamMoveCancelled("YAM move cancelled; followers remain connected")

        def send(setpoint: np.ndarray) -> None:
            result = self.command(setpoint)
            if result.rejection_reason is not None:
                raise RuntimeError(
                    f"YAM move command rejected: {result.rejection_reason}"
                )

        try:
            for step in range(1, steps + 1):
                progress = step / steps
                alpha = progress * progress * (3.0 - 2.0 * progress)
                check_cancelled()
                send(start + alpha * (target - start))
                if step < steps:
                    remaining_s = deadline_s - time.monotonic()
                    if remaining_s <= 0:
                        raise TimeoutError("YAM move timed out during interpolation")
                    time.sleep(min(period_s, remaining_s))
            while True:
                max_error = float(
                    np.max(np.abs(target - self.read_state().as_vector()))
                )
                check_cancelled()
                if max_error <= tolerance:
                    held = self.hold()
                    check_cancelled()
                    _LOGGER.info("YAM move target reached (max error %.4f)", max_error)
                    return held
                remaining_s = deadline_s - time.monotonic()
                if remaining_s <= 0:
                    raise TimeoutError(
                        f"YAM move did not converge within {timeout_s:.2f}s "
                        f"(max error {max_error:.4f}, tolerance {tolerance:.4f})"
                    )
                check_cancelled()
                send(target)
                time.sleep(min(period_s, remaining_s))
        except Exception:
            self.emergency_hold()
            raise

    def hold(self) -> np.ndarray:
        """Hold both followers at their measured pose and return it."""
        try:
            state = self.read_state()
        except Exception:
            self.emergency_hold()
            raise
        error = self._hold_each()
        if error is not None:
            raise error
        return state.as_vector()

    def emergency_hold(self) -> None:
        """Hold every connected follower, logging rather than raising failures."""
        self._hold_each()

    def close(self) -> None:
        """Release both followers.

        A failed connection or release leaves the follower's CAN chain state
        unknown, so the caller must not hand the hardware to anyone else.
        """
        if self._release_error is not None:
            raise RuntimeError(
                "a YAM follower's release state is unknown; "
                "its CAN chain may still be open"
            ) from self._release_error
        self._closed = True
        followers, self.followers = self.followers, None
        if followers is not None:
            self._release(followers)
        if self._release_error is not None:
            raise RuntimeError(
                f"failed to release the YAM followers: {self._release_error}"
            ) from self._release_error

    def _require_followers(self) -> tuple[Any, Any]:
        if self.followers is None:
            raise RuntimeError("YAM followers are not connected")
        return self.followers

    def _check_joint_limits(self) -> None:
        for index, arm in enumerate(self._require_followers()):
            sdk = arm.joint_limits()
            if np.any(self._lower[index] < sdk[:, 0] - 1e-6) or np.any(
                self._upper[index] > sdk[:, 1] + 1e-6
            ):
                raise ValueError(
                    f"configured YAM {_SIDES[index]} joint limits must be within "
                    f"the i2rt limits {sdk.tolist()}"
                )

    def _hold_each(self) -> BaseException | None:
        first_error = None
        for arm in self.followers or ():
            try:
                arm.hold()
            except Exception as error:
                first_error = first_error or error
                _LOGGER.exception("Failed to hold a YAM follower")
        return first_error

    def _release(self, arms: Any) -> None:
        for arm in reversed(tuple(arms)):
            try:
                arm.disconnect()
            except Exception as error:
                self._release_error = self._release_error or error
                _LOGGER.exception("Failed to release a YAM follower")
