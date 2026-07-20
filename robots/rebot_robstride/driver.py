"""Safety-focused motorbridge driver for the seven-motor reBot DevArm."""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable, Sequence
from typing import Any, NoReturn

from robots.rebot_robstride.config import GripperConfig, JointConfig, RebotConfig

MECH_POS = 0x7019
ROBSTRIDE_HOST_ID = 0xFD
MIT_MODE = 1
MIN_JERK_PEAK_VELOCITY = 1.875


class MotionCancelled(RuntimeError):
    """Raised inside a trajectory after a preemptive stop request."""


def _default_controller_factory(channel: str):
    try:
        from motorbridge import Controller
    except ImportError as exc:  # pragma: no cover - installation error path
        raise RuntimeError(
            "motorbridge is required; install the rpent[rebot-robstride] extra"
        ) from exc
    return Controller(channel)


def _min_jerk(value: float) -> float:
    return 10 * value**3 - 15 * value**4 + 6 * value**5


class RebotRobstrideDriver:
    """Own one RobStride CAN session and fail closed around every command."""

    def __init__(
        self,
        config: RebotConfig,
        *,
        controller_factory: Callable[[str], Any] | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.config = config
        self._controller_factory = controller_factory or _default_controller_factory
        self._clock = clock
        self._sleep = sleep

        # Lock order is motion -> I/O -> state. The emergency-stop path never
        # waits for the motion lock and sets cancellation before waiting for I/O.
        self._motion_lock = threading.Lock()
        self._io_lock = threading.RLock()
        self._state_lock = threading.RLock()
        self._cancel_event = threading.Event()

        self._controller = None
        self._motors: dict[int, Any] = {}
        self._enabled = False
        self._gripper_enabled = False
        self._stopped = False
        self._last_targets: list[float] | None = None
        self._previous_positions: dict[int, float] | None = None
        self._previous_sample_time: float | None = None
        self._last_heartbeat = self._clock()

    @property
    def connected(self) -> bool:
        with self._state_lock:
            return self._controller is not None

    @property
    def enabled(self) -> bool:
        with self._state_lock:
            return self._enabled

    @property
    def stopped(self) -> bool:
        with self._state_lock:
            return self._stopped

    def connect(self) -> dict[str, Any]:
        """Open CAN, register all motors, and take a passive state snapshot."""
        with self._motion_lock:
            with self._state_lock:
                if self._controller is not None:
                    already_connected = True
                else:
                    already_connected = False
            if already_connected:
                return self.state()

            controller = self._controller_factory(self.config.channel)
            try:
                motor_configs = [*self.config.joints, self.config.gripper]
                motors = {
                    motor.motor_id: controller.add_robstride_motor(
                        motor_id=motor.motor_id,
                        feedback_id=ROBSTRIDE_HOST_ID,
                        model=motor.model,
                    )
                    for motor in motor_configs
                }
                with self._state_lock:
                    self._controller = controller
                    self._motors = motors
                snapshot = self.state()
                self._last_targets = list(snapshot["joint_positions"])
                return snapshot
            except Exception:
                try:
                    controller.close()
                finally:
                    with self._state_lock:
                        self._controller = None
                        self._motors = {}
                raise

    def state(self) -> dict[str, Any]:
        """Return fresh positions, estimated velocities, and fault reports."""
        snapshot = self._sample_feedback()
        with self._state_lock:
            enabled = self._enabled
            gripper_enabled = self._gripper_enabled
            stopped = self._stopped
        joint_ids = [joint.motor_id for joint in self.config.joints]
        gripper_id = self.config.gripper.motor_id
        return {
            "connected": True,
            "enabled": enabled,
            "gripper_enabled": gripper_enabled,
            "stopped": stopped,
            "channel": self.config.channel,
            "joint_names": [joint.name for joint in self.config.joints],
            "joint_positions": [
                snapshot["positions"][motor_id] for motor_id in joint_ids
            ],
            "joint_velocities": [
                snapshot["velocities"][motor_id] for motor_id in joint_ids
            ],
            "gripper_position": snapshot["positions"][gripper_id],
            "gripper_velocity": snapshot["velocities"][gripper_id],
            "faults": snapshot["faults"],
            "timestamp": snapshot["timestamp"],
        }

    def enable(self) -> dict[str, Any]:
        """Validate startup state, enable arm joints, and hold observed pose."""
        with self._motion_lock:
            self._require_connected()
            with self._state_lock:
                if self._stopped:
                    raise RuntimeError(
                        "arm is stopped; call reset_stop before enabling"
                    )
                self._cancel_event.clear()
            first = self._sample_feedback()
            self._validate_startup_positions(first)
            self._sleep(self.config.startup_sample_interval_s)
            second = self._sample_feedback()
            self._validate_startup_positions(second)
            self._validate_startup_velocity(second)

            arm_configs = list(self.config.joints)
            try:
                for motor_config in arm_configs:
                    with self._io_lock:
                        self._raise_if_cancelled()
                        motor = self._motors[motor_config.motor_id]
                        motor.clear_error()
                with self._io_lock:
                    self._raise_if_cancelled()
                    self._assert_no_faults(
                        self._read_faults_locked(),
                        {joint.motor_id for joint in arm_configs},
                        context="startup",
                    )
                for motor_config in arm_configs:
                    with self._io_lock:
                        self._raise_if_cancelled()
                        self._motors[motor_config.motor_id].ensure_mode(
                            MIT_MODE, timeout_ms=1000
                        )
                for motor_config in arm_configs:
                    with self._io_lock:
                        self._raise_if_cancelled()
                        self._motors[motor_config.motor_id].enable()
                for motor_config in arm_configs:
                    with self._io_lock:
                        self._raise_if_cancelled()
                        self._send_mit_locked(
                            motor_config,
                            second["positions"][motor_config.motor_id],
                        )
            except Exception as exc:
                self._fail_closed(exc)

            with self._state_lock:
                self._enabled = True
                self._gripper_enabled = False
                self._stopped = False
                self._cancel_event.clear()
                self._last_heartbeat = self._clock()
                self._last_targets = [
                    second["positions"][joint.motor_id] for joint in arm_configs
                ]
            return {
                "enabled": True,
                "gripper_enabled": False,
                "hold_positions": list(self._last_targets),
            }

    def move_joints(
        self,
        positions: Sequence[float],
        *,
        duration_s: float = 2.0,
    ) -> dict[str, Any]:
        """Execute a monitored minimum-jerk trajectory and verify settlement."""
        with self._motion_lock:
            self._require_motion_ready()
            target = [float(value) for value in positions]
            if len(target) != len(self.config.joints):
                raise ValueError("positions must contain exactly six joint values")
            if not all(math.isfinite(value) for value in target):
                raise ValueError("joint targets must be finite")
            self._validate_joint_limits(target)
            requested_duration = self._validate_duration(duration_s)

            self._cancel_event.clear()
            try:
                start_state = self._sample_feedback()
                start = [
                    start_state["positions"][joint.motor_id]
                    for joint in self.config.joints
                ]
                self._validate_motion_feedback(start_state, start)
            except Exception as exc:
                self._fail_closed(exc)
            minimum_duration = MIN_JERK_PEAK_VELOCITY * max(
                abs(goal - initial) / joint.max_velocity
                for initial, goal, joint in zip(start, target, self.config.joints)
            )
            actual_duration = max(requested_duration, minimum_duration)
            self._check_actual_duration(actual_duration)
            steps = max(2, math.ceil(actual_duration * self.config.control_rate_hz))
            interval = actual_duration / steps
            feedback_stride = max(
                1,
                round(self.config.control_rate_hz / self.config.feedback_rate_hz),
            )
            start_time = self._clock()

            try:
                for index in range(1, steps + 1):
                    self._raise_if_cancelled()
                    scale = _min_jerk(index / steps)
                    waypoint = [
                        initial + (goal - initial) * scale
                        for initial, goal in zip(start, target)
                    ]
                    with self._io_lock:
                        for joint, position in zip(self.config.joints, waypoint):
                            self._send_mit_locked(joint, position)
                    with self._state_lock:
                        self._last_targets = waypoint

                    remaining = start_time + index * interval - self._clock()
                    if remaining > 0 and self._sleep_interruptible(remaining):
                        raise MotionCancelled("joint motion cancelled by stop request")
                    if index % feedback_stride == 0 or index == steps:
                        self._validate_motion_feedback(
                            self._sample_feedback(), waypoint
                        )

                settled = self._wait_for_joint_target(target)
            except MotionCancelled:
                raise
            except Exception as exc:
                self._fail_closed(exc)

            errors = [
                abs(observed - goal)
                for observed, goal in zip(settled["positions"], target)
            ]
            max_error = max(errors)
            reached = bool(settled["reached"])
            if not reached:
                self._disable_after_timeout("joint settlement timed out")
            return {
                "target_positions": target,
                "final_positions": settled["positions"],
                "final_velocities": settled["velocities"],
                "max_error": max_error,
                "reached": reached,
                "enabled": self.enabled,
                "requested_duration_s": requested_duration,
                "actual_duration_s": actual_duration,
            }

    def set_gripper(
        self,
        position: float,
        *,
        duration_s: float = 1.0,
    ) -> dict[str, Any]:
        """Enable and move the calibrated gripper; 0.0=open and 1.0=closed."""
        with self._motion_lock:
            self._require_motion_ready()
            normalized = float(position)
            if not math.isfinite(normalized) or not 0.0 <= normalized <= 1.0:
                raise ValueError("gripper position must be in [0.0, 1.0]")
            requested_duration = self._validate_duration(duration_s)
            gripper = self.config.gripper
            if gripper.open_position is None or gripper.closed_position is None:
                raise RuntimeError(
                    "gripper is not calibrated; configure open_position and closed_position"
                )

            self._cancel_event.clear()
            try:
                snapshot = self._sample_feedback()
            except Exception as exc:
                self._fail_closed(exc)
            start = snapshot["positions"][gripper.motor_id]
            travel_lower = min(gripper.open_position, gripper.closed_position)
            travel_upper = max(gripper.open_position, gripper.closed_position)
            if not travel_lower <= start <= travel_upper:
                raise RuntimeError(
                    f"gripper feedback {start:.6f} is outside calibrated travel "
                    f"[{travel_lower:.6f}, {travel_upper:.6f}]"
                )
            target = gripper.open_position + normalized * (
                gripper.closed_position - gripper.open_position
            )
            self._ensure_gripper_enabled(gripper, start)

            minimum_duration = (
                MIN_JERK_PEAK_VELOCITY * abs(target - start) / gripper.max_velocity
            )
            actual_duration = max(requested_duration, minimum_duration)
            self._check_actual_duration(actual_duration)
            steps = max(2, math.ceil(actual_duration * self.config.control_rate_hz))
            interval = actual_duration / steps
            feedback_stride = max(
                1,
                round(self.config.control_rate_hz / self.config.feedback_rate_hz),
            )
            start_time = self._clock()

            try:
                for index in range(1, steps + 1):
                    self._raise_if_cancelled()
                    waypoint = start + (target - start) * _min_jerk(index / steps)
                    with self._io_lock:
                        self._send_mit_locked(gripper, waypoint)
                    remaining = start_time + index * interval - self._clock()
                    if remaining > 0 and self._sleep_interruptible(remaining):
                        raise MotionCancelled(
                            "gripper motion cancelled by stop request"
                        )
                    if index % feedback_stride == 0 or index == steps:
                        feedback = self._sample_feedback()
                        self._validate_gripper_feedback(feedback, waypoint)

                settled = self._wait_for_gripper_target(target)
            except MotionCancelled:
                raise
            except Exception as exc:
                self._fail_closed(exc)

            error = abs(settled["position"] - target)
            reached = bool(settled["reached"])
            if not reached:
                self._disable_after_timeout("gripper settlement timed out")
            return {
                "normalized_position": normalized,
                "target_position": target,
                "final_position": settled["position"],
                "final_velocity": settled["velocity"],
                "max_error": error,
                "reached": reached,
                "enabled": self.enabled,
                "actual_duration_s": actual_duration,
            }

    def stop_motion(self) -> dict[str, Any]:
        """Preempt motion, latch a software stop, and hold fresh arm feedback."""
        self._cancel_event.set()
        with self._state_lock:
            self._require_connected_locked()
            self._stopped = True
            enabled = self._enabled
        if enabled:
            try:
                snapshot = self._sample_feedback()
                self._validate_startup_positions(snapshot)
                self._assert_no_faults(
                    snapshot["faults"],
                    {joint.motor_id for joint in self.config.joints},
                    context="soft stop",
                )
                with self._io_lock:
                    for joint in self.config.joints:
                        self._send_mit_locked(
                            joint, snapshot["positions"][joint.motor_id]
                        )
                with self._state_lock:
                    self._last_targets = [
                        snapshot["positions"][joint.motor_id]
                        for joint in self.config.joints
                    ]
            except Exception as exc:
                self._fail_closed(exc)
        return {"enabled": self.enabled, "stopped": True}

    def reset_stop(self) -> dict[str, Any]:
        """Clear only the software-stop latch; it never enables motors."""
        with self._state_lock:
            self._require_connected_locked()
            self._stopped = False
            self._cancel_event.clear()
            return {"enabled": self._enabled, "stopped": False}

    def emergency_stop(self) -> dict[str, Any]:
        """Preempt motion and disable every motor without waiting for trajectory locks."""
        self._cancel_event.set()
        with self._state_lock:
            self._require_connected_locked()
            self._stopped = True
        self._disable_all()
        return {"enabled": False, "stopped": True}

    def heartbeat(self) -> dict[str, Any]:
        """Refresh the agent-process deadman without changing motor state."""
        with self._state_lock:
            self._require_connected_locked()
            self._last_heartbeat = self._clock()
            return {"ok": True, "enabled": self._enabled}

    def enforce_heartbeat_deadman(self) -> bool:
        """Disable enabled motors after the agent heartbeat expires."""
        with self._state_lock:
            expired = (
                self._enabled or self._gripper_enabled
            ) and self._clock() - self._last_heartbeat > self.config.heartbeat_timeout_s
        if not expired:
            return False
        self.emergency_stop()
        return True

    def close(self) -> None:
        """Cancel motion, disable enabled motors, and release the controller."""
        self._cancel_event.set()
        with self._motion_lock:
            with self._state_lock:
                controller = self._controller
                should_disable = self._enabled or self._gripper_enabled
            if controller is None:
                return
            try:
                if should_disable:
                    self._disable_all()
            finally:
                with self._io_lock:
                    controller.close()
                with self._state_lock:
                    self._controller = None
                    self._motors = {}
                    self._enabled = False
                    self._gripper_enabled = False
                    self._stopped = True
                    self._previous_positions = None
                    self._previous_sample_time = None

    def _sample_feedback(self) -> dict[str, Any]:
        self._require_connected()
        with self._io_lock:
            positions = {
                motor_id: self._read_position_locked(motor_id)
                for motor_id in self._motors
            }
            faults = self._read_faults_locked()
        timestamp = self._clock()
        with self._state_lock:
            previous_positions = self._previous_positions
            previous_time = self._previous_sample_time
            if (
                previous_positions is None
                or previous_time is None
                or timestamp <= previous_time
            ):
                velocities = dict.fromkeys(positions, 0.0)
            else:
                elapsed = timestamp - previous_time
                velocities = {
                    motor_id: (position - previous_positions[motor_id]) / elapsed
                    for motor_id, position in positions.items()
                }
            self._previous_positions = dict(positions)
            self._previous_sample_time = timestamp
        return {
            "positions": positions,
            "velocities": velocities,
            "faults": faults,
            "timestamp": timestamp,
        }

    def _read_position_locked(self, motor_id: int) -> float:
        position = float(
            self._motors[motor_id].robstride_get_param_f32(
                MECH_POS, timeout_ms=self.config.read_timeout_ms
            )
        )
        if not math.isfinite(position):
            raise RuntimeError(
                f"motor {motor_id} returned non-finite position feedback"
            )
        return position

    def _read_faults_locked(self) -> dict[int, dict[str, int]]:
        reports: dict[int, dict[str, int]] = {}
        for motor_id, motor in self._motors.items():
            fault_raw, warning_raw = motor.robstride_get_fault_report()
            reports[motor_id] = {
                "fault_raw": int(fault_raw),
                "warning_raw": int(warning_raw),
            }
        return reports

    def _send_mit_locked(
        self, config: JointConfig | GripperConfig, position: float
    ) -> None:
        self._motors[config.motor_id].send_mit(
            float(position), 0.0, float(config.kp), float(config.kd), 0.0
        )

    def _validate_startup_positions(self, snapshot: dict[str, Any]) -> None:
        positions = [
            snapshot["positions"][joint.motor_id] for joint in self.config.joints
        ]
        self._validate_joint_limits(positions)

    def _validate_startup_velocity(self, snapshot: dict[str, Any]) -> None:
        for joint in self.config.joints:
            velocity = abs(snapshot["velocities"][joint.motor_id])
            if velocity > self.config.startup_velocity_limit_rad_s:
                raise RuntimeError(
                    f"{joint.name} startup velocity {velocity:.6f} rad/s exceeds "
                    f"{self.config.startup_velocity_limit_rad_s:.6f} rad/s"
                )

    def _validate_motion_feedback(
        self, snapshot: dict[str, Any], waypoint: Sequence[float]
    ) -> None:
        arm_ids = {joint.motor_id for joint in self.config.joints}
        self._assert_no_faults(snapshot["faults"], arm_ids, context="motion")
        for joint, expected in zip(self.config.joints, waypoint):
            observed = snapshot["positions"][joint.motor_id]
            error = abs(observed - expected)
            if error > self.config.max_tracking_error_rad:
                raise RuntimeError(
                    f"{joint.name} tracking error {error:.6f} rad exceeds "
                    f"{self.config.max_tracking_error_rad:.6f} rad"
                )
            velocity = abs(snapshot["velocities"][joint.motor_id])
            velocity_limit = joint.max_velocity * self.config.velocity_abort_multiplier
            if velocity > velocity_limit:
                raise RuntimeError(
                    f"{joint.name} feedback velocity {velocity:.6f} rad/s exceeds "
                    f"abort limit {velocity_limit:.6f} rad/s"
                )

    def _validate_gripper_feedback(
        self, snapshot: dict[str, Any], waypoint: float
    ) -> None:
        gripper = self.config.gripper
        self._assert_no_faults(
            snapshot["faults"], {gripper.motor_id}, context="gripper motion"
        )
        error = abs(snapshot["positions"][gripper.motor_id] - waypoint)
        if error > self.config.max_tracking_error_rad:
            raise RuntimeError(
                f"gripper tracking error {error:.6f} rad exceeds "
                f"{self.config.max_tracking_error_rad:.6f} rad"
            )
        velocity = abs(snapshot["velocities"][gripper.motor_id])
        limit = gripper.max_velocity * self.config.velocity_abort_multiplier
        if velocity > limit:
            raise RuntimeError(
                f"gripper feedback velocity {velocity:.6f} rad/s exceeds "
                f"abort limit {limit:.6f} rad/s"
            )

    @staticmethod
    def _assert_no_faults(
        reports: dict[int, dict[str, int]], motor_ids: set[int], *, context: str
    ) -> None:
        active = {
            motor_id: reports[motor_id]
            for motor_id in sorted(motor_ids)
            if reports[motor_id]["fault_raw"] or reports[motor_id]["warning_raw"]
        }
        if active:
            raise RuntimeError(f"RobStride fault during {context}: {active}")

    def _ensure_gripper_enabled(self, gripper: GripperConfig, hold: float) -> None:
        with self._state_lock:
            if self._gripper_enabled:
                return
        try:
            motor = self._motors[gripper.motor_id]
            with self._io_lock:
                self._raise_if_cancelled()
                motor.clear_error()
            with self._io_lock:
                self._raise_if_cancelled()
                self._assert_no_faults(
                    self._read_faults_locked(),
                    {gripper.motor_id},
                    context="gripper startup",
                )
            with self._io_lock:
                self._raise_if_cancelled()
                motor.ensure_mode(MIT_MODE, timeout_ms=1000)
            with self._io_lock:
                self._raise_if_cancelled()
                motor.enable()
            with self._io_lock:
                self._raise_if_cancelled()
                self._send_mit_locked(gripper, hold)
            with self._state_lock:
                self._gripper_enabled = True
        except Exception as exc:
            self._fail_closed(exc)

    def _wait_for_joint_target(self, target: list[float]) -> dict[str, Any]:
        deadline = self._clock() + self.config.settle_timeout_s
        consecutive = 0
        final_positions = target
        final_velocities = [math.inf] * len(target)
        while True:
            self._raise_if_cancelled()
            snapshot = self._sample_feedback()
            self._validate_motion_feedback(snapshot, target)
            final_positions = [
                snapshot["positions"][joint.motor_id] for joint in self.config.joints
            ]
            final_velocities = [
                snapshot["velocities"][joint.motor_id] for joint in self.config.joints
            ]
            positions_ok = (
                max(abs(value - goal) for value, goal in zip(final_positions, target))
                <= self.config.settle_tolerance
            )
            velocities_ok = max(abs(value) for value in final_velocities) <= (
                self.config.settle_velocity_rad_s
            )
            consecutive = consecutive + 1 if positions_ok and velocities_ok else 0
            if consecutive >= self.config.settle_samples:
                return {
                    "positions": final_positions,
                    "velocities": final_velocities,
                    "reached": True,
                }
            if self._clock() >= deadline:
                return {
                    "positions": final_positions,
                    "velocities": final_velocities,
                    "reached": False,
                }
            if self._sleep_interruptible(1.0 / self.config.feedback_rate_hz):
                raise MotionCancelled("joint settlement cancelled by stop request")

    def _wait_for_gripper_target(self, target: float) -> dict[str, Any]:
        deadline = self._clock() + self.config.settle_timeout_s
        consecutive = 0
        motor_id = self.config.gripper.motor_id
        final_position = target
        final_velocity = math.inf
        while True:
            self._raise_if_cancelled()
            snapshot = self._sample_feedback()
            self._validate_gripper_feedback(snapshot, target)
            final_position = snapshot["positions"][motor_id]
            final_velocity = snapshot["velocities"][motor_id]
            settled = (
                abs(final_position - target) <= self.config.settle_tolerance
                and abs(final_velocity) <= self.config.settle_velocity_rad_s
            )
            consecutive = consecutive + 1 if settled else 0
            if consecutive >= self.config.settle_samples:
                return {
                    "position": final_position,
                    "velocity": final_velocity,
                    "reached": True,
                }
            if self._clock() >= deadline:
                return {
                    "position": final_position,
                    "velocity": final_velocity,
                    "reached": False,
                }
            if self._sleep_interruptible(1.0 / self.config.feedback_rate_hz):
                raise MotionCancelled("gripper settlement cancelled by stop request")

    def _sleep_interruptible(self, duration_s: float) -> bool:
        deadline = self._clock() + max(0.0, duration_s)
        while True:
            if self._cancel_event.is_set():
                return True
            remaining = deadline - self._clock()
            if remaining <= 0:
                return False
            self._sleep(min(remaining, 0.01))

    def _raise_if_cancelled(self) -> None:
        if self._cancel_event.is_set():
            raise MotionCancelled("motion cancelled by stop request")

    def _validate_duration(self, duration_s: float) -> float:
        duration = float(duration_s)
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError("duration_s must be positive and finite")
        if duration > self.config.max_motion_duration_s:
            raise ValueError(
                f"duration_s must not exceed {self.config.max_motion_duration_s:.3f}"
            )
        return duration

    def _check_actual_duration(self, duration_s: float) -> None:
        if duration_s > self.config.max_motion_duration_s:
            raise ValueError(
                "velocity-limited trajectory duration exceeds max_motion_duration_s"
            )

    def _disable_after_timeout(self, reason: str) -> None:
        try:
            self._disable_all()
        except Exception as exc:
            raise RuntimeError(f"{reason}; disable_all also failed: {exc}") from exc

    def _fail_closed(self, error: Exception) -> NoReturn:
        self._cancel_event.set()
        with self._state_lock:
            self._stopped = True
        try:
            self._disable_all()
        except Exception as disable_error:
            raise RuntimeError(
                f"{error}; fail-closed disable_all also failed: {disable_error}"
            ) from error
        raise error

    def _disable_all(self) -> None:
        with self._io_lock:
            controller = self._controller
            if controller is None:
                return
            controller.disable_all()
        with self._state_lock:
            self._enabled = False
            self._gripper_enabled = False
            self._stopped = True

    def _validate_joint_limits(self, target: Sequence[float]) -> None:
        for joint, value in zip(self.config.joints, target):
            if not joint.lower <= value <= joint.upper:
                raise ValueError(
                    f"{joint.name} target {value:.6f} outside "
                    f"[{joint.lower:.6f}, {joint.upper:.6f}]"
                )

    def _require_connected(self) -> None:
        with self._state_lock:
            self._require_connected_locked()

    def _require_connected_locked(self) -> None:
        if self._controller is None:
            raise RuntimeError("arm is not connected")

    def _require_motion_ready(self) -> None:
        with self._state_lock:
            self._require_connected_locked()
            if not self._enabled:
                raise RuntimeError("arm is not enabled")
            if self._stopped:
                raise RuntimeError("arm is stopped; call reset_stop before moving")
