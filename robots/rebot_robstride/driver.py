"""Safe motorbridge driver for the seven-motor reBot DevArm RobStride arm."""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable, Sequence
from typing import Any

from robots.rebot_robstride.config import JointConfig, RebotConfig

MECH_POS = 0x7019
MECH_VEL = 0x701A
ROBSTRIDE_HOST_ID = 0xFD
MIT_MODE = 1


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
        self._lock = threading.RLock()
        self._controller = None
        self._motors: dict[int, Any] = {}
        self._enabled = False
        self._stopped = False
        self._last_targets: list[float] | None = None

    @property
    def connected(self) -> bool:
        return self._controller is not None

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def stopped(self) -> bool:
        return self._stopped

    def connect(self) -> dict[str, Any]:
        """Open CAN, register all motors, and take a passive state snapshot."""
        with self._lock:
            if self.connected:
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
                self._controller = controller
                self._motors = motors
                snapshot = self.state()
                self._last_targets = list(snapshot["joint_positions"])
                return snapshot
            except Exception:
                try:
                    controller.close()
                finally:
                    self._controller = None
                    self._motors = {}
                raise

    def state(self) -> dict[str, Any]:
        """Return a fresh all-motor snapshot using RobStride parameter reads."""
        with self._lock:
            self._require_connected()
            positions: list[float] = []
            velocities: list[float] = []
            for joint in self.config.joints:
                position, velocity = self._read_motor(joint.motor_id)
                positions.append(position)
                velocities.append(velocity)
            gripper_position, gripper_velocity = self._read_motor(
                self.config.gripper.motor_id
            )
            return {
                "connected": True,
                "enabled": self._enabled,
                "stopped": self._stopped,
                "channel": self.config.channel,
                "joint_names": [joint.name for joint in self.config.joints],
                "joint_positions": positions,
                "joint_velocities": velocities,
                "gripper_position": gripper_position,
                "gripper_velocity": gripper_velocity,
                "timestamp": self._clock(),
            }

    def enable(self) -> dict[str, Any]:
        """Clear faults, select MIT mode, and hold the observed pose."""
        with self._lock:
            self._require_connected()
            controller = self._controller
            assert controller is not None
            snapshot = self.state()
            hold_targets = [
                *snapshot["joint_positions"],
                snapshot["gripper_position"],
            ]
            motor_configs = [*self.config.joints, self.config.gripper]
            for motor_config in motor_configs:
                motor = self._motors[motor_config.motor_id]
                motor.clear_error()
                motor.ensure_mode(MIT_MODE, timeout_ms=1000)
            try:
                controller.enable_all()
                self._enabled = True
                for motor_config, target in zip(motor_configs, hold_targets):
                    self._send_mit(motor_config, target)
            except Exception:
                self._stopped = True
                try:
                    controller.disable_all()
                finally:
                    self._enabled = False
                raise
            self._stopped = False
            self._last_targets = list(snapshot["joint_positions"])
            return {
                "enabled": True,
                "hold_positions": list(snapshot["joint_positions"]),
                "gripper_hold_position": snapshot["gripper_position"],
            }

    def move_joints(
        self,
        positions: Sequence[float],
        *,
        duration_s: float = 2.0,
    ) -> dict[str, Any]:
        """Execute a minimum-jerk joint trajectory and verify final feedback."""
        with self._lock:
            self._require_motion_ready()
            target = [float(value) for value in positions]
            if len(target) != len(self.config.joints):
                raise ValueError("positions must contain exactly six joint values")
            if not all(math.isfinite(value) for value in target):
                raise ValueError("joint targets must be finite")
            self._validate_joint_limits(target)
            if not math.isfinite(duration_s) or duration_s <= 0:
                raise ValueError("duration_s must be positive and finite")

            start_state = self.state()
            start = list(start_state["joint_positions"])
            minimum_duration = max(
                abs(goal - initial) / joint.max_velocity
                for initial, goal, joint in zip(start, target, self.config.joints)
            )
            actual_duration = max(float(duration_s), minimum_duration)
            steps = max(2, math.ceil(actual_duration * self.config.control_rate_hz))
            interval = actual_duration / steps
            start_time = self._clock()
            try:
                for index in range(1, steps + 1):
                    scale = _min_jerk(index / steps)
                    waypoint = [
                        initial + (goal - initial) * scale
                        for initial, goal in zip(start, target)
                    ]
                    for joint, position in zip(self.config.joints, waypoint):
                        self._send_mit(joint, position)
                    self._last_targets = waypoint
                    remaining = start_time + index * interval - self._clock()
                    if remaining > 0:
                        self._sleep(remaining)

                final = self._wait_for_target(target)
            except Exception:
                self._stopped = True
                raise
            errors = [abs(observed - goal) for observed, goal in zip(final, target)]
            max_error = max(errors)
            return {
                "target_positions": target,
                "final_positions": final,
                "max_error": max_error,
                "reached": max_error <= self.config.settle_tolerance,
                "requested_duration_s": float(duration_s),
                "actual_duration_s": actual_duration,
            }

    def set_gripper(
        self,
        position: float,
        *,
        duration_s: float = 1.0,
    ) -> dict[str, Any]:
        """Move the calibrated gripper; 0.0=open and 1.0=closed."""
        with self._lock:
            self._require_motion_ready()
            normalized = float(position)
            if not math.isfinite(normalized) or not 0.0 <= normalized <= 1.0:
                raise ValueError("gripper position must be in [0.0, 1.0]")
            if not math.isfinite(duration_s) or duration_s <= 0:
                raise ValueError("duration_s must be positive and finite")
            gripper = self.config.gripper
            if gripper.open_position is None or gripper.closed_position is None:
                raise RuntimeError(
                    "gripper is not calibrated; configure open_position and closed_position"
                )
            target = gripper.open_position + normalized * (
                gripper.closed_position - gripper.open_position
            )
            start, _ = self._read_motor(gripper.motor_id)
            minimum_duration = abs(target - start) / gripper.max_velocity
            actual_duration = max(float(duration_s), minimum_duration)
            steps = max(2, math.ceil(actual_duration * self.config.control_rate_hz))
            interval = actual_duration / steps
            start_time = self._clock()
            try:
                for index in range(1, steps + 1):
                    scale = _min_jerk(index / steps)
                    waypoint = start + (target - start) * scale
                    self._send_mit(gripper, waypoint)
                    remaining = start_time + index * interval - self._clock()
                    if remaining > 0:
                        self._sleep(remaining)
                final, _ = self._read_motor(gripper.motor_id)
            except Exception:
                self._stopped = True
                raise
            return {
                "normalized_position": normalized,
                "target_position": target,
                "final_position": final,
                "actual_duration_s": actual_duration,
            }

    def stop_motion(self) -> dict[str, Any]:
        """Latch a torque-holding software stop and reject new trajectories."""
        with self._lock:
            self._require_connected()
            self._stopped = True
            if self._enabled:
                snapshot = self.state()
                for joint, position in zip(
                    self.config.joints, snapshot["joint_positions"]
                ):
                    self._send_mit(joint, position)
                self._last_targets = list(snapshot["joint_positions"])
            return {"enabled": self._enabled, "stopped": True}

    def reset_stop(self) -> dict[str, Any]:
        """Clear only the software-stop latch; it never enables motors."""
        with self._lock:
            self._require_connected()
            self._stopped = False
            return {"enabled": self._enabled, "stopped": False}

    def emergency_stop(self) -> dict[str, Any]:
        """Disable every motor immediately; the unsupported arm may fall."""
        with self._lock:
            self._require_connected()
            self._controller.disable_all()
            self._enabled = False
            self._stopped = True
            return {"enabled": False, "stopped": True}

    def close(self) -> None:
        """Apply the configured shutdown policy and release the CAN controller."""
        with self._lock:
            if self._controller is None:
                return
            try:
                if self.config.shutdown_policy == "disable" and self._enabled:
                    self._controller.disable_all()
                    self._enabled = False
                elif self._enabled and self._last_targets is not None:
                    for joint, target in zip(self.config.joints, self._last_targets):
                        self._send_mit(joint, target)
            finally:
                self._controller.close()
                self._controller = None
                self._motors = {}
                self._enabled = False
                self._stopped = True

    def _read_motor(self, motor_id: int) -> tuple[float, float]:
        motor = self._motors[motor_id]
        position = float(
            motor.robstride_get_param_f32(
                MECH_POS, timeout_ms=self.config.read_timeout_ms
            )
        )
        velocity = float(
            motor.robstride_get_param_f32(
                MECH_VEL, timeout_ms=self.config.read_timeout_ms
            )
        )
        if not math.isfinite(position) or not math.isfinite(velocity):
            raise RuntimeError(f"motor {motor_id} returned non-finite feedback")
        return position, velocity

    def _send_mit(self, config: JointConfig | Any, position: float) -> None:
        self._motors[config.motor_id].send_mit(
            float(position), 0.0, float(config.kp), float(config.kd), 0.0
        )

    def _wait_for_target(self, target: list[float]) -> list[float]:
        deadline = self._clock() + self.config.settle_timeout_s
        final: list[float] = []
        while True:
            final = list(self.state()["joint_positions"])
            if (
                max(abs(value - goal) for value, goal in zip(final, target))
                <= self.config.settle_tolerance
            ):
                return final
            if self._clock() >= deadline:
                return final
            self._sleep(0.05)

    def _validate_joint_limits(self, target: list[float]) -> None:
        for joint, value in zip(self.config.joints, target):
            if not joint.lower <= value <= joint.upper:
                raise ValueError(
                    f"{joint.name} target {value:.6f} outside "
                    f"[{joint.lower:.6f}, {joint.upper:.6f}]"
                )

    def _require_connected(self) -> None:
        if self._controller is None:
            raise RuntimeError("arm is not connected")

    def _require_motion_ready(self) -> None:
        self._require_connected()
        if not self._enabled:
            raise RuntimeError("arm is not enabled")
        if self._stopped:
            raise RuntimeError("arm is stopped; call reset_stop before moving")
