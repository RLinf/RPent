# Copyright 2026 The RPent Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at https://www.apache.org/licenses/LICENSE-2.0

"""Bounded, optional outer joint PI; all commands still pass the env guards."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, fields
from typing import Any

import numpy as np

from robots.yam.geometry import enforce_hard_limits, joint_limits_from_config


@dataclass(frozen=True)
class JointServoConfig:
    enabled: bool = False
    kp: float = 0.0
    ki: float = 0.5
    period_s: float = 0.1
    timeout_s: float = 3.0
    max_bias_rad: float = 0.025
    max_bias_step_rad: float = 0.002
    joint_tolerance_rad: float = 0.008
    position_tolerance_m: float = 0.005
    rotation_tolerance_rad: float = 0.05
    stable_samples: int = 3
    progress_timeout_s: float = 1.0
    progress_epsilon_rad: float = 0.0002

    @classmethod
    def from_config(cls, value: Any) -> JointServoConfig:
        value = {} if value is None else value
        if not isinstance(value, dict) or set(value) - {f.name for f in fields(cls)}:
            raise ValueError("joint_servo must be a mapping with known fields only")
        config = cls(**value)
        if type(config.enabled) is not bool:
            raise ValueError("joint_servo.enabled must be boolean")
        bounds = {
            "kp": (0, 0.5),
            "ki": (0, 2),
            "period_s": (0.05, 0.2),
            "timeout_s": (0.3, 15),
            "max_bias_rad": (0, 0.025),
            "max_bias_step_rad": (0, 0.002),
            "joint_tolerance_rad": (0, 0.020),
            "position_tolerance_m": (0, 0.010),
            "rotation_tolerance_rad": (0, 0.05),
            "progress_timeout_s": (0.3, 5),
            "progress_epsilon_rad": (0, 0.001),
        }
        for name, (lower, upper) in bounds.items():
            number = getattr(config, name)
            if (
                isinstance(number, bool)
                or not isinstance(number, (int, float))
                or not np.isfinite(number)
                or not lower <= number <= upper
                or (lower == 0 and name not in ("kp", "ki") and number == 0)
            ):
                raise ValueError(
                    f"joint_servo.{name} outside supported range [{lower}, {upper}]"
                )
        if (
            type(config.stable_samples) is not int
            or not 3 <= config.stable_samples <= 10
        ):
            raise ValueError("joint_servo.stable_samples must be integer in [3,10]")
        if (
            config.progress_timeout_s > config.timeout_s
            or config.max_bias_step_rad > config.max_bias_rad
        ):
            raise ValueError(
                "joint_servo progress timeout/bias step exceeds its total bound"
            )
        return config

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class BoundedJointPI:
    """Back-calculate integral from applied bias to prevent hidden windup."""

    def __init__(self, config: JointServoConfig) -> None:
        self.config = config
        self.integral = np.zeros(6)
        self.bias = np.zeros(6)

    def update(self, error: np.ndarray, dt: float) -> np.ndarray:
        error = np.asarray(error, dtype=np.float64)
        if (
            error.shape != (6,)
            or not np.isfinite(error).all()
            or not np.isfinite(dt)
            or dt <= 0
        ):
            raise ValueError(
                "joint servo requires finite six-joint error and positive dt"
            )
        cfg = self.config
        # Lost time never grants an accumulated motion allowance.
        desired = (
            cfg.kp * error + self.integral + cfg.ki * error * min(dt, cfg.period_s)
        )
        bias = np.clip(desired, -cfg.max_bias_rad, cfg.max_bias_rad)
        bias = np.clip(
            bias, self.bias - cfg.max_bias_step_rad, self.bias + cfg.max_bias_step_rad
        )
        self.accept(bias, error)
        return bias.copy()

    def accept(self, applied_bias: np.ndarray, error: np.ndarray) -> None:
        bias = np.asarray(applied_bias, dtype=np.float64)
        if (
            bias.shape != (6,)
            or not np.isfinite(bias).all()
            or np.any(abs(bias) > self.config.max_bias_rad + 1e-9)
        ):
            raise ValueError("applied joint servo bias exceeded total bound")
        self.bias = bias.copy()
        self.integral = self.bias - self.config.kp * error


def _pose_errors(value: Any, target: np.ndarray) -> tuple[np.ndarray, float, float]:
    pose = np.asarray(value, dtype=np.float64)
    if (
        pose.shape != (7,)
        or not np.isfinite(pose).all()
        or np.linalg.norm(pose[3:]) < 1e-8
        or np.linalg.norm(target[3:]) < 1e-8
    ):
        raise ValueError("joint servo requires finite xyz+wxyz feedback")
    cosine = abs(
        float(
            np.dot(
                pose[3:] / np.linalg.norm(pose[3:]),
                target[3:] / np.linalg.norm(target[3:]),
            )
        )
    )
    return (
        pose,
        float(np.linalg.norm(pose[:3] - target[:3])),
        float(2 * np.arccos(np.clip(cosine, 0, 1))),
    )


def _driver_trace(info: dict[str, Any], arm: str) -> dict[str, Any]:
    """Copy only active-arm motor diagnostics already included in feedback."""
    diagnostics = info.get("control_diagnostics")
    if not isinstance(diagnostics, dict):
        return {}
    arms = diagnostics.get("arms")
    if not isinstance(arms, dict) or not isinstance(arms.get(arm), dict):
        return {}
    diagnostics = arms[arm]
    result = {}
    for group, names in (
        (
            "observations",
            ("joint_position_rad", "joint_velocity_rad_s", "joint_effort_nm"),
        ),
        ("active_command", ("target_qpos", "kp_motor_space", "kd_motor_space")),
        ("feedforward", ("motor_torque_nm",)),
    ):
        values = diagnostics.get(group, {})
        if not isinstance(values, dict):
            continue
        for name in names:
            value = values.get(name)
            if value is None:
                continue
            try:
                vector = np.asarray(value, dtype=np.float64)
                if (
                    vector.ndim == 1
                    and len(vector) in (6, 7)
                    and np.isfinite(vector).all()
                ):
                    key = f"feedforward_{name}" if group == "feedforward" else name
                    result[key] = vector[:6].tolist()
            except (TypeError, ValueError):
                continue
    return result


def run_joint_servo(
    *,
    env: Any,
    apply_updates: Any,
    check_cancelled: Any,
    arm: str,
    nominal: np.ndarray,
    target_pose: np.ndarray,
    episode_id: str,
    config: JointServoConfig,
) -> dict[str, Any]:
    """Settle around a fixed nominal solution; never replace it with corrected q.

    RPC calls have their own transport timeouts. This loop checks its wall-time
    deadline after each call; it cannot preempt a blocked transport operation.
    """
    controller = BoundedJointPI(config)
    offset = 0 if arm == "left" else 7
    started = time.monotonic()
    last_update = started
    last_progress = started
    best_error = float("inf")
    stable = executed = requested = 0
    compact = env.execution_capabilities.get("compact_control") is True
    previous_observation = started
    info = env.last_info
    trace = []
    reason = "timeout"
    final_pose = None
    position_error = rotation_error = joint_error = None
    stop_error = None
    stop_receipt = None
    try:
        lower, upper = joint_limits_from_config(env.execution_capabilities)
        if arm not in ("left", "right"):
            raise ValueError("joint servo arm must be left or right")
        nominal = np.asarray(nominal, dtype=np.float64).copy()
        target_pose = np.asarray(target_pose, dtype=np.float64)
        if (
            nominal.shape != (6,)
            or not np.isfinite(nominal).all()
            or target_pose.shape != (7,)
            or not np.isfinite(target_pose).all()
        ):
            raise ValueError(
                "joint servo target must be finite six-joint nominal and xyz+wxyz"
            )
        while True:
            reason = "cancelled"
            check_cancelled()
            reason = "feedback_error"
            read_started = time.monotonic()
            if compact:
                env.read_control_state()
                observation, info = env.last_control_obs, env.last_control_info
            else:
                env.observe()
                observation, info = env.last_obs, env.last_info
            now = time.monotonic()
            status = info["episode_status"]
            if status["episode_id"] != episode_id:
                reason = "episode_changed"
                break
            if (
                status.get("stop_requested")
                or status.get("eval_success")
                or status.get("terminal_event") is not None
            ):
                reason = "episode_stopped"
                break
            if int(status["take_action_cnt"]) >= int(status["step_lim"]):
                reason = "budget_exhausted"
                break
            measured = enforce_hard_limits(
                observation["state"]["joint_position"], lower, upper
            )
            commanded = enforce_hard_limits(info["commanded_qpos"], lower, upper)
            expected = commanded.copy()
            expected[offset : offset + 6] = nominal
            enforce_hard_limits(expected, lower, upper, name="servo nominal")
            error = nominal - measured[offset : offset + 6]
            joint_error = float(np.max(abs(error)))
            final_pose, position_error, rotation_error = _pose_errors(
                info["robot_state"][f"{arm}_eef_pose"], target_pose
            )
            applied_bias = commanded[offset : offset + 6] - nominal
            if np.any(abs(applied_bias - controller.bias) > 1e-7):
                # A competing writer or unexpected downstream clipping must not
                # be hidden by resetting the nominal/integrator reference.
                reason = "command_mismatch"
                break
            entry = {
                "elapsed_s": now - started,
                "cycle_s": now - previous_observation,
                "feedback_read_s": now - read_started,
                "driver": _driver_trace(info, arm),
                "joint_error_rad": error.tolist(),
                "position_error_m": position_error,
                "rotation_error_rad": rotation_error,
                "bias_rad": controller.bias.tolist(),
            }
            trace.append(entry)
            previous_observation = now
            reached = (
                joint_error <= config.joint_tolerance_rad
                and position_error <= config.position_tolerance_m
                and rotation_error <= config.rotation_tolerance_rad
            )
            stable = stable + 1 if reached else 0
            if stable >= config.stable_samples:
                return {
                    "enabled": True,
                    "success": True,
                    "stop_reason": "settled",
                    "executed_steps": executed,
                    "requested_steps": requested,
                    "compact_control": compact,
                    "episode_status": dict(info["episode_status"]),
                    "trace": trace,
                    "measured_pose": final_pose.tolist(),
                    "position_error_m": position_error,
                    "rotation_error_rad": rotation_error,
                    "max_joint_error_rad": joint_error,
                    "final_bias_rad": controller.bias.tolist(),
                    "nominal_qpos": nominal.tolist(),
                }
            if now - started >= config.timeout_s:
                reason = "timeout"
                break
            if joint_error < best_error - config.progress_epsilon_rad:
                best_error, last_progress = joint_error, now
            if not reached and now - last_progress >= config.progress_timeout_s:
                reason = "no_progress"
                break
            if not reached:
                previous_bias = controller.bias.copy()
                bias = controller.update(
                    error, max(1e-6, min(now - last_update, config.period_s))
                )
                corrected = nominal + bias
                candidate = commanded.copy()
                candidate[offset : offset + 6] = corrected
                enforce_hard_limits(candidate, lower, upper, name="servo correction")
                reason = "execution_error"
                requested += 1
                options = {"compact_control": True} if compact else {}
                result = apply_updates(
                    [{"arm": arm, "arm_qpos": corrected}],
                    expected_episode_id=episode_id,
                    **options,
                )
                info = env.last_control_info if compact else env.last_info
                count = int(result.get("executed_actions", 0))
                executed += count
                entry["requested_bias_rad"] = bias.tolist()
                if count != 1:
                    reason = "execution_incomplete"
                    break
                applied = (
                    enforce_hard_limits(info["commanded_qpos"], lower, upper)[
                        offset : offset + 6
                    ]
                    - nominal
                )
                if np.any(
                    abs(applied - previous_bias) > config.max_bias_step_rad + 1e-9
                ):
                    raise ValueError(
                        "applied joint servo bias exceeded per-cycle bound"
                    )
                controller.accept(applied, error)
                last_update = now
            time.sleep(max(0.0, config.period_s - (time.monotonic() - now)))
    except Exception as error:
        trace.append({"failure": f"{type(error).__name__}: {error}", "stage": reason})
    # A bounded stationary residual is not convergence or evidence of free
    # space. Preserve the accepted command so the planner can observe before
    # choosing a new approach; all faults still latch the normal stop.
    recoverable = (
        reason in {"no_progress", "timeout"}
        and position_error is not None
        and position_error <= 0.015
        and rotation_error <= config.rotation_tolerance_rad
        and joint_error <= config.max_bias_rad
        and len(trace) >= config.stable_samples
        and all("joint_error_rad" in row for row in trace[-config.stable_samples :])
        and np.max(
            np.ptp(
                [row["joint_error_rad"] for row in trace[-config.stable_samples :]],
                axis=0,
            )
        )
        <= 0.001
    )
    if not recoverable:
        try:
            stop_receipt = env.request_stop()
        except Exception as error:
            stop_error = f"{type(error).__name__}: {error}"
    return {
        "enabled": True,
        "success": False,
        "recoverable": bool(recoverable),
        "stop_reason": reason,
        "executed_steps": executed,
        "requested_steps": requested,
        "compact_control": compact,
        "episode_status": dict(
            (
                env.last_control_info
                if compact and getattr(env, "last_control_info", None) is not None
                else env.last_info
            )["episode_status"]
        ),
        "trace": trace,
        "measured_pose": None if final_pose is None else final_pose.tolist(),
        "position_error_m": position_error,
        "rotation_error_rad": rotation_error,
        "max_joint_error_rad": joint_error,
        "final_bias_rad": controller.bias.tolist(),
        "stop_error": stop_error,
        "stop_receipt": stop_receipt,
    }
