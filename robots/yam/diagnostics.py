# Copyright 2026 The RPent Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at https://www.apache.org/licenses/LICENSE-2.0

"""Read cached control diagnostics from already connected RLinf/i2rt followers.

Private compatibility access is confined here. Nothing connects hardware, sends
commands, changes gains, clears faults, or computes new control torques.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import numpy as np


def _vector(value: Any, size: int) -> list[float]:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (size,) or not np.isfinite(array).all():
        raise ValueError(f"expected {size} finite values")
    return array.tolist()


def _section(reader: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    try:
        return {"status": "available", **reader()}
    except Exception as error:
        return {"status": "unsupported", "reason": f"{type(error).__name__}: {error}"}


def _active_command(robot: Any) -> dict[str, Any]:
    lock = getattr(robot, "_command_lock", None)
    if lock is None or not lock.acquire(timeout=0.05):
        raise RuntimeError("SDK command snapshot lock unavailable")
    try:
        command = robot._commands
        if getattr(command, "indices", None) is not None:
            raise ValueError("partial indexed SDK commands are unsupported")
        # Copies prevent a mapper or subsequent SDK update from mutating the
        # captured command. Never expose live SDK arrays to serialization.
        raw_position = np.asarray(_vector(command.pos, 7))
        raw_velocity = np.asarray(_vector(command.vel, 7))
        kp = _vector(command.kp, 7)
        kd = _vector(command.kd, 7)
        torques = _vector(command.torques, 7)
    finally:
        lock.release()
    mapper = robot.remapper
    return {
        "target_qpos": _vector(mapper.to_command_joint_pos_space(raw_position), 7),
        "target_velocity": _vector(mapper.to_command_joint_vel_space(raw_velocity), 7),
        "kp_motor_space": kp,
        "kd_motor_space": kd,
        "explicit_torque_nm": torques,
        "source": "i2rt._commands copied under _command_lock",
        "position_units": "J1-J6 rad; gripper normalized 0=closed, 1=open",
        "velocity_units": "J1-J6 rad/s; gripper normalized units/s",
        "gain_units": "SDK motor-space Nm/rad and Nm/(rad/s), including gripper motor",
        "limitation": "queued SDK command before per-cycle gripper force limiting; not bus acknowledgement",
    }


def _observations(robot: Any) -> dict[str, Any]:
    # The public method takes the SDK's non-reentrant state lock itself. Do
    # not wrap it in _state_lock, which would deadlock MotorChainRobot.
    observation = robot.get_observations()
    return {
        "joint_position_rad": _vector(observation["joint_pos"], 6),
        "joint_velocity_rad_s": _vector(observation["joint_vel"], 6),
        "joint_effort_nm": _vector(observation["joint_eff"], 6),
        "gripper_position_normalized": _vector(observation["gripper_pos"], 1)[0],
        "gripper_velocity_normalized_s": _vector(observation["gripper_vel"], 1)[0],
        "gripper_motor_effort_nm": _vector(observation["gripper_eff"], 1)[0],
        "source": "i2rt.get_observations cached joint feedback; SDK state lock",
        "effort_convention": "SDK motor feedback effort; not Cartesian wrench or measured contact force",
    }


def _feedback(robot: Any) -> dict[str, Any]:
    lock = getattr(robot, "_state_lock", None)
    if lock is None or not lock.acquire(timeout=0.05):
        raise RuntimeError("SDK feedback snapshot lock unavailable")
    try:
        stamp = float(robot._joint_state.timestamp)
    finally:
        lock.release()
    if not np.isfinite(stamp):
        raise ValueError("non-finite SDK feedback timestamp")
    return {
        "timestamp_s": stamp,
        "host_age_s": time.time() - stamp,
        "source": "i2rt._joint_state.timestamp copied under _state_lock",
        "limitation": "sampled after public observations; not an atomic observation/command snapshot",
    }


def _configured_control(robot: Any) -> dict[str, Any]:
    info = robot.get_robot_info()
    return {
        "kp_motor_space": _vector(info["kp"], 7),
        "kd_motor_space": _vector(info["kd"], 7),
        "gravity_comp_factor": _vector(info["gravity_comp_factor"], 7)
        if np.asarray(info["gravity_comp_factor"]).shape == (7,)
        else _vector(info["gravity_comp_factor"], 6),
        "use_coulomb_friction": bool(info["use_coulomb_friction"]),
        "source": "i2rt.get_robot_info; configured gains may differ from active command",
    }


def _cached_motor_telemetry(robot: Any) -> dict[str, Any]:
    # DMChainCanInterface.read_states copies its existing state under
    # state_lock; it does not send CAN frames. Its timestamp is generated at
    # this read, not when each motor feedback frame arrived.
    states = robot.motor_chain.read_states()
    if len(states) != 7:
        raise ValueError("expected seven cached motor states")
    motors = []
    for state in states:
        # Actual SDK stores hex strings despite MotorInfo's int annotation.
        code = state.error_code
        if isinstance(code, str):
            code = int(code, 16)
        values = _vector([state.id, code, state.temp_mos, state.temp_rotor], 4)
        motor_id, error_code, temp_mos, temp_rotor = values
        if motor_id != int(motor_id) or not 1 <= motor_id <= 127:
            raise ValueError("invalid cached motor ID")
        if error_code != int(error_code) or not 0 <= error_code <= 15:
            raise ValueError("invalid cached motor error code")
        motors.append(
            {
                "motor_id": int(motor_id),
                "error_code": int(error_code),
                "temp_mos_c": temp_mos,
                "temp_rotor_c": temp_rotor,
            }
        )
    if len({motor["motor_id"] for motor in motors}) != 7:
        raise ValueError("duplicate cached motor IDs")
    stamps = _vector([state.timestamp for state in states], 7)
    return {
        "motors": motors,
        "cache_read_timestamp_s": stamps[0],
        "source": "i2rt.motor_chain.read_states; cached copy under state_lock",
        "error_code_convention": "raw MIT feedback status nibble as integer; SDK treats 1 as enabled, not 0",
        "limitation": "timestamp marks cache read, not CAN reception; freshness and present fault-free state are not established",
    }


def read_control_diagnostics(
    runtime: Any, *, include_cached_motor_telemetry: bool = False
) -> dict[str, Any]:
    """Return explicit available/unsupported sections without opening devices.

    Call from the environment's existing serialized observation path. Each SDK
    command snapshot is locked, but arms and feedback/commands are sampled
    sequentially. These diagnostics neither validate health nor enable motion.
    """
    started = time.time()
    followers = getattr(runtime, "_followers", None)
    if not isinstance(followers, (list, tuple)) or len(followers) != 2:
        return {
            "status": "unsupported",
            "reason": "runtime has no connected follower pair",
        }
    arms = {}
    for side, backend in zip(("left", "right"), followers, strict=True):
        robot = getattr(backend, "_robot", None)
        if robot is None:
            arms[side] = {
                "status": "unsupported",
                "reason": "backend exposes no connected i2rt robot",
            }
            continue
        sections = {
            "observations": _section(lambda: _observations(robot)),
            "feedback": _section(lambda: _feedback(robot)),
            "active_command": _section(lambda: _active_command(robot)),
            "configured_control": _section(lambda: _configured_control(robot)),
            "feedforward": _section(
                lambda: {
                    "motor_torque_nm": _vector(robot.get_motor_torques(), 7),
                    "source": "i2rt.get_motor_torques",
                    "limitation": "last computed feedforward includes gravity and optional friction/explicit torque; excludes motor-internal PD and is not measured effort",
                }
            ),
        }
        if include_cached_motor_telemetry:
            sections["cached_motor_telemetry"] = _section(
                lambda: _cached_motor_telemetry(robot)
            )
        arms[side] = {
            "status": "available"
            if all(s["status"] == "available" for s in sections.values())
            else "partial",
            **sections,
        }
    return {
        "status": "available"
        if all(a["status"] == "available" for a in arms.values())
        else "partial",
        "host_read_start_s": started,
        "host_read_end_s": time.time(),
        "arms": arms,
        "limitation": "cached sequential samples, not simultaneous telemetry; no fault-code capability inferred",
    }
