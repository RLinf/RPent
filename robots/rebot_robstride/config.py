"""Validated configuration for the reBot DevArm RobStride backend."""

from __future__ import annotations

import math
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class JointConfig:
    """One RobStride arm joint in raw motor coordinates."""

    name: str
    motor_id: int
    model: str
    lower: float
    upper: float
    kp: float
    kd: float
    max_velocity: float = 0.5


@dataclass(frozen=True)
class GripperConfig:
    """RobStride gripper motor and optional calibrated travel endpoints."""

    motor_id: int = 7
    model: str = "rs-00"
    kp: float = 20.0
    kd: float = 1.0
    max_velocity: float = 1.0
    open_position: float | None = None
    closed_position: float | None = None


@dataclass(frozen=True)
class RebotConfig:
    """Complete hardware and safety configuration."""

    channel: str
    bitrate: int
    control_rate_hz: float
    feedback_rate_hz: float
    read_timeout_ms: int
    settle_tolerance: float
    settle_timeout_s: float
    settle_samples: int
    settle_velocity_rad_s: float
    startup_sample_interval_s: float
    startup_velocity_limit_rad_s: float
    max_tracking_error_rad: float
    velocity_abort_multiplier: float
    max_motion_duration_s: float
    heartbeat_timeout_s: float
    joints: tuple[JointConfig, ...]
    gripper: GripperConfig


def default_config() -> RebotConfig:
    """Return conservative defaults for the seven-motor B601-RS build."""
    joints = (
        JointConfig("joint1", 1, "rs-06", -2.8, 2.8, 50.0, 3.0),
        JointConfig("joint2", 2, "rs-06", -3.14, 0.0, 150.0, 10.0),
        JointConfig("joint3", 3, "rs-06", -3.14, 0.0, 150.0, 10.0),
        JointConfig("joint4", 4, "rs-00", -1.57, 1.57, 50.0, 5.0),
        JointConfig("joint5", 5, "rs-00", -1.57, 1.57, 50.0, 4.0),
        JointConfig("joint6", 6, "rs-00", -3.14, 3.14, 50.0, 4.0),
    )
    return _validate(
        RebotConfig(
            channel="can0",
            bitrate=1_000_000,
            control_rate_hz=50.0,
            feedback_rate_hz=10.0,
            read_timeout_ms=100,
            settle_tolerance=0.03,
            settle_timeout_s=2.0,
            settle_samples=3,
            settle_velocity_rad_s=0.05,
            startup_sample_interval_s=0.1,
            startup_velocity_limit_rad_s=0.1,
            max_tracking_error_rad=0.35,
            velocity_abort_multiplier=2.5,
            max_motion_duration_s=60.0,
            heartbeat_timeout_s=2.0,
            joints=joints,
            gripper=GripperConfig(),
        )
    )


def load_config(path: str | Path | None = None) -> RebotConfig:
    """Load a YAML override on top of :func:`default_config`."""
    config = default_config()
    if path is None:
        return config

    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - installation error path
        raise RuntimeError(
            "PyYAML is required to load a reBot config; install rpent[rebot-robstride]"
        ) from exc

    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError("reBot config must contain a YAML mapping")

    allowed_root = {field.name for field in fields(RebotConfig)}
    unknown_root = sorted(set(raw) - allowed_root)
    if unknown_root:
        raise ValueError(f"unknown reBot config fields: {', '.join(unknown_root)}")

    scalar_fields = {
        "channel",
        "bitrate",
        "control_rate_hz",
        "feedback_rate_hz",
        "read_timeout_ms",
        "settle_tolerance",
        "settle_timeout_s",
        "settle_samples",
        "settle_velocity_rad_s",
        "startup_sample_interval_s",
        "startup_velocity_limit_rad_s",
        "max_tracking_error_rad",
        "velocity_abort_multiplier",
        "max_motion_duration_s",
        "heartbeat_timeout_s",
    }
    updates = {key: raw[key] for key in scalar_fields if key in raw}

    joints = config.joints
    if "joints" in raw:
        joint_rows = raw["joints"]
        if not isinstance(joint_rows, list) or len(joint_rows) != 6:
            raise ValueError("joints must be a list containing exactly six entries")
        joints = tuple(_joint_from_mapping(row) for row in joint_rows)

    gripper = config.gripper
    if "gripper" in raw:
        row = raw["gripper"] or {}
        if not isinstance(row, dict):
            raise ValueError("gripper must be a mapping")
        allowed = {field.name for field in GripperConfig.__dataclass_fields__.values()}
        unknown = sorted(set(row) - allowed)
        if unknown:
            raise ValueError(f"unknown gripper fields: {', '.join(unknown)}")
        gripper = replace(gripper, **row)

    return _validate(replace(config, joints=joints, gripper=gripper, **updates))


def _joint_from_mapping(row: Any) -> JointConfig:
    if not isinstance(row, dict):
        raise ValueError("each joint entry must be a mapping")
    required = {"name", "motor_id", "model", "lower", "upper", "kp", "kd"}
    missing = sorted(required - set(row))
    if missing:
        raise ValueError(f"joint entry missing fields: {', '.join(missing)}")
    allowed = required | {"max_velocity"}
    unknown = sorted(set(row) - allowed)
    if unknown:
        raise ValueError(f"unknown joint fields: {', '.join(unknown)}")
    return JointConfig(**row)


def _validate(config: RebotConfig) -> RebotConfig:
    if not isinstance(config.channel, str) or not config.channel:
        raise ValueError("channel must be a non-empty string")
    if (
        isinstance(config.bitrate, bool)
        or not isinstance(config.bitrate, int)
        or config.bitrate <= 0
    ):
        raise ValueError("bitrate must be a positive integer")

    if (
        isinstance(config.read_timeout_ms, bool)
        or not isinstance(config.read_timeout_ms, int)
        or not 1 <= config.read_timeout_ms <= 1000
    ):
        raise ValueError("read_timeout_ms must be an integer in [1, 1000]")

    finite_positive = {
        "control_rate_hz": config.control_rate_hz,
        "feedback_rate_hz": config.feedback_rate_hz,
        "settle_tolerance": config.settle_tolerance,
        "settle_timeout_s": config.settle_timeout_s,
        "settle_velocity_rad_s": config.settle_velocity_rad_s,
        "startup_sample_interval_s": config.startup_sample_interval_s,
        "startup_velocity_limit_rad_s": config.startup_velocity_limit_rad_s,
        "max_tracking_error_rad": config.max_tracking_error_rad,
        "velocity_abort_multiplier": config.velocity_abort_multiplier,
        "max_motion_duration_s": config.max_motion_duration_s,
        "heartbeat_timeout_s": config.heartbeat_timeout_s,
    }
    for name, value in finite_positive.items():
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0
        ):
            raise ValueError(f"{name} must be finite and positive")
    if not 10 <= config.control_rate_hz <= 200:
        raise ValueError("control_rate_hz must be in [10, 200]")
    if not 5 <= config.feedback_rate_hz <= config.control_rate_hz:
        raise ValueError("feedback_rate_hz must be in [5, control_rate_hz]")
    if (
        isinstance(config.settle_samples, bool)
        or not isinstance(config.settle_samples, int)
        or not 1 <= config.settle_samples <= 10
    ):
        raise ValueError("settle_samples must be an integer in [1, 10]")
    bounded_values = {
        "settle_tolerance": (config.settle_tolerance, 0.2),
        "settle_timeout_s": (config.settle_timeout_s, 5.0),
        "settle_velocity_rad_s": (config.settle_velocity_rad_s, 0.5),
        "startup_sample_interval_s": (config.startup_sample_interval_s, 1.0),
        "startup_velocity_limit_rad_s": (
            config.startup_velocity_limit_rad_s,
            0.5,
        ),
        "max_tracking_error_rad": (config.max_tracking_error_rad, 1.0),
        "velocity_abort_multiplier": (config.velocity_abort_multiplier, 5.0),
        "max_motion_duration_s": (config.max_motion_duration_s, 60.0),
        "heartbeat_timeout_s": (config.heartbeat_timeout_s, 5.0),
    }
    for name, (value, maximum) in bounded_values.items():
        if value > maximum:
            raise ValueError(f"{name} must not exceed {maximum}")
    if config.heartbeat_timeout_s < 0.25:
        raise ValueError("heartbeat_timeout_s must be at least 0.25")
    if config.max_motion_duration_s + config.settle_timeout_s > 65.0:
        raise ValueError(
            "max_motion_duration_s + settle_timeout_s must not exceed 65 seconds"
        )
    if len(config.joints) != 6:
        raise ValueError("exactly six arm joints are required")

    motor_ids = [joint.motor_id for joint in config.joints] + [config.gripper.motor_id]
    if any(
        isinstance(motor_id, bool) or not isinstance(motor_id, int)
        for motor_id in motor_ids
    ):
        raise ValueError("motor IDs must be integers")
    if len(set(motor_ids)) != len(motor_ids):
        raise ValueError("motor IDs must be unique")
    if any(not 1 <= motor_id <= 0xFF for motor_id in motor_ids):
        raise ValueError("motor IDs must be in 1..255")

    names = [joint.name for joint in config.joints]
    if len(set(names)) != len(names):
        raise ValueError("joint names must be unique")

    for joint in config.joints:
        values = (joint.lower, joint.upper, joint.kp, joint.kd, joint.max_velocity)
        if not all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
            for value in values
        ):
            raise ValueError(f"{joint.name} contains a non-finite value")
        if joint.lower >= joint.upper:
            raise ValueError(f"{joint.name} lower limit must be below upper limit")
        if max(abs(joint.lower), abs(joint.upper)) > 2 * math.pi:
            raise ValueError(f"{joint.name} limits must remain within +/-2*pi")
        if not 0 <= joint.kp <= 200 or not 0 <= joint.kd <= 20:
            raise ValueError(
                f"{joint.name} gains must satisfy kp in [0, 200], kd in [0, 20]"
            )
        if not 0 < joint.max_velocity <= 1.0:
            raise ValueError(f"{joint.name} max_velocity must be in (0, 1.0]")

    gripper = config.gripper
    if (gripper.open_position is None) != (gripper.closed_position is None):
        raise ValueError(
            "gripper open_position and closed_position must both be set or both be null"
        )
    gripper_values = (gripper.kp, gripper.kd, gripper.max_velocity)
    if not all(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        for value in gripper_values
    ):
        raise ValueError("gripper contains a non-finite value")
    if not 0 <= gripper.kp <= 200 or not 0 <= gripper.kd <= 20:
        raise ValueError("gripper gains must satisfy kp in [0, 200], kd in [0, 20]")
    if not 0 < gripper.max_velocity <= 1.0:
        raise ValueError("gripper max_velocity must be in (0, 1.0]")
    if gripper.open_position is not None:
        assert gripper.closed_position is not None
        endpoints = (gripper.open_position, gripper.closed_position)
        if not all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
            for value in endpoints
        ):
            raise ValueError("gripper endpoints must be finite")
        if gripper.open_position == gripper.closed_position:
            raise ValueError("gripper endpoints must differ")
        if max(abs(value) for value in endpoints) > 4 * math.pi:
            raise ValueError("gripper endpoints must remain within +/-4*pi")
    return config
