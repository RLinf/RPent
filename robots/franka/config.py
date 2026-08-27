"""Unified single-Franka robot configuration.

The user-facing ``robot_config.yaml`` is the only file users edit. Its shape
is defined by the dataclasses below and validated on load, so a typo or missing
key fails with an exact path.

The internal RLinf adapter (``FrankaConfig`` for the hardware slot and
``PhysicalAgentFrankaConfig`` for ``env.eval.override_cfg``) is derived by
introspecting RLinf's own dataclass fields, so the mapping can never silently
drift from the installed RLinf branch: an unknown key raises and lists every
valid key.

The generic helpers at the top are reused by :mod:`robots.dual_franka`.
"""

from __future__ import annotations

import dataclasses
import os
import re
from dataclasses import MISSING, dataclass, field
from pathlib import Path
from typing import (
    Any,
    Optional,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)

from omegaconf import OmegaConf

DEFAULT_CONFIG = Path(__file__).with_name("robot_config.yaml")


# ---------------------------------------------------------------------------
# Generic dataclass-introspection helpers (shared with dual_franka)
# ---------------------------------------------------------------------------


def field_names(config_cls: type) -> frozenset:
    """Return every constructor kwarg a dataclass accepts (incl. inherited)."""
    return frozenset(f.name for f in dataclasses.fields(config_cls))


def strict_mapping(
    config_cls: type,
    mapping: dict[str, Any],
    *,
    where: str,
) -> dict[str, Any]:
    """Return ``mapping`` as a plain dict, rejecting keys not on ``config_cls``.

    Args:
        config_cls: The RLinf dataclass whose fields define the valid keys.
        mapping: The adapter keys built from the RPent YAML.
        where: Human-readable location for error messages.

    Raises:
        ValueError: listing the offending keys and the full valid key set.
    """
    valid = field_names(config_cls)
    unknown = sorted(set(mapping) - valid)
    if unknown:
        raise ValueError(
            f"{where}: unknown key(s) {unknown}. Valid keys: {sorted(valid)}."
        )
    return dict(mapping)


def _require_mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    return value


# ---------------------------------------------------------------------------
# RPent user-facing schema
# ---------------------------------------------------------------------------


@dataclass
class EndEffectorConfig:
    type: str
    connection: Optional[str] = None


@dataclass
class RobotConfig:
    ip: str
    node: int = 0
    end_effector: EndEffectorConfig = field(
        default_factory=lambda: EndEffectorConfig(type="franka")
    )


@dataclass
class CamerasConfig:
    devices: dict[str, Any]
    depth: bool = True
    output_size: Optional[int] = None


@dataclass
class BoundsConfig:
    x: float
    y: float
    z_below: float
    z_above: float
    roll_pitch: float
    yaw: float


@dataclass
class WorkspaceConfig:
    target_pose: list[float]
    bounds: BoundsConfig


@dataclass
class ControlConfig:
    action_scale: list[float]
    success_position_tolerance: list[float]
    move: dict[str, Any]
    rotate: dict[str, Any]
    servo: dict[str, Any]
    gripper: dict[str, Any]
    success_hold_steps: int = 1
    episode_steps: Optional[int] = None


@dataclass
class FrankaConfigFile:
    robot: RobotConfig
    cameras: CamerasConfig
    workspace: WorkspaceConfig
    control: ControlConfig


def _optional_inner(tp: type) -> Optional[type]:
    """Return the non-``None`` type of an ``Optional[X]`` annotation, else None."""
    if get_origin(tp) is Union and type(None) in get_args(tp):
        return next(a for a in get_args(tp) if a is not type(None))
    return None


def _coerce(tp: type, value: Any, path: str) -> Any:
    """Validate ``value`` against ``tp`` at ``path`` and return it (coerced)."""
    inner = _optional_inner(tp)
    if inner is not None:
        if value is None:
            return None
        return _coerce(inner, value, path)
    if dataclasses.is_dataclass(tp):
        return _validate(tp, value, path)
    origin = get_origin(tp)
    if origin is dict:
        if not isinstance(value, dict):
            raise ValueError(f"{path} must be a mapping")
        return dict(value)
    if origin is list:
        if not isinstance(value, (list, tuple)):
            raise ValueError(f"{path} must be a list")
        return list(value)
    return value


def _validate(cls: type, data: Any, path: str):
    """Validate a plain mapping ``data`` against dataclass ``cls`` recursively."""
    if not isinstance(data, dict):
        raise ValueError(f"{path} must be a mapping")
    valid = field_names(cls)
    unknown = sorted(set(data) - valid)
    if unknown:
        raise ValueError(
            f"{path}: unknown key(s) {unknown}. Valid keys: {sorted(valid)}."
        )
    hints = get_type_hints(cls)
    kwargs: dict[str, Any] = {}
    for f in dataclasses.fields(cls):
        if f.name in data:
            kwargs[f.name] = _coerce(hints[f.name], data[f.name], f"{path}.{f.name}")
        elif f.default is MISSING and f.default_factory is MISSING:
            raise ValueError(f"{path}: missing required key '{f.name}'")
    return cls(**kwargs)


def load_mapping(path: str | Path) -> dict[str, Any]:
    """Load a robot YAML config as a plain dict (shared with ``dual_franka``)."""
    config_path = Path(path).expanduser().resolve()
    raw = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    if not isinstance(raw, dict):
        raise ValueError(f"robot config must be a mapping: {config_path}")
    return raw


def load_schema(path: Optional[str] = None) -> FrankaConfigFile:
    """Load and validate ``robot_config.yaml`` against the RPent schema."""
    return _validate(
        FrankaConfigFile, load_mapping(path or DEFAULT_CONFIG), "robot_config"
    )


# ---------------------------------------------------------------------------
# Machine-identity resolution (environment variables / command-line flags)
# ---------------------------------------------------------------------------

_ENV_TOKEN = re.compile(r"^[A-Z][A-Z0-9_]*$")


def _is_env_token(value: Any) -> bool:
    """True when ``value`` looks like an environment-variable name."""
    return isinstance(value, str) and _ENV_TOKEN.match(value) is not None


def _resolve(
    value: Any,
    *,
    override: Optional[str],
    path: str,
    flag: str,
    required: bool,
) -> Any:
    """Resolve one identity value: flag > environment variable > YAML value.

    A YAML value that looks like an environment-variable name (e.g.
    ``ROBOT_IP``) is read from that variable; anything else is used literally.
    """
    if override is not None:
        return override
    resolved = os.environ.get(value) if _is_env_token(value) else value
    if required and not resolved:
        if _is_env_token(value):
            raise ValueError(
                f"{path}: {value!r} is an environment-variable reference that "
                f"is not set. Export {value} or pass {flag}."
            )
        raise ValueError(
            f"{path}: identity value is not set. Set it in robot_config.yaml, "
            f"export the corresponding environment variable, or pass {flag}."
        )
    return resolved


def resolve_identity(
    config: FrankaConfigFile,
    *,
    robot_ip: Optional[str] = None,
    camera_serial_wrist: Optional[str] = None,
    camera_serial_external: Optional[str] = None,
    gripper_connection: Optional[str] = None,
) -> FrankaConfigFile:
    """Fill machine-identity fields from flags / environment variables.

    ``robot.ip``, each camera ``serial`` and the gripper ``connection`` are
    machine-specific and belong outside the committed ``robot_config.yaml``.
    Precedence per field: command-line flag > environment variable > the YAML
    value (a literal, or an env-var token). The wrist camera is the device with
    ``main: true``; the remaining device is treated as the external camera.
    """
    config.robot.ip = _resolve(
        config.robot.ip,
        override=robot_ip,
        path="robot.ip",
        flag="--robot-ip",
        required=True,
    )
    config.robot.end_effector.connection = _resolve(
        config.robot.end_effector.connection,
        override=gripper_connection,
        path="robot.end_effector.connection",
        flag="--gripper-connection",
        required=False,
    )
    for name, device in config.cameras.devices.items():
        if device.get("main"):
            override = camera_serial_wrist
            flag = "--camera-serial-wrist"
        else:
            override = camera_serial_external
            flag = "--camera-serial-external"
        device["serial"] = _resolve(
            device["serial"],
            override=override,
            path=f"cameras.devices.{name}.serial",
            flag=flag,
            required=True,
        )
    return config
