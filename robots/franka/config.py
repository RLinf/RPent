"""User-facing config loading and developer defaults for one physical Franka.

Users edit only ``robot_config.yaml``: machine identity (robot IP, camera
serials, gripper) and workspace geometry (target/reset poses, limits). The
primitive-control knobs and the RLinf field values RPent deliberately tunes
live here as developer defaults and are applied as ``override_cfg`` over
RLinf's own dataclass defaults, so they are never restated in the YAML.

The generic helpers are shared with :mod:`robots.dual_franka`.
"""

from __future__ import annotations

import dataclasses
import os
import re
from pathlib import Path
from typing import Any, Optional

from omegaconf import OmegaConf

DEFAULT_CONFIG = Path(__file__).with_name("robot_config.yaml")


# ---------------------------------------------------------------------------
# Developer defaults
# ---------------------------------------------------------------------------

# Primitive-control knobs consumed by the RPent Franka env server. RLinf has no
# equivalent fields, so these live here rather than in any RLinf dataclass.
CONTROL = {
    "move": {"timeout_s": 15.0, "tolerance_m": 0.005},
    "rotate": {"timeout_s": 15.0, "tolerance_rad": 0.04},
    "servo": {"iteration_multiplier": 4, "min_iterations": 8},
    "gripper": {"settle_s": 0.25, "timeout_s": 8.0, "max_iterations": 4},
}

_COMPLIANCE_PARAM = {
    "translational_stiffness": 2000,
    "translational_damping": 89,
    "rotational_stiffness": 150,
    "rotational_damping": 7,
    "translational_Ki": 0,
    "translational_clip_x": 0.01,
    "translational_clip_y": 0.01,
    "translational_clip_z": 0.01,
    "translational_clip_neg_x": 0.01,
    "translational_clip_neg_y": 0.01,
    "translational_clip_neg_z": 0.01,
    "rotational_clip_x": 0.02,
    "rotational_clip_y": 0.02,
    "rotational_clip_z": 0.02,
    "rotational_clip_neg_x": 0.02,
    "rotational_clip_neg_y": 0.02,
    "rotational_clip_neg_z": 0.02,
    "rotational_Ki": 0,
}

_PRECISION_PARAM = {
    "translational_stiffness": 3000,
    "translational_damping": 89,
    "rotational_stiffness": 300,
    "rotational_damping": 9,
    "translational_Ki": 0.1,
    "translational_clip_x": 0.01,
    "translational_clip_y": 0.01,
    "translational_clip_z": 0.01,
    "translational_clip_neg_x": 0.01,
    "translational_clip_neg_y": 0.01,
    "translational_clip_neg_z": 0.01,
    "rotational_clip_x": 0.05,
    "rotational_clip_y": 0.05,
    "rotational_clip_z": 0.05,
    "rotational_clip_neg_x": 0.05,
    "rotational_clip_neg_y": 0.05,
    "rotational_clip_neg_z": 0.05,
    "rotational_Ki": 0.1,
}

# ``env.eval.override_cfg`` values RPent sets away from RLinf's
# ``FrankaRobotConfig`` defaults. Keys are RLinf field names; anything omitted
# here keeps RLinf's default.
ENV_DEFAULTS = {
    "enable_camera_player": False,  # RLinf default True
    "enable_camera_depth": True,  # RLinf default False
    "camera_resize": False,  # RLinf default True (keep native resolution)
    "max_num_steps": 200_000_000,  # effectively no step cap
    "reward_threshold": [0.01, 0.01, 0.01, 0.0, 0.0, 0.0],
    "action_scale": [0.02, 0.1, 1.0],
    "enable_gripper_penalty": False,  # RLinf default True
    "compliance_param": _COMPLIANCE_PARAM,
    "precision_param": _PRECISION_PARAM,
}


# ---------------------------------------------------------------------------
# Generic helpers (shared with dual_franka)
# ---------------------------------------------------------------------------


def strict_mapping(
    config_cls: type,
    mapping: dict[str, Any],
    *,
    where: str,
) -> dict[str, Any]:
    """Return ``mapping`` as a plain dict, rejecting keys not on ``config_cls``.

    Args:
        config_cls: The RLinf dataclass whose fields define the valid keys.
        mapping: The adapter keys built from the RPent YAML and defaults.
        where: Human-readable location for error messages.

    Raises:
        ValueError: listing the offending keys and the full valid key set.
    """
    valid = frozenset(f.name for f in dataclasses.fields(config_cls))
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


def load_mapping(path: str | Path) -> dict[str, Any]:
    """Load a robot YAML config as a plain dict (shared with ``dual_franka``)."""
    config_path = Path(path).expanduser().resolve()
    raw = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    if not isinstance(raw, dict):
        raise ValueError(f"robot config must be a mapping: {config_path}")
    return raw


def flatten_control(control: dict[str, Any]) -> dict[str, Any]:
    """Flatten nested ``CONTROL`` defaults into the flat env-server keys.

    ``move.max_step_m`` / ``rotate.max_step_rad`` are optional (dual-arm only).
    """
    flat = {
        "move_timeout_s": float(control["move"]["timeout_s"]),
        "move_tolerance_m": float(control["move"]["tolerance_m"]),
        "rotate_timeout_s": float(control["rotate"]["timeout_s"]),
        "rotate_tolerance_rad": float(control["rotate"]["tolerance_rad"]),
        "iteration_multiplier": int(control["servo"]["iteration_multiplier"]),
        "min_iterations": int(control["servo"]["min_iterations"]),
        "gripper_settle_s": float(control["gripper"]["settle_s"]),
        "gripper_timeout_s": float(control["gripper"]["timeout_s"]),
        "gripper_max_iterations": int(control["gripper"]["max_iterations"]),
    }
    if "max_step_m" in control["move"]:
        flat["move_max_step_m"] = float(control["move"]["max_step_m"])
    if "max_step_rad" in control["rotate"]:
        flat["rotate_max_step_rad"] = float(control["rotate"]["max_step_rad"])
    return flat


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
    mapping: dict[str, Any],
    *,
    robot_ip: Optional[str] = None,
    camera_serial_wrist: Optional[str] = None,
    camera_serial_external: Optional[str] = None,
    gripper_connection: Optional[str] = None,
) -> dict[str, Any]:
    """Fill machine-identity values from flags / environment variables.

    ``robot.ip``, each camera ``serial`` and the gripper ``connection`` are
    machine-specific. Precedence per field: command-line flag > environment
    variable > the YAML value (a literal, or an env-var token). The wrist
    camera is the device with ``main: true``; the remaining device is the
    external camera. Mutates and returns ``mapping``.
    """
    robot = _require_mapping(mapping.get("robot"), "robot")
    robot["ip"] = _resolve(
        robot.get("ip"),
        override=robot_ip,
        path="robot.ip",
        flag="--robot-ip",
        required=True,
    )
    end_effector = _require_mapping(
        robot.get("end_effector"), "robot.end_effector"
    )
    end_effector["connection"] = _resolve(
        end_effector.get("connection"),
        override=gripper_connection,
        path="robot.end_effector.connection",
        flag="--gripper-connection",
        required=False,
    )
    cameras = _require_mapping(mapping.get("cameras"), "cameras")
    devices = _require_mapping(cameras.get("devices"), "cameras.devices")
    for name, device in devices.items():
        device = _require_mapping(device, f"cameras.devices.{name}")
        if device.get("main"):
            override, flag = camera_serial_wrist, "--camera-serial-wrist"
        else:
            override, flag = camera_serial_external, "--camera-serial-external"
        device["serial"] = _resolve(
            device.get("serial"),
            override=override,
            path=f"cameras.devices.{name}.serial",
            flag=flag,
            required=True,
        )
    return mapping
