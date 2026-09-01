"""User-facing config, developer defaults, and RLinf adapter for one Franka.

Users edit only ``example.yaml``: machine identity (robot IP, camera
serials, gripper) and workspace geometry (target/reset poses, limits). The
primitive-control knobs and the RLinf field values RPent deliberately tunes
live here as developer defaults and are applied as ``override_cfg`` over
RLinf's own dataclass defaults, so they are never restated in the YAML.

``load_runtime_config`` turns that into the RLinf adapter config. The generic
helpers are shared with :mod:`robots.dual_franka`.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

DEFAULT_CONFIG = Path(__file__).with_name("config") / "example.yaml"

# easy_handeye's default save directory is ``~/.ros/easy_handeye``; RPent reads
# its ``hand_eye_calibration.json`` bundle from there by default.
DEFAULT_CALIBRATION_PATH = (
    Path("~/.ros/easy_handeye/hand_eye_calibration.json").expanduser()
)

_calibration_path: Path | None = None


def set_calibration_path(path: str | Path | None) -> None:
    """Configure the hand-eye calibration bundle path (once, at runtime init)."""
    global _calibration_path
    _calibration_path = Path(path).expanduser() if path else None


def get_calibration_path() -> Path:
    """Return the configured calibration bundle path, or the easy_handeye default."""
    return Path(_calibration_path or DEFAULT_CALIBRATION_PATH)


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


@dataclass(frozen=True)
class FrankaRuntimeConfig:
    """Generated RLinf adapter config and RPent primitive settings."""

    rlinf: DictConfig
    controller: dict[str, Any]


def load_runtime_config(
    path: str | Path | None,
    *,
    task_description: str,
) -> FrankaRuntimeConfig:
    """Load the user YAML, apply developer defaults, and build the adapter."""
    # Lazy RLinf imports: keys are validated against these dataclasses (drift
    # guard), deferred so importing this module stays RLinf-free.
    from rlinf.envs.realworld.franka.franka_env import FrankaRobotConfig
    from rlinf.scheduler.hardware.robots.franka import FrankaConfig

    raw = load_mapping(path or DEFAULT_CONFIG)
    robot = _require_mapping(raw.get("robot"), "robot")
    end_effector = _require_mapping(
        robot.get("end_effector"), "robot.end_effector"
    )
    cameras = _require_mapping(raw.get("cameras"), "cameras")
    devices = _require_mapping(cameras.get("devices"), "cameras.devices")
    workspace = _require_mapping(raw.get("workspace"), "workspace")

    camera_serials: list[str] = []
    camera_names: dict[str, str] = {}
    camera_types: set[str] = set()
    main_image_keys: list[str] = []
    for name, value in devices.items():
        device = _require_mapping(value, f"cameras.devices.{name}")
        serial = str(device["serial"])
        camera_serials.append(serial)
        camera_names[serial] = str(name)
        camera_types.add(str(device.get("type", "realsense")))
        if bool(device.get("main", False)):
            main_image_keys.append(str(name))
    if len(main_image_keys) != 1:
        raise ValueError("exactly one camera device must set main: true")
    if len(camera_types) != 1:
        raise ValueError("single Franka currently requires one camera type")

    hardware = strict_mapping(
        FrankaConfig,
        {
            "robot_ip": robot["ip"],
            "camera_serials": camera_serials,
            "camera_type": camera_types.pop(),
            "gripper_type": end_effector.get("type", "franka"),
            "gripper_connection": end_effector.get("connection"),
            "node_rank": 0,
        },
        where="cluster.node_groups[].hardware.configs[]",
    )
    override_cfg = strict_mapping(
        FrankaRobotConfig,
        {
            **ENV_DEFAULTS,
            "task_description": task_description,
            "camera_names": camera_names,
            "target_ee_pose": list(workspace["target_ee_pose"]),
            "reset_ee_pose": list(workspace["reset_ee_pose"]),
            "ee_pose_limit_min": list(workspace["ee_pose_limit_min"]),
            "ee_pose_limit_max": list(workspace["ee_pose_limit_max"]),
        },
        where="env.eval.override_cfg",
    )

    rlinf = OmegaConf.create(
        {
            "cluster": {
                "num_nodes": 1,
                "component_placement": {
                    "env": {"node_group": "franka", "placement": 0}
                },
                "node_groups": [
                    {
                        "label": "franka",
                        "node_ranks": 0,
                        "hardware": {"type": "Franka", "configs": [hardware]},
                    }
                ],
            },
            "env": {
                "eval": {
                    "seed": 0,
                    "group_size": 1,
                    "auto_reset": False,
                    "ignore_terminations": False,
                    "use_fixed_reset_state_ids": False,
                    "max_episode_steps": None,
                    "use_spacemouse": False,
                    "no_gripper": False,
                    "main_image_key": main_image_keys[0],
                    "keyboard_reward_wrapper": None,
                    "use_relative_frame": True,
                    "video_cfg": {},
                    "init_params": {"id": "PhysicalAgentFrankaEnv-v1"},
                    "override_cfg": override_cfg,
                }
            },
        }
    )
    return FrankaRuntimeConfig(
        rlinf=rlinf,
        controller=flatten_control(CONTROL),
    )
