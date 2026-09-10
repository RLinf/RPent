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

import yaml
from omegaconf import DictConfig, OmegaConf

DEFAULT_CONFIG = Path(__file__).with_name("config") / "example.yaml"

_robot_config_path: Path | None = None


def set_robot_config_path(path: str | Path | None) -> None:
    """Record the ``--robot-config`` CLI value (``None`` when absent).

    Called once at CLI parse time (robot spec ``parse_config``), before any
    runtime or toolkit construction.
    """
    global _robot_config_path
    _robot_config_path = Path(path).expanduser() if path else None


def get_robot_config_path() -> Path:
    """Return the ``--robot-config`` override, or the robot's packaged default."""
    return Path(_robot_config_path or DEFAULT_CONFIG)


def _calibration_mapping_from(perception: Any) -> dict[str, str]:
    """Return the validated ``perception.calibration`` source mapping.

    Shared by the single- and dual-Franka loaders. Returns an empty mapping
    when the section is absent.

    Raises:
        ValueError: when a mapping value is not a path-like string.
    """
    if not isinstance(perception, dict):
        return {}
    mapping = perception.get("calibration")
    if not isinstance(mapping, dict):
        return {}
    invalid = sorted(
        key for key, value in mapping.items() if not isinstance(value, (str, Path))
    )
    if invalid:
        raise ValueError(
            "perception.calibration values must be easy_handeye YAML paths; "
            f"got non-path value(s) for {invalid}"
        )
    return {str(key): str(value) for key, value in mapping.items()}


def get_perception_calibration_mapping() -> dict[str, str]:
    """Return the robot-config ``perception.calibration`` YAML-source mapping.

    Maps RPent camera keys (``base_camera``/``d455_camera`` for dual Franka,
    ``external``/``wrist`` for single Franka) to easy_handeye YAML paths.
    Returns an empty mapping when the robot config has no such section.

    Raises:
        ValueError: when a mapping value is not a path-like string.
    """
    raw = load_mapping(get_robot_config_path())
    return _calibration_mapping_from(raw.get("perception"))


def load_easy_handeye_yaml(path: str | Path) -> dict[str, Any]:
    """Load one easy_handeye calibration YAML as a bundle entry.

    easy_handeye saves each calibration as ``~/.ros/easy_handeye/<name>.yaml``
    with a ``parameters`` section (frame names, ``eye_on_hand``, ...) and a
    ``transformation`` section (``x/y/z`` plus ``qx/qy/qz/qw``). The returned
    entry carries the YAML's file name as ``source_name`` plus its verbatim
    ``parameters`` and ``transformation`` sections.

    Raises:
        ValueError: when the file is missing or not an easy_handeye YAML.
    """
    yaml_path = Path(path).expanduser()
    if not yaml_path.exists():
        raise ValueError(f"easy_handeye calibration YAML not found: {yaml_path}")
    try:
        data = yaml.safe_load(yaml_path.read_text(errors="replace"))
    except yaml.YAMLError as exc:
        raise ValueError(
            f"invalid easy_handeye calibration YAML {yaml_path}: {exc}"
        ) from exc
    if (
        not isinstance(data, dict)
        or not isinstance(data.get("parameters"), dict)
        or not isinstance(data.get("transformation"), dict)
    ):
        raise ValueError(
            f"{yaml_path} must be an easy_handeye calibration YAML (a mapping "
            "with 'parameters' and 'transformation' sections)"
        )
    transformation = data["transformation"]
    missing = {"x", "y", "z", "qx", "qy", "qz", "qw"} - set(transformation)
    if missing:
        raise ValueError(f"{yaml_path} missing transform fields: {sorted(missing)}")
    return {
        "source_name": yaml_path.name,
        "parameters": data["parameters"],
        "transformation": dict(transformation),
    }


def describe_calibration_source() -> str:
    """Describe the robot-config easy_handeye YAML calibration mapping."""
    mapping = get_perception_calibration_mapping()
    if not mapping:
        return (
            "no perception.calibration easy_handeye YAML mapping configured in "
            f"{get_robot_config_path()}"
        )
    listed = ", ".join(
        f"{key}={Path(value).expanduser()}" for key, value in sorted(mapping.items())
    )
    return f"easy_handeye YAMLs (robot-config perception.calibration): {listed}"


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
    # RPent drives the cameras through its own env server; it does not run the
    # in-process camera player.
    "enable_camera_player": False,
    # The back-projection primitives need per-pixel depth.
    "enable_camera_depth": True,
    # Native resolution keeps pixel back-projection aligned with calibration.
    "camera_resize": False,
    # Episodes are bounded by the planner, not a step budget.
    "max_num_steps": 200_000_000,
    # Success is decided by the planner; these thresholds are nominal.
    "reward_threshold": [0.01, 0.01, 0.01, 0.0, 0.0, 0.0],
    # Bounded per-step motion increments (xyz m, rpy rad, gripper) for safety.
    "action_scale": [0.02, 0.1, 1.0],
    # Grasp timing is planner-controlled; no fixed per-step gripper penalty.
    "enable_gripper_penalty": False,
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

    raw = load_mapping(path or get_robot_config_path())
    robot = _require_mapping(raw.get("robot"), "robot")
    end_effector = _require_mapping(robot.get("end_effector"), "robot.end_effector")
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
                    "init_params": {"id": "RPentFrankaEnv-v1"},
                    "override_cfg": override_cfg,
                }
            },
        }
    )
    return FrankaRuntimeConfig(
        rlinf=rlinf,
        controller=flatten_control(CONTROL),
    )
