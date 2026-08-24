"""RPent-owned configuration for one Franka runtime."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

DEFAULT_CONFIG = Path(__file__).with_name("robot_config.yaml")


@dataclass(frozen=True)
class FrankaRuntimeConfig:
    """Generated RLinf adapter config and RPent primitive settings."""

    rlinf: DictConfig
    controller: dict[str, Any]


def _require_mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    return value


def _load_mapping(path: str | Path | None) -> dict[str, Any]:
    config_path = Path(path or DEFAULT_CONFIG).expanduser().resolve()
    raw = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    if not isinstance(raw, dict):
        raise ValueError(f"robot config must be a mapping: {config_path}")
    return raw


def load_runtime_config(
    path: str | Path | None,
    *,
    task_description: str,
) -> FrankaRuntimeConfig:
    """Load the RPent schema and build the internal RLinf adapter config."""
    raw = _load_mapping(path)
    robot = _require_mapping(raw.get("robot"), "robot")
    end_effector = _require_mapping(robot.get("end_effector"), "robot.end_effector")
    cameras = _require_mapping(raw.get("cameras"), "cameras")
    devices = _require_mapping(cameras.get("devices"), "cameras.devices")
    workspace = _require_mapping(raw.get("workspace"), "workspace")
    bounds = _require_mapping(workspace.get("bounds"), "workspace.bounds")
    control = _require_mapping(raw.get("control"), "control")

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

    node_rank = int(robot.get("node", 0))
    output_size = cameras.get("output_size")
    position_tolerance = list(control["success_position_tolerance"])
    if len(position_tolerance) != 3:
        raise ValueError("control.success_position_tolerance must have 3 values")
    episode_steps = control.get("episode_steps")
    max_num_steps = int(episode_steps) if episode_steps is not None else 200_000_000

    rlinf = OmegaConf.create(
        {
            "cluster": {
                "num_nodes": node_rank + 1,
                "component_placement": {
                    "env": {"node_group": "franka", "placement": 0}
                },
                "node_groups": [
                    {
                        "label": "franka",
                        "node_ranks": node_rank,
                        "hardware": {
                            "type": "Franka",
                            "configs": [
                                {
                                    "robot_ip": str(robot["ip"]),
                                    "camera_serials": camera_serials,
                                    "camera_type": camera_types.pop(),
                                    "gripper_type": str(end_effector["type"]),
                                    "gripper_connection": end_effector.get(
                                        "connection"
                                    ),
                                    "node_rank": node_rank,
                                }
                            ],
                        },
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
                    "max_episode_steps": episode_steps,
                    "use_spacemouse": False,
                    "no_gripper": False,
                    "main_image_key": main_image_keys[0],
                    "keyboard_reward_wrapper": None,
                    "use_relative_frame": True,
                    "video_cfg": {},
                    "init_params": {"id": "PhysicalAgentFrankaEnv-v1"},
                    "override_cfg": {
                        "is_dummy": False,
                        "task_description": task_description,
                        "camera_names": camera_names,
                        "enable_camera_player": False,
                        "enable_camera_depth": bool(cameras.get("depth", False)),
                        "camera_resize": output_size is not None,
                        "camera_observation_size": int(output_size or 128),
                        "target_ee_pose": list(workspace["target_pose"]),
                        "action_scale": list(control["action_scale"]),
                        "clip_x_range": float(bounds["x"]),
                        "clip_y_range": float(bounds["y"]),
                        "clip_z_range_low": float(bounds["z_below"]),
                        "clip_z_range_high": float(bounds["z_above"]),
                        "clip_roll_pitch_range": float(bounds["roll_pitch"]),
                        "clip_rz_range": float(bounds["yaw"]),
                        "max_num_steps": max_num_steps,
                        "success_hold_steps": int(control["success_hold_steps"]),
                        "reward_threshold": position_tolerance + [0.0, 0.0, 0.0],
                        "enable_gripper_penalty": False,
                    },
                }
            },
        }
    )
    controller = {
        name: _require_mapping(control.get(name), f"control.{name}")
        for name in ("move", "rotate", "servo", "gripper")
    }
    return FrankaRuntimeConfig(rlinf=rlinf, controller=controller)
