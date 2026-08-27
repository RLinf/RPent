"""RPent -> RLinf adapter for one Franka runtime."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from robots.franka.config import (
    _require_mapping,
    load_schema,
    resolve_identity,
    strict_mapping,
)


@dataclass(frozen=True)
class FrankaRuntimeConfig:
    """Generated RLinf adapter config and RPent primitive settings."""

    rlinf: DictConfig
    controller: dict[str, Any]


def _hardware_config_cls() -> type:
    """RLinf dataclass whose fields define the valid hardware-config keys."""
    from rlinf.scheduler.hardware.robots.franka import FrankaConfig

    return FrankaConfig


def _env_config_cls() -> type:
    """RLinf dataclass whose fields define the valid ``override_cfg`` keys."""
    from robots.franka.physical_agent_env import PhysicalAgentFrankaConfig

    return PhysicalAgentFrankaConfig


def load_runtime_config(
    path: str | Path | None,
    *,
    task_description: str,
    robot_ip: str | None = None,
    camera_serial_wrist: str | None = None,
    camera_serial_external: str | None = None,
    gripper_connection: str | None = None,
) -> FrankaRuntimeConfig:
    """Load the RPent schema and build the internal RLinf adapter config."""
    raw = resolve_identity(
        load_schema(path),
        robot_ip=robot_ip,
        camera_serial_wrist=camera_serial_wrist,
        camera_serial_external=camera_serial_external,
        gripper_connection=gripper_connection,
    )
    robot = raw.robot
    cameras = raw.cameras
    workspace = raw.workspace
    control = raw.control

    camera_serials: list[str] = []
    camera_names: dict[str, str] = {}
    camera_types: set[str] = set()
    main_image_keys: list[str] = []
    for name, value in cameras.devices.items():
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

    node_rank = robot.node
    output_size = cameras.output_size
    position_tolerance = list(control.success_position_tolerance)
    if len(position_tolerance) != 3:
        raise ValueError("control.success_position_tolerance must have 3 values")
    episode_steps = control.episode_steps
    max_num_steps = int(episode_steps) if episode_steps is not None else 200_000_000

    hardware = strict_mapping(
        _hardware_config_cls(),
        {
            "robot_ip": robot.ip,
            "camera_serials": camera_serials,
            "camera_type": camera_types.pop(),
            "gripper_type": robot.end_effector.type,
            "gripper_connection": robot.end_effector.connection,
            "node_rank": node_rank,
        },
        where="cluster.node_groups[].hardware.configs[]",
    )
    override_cfg = strict_mapping(
        _env_config_cls(),
        {
            "is_dummy": False,
            "task_description": task_description,
            "camera_names": camera_names,
            "enable_camera_player": False,
            "enable_camera_depth": cameras.depth,
            "camera_resize": output_size is not None,
            "camera_observation_size": int(output_size or 128),
            "target_ee_pose": list(workspace.target_pose),
            "action_scale": list(control.action_scale),
            "clip_x_range": float(workspace.bounds.x),
            "clip_y_range": float(workspace.bounds.y),
            "clip_z_range_low": float(workspace.bounds.z_below),
            "clip_z_range_high": float(workspace.bounds.z_above),
            "clip_roll_pitch_range": float(workspace.bounds.roll_pitch),
            "clip_rz_range": float(workspace.bounds.yaw),
            "max_num_steps": max_num_steps,
            "success_hold_steps": int(control.success_hold_steps),
            "reward_threshold": position_tolerance + [0.0, 0.0, 0.0],
            "enable_gripper_penalty": False,
        },
        where="env.eval.override_cfg",
    )

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
                    "max_episode_steps": episode_steps,
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
    controller = {
        name: getattr(control, name)
        for name in ("move", "rotate", "servo", "gripper")
    }
    return FrankaRuntimeConfig(rlinf=rlinf, controller=controller)
