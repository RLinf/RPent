"""RPent -> RLinf adapter for one Franka runtime."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from robots.franka.config import (
    CONTROL,
    DEFAULT_CONFIG,
    ENV_DEFAULTS,
    _require_mapping,
    flatten_control,
    load_mapping,
    resolve_identity,
    strict_mapping,
)


@dataclass(frozen=True)
class FrankaRuntimeConfig:
    """Generated RLinf adapter config and RPent primitive settings."""

    rlinf: DictConfig
    controller: dict[str, Any]


def load_runtime_config(
    path: str | Path | None,
    *,
    task_description: str,
    robot_ip: str | None = None,
    camera_serial_wrist: str | None = None,
    camera_serial_external: str | None = None,
    gripper_connection: str | None = None,
) -> FrankaRuntimeConfig:
    """Load the user YAML, apply developer defaults, and build the adapter."""
    # Lazy RLinf imports: keys are validated against these dataclasses (drift
    # guard), deferred so importing this module stays RLinf-free.
    from rlinf.envs.realworld.franka.franka_env import FrankaRobotConfig
    from rlinf.scheduler.hardware.robots.franka import FrankaConfig

    raw = resolve_identity(
        load_mapping(path or DEFAULT_CONFIG),
        robot_ip=robot_ip,
        camera_serial_wrist=camera_serial_wrist,
        camera_serial_external=camera_serial_external,
        gripper_connection=gripper_connection,
    )
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
