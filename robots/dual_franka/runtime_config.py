"""RPent-owned configuration for a dual-Franka runtime."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from robots.dual_franka.config import (
    CONTROL,
    EPISODE_STEPS,
    HARDWARE_NODE,
    LEFT_CONTROLLER_NODE,
    NODES,
    PERCEPTION_DEFAULTS,
    RIGHT_CONTROLLER_NODE,
)
from robots.franka.config import (
    _require_mapping,
    flatten_control,
    load_mapping,
    strict_mapping,
)
from robots.franka.runtime_config import FrankaRuntimeConfig

DEFAULT_CONFIG = Path(__file__).with_name("robot_config.yaml")


def _camera_slot(observation: dict[str, Any], slot: str) -> tuple[list[str], str]:
    values = observation.get(slot, [])
    if not isinstance(values, list) or not values:
        raise ValueError(f"cameras.observation.{slot} must be a non-empty list")
    serials: list[str] = []
    camera_type: str | None = None
    for index, value in enumerate(values):
        camera = _require_mapping(value, f"cameras.observation.{slot}[{index}]")
        serials.append(str(camera["serial"]))
        current_type = str(camera.get("type", "realsense"))
        if camera_type is None:
            camera_type = current_type
        elif current_type != camera_type:
            raise ValueError(f"all cameras in {slot} must use one camera type")
    return serials, camera_type or "realsense"


def _hardware_config_cls() -> type:
    """RLinf dataclass whose fields define the valid hardware-config keys."""
    from rlinf.scheduler.hardware.robots.dual_franka import DualFrankaConfig

    return DualFrankaConfig


def _env_config_cls() -> type:
    """RLinf dataclass whose fields define the valid ``override_cfg`` keys."""
    from rlinf.envs.realworld.franka.tasks.dual_franka_tcp_env import (
        DualFrankaTCPRobotConfig,
    )

    return DualFrankaTCPRobotConfig


def _perception_cameras(cameras: dict[str, Any]) -> dict[str, Any]:
    """Build the flat perception-camera config from YAML identity + defaults."""
    perception = _require_mapping(
        cameras.get("perception", {}), "cameras.perception"
    )
    output: dict[str, Any] = {}
    for name, value in perception.items():
        camera = _require_mapping(value, f"cameras.perception.{name}")
        output[str(name)] = {
            "enabled": True,
            "serial_number": str(camera["serial"]),
            "camera_type": str(camera.get("type", "realsense")),
            "enable_depth": bool(PERCEPTION_DEFAULTS["enable_depth"]),
        }
    return {"cameras": output}


def load_runtime_config(
    path: str | Path | None,
    *,
    task_description: str,
) -> FrankaRuntimeConfig:
    """Load the user YAML, apply developer defaults, and build the adapter."""
    raw = load_mapping(path or DEFAULT_CONFIG)
    robot = _require_mapping(raw.get("robot"), "robot")
    arms = _require_mapping(robot.get("arms"), "robot.arms")
    left = _require_mapping(arms.get("left"), "robot.arms.left")
    right = _require_mapping(arms.get("right"), "robot.arms.right")
    left_gripper = _require_mapping(left.get("gripper"), "robot.arms.left.gripper")
    right_gripper = _require_mapping(
        right.get("gripper"), "robot.arms.right.gripper"
    )
    cameras = _require_mapping(raw.get("cameras"), "cameras")
    observation = _require_mapping(cameras.get("observation"), "cameras.observation")
    workspace = _require_mapping(raw.get("workspace"), "workspace")

    base_serials, base_type = _camera_slot(observation, "base")
    left_serials, left_type = _camera_slot(observation, "left_wrist")
    right_serials, right_type = _camera_slot(observation, "right_wrist")

    hardware = strict_mapping(
        _hardware_config_cls(),
        {
            "left_robot_ip": str(left["ip"]),
            "right_robot_ip": str(right["ip"]),
            "base_camera_serials": base_serials,
            "base_camera_type": base_type,
            "left_camera_serials": left_serials,
            "left_camera_type": left_type,
            "right_camera_serials": right_serials,
            "right_camera_type": right_type,
            "left_gripper_type": str(left_gripper["type"]),
            "right_gripper_type": str(right_gripper["type"]),
            "left_gripper_connection": left_gripper.get("connection"),
            "right_gripper_connection": right_gripper.get("connection"),
            "left_controller_node_rank": LEFT_CONTROLLER_NODE,
            "right_controller_node_rank": RIGHT_CONTROLLER_NODE,
            "node_rank": HARDWARE_NODE,
        },
        where="cluster.node_groups[].hardware.configs[]",
    )
    override_cfg = strict_mapping(
        _env_config_cls(),
        {
            "max_num_steps": EPISODE_STEPS,
            "task_description": task_description,
            "target_ee_pose": list(workspace["target_ee_pose"]),
            "ee_pose_limit_min": list(workspace["ee_pose_limit_min"]),
            "ee_pose_limit_max": list(workspace["ee_pose_limit_max"]),
        },
        where="env.eval.override_cfg",
    )

    rlinf = OmegaConf.create(
        {
            "cluster": {
                "num_nodes": max(NODES) + 1,
                "component_placement": {
                    "env": {"node_group": "dual_franka", "placement": 0}
                },
                "node_groups": [
                    {
                        "label": "dual_franka",
                        "node_ranks": ",".join(str(node) for node in NODES),
                        "hardware": {"type": "DualFranka", "configs": [hardware]},
                    }
                ],
            },
            "env": {
                "eval": {
                    "seed": 0,
                    "group_size": 1,
                    "auto_reset": True,
                    "ignore_terminations": False,
                    "use_fixed_reset_state_ids": False,
                    "max_episode_steps": EPISODE_STEPS,
                    "use_spacemouse": False,
                    "use_gello": False,
                    "use_gello_joint": False,
                    "no_gripper": False,
                    "main_image_key": str(cameras["main"]),
                    "keyboard_reward_wrapper": None,
                    "use_relative_frame": False,
                    "video_cfg": {},
                    "init_params": {"id": "DualFrankaTCPEnv-v1"},
                    "override_cfg": override_cfg,
                }
            },
        }
    )
    controller = flatten_control(CONTROL)
    controller["perception"] = _perception_cameras(cameras)
    return FrankaRuntimeConfig(rlinf=rlinf, controller=controller)
