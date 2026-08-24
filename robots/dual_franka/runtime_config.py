"""RPent-owned configuration for a dual-Franka runtime."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from robots.franka.runtime_config import (
    FrankaRuntimeConfig,
    _load_mapping,
    _require_mapping,
)

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


def load_runtime_config(
    path: str | Path | None,
    *,
    task_description: str,
) -> FrankaRuntimeConfig:
    """Load the RPent schema and build the internal RLinf adapter config."""
    raw = _load_mapping(path or DEFAULT_CONFIG)
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
    bounds = _require_mapping(workspace.get("bounds"), "workspace.bounds")
    control = _require_mapping(raw.get("control"), "control")
    if control.get("mode") != "tcp_rot6d":
        raise ValueError("dual Franka currently requires control.mode: tcp_rot6d")

    base_serials, base_type = _camera_slot(observation, "base")
    left_serials, left_type = _camera_slot(observation, "left_wrist")
    right_serials, right_type = _camera_slot(observation, "right_wrist")
    nodes = [int(node) for node in robot["nodes"]]
    if nodes != [0, 1]:
        raise ValueError("dual Franka currently requires robot.nodes: [0, 1]")

    hardware_node = int(robot.get("hardware_node", 0))
    rlinf = OmegaConf.create(
        {
            "cluster": {
                "num_nodes": max(nodes) + 1,
                "component_placement": {
                    "env": {"node_group": "dual_franka", "placement": 0}
                },
                "node_groups": [
                    {
                        "label": "dual_franka",
                        "node_ranks": ",".join(str(node) for node in nodes),
                        "hardware": {
                            "type": "DualFranka",
                            "configs": [
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
                                    "left_gripper_connection": left_gripper.get(
                                        "connection"
                                    ),
                                    "right_gripper_connection": right_gripper.get(
                                        "connection"
                                    ),
                                    "left_controller_node_rank": int(
                                        left["controller_node"]
                                    ),
                                    "right_controller_node_rank": int(
                                        right["controller_node"]
                                    ),
                                    "node_rank": hardware_node,
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
                    "auto_reset": True,
                    "ignore_terminations": False,
                    "use_fixed_reset_state_ids": False,
                    "max_episode_steps": int(control["episode_steps"]),
                    "use_spacemouse": False,
                    "use_gello": False,
                    "use_gello_joint": False,
                    "no_gripper": False,
                    "main_image_key": str(cameras["main"]),
                    "keyboard_reward_wrapper": None,
                    "use_relative_frame": False,
                    "video_cfg": {},
                    "init_params": {"id": "DualFrankaTcpEnv-v1"},
                    "override_cfg": {
                        "is_dummy": False,
                        "task_description": task_description,
                        "enable_camera_player": False,
                        "enable_camera_depth": True,
                        "rotation_repr": "rot6d",
                        "joint_reset_qpos": list(control["joint_reset_qpos"]),
                        "target_ee_pose": list(workspace["target_pose"]),
                        "max_num_steps": int(control["episode_steps"]),
                        "action_scale": list(control["action_scale"]),
                        "ee_pose_limit_min": list(bounds["min"]),
                        "ee_pose_limit_max": list(bounds["max"]),
                        "success_hold_steps": int(control["success_hold_steps"]),
                    },
                }
            },
        }
    )
    perception = _require_mapping(cameras.get("perception", {}), "cameras.perception")
    controller = {
        name: _require_mapping(control.get(name), f"control.{name}")
        for name in ("move", "rotate", "servo", "gripper")
    }
    controller["perception"] = {"cameras": {}}
    for name, value in perception.items():
        camera = _require_mapping(value, f"cameras.perception.{name}")
        controller["perception"]["cameras"][str(name)] = {
            "enabled": bool(camera.get("enabled", True)),
            "serial_number": str(camera["serial"]),
            "camera_type": str(camera.get("type", "realsense")),
            "resolution": list(camera.get("resolution", [640, 480])),
            "fps": int(camera.get("fps", 15)),
            "enable_depth": bool(camera.get("depth", True)),
        }
    return FrankaRuntimeConfig(rlinf=rlinf, controller=controller)
