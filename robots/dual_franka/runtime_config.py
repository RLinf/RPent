"""RPent configuration and RLinf adapter for a dual-Franka runtime.

Users edit only ``example.yaml`` (machine identity + workspace geometry).
Developer defaults (node placement, primitive control, perception tuning,
episode length) live here and are applied over RLinf's own dataclass defaults.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from robots.franka.runtime_config import (
    FrankaRuntimeConfig,
    _require_mapping,
    flatten_control,
    load_mapping,
    strict_mapping,
)

# Fixed two-node placement (the RLinf cluster/placement is a training concern;
# RPent only evaluates). Users do not change these.
NODES = [0, 1]
HARDWARE_NODE = 0
LEFT_CONTROLLER_NODE = 0
RIGHT_CONTROLLER_NODE = 1

# Primitive-control knobs consumed by the RPent dual-Franka env server. RLinf
# has no equivalent fields; ``max_step_*`` bound each interpolation step.
CONTROL = {
    "move": {"timeout_s": 20.0, "tolerance_m": 0.005, "max_step_m": 0.02},
    "rotate": {"timeout_s": 20.0, "tolerance_rad": 0.04, "max_step_rad": 0.1},
    "servo": {"iteration_multiplier": 4, "min_iterations": 8},
    "gripper": {"settle_s": 0.4, "timeout_s": 10.0, "max_iterations": 4},
}

# Episode length, used for both ``override_cfg.max_num_steps`` (RLinf default
# 100) and ``env.eval.max_episode_steps``.
EPISODE_STEPS = 300

DEFAULT_CONFIG = Path(__file__).with_name("config") / "example.yaml"


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
            "enable_depth": True,
        }
    return {"cameras": output}


def load_runtime_config(
    path: str | Path | None,
    *,
    task_description: str,
) -> FrankaRuntimeConfig:
    """Load the user YAML, apply developer defaults, and build the adapter."""
    # Lazy RLinf imports: keys are validated against these dataclasses (drift
    # guard), deferred so importing this module stays RLinf-free.
    from rlinf.envs.realworld.franka.tasks.dual_franka_tcp_env import (
        DualFrankaTCPRobotConfig,
    )
    from rlinf.scheduler.hardware.robots.dual_franka import DualFrankaConfig

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
        DualFrankaConfig,
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
        DualFrankaTCPRobotConfig,
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
