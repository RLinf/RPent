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
    get_calibration_path,
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
#
# PhysicalAgent alignment note: the live Frankas often need several 10 Hz
# closed-loop corrections before the reported TCP reaches the requested target.
# The high iteration guard and longer gripper settle time keep RPent from
# declaring failure before the real robot has physically settled.  These are
# deployment-tuned controller-side tolerances, not RPent-wide defaults.
CONTROL = {
    "move": {"timeout_s": 20.0, "tolerance_m": 0.006, "max_step_m": 0.02},
    "rotate": {"timeout_s": 20.0, "tolerance_rad": 0.04, "max_step_rad": 0.1},
    # RLinf paces Cartesian steps at 10 Hz.  Keep the iteration guard high
    # enough that the timeouts, rather than this guard, normally terminate a
    # slowly converging closed-loop move.  Larger moves also receive a budget
    # proportional to the number of bounded interpolation segments.
    "servo": {"iteration_multiplier": 50, "min_iterations": 200},
    "gripper": {"settle_s": 1.5, "timeout_s": 10.0, "max_iterations": 4},
}

RECOVERY = {
    "return_timeout_s": 20.0,
    "return_tolerance_m": 0.006,
    "return_tolerance_rad": 0.04,
}

# Raw env safety horizon. RPent owns task-level episode boundaries; RLinf
# auto-reset must stay disabled during long, multi-skill physical tasks because
# reset opens both grippers and moves both arms home.
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
    perception = _require_mapping(cameras.get("perception", {}), "cameras.perception")
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


def _agent_observation(cameras: dict[str, Any]) -> dict[str, list[str]]:
    """Load planner-facing camera display policy from robot config."""
    # PhysicalAgent alignment note: the old clean-desk logs treated the fixed
    # D455 view as the main semantic/localization view, while wrist/base images
    # were mainly auxiliary checks.  Keep that as the default for this deployed
    # config, but allow the YAML to register different inline/auxiliary views.
    default = {
        "inline_cameras": ["d455"],
        "auxiliary_cameras": ["left_wrist", "base", "right_wrist"],
    }
    raw = cameras.get("agent_observation", default)
    policy = _require_mapping(raw, "cameras.agent_observation")
    out: dict[str, list[str]] = {}
    for key in ("inline_cameras", "auxiliary_cameras"):
        value = policy.get(key, default[key])
        if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
            raise ValueError(f"cameras.agent_observation.{key} must be a list of strings")
        out[key] = [str(item) for item in value]
    return out


def _projection_views(raw: dict[str, Any]) -> dict[str, Any]:
    # PhysicalAgent alignment note: D455 is not part of RLinf's original
    # three-camera dual-Franka observation contract, so RPent keeps an explicit
    # projection registry for extra RGBD views that can localize pixels in the
    # shared right_base frame.
    perception = _require_mapping(raw.get("perception"), "perception")
    views = _require_mapping(perception.get("projection_views"), "perception.projection_views")
    return {str(alias): dict(_require_mapping(value, f"perception.projection_views.{alias}")) for alias, value in views.items()}


def _joint_health_thresholds(raw: dict[str, Any]) -> dict[str, dict[str, float]]:
    """Load per-arm joint-health thresholds from the user robot config."""
    # PhysicalAgent alignment note: long multi-stage VLA runs can drift into
    # awkward Franka joint postures even when TCP-space task progress still
    # looks fine.  These thresholds reproduce the operator guard rails used
    # during live debugging; they should be tuned per mounting/task, or replaced
    # by native Franka green/yellow/red health signals when available.
    joint_health = _require_mapping(raw.get("joint_health"), "joint_health")
    thresholds = _require_mapping(
        joint_health.get("thresholds"), "joint_health.thresholds"
    )
    out: dict[str, dict[str, float]] = {}
    for arm in ("left", "right"):
        values = _require_mapping(
            thresholds.get(arm), f"joint_health.thresholds.{arm}"
        )
        out[arm] = {str(key): float(value) for key, value in values.items()}
    return out


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
    right_gripper = _require_mapping(right.get("gripper"), "robot.arms.right.gripper")
    cameras = _require_mapping(raw.get("cameras"), "cameras")
    observation = _require_mapping(cameras.get("observation"), "cameras.observation")
    workspace = _require_mapping(raw.get("workspace"), "workspace")
    joint_health_thresholds = _joint_health_thresholds(raw)

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
            "joint_reset_qpos": list(workspace["joint_reset_qpos"]),
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
                    "auto_reset": False,
                    "ignore_terminations": False,
                    "use_fixed_reset_state_ids": False,
                    "max_episode_steps": None,
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
    controller["calibration_path"] = str(get_calibration_path())
    controller["perception"] = _perception_cameras(cameras)
    controller["agent_observation"] = _agent_observation(cameras)
    controller["projection_views"] = _projection_views(raw)
    controller["recovery_return_timeout_s"] = RECOVERY["return_timeout_s"]
    controller["recovery_return_tolerance_m"] = RECOVERY["return_tolerance_m"]
    controller["recovery_return_tolerance_rad"] = RECOVERY["return_tolerance_rad"]
    controller["joint_health_thresholds"] = joint_health_thresholds
    return FrankaRuntimeConfig(rlinf=rlinf, controller=controller)
