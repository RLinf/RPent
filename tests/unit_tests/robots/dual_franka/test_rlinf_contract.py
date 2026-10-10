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

"""Contract tests guarding the dual-Franka config keys against RLinf."""

from __future__ import annotations

import dataclasses

import pytest
import yaml

# Keys RPent builds into ``env.eval.override_cfg`` and the ``DualFranka``
# hardware config. Kept here (not a runtime constant) so this test doubles as
# the authoritative drift guard.
_OVERRIDE_KEYS = {
    "max_num_steps",
    "task_description",
    "joint_reset_qpos",
    "target_ee_pose",
    "ee_pose_limit_min",
    "ee_pose_limit_max",
}

_HARDWARE_KEYS = {
    "left_robot_ip",
    "right_robot_ip",
    "base_camera_serials",
    "base_camera_type",
    "left_camera_serials",
    "left_camera_type",
    "right_camera_serials",
    "right_camera_type",
    "left_gripper_type",
    "right_gripper_type",
    "left_gripper_connection",
    "right_gripper_connection",
    "realtime_config",
    "left_controller_node_rank",
    "right_controller_node_rank",
    "node_rank",
    "compliance",
}


def test_override_keys_are_valid_rlinf_fields(fake_rlinf_realworld_modules):
    from rlinf.envs.real.franka.dual_franka_tcp import (
        DualFrankaTCPEnvConfig,
    )

    valid = {field.name for field in dataclasses.fields(DualFrankaTCPEnvConfig)}
    unknown = sorted(_OVERRIDE_KEYS - valid)
    assert not unknown, f"override keys not in DualFrankaTCPEnvConfig: {unknown}"


def test_hardware_keys_are_valid_rlinf_fields(fake_rlinf_realworld_modules):
    from rlinf.robotics.robots.dual_franka import DualFrankaConfig

    valid = {field.name for field in dataclasses.fields(DualFrankaConfig)}
    unknown = sorted(_HARDWARE_KEYS - valid)
    assert not unknown, f"hardware keys not in DualFrankaConfig: {unknown}"


def test_runtime_config_sets_realtime_mode(fake_rlinf_realworld_modules):
    from robots.dual_franka.runtime_config import load_runtime_config

    runtime = load_runtime_config(None, task_description="test task")
    hardware = runtime.rlinf.cluster.node_groups[0].hardware.configs[0]
    assert hardware["realtime_config"] == "ignore"


def test_controller_carries_calibration_mapping_for_ray_worker(
    fake_rlinf_realworld_modules,
):
    from robots.dual_franka.runtime_config import load_runtime_config
    from robots.franka.runtime_config import (
        get_perception_calibration_mapping,
        set_robot_config_path,
    )

    runtime = load_runtime_config(None, task_description="test task")

    # The Ray worker resolves hand-eye calibration through the robot config
    # path carried in the controller: the config's perception.calibration
    # mapping lists the easy_handeye YAMLs, so no separate calibration file
    # is threaded through the controller.
    set_robot_config_path(runtime.controller["robot_config_path"])
    try:
        mapping = get_perception_calibration_mapping()
    finally:
        set_robot_config_path(None)

    assert mapping, (
        "robot config must list easy_handeye YAMLs under "
        "perception.calibration so the Ray worker can resolve hand-eye "
        "calibration"
    )


@pytest.mark.parametrize("custom_config", [False, True])
def test_runtime_config_preserves_explicit_overrides(
    fake_rlinf_realworld_modules, tmp_path, custom_config
):
    from robots.dual_franka.runtime_config import (
        DEFAULT_CONFIG,
        load_mapping,
        load_runtime_config,
    )

    config = load_mapping(DEFAULT_CONFIG)
    path = None
    if custom_config:
        config["robot"]["compliance"] = {
            "nullspace_stiffness": 1.5,
            "translational_clip": 0.02,
            "rotational_clip": 0.12,
            "max_step_rad": 0.08,
        }
        config["joint_health"]["thresholds"] = {
            "left": {"warning_min_joint_margin": 0.2, "critical_min_joint_margin": 0.1},
            "right": {
                "warning_min_joint_margin": 0.3,
                "critical_min_joint_margin": 0.15,
            },
        }
        path = tmp_path / "robot.yaml"
        path.write_text(yaml.safe_dump(config))

    runtime = load_runtime_config(path, task_description="test task")
    hardware = runtime.rlinf.cluster.node_groups[0].hardware.configs[0]
    assert hardware["realtime_config"] == "ignore"
    assert set(hardware) == (
        _HARDWARE_KEYS if custom_config else _HARDWARE_KEYS - {"compliance"}
    )
    assert hardware["left_controller_node_rank"] == 0
    assert hardware["right_controller_node_rank"] == 1
    if custom_config:
        assert hardware["compliance"] == config["robot"]["compliance"]
    assert (
        runtime.controller["joint_health_thresholds"]
        == config["joint_health"]["thresholds"]
    )
