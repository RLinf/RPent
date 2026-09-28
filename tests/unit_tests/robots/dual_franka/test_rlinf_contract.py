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
    "left_controller_node_rank",
    "right_controller_node_rank",
    "node_rank",
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
