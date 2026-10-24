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

"""Physical LIBERO observations usable by both VLA and WAM encoders."""

from typing import Any

import numpy as np

from rpent.robots.components.am_contracts import AMObservation


def physical_observation(raw_obs: dict[str, Any]) -> AMObservation:
    """Extract top-left-origin RGB and named state without a model representation.

    End-effector position and xyzw orientation are in the world frame; position
    and finger joint positions are in metres. No normalization is applied.
    """
    cameras = {"agentview_image": "primary", "robot0_eye_in_hand_image": "wrist"}
    fields = {
        "robot0_eef_pos": "eef_position",
        "robot0_eef_quat": "eef_quaternion_xyzw",
        "robot0_gripper_qpos": "gripper_qpos",
        "robot0_gripper_qvel": "gripper_qvel",
        "robot0_joint_pos": "joint_positions",
        "robot0_joint_vel": "joint_velocities",
        "robot0_eef_vel_lin": "eef_linear_velocity",
        "robot0_eef_vel_ang": "eef_angular_velocity",
    }
    return {
        "images": {
            cameras.get(name, name.removesuffix("_image")): np.ascontiguousarray(
                value[::-1]
            )
            for name, value in raw_obs.items()
            if name.endswith("_image")
        },
        "state": {
            target: raw_obs[source]
            for source, target in fields.items()
            if source in raw_obs
        },
        "instruction": raw_obs["task_descriptions"],
    }
