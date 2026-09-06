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

"""Read kinematics in the existing RLinf simulation actor's owning thread.

Ray's built-in actor call executes this importable function serially alongside
reset/step. No simulator object crosses the process boundary. Actions continue
through OfficialBehaviorBackend.chunk_step, including official-success checks.
"""

from __future__ import annotations

import numpy as np


def get_motion_state(process, env_index: int = 0) -> dict:
    import omnigibson as og
    from omnigibson.utils.transform_utils import quat2mat

    from robots.behavior.rlinf_env import _torch_to_numpy

    env = process.env.envs[env_index]
    robot = env.robots[0]
    controls = robot.get_control_dict()
    joint_positions = robot.get_joint_positions()
    _, base_quat = robot.get_position_orientation()

    def array(value):
        return np.asarray(_torch_to_numpy(value)).copy()

    rotation = array(quat2mat(base_quat))
    result = {"control_dt": float(og.sim.get_sim_step_dt()), "hands": {}}
    for hand in ("left", "right"):
        controller = robot.controllers[f"arm_{hand}"]
        if controller.motor_type != "position" or controller.use_delta_commands:
            raise RuntimeError("motion requires absolute position arm controllers")
        idx = robot.arm_control_idx[hand]
        pos, quat = robot.get_eef_pose(hand)
        jacobian = array(controls[f"eef_{hand}_jacobian_relative"][:, idx])
        world_jacobian = np.concatenate(
            (rotation @ jacobian[:3], rotation @ jacobian[3:]), axis=0
        )
        palm_pos, _ = robot.links[f"{hand}_gripper_link"].get_position_orientation()
        direction = array(pos - palm_pos)
        length = float(np.linalg.norm(direction))
        if length < 1e-6:
            raise RuntimeError("EEF and gripper origins do not define an approach axis")
        contacts = sorted(
            {
                str(body)
                for link in robot.finger_links[hand]
                for contact in link.contact_list()
                for body in (contact.body0, contact.body1)
                if not str(body).startswith(robot.prim_path + "/")
            }
        )
        result["hands"][hand] = {
            "position": array(pos),
            "quaternion_xyzw": array(quat),
            "joint_positions": array(joint_positions[idx]),
            "joint_lower_limits": array(robot.joint_lower_limits[idx]),
            "joint_upper_limits": array(robot.joint_upper_limits[idx]),
            "jacobian": world_jacobian,
            "approach_direction": direction / length,
            "contacts": contacts,
        }
    return result
