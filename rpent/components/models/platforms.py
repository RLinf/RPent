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

"""Model-side copies of the wire contracts a platform integration defines.

These values describe the native action layout and camera roles a model
adapter expects. The operational definitions belong to the platform side
(``components/sim/libero/control.py``, ``components/sim/robotwin/control.py``,
and ``robots/yam/contracts.py``); a model service keeps its own copy rather
than depend on a simulator install. The two sides must stay equal, which the
platform contract test enforces.
"""

from __future__ import annotations

from rpent.components.models.contracts import AMControlSpec

# Values use LIBERO's native OSC scaling and gripper convention. Model
# adapters restore this representation before handing actions to the env.
LIBERO_OSC = AMControlSpec(
    embodiment="libero_7d",
    action_space="libero.osc_pose.delta.v1",
    action_schema=[
        "delta_x",
        "delta_y",
        "delta_z",
        "delta_axis_angle_x",
        "delta_axis_angle_y",
        "delta_axis_angle_z",
        "gripper",
    ],
    action_type=None,
)

# Native joint_action.vector order: left six joint targets and gripper,
# then right six joint targets and gripper. Preserve native gripper values.
ROBOTWIN_QPOS = AMControlSpec(
    embodiment="robotwin_qpos14",
    action_space="robotwin.joint_action.qpos14.v1",
    action_schema=[f"joint_action_{i}" for i in range(14)],
    action_type="qpos",
)

# Absolute native end-effector poses with wxyz quaternions, left arm first.
ROBOTWIN_EEF = AMControlSpec(
    embodiment="robotwin_eef16",
    action_space="robotwin.eef_pose.absolute.v1",
    action_schema=[
        f"{arm}_{field}"
        for arm in ("left", "right")
        for field in ("x", "y", "z", "qw", "qx", "qy", "qz", "gripper")
    ],
    action_type="ee",
)

ROBOTWIN_CONTROLLERS = {
    spec["action_space"]: spec for spec in (ROBOTWIN_QPOS, ROBOTWIN_EEF)
}

ROBOTWIN_CAMERAS = ("head", "left_wrist", "right_wrist")

# Identity a YAM Pi0.5 VLA server must report back to the control station.
YAM_VLA_CONTRACT = {
    "runtime": "yam_vla",
    "policy_name": "pi05_yam_joint",
    "camera_order": ["top", "left", "right"],
    "state_layout": "qpos14",
    "action_layout": "qpos14",
    "action_dim": 14,
    "action_horizon": 30,
    "use_length": 5,
}
