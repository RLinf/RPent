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

"""Code-policy instructions for dual_franka."""

from rpent.prompt import code_policy as common
from rpent.prompt.utils import PromptNode

FRAMES_AND_UNITS = r"""
Both arms use the calibrated right-base coordinate frame. Positions are metres;
quaternions are [qx, qy, qz, qw]; relative rotations are XYZ Euler radians and
left-multiply the current rotation. Pose refers to the configured robot TCP,
which may differ from the fingertip grasp centre. Account for the tool offset.
"""

OBSERVATION = r"""
Start with `robo.view_env_state(step=-1)`. The observation's `state` contains
`left_arm` and `right_arm`, with measured `tcp_pose` and gripper fields.
If `observe` is registered, call it to capture fresh TCPs and camera frames
without motion. Observation reads, `back_project` and `segment` use recorded
snapshots. Use row/column pixels from the original image and the matching step.
Inspect available images and calibration; do not assume wrist depth exists.
"""

MOTION = r"""
Use `move_delta` for relative XYZ translation and `rotate_delta` for relative
XYZ Euler rotation. Supply keyword arguments from the API reference. To reach
an absolute position, observe fresh feedback and compute target minus current
position. Supply intermediate waypoints to clear the table and the other arm.
The controller may drift during rotation; inspect the final pose and the raw
motion result before proceeding. Each Cartesian command is limited to 0.20 m
translation or 0.35 rad rotation. Respect the runtime's motion-call budget.
"""

VLA_SKILLS = r"""
Optional VLA skills are `vla_right_grasp`, `vla_handoff` and `vla_left_place`.
Use them only when registered with a configured endpoint; otherwise use Cartesian
primitives. Inference uses the host's configured checkpoint instruction. These
skills can affect both arms and do not use Cartesian delta limits. Inspect
`boundary_reached` and visual evidence: a segment boundary is not proof of task
success. Call handoff directly after a confirmed grasp without speculative arm
preparation. For Cartesian grasping, localize, choose an approach and tool
orientation, open, approach, close, lift and verify. Do not use example coordinates.
"""

SAFETY_AND_RECOVERY = r"""
Stop on conflicting observations, failed motion, joint-health critical status,
or unintended contact. Software cancellation takes effect between blocking robot
calls; it is not an emergency stop. The operator controls the physical emergency
stop. Use only the reset/recovery controls actually registered for this run.
Obtain scene confirmation before motion and refresh observations after any reset.
"""

TASK_SUCCESS = r"""
`finish` reports the agent's assessment. Only the operator verdict determines
physical success. Obtain the verdict through the provided operator controls
before completing the task; do not infer it from a successful tool call.
"""


def system_prompt() -> dict[str, PromptNode]:
    """Assemble code execution rules and dual_franka operating knowledge."""
    return {
        **common.system_prompt(),
        "Frames and units": FRAMES_AND_UNITS,
        "Observation": OBSERVATION,
        "Motion": MOTION,
        "VLA skills": VLA_SKILLS,
        "Safety and recovery": SAFETY_AND_RECOVERY,
        "Task success": TASK_SUCCESS,
        "API reference": "{{code_policy_api}}",
    }
