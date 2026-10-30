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

"""Code-policy instructions for robodojo."""

from rpent.prompt import code_policy as common
from rpent.prompt.utils import PromptNode

OBSERVE_AND_LOCALIZE = r"""
Two ARX-X5 arms, left and right, stand side by side at the near edge of a table.
World positions use metres, with z up. Read the current frame, end-effector poses
and camera calibration from `robo.view_env_state(step=-1)`; its `task_language`
is authoritative. Read object heights from the scene, never assume them.
Use `segment` and `back_project` on the recorded images, with keyword arguments
from the API reference. Head and wrist views must describe the same candidate.
Check returned fields before using a localization: a point on a rim, an arm or
background is not the object's centre. Re-localize after contact or scene changes.
"""

MOTION_AND_GRIPPER = r"""
Use the registered `move_to(xyz=..., arm=..., gripper=...)` tool. The current
RPent tool commands an end-effector position; do not apply a separate SDK grasp
point offset or pass unsupported orientation, path or speed arguments.
Approach high, descend in small observed increments and retreat straight up
before travelling. Keep the other arm and surrounding objects clear.

Commands are signed: gripper=1 closes/holds, -1 opens and 0 keeps the command
where supported by the tool schema. Carry with gripper=1. A closed gripper alone
proves nothing: verify that the target leaves its place and moves with the arm.
Use `set_gripper(arm=..., gripper=...)` to open or close. Inspect the actual result
and fresh visual evidence; do not assume a full stroke or an adjustable opening.
"""

SKILLS_AND_FEEDBACK = r"""
Use `pi0_pick` for grasping when registered. Treat its success signal as a
provisional heuristic, then verify the object in the wrist/head images and
whether it moves with the arm. Use bounded recovery: change the description,
approach or staging after a failure, rather than repeating the same attempt.
The original motion result is in the returned observation's `log.result`.
Check its error/outcome fields and the observation after every step.
"""

JUDGING_PROGRESS = r"""
The benchmark judges the final scene. Do not invent a live task-success flag.
A named lift amount is measured on the object, not the gripper: held objects can
sag or slip. During transport the hold must remain stable. For placement, the
object must rest at the requested destination after opening and withdrawing;
an object on a rim or beside its destination is not placed.

Check every object named by the instruction, the exact requested relation and
orientation, and whether other fixtures or objects were disturbed. A nearly
completed arrangement is insufficient. For pushing/sliding tasks, lifting the
object may violate the instruction. Re-localize and correct the remaining error
while budget remains. Restore arms and open grippers when required by the task,
using only registered tools; there is no assumed `home` helper.
"""

RECOVERY_AND_BUDGET = r"""
Inspect the scene after motion. Stop the current plan if an object is unstable
or an arm is obstructed. Identify the failed phase: wrong object, missed grasp,
lost hold, unreachable target or premature release. Change one meaningful
variable and verify. Do not repeat the same unsuccessful recovery indefinitely.
Recover in the current scene; use a reset only when explicitly provided and
permitted by this run. Finish when the scene meets the instruction, or report
failure/stuck when no justified recovery remains.
"""


def system_prompt() -> dict[str, PromptNode]:
    """Assemble code execution rules and robodojo operating knowledge."""
    return {
        **common.system_prompt(),
        "Observe and localize": OBSERVE_AND_LOCALIZE,
        "Motion and gripper": MOTION_AND_GRIPPER,
        "Skills and feedback": SKILLS_AND_FEEDBACK,
        "Judging progress": JUDGING_PROGRESS,
        "Recovery and budget": RECOVERY_AND_BUDGET,
        "API reference": "{{code_policy_api}}",
    }
