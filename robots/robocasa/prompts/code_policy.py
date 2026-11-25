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

"""RoboCasa operating knowledge for code policies."""

from robots.robocasa.prompts.evaluate import GRIPPER_RULES, LOCALIZATION, NAVIGATION
from rpent.prompt.utils import PromptNode


def system_prompt() -> dict[str, PromptNode]:
    """Return RoboCasa's observation and control conventions."""
    return {
        "State and progress": """
Read `task_language` and `task_progress` from the observation. Native success
is `robocasa_terminated`. The `state` fields `robot0_eef_pos`,
`robot0_eef_quat`, `robot0_base_pos`, `robot0_base_quat` and
`robot0_gripper_qpos` describe the robot. Positions are in world metres, z up,
with [x, y, z, w] quaternions. Pair pixels with the same camera and step;
world maps older than 25 steps are deleted. Re-localize after base motion,
contact or a VLA run. Use wrist refinement for the same agentview target.
""",
        "Localization": LOCALIZATION,
        "Navigation": NAVIGATION
        + """
The arm reaches about 0.8 m from the base. Navigation has no path planner.
On `stuck`, back off and try a clear approach. Positive raw lateral moves
right and positive turn rotates counter-clockwise in the base frame.
""",
        "Gripper": GRIPPER_RULES,
        "Arm": """
Read `ok`, `final_dist` and `eef` from `log.result` after a move. Split
moves longer than 0.30 m. Approach vertically with 5–10 mm increments near
contact and `step_clip=0.012` for fine moves and retreats. On a stall, change
the approach or move the base closer. `rotate_pitch` is relative. Confirm
that a grasp moves the target during a small lift before carrying it.
""",
        "Action model": """
Pass the live `task_language` as the prompt to `rldx_skill` (arm and base) or
`rldx_arm` (base clamped). Keep chunk and settling defaults. Manual moves
invalidate frame history (`vla_desync`), so continue a capped skill while
contact (`grasp_contact`, `held_apart`) or `task_progress` supports it.
After two unproductive calls, re-observe and change one staging variable.
Use task progress to verify articulation and control changes. Preserve
completed relations, verify position after release and recover in place.
""",
    }
