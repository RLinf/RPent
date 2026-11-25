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

"""RoboTwin operating knowledge for code policies."""

from robots.robotwin.prompts.evaluate import PERCEPTION, TASK_FAMILIES, WAM_WORKFLOW
from rpent.prompt.utils import PromptNode


def system_prompt(*, policy_kind: str | None) -> dict[str, PromptNode]:
    """Return RoboTwin's observation and control conventions."""
    return {
        "State and success": """
Follow `task_language` and the named arm. Read TCP poses and gripper
commands from `state["robot_state"]`, and success and budget from
`state["episode_status"]` (`eval_success`, `step_lim - take_action_cnt`).
Check `log.result` after actions. Stop motion on native success or truncation.
There is no reset.
""",
        "Frames and motion": """
World x runs across the table (left arm at negative x), y away from the
robot, z up. Use metres and [w, x, y, z] quaternions. Sample the scene's table
height. `move_to` targets the wrist link, 0.12 m behind the TCP along the
observed approach axis. `rotate_wrist` sweeps that same offset, so rotate
a held object in small increments at clearance height and verify the hold.

Gripper 1.0 opens, 0.0 closes. Motion preserves the command by default.
Gripper state is a command. Confirm that the target moves with the TCP
during a small lift. Check `stop_reason` and `final_dist_m`; a plan can fail
or stop early. Near contact, change z by 5–10 mm with at most 8 substeps per
move. Stop on failure, no progress, drift or native success.
""",
        "Perception": PERCEPTION
        + """
`sample_world_xyz` returns a local window median; `query_world_map` gives
box extent and median. Use interior pixels and refresh with `render` if
needed before sampling.
""",
        "Task checks": TASK_FAMILIES,
        "Action model": WAM_WORKFLOW
        if policy_kind == "wam"
        else """
`lingbot_act` uses the original instruction and current images in chunks of
50 native actions. Use one chunk near contact, two for stable progress, and
three only for a continuous phase already progressing. Use it for grasp,
handover and contact-rich motion. Script free-space transport, staging,
release and retreat after verifying the previous phase. When progress stalls,
observe and change the staging variable that blocks the next phase.
""",
    }
