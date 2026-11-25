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

"""LIBERO operating knowledge for code policies."""

from rpent.prompt.utils import PromptNode

CONTROL = """
Read `task_language` from the observation and `robot0_eef_pos`,
`robot0_eef_quat`, `robot0_gripper_qpos` and `object_names` from its
`state`. The action payload is in `log.result`. Task success is the native
`terminated` flag. Compare motion `final_dist_m` with your requested `tol`.

World z is up, positions are in metres and angles in radians. Robot left
is +y and appears on the right of agentview. Use agentview to select the
target and destination by appearance and relation, then the wrist to refine
the same candidate. `object_names` supplies no pose.

Use `segment` (SAM3) and inspect `found`, `score` and `world_xyz`. The point
is a visible surface. Use `back_project` on several interior pixels or a
window with `row_range` and `col_range` for a container. Read table height
from the scene. Re-localize after contact. Open with gripper=-1, close and
carry with gripper=1 on every move, including `move_pose`. The sum of the
absolute finger positions is about 0.08 open, 0 closed on air and 0.01–0.05
when holding.

Use waypoints for x/y travel over 0.30 m and approach and retreat vertically.
Use `step_clip` of 0.025 empty, 0.015 for cans and 0.012 for tall bottles or
fine approaches. Change the approach or use `move_pose` when `move_to` stalls
near a cabinet or shelf. Recover in place from fresh observations.
"""

SCRIPTED_GRASP = """
Localize the object and destination, choose a grasp height below the object's
top surface, and check the numeric outcome after each motion. Move above
the target with the gripper open, descend, close with `set_gripper` and
lift vertically. Check finger separation. Under --no-am, action calls return
state without images. Use `robo.view_env_state(step=-1)` to check the latest
images when the grasp is uncertain. Move above the destination, descend, release and
check native termination.
"""

VLA_SKILLS = """
Pre-position 15–20 cm above the object for `pi0_pick` (try the home pose if
needed). Start with the object name, then the live instruction or a spatial
qualifier. Use `lift_thresh=0.05` for flat objects and 0.08 for tall slippery
bottles. Verify lift and grip independently, then firm the hold with
`set_gripper(gripper=1, steps=8)` (at most 5 for a laterally weak can).
Script carry and release. For a bowl on a plate, start with
`eef_y = plate_y + 0.045`; carry tall bottles at z=0.30 and release from
above. Use bounded `pi0_doubled` calls for contact-rich motion with scripted
alignment between calls.
"""


def system_prompt(*, policy_kind: str | None = "vla") -> dict[str, PromptNode]:
    """Return LIBERO conventions and the selected control strategy."""
    return {
        "Observation, perception and motion": CONTROL,
        "Grasp strategy": {
            None: SCRIPTED_GRASP,
            "wam": "Use bounded `robo.wam_act` chunks and check native termination.",
            "vla": VLA_SKILLS,
        }[policy_kind],
    }
