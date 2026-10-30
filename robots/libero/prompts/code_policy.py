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

"""Code-policy instructions for libero."""

from rpent.prompt import code_policy as common
from rpent.prompt.utils import PromptNode

OBSERVATION_AND_TASK_SUCCESS = r"""
Read `observation = robo.view_env_state(step=-1)`. Its `state` contains
`robot0_eef_pos`, `robot0_eef_quat`, `robot0_gripper_qpos` and `object_names`.
After motion, the action payload is in `observation["log"]["result"]`.
Compare `final_dist_m` with your requested tolerance; there is no `Move` class
or derived `reached` property. Use keyword arguments for every tool call.
Task success is the native `terminated` flag, not a grasp heuristic.
"""

FRAMES_UNITS_SCENE = r"""
- World frame: x/y across the table, z up, metres, radians.  `robo.view_env_state(step=-1)["state"]["robot0_eef_pos"]`
  is the gripper position in this frame; `back_project`, `segment` and
  `back_project` with `row_range` and `col_range` return points in the same frame, directly comparable.
- Read heights from the scene: `robo.view_env_state(step=-1)` for the gripper, a bare-table
  pixel through `back_project` for the table surface.  Never hard-code them.
- "left"/"right" in task text is egocentric to the robot: +y is the robot's
  LEFT, which appears on the RIGHT of the agentview image.
- `robo.view_env_state(step=-1)["state"]["object_names"]` (e.g. `akita_black_bowl_1`) carry no pose and no
  visual difference between `_1`/`_2`; identify targets by what they look
  like and where they are, and obey the task instruction verbatim.
"""

GRIPPER = r"""
- `gripper=+1` closes / holds, `gripper=-1` opens.  Every motion call holds
  its `gripper` value for the whole motion: carry an object with
  `gripper=+1` in every `move_to`; a carry with `-1` silently drops it.
- `move_pose` also defaults to open - pass `gripper=+1` while holding.
- `robo.set_gripper(gripper=1, steps=8)` after a pick firms the grip before carrying.
- the sum of absolute `robot0_gripper_qpos` values: ~0.08 fully open; ~0.0 closed on nothing;
  roughly 0.01-0.05 means an object is between the fingers.
"""

MOTION = r"""
- One `move_to`/`move_pose` may travel at most 0.30 m in x/y.  Split long traversals into 2-3 waypoints
  at carrying height.
- `step_clip` (per-step travel cap, m): 0.025 with an empty gripper or a
  box, 0.015 for cans, 0.012 for tall bottles and for fine approaches or
  vertical retreats.
- Approach high, then descend vertically; retreat straight up before
  travelling.
- `final_dist_m` greater than the requested `tol` means the servo stalled (IK limit, contact, budget).
  Near cabinet fronts, low shelves and deep reaches `move_to` can wall at an
  IK singularity; `move_pose` (co-varying xyz with pitch/yaw) reaches deeper.
- Long pushes can destabilise the simulator; keep pushes short and capped or
  use `pi0_doubled`.
"""

VLA_SKILLS = r"""
- `pi0_pick` runs the frozen Pi0.5 policy for the grasp only; you do every
  `move_to` and the `release` yourself.  Typical call:
  `robo.pi0_pick(prompt="pick up the <object>", max_chunks=20, lift_thresh=0.05,
  gripper_closed_thresh=0.06)`; use `lift_thresh=0.08` for slippery tall
  bottles.  Pre-position the gripper 15-20 cm above the target first (some
  objects grasp best straight from the home pose: the VLA has its own
  approach trajectory).
- Prompt ladder when a pick does not lift: 1) "pick up the <object>",
  2) the task instruction verbatim, 3) a spatial qualifier, 4) re-position
  the pre-pose (lower z, 5 cm offset in xy) and retry.
- Judge the grasp yourself: the pick result’s `success` field (lift + closure heuristic) is a
  hint, not proof.  After the pick read the sum of absolute `robot0_gripper_qpos` values
  (0.01-0.05 = holding, ~0 = closed on nothing) and, if in doubt, `segment`
  the object in the wrist image: it should now be raised into the gripper
  and its original spot empty.
- Once lifted, `robo.set_gripper(gripper=1, steps=8)` (8-12 steps; `<=5` for a
  laterally weak can) firms the grip; then script the carry
  (`robo.move_to(xyz=waypoint, gripper=1)` waypoints) and the `release`.
- `robo.pi0_pick(prompt=instruction, lift_thresh=999, gripper_closed_thresh=0)` turns it into a
  generic closed-loop contact skill; `pi0_doubled` is the dedicated contact
  skill for drawers, doors, knobs and insertions (its `success` only mirrors
  the task predicate); call it repeatedly, alternating with short scripted
  aligns.
"""

PERCEPTION = r"""
- Two cameras: `agentview` (fixed, global layout; best for identifying the
  right object and the right destination) and `wrist` (on the gripper;
  close-range geometry once the gripper is 15-20 cm above the target).
- `robo.segment(prompt=description)` runs SAM3 on a recorded image: describe colour + shape +
  relation ("the black bowl on the stove"); brand or internal names score
  badly.  Check `found` and `score`; `world_xyz` is the visible surface
  median (use its x/y; take z from the resting height).
- `robo.back_project(row=row, col=col)`: sample several pixels firmly on the object's top
  surface and take the median; pixels on rims, edges, shadows or the gap to
  the table return background points metres away.
- `back_project` with `row_range` and `col_range` gives the interior centre of a container from a pixel
  window (a mask median is rim-biased). Use the returned region statistics for in-code checks.
- Flat discs at table height (plate, burner, cabinet top, lid) look alike in
  depth; identify the destination in the RGB image by the noun in the task.
- Refine with the wrist only when its estimate agrees with the agentview
  estimate to within a few cm; a large jump means it locked onto a look-alike.
"""

PLACING_AND_RECOVERY = r"""
- Descend to place, `release`, then retreat straight up with a small
  `step_clip`; `release` returns `terminated=True` when the placement
  satisfied the task and the episode is then over.
- Key numbers (the same ones the tool-calling agent is given): single-step
  x/y within 0.30 m; `lift_thresh` 0.05 for flat objects / 0.08 for slippery
  tall bottles; `step_clip` 0.025 empty or box / 0.015 cans / 0.012 tall
  bottles; gripper home z is about 0.68 in living-room scenes, 1.17 in
  kitchen scenes, 0.26 in object scenes; BOWL: `eef_y = plate_y + 0.045`;
  TALL BOTTLES: carry at z = 0.30 and drop without descending; approach
  high then vertical; recover by re-picking, not by hovering.
- Containers can slide when bumped; re-localise after any contact.
- There is no reset.  Recover in place: re-localise, re-pre-position,
  re-issue the pick, re-firm the grip.
"""


def system_prompt() -> dict[str, PromptNode]:
    """Assemble code execution rules and libero operating knowledge."""
    return {
        **common.system_prompt(),
        "Observation and task success": OBSERVATION_AND_TASK_SUCCESS,
        "Frames, units, scene": FRAMES_UNITS_SCENE,
        "Gripper": GRIPPER,
        "Motion": MOTION,
        "The VLA skills": VLA_SKILLS,
        "Perception": PERCEPTION,
        "Placing and recovery": PLACING_AND_RECOVERY,
        "API reference": "{{code_policy_api}}",
    }
