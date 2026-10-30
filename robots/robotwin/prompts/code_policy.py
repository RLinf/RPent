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

"""Code-policy instructions for robotwin."""

from rpent.prompt import code_policy as common
from rpent.prompt.utils import PromptNode

OBSERVATION_AND_TASK_SUCCESS = r"""
Read `observation = robo.view_env_state(step=-1)` and `state = observation["state"]`.
`state["robot_state"]` contains `left_eef_pose`, `right_eef_pose`, TCP poses
and gripper commands. `state["episode_status"]` contains `eval_success`,
`take_action_cnt` and `step_lim`. Action results are in `observation["log"]["result"]`.
Only the native `eval_success` flag proves task success. Use keyword arguments;
there are no typed SDK results, constants or `robo.done` property.
"""

FRAMES_UNITS_ROBOT = r"""
- Two arms, `"left"` and `"right"`, on one base at the near edge of a table.
  World frame: x across the table (the LEFT arm starts at negative x and is
  on the LEFT side of the head image), y away from the robot towards the far
  edge, z up; metres and radians; quaternions are `[w, x, y, z]`.
- Both arms start above the table with the grippers open (1.0) and pointing
  along +y (the approach axis).  Read the table height from the head world
  map using `sample_world_xyz` or `query_world_map` on visible table pixels; it is randomised per
  scene, never hard-code it.
- EEF vs TCP: `move_to` targets the EEF (wrist link).  The TCP - the gripper
  centre, where an object is held - is 0.12 m further along the approach
  axis derived from the observed left TCP quaternion.  To put the gripper centre on a point p,
  command the EEF at `p - 0.12 * approach`; sending an object's surface point
  as an EEF target parks the gripper 12 cm past it.
- The task instruction (`task_language` in the observation) is authoritative: obey it verbatim,
  including which arm it names.  When it names none, the arm on the object's
  side (left for x < 0) is the natural choice; the other arm stays out of the
  way.
"""

GRIPPER = r"""
- Gripper values are floats in [0, 1]: 1.0 open, 0.0 closed.  `set_gripper` interpolates linearly over
  `steps` native actions; `release` opens to 1.0.
- `move_to` keeps the current command, so a held object
  stays held by default; pass a value only to change it during the motion.
- `state["robot_state"]["left_gripper"]` reports the commanded value, not contact.  Closure
  alone proves nothing: a grasp is verified when the object leaves its
  resting place and moves with the TCP (compare wrist images / world samples
  before and after a small lift).
"""

MOTION_PRIMITIVES = r"""
- `move_to`: a motion planner
  computes a collision-free joint path to the EEF pose and executes it as
  `substeps` native actions.  the move result’s `stop_reason="plan_failed"` means nothing moved (the
  target is unreachable, in collision with the table, the other arm or an
  object): do not repeat the same target; retreat or change one variable
  (approach height, waypoint, orientation).
- A successful plan does not prove arrival: compare `final_eef_xyz` with the
  target (compute the target distance or inspect `final_dist_m`) and check the images for the intended
  scene change.
- Guarded low approaches: near the table, a container rim, a button, a
  hinge, a stacked object or the other arm never queue several unobserved low
  waypoints.  Keep x/y, orientation and gripper, change only z by 5-10 mm per
  call with at most 8 substeps, and check after every increment.  Do the
  increments in a loop inside one cell: after each `move_to` test the result
  and the state in code (the plan failed, z made no progress, x/y drifted,
  the observed `eval_success` flag) and stop the loop on any of them; look at an image only when
  the hold becomes uncertain or a contact cannot be interpreted.  Prefer one
  VLA chunk when the terminal motion needs contact feedback.
- `rotate_wrist` keeps the EEF position but sweeps the
  TCP and a held object on a 0.12 m arc.  Rotate only after a verified hold,
  with clearance around the gripper, at a safe transport height, in small
  increments; then re-check the hold and the object's actual orientation
  (EEF yaw change is not object yaw change).
- Every native action counts against `step_lim`; the limit is a safety
  ceiling, not a target.  `step_lim - take_action_cnt` from `episode_status` and the truncated flag tell you
  where you stand.
"""

VLA = r"""
- `lingbot_act` runs the frozen LingBot-VLA on the benchmark's own
  task instruction and the three current images for `chunks` x 50 native
  actions, stopping early on success.  There is no prompt to tune: it always
  receives `task_language` in the observation.
- Chunk pattern: one chunk near contact, near success, when the scene is
  unstable or for a small correction; two for ordinary stable progress; three
  only for a continuity-sensitive phase (bimanual, articulated, hanging,
  insertion, tool use) that is already moving correctly.  Several chunks can
  run from one cell: loop over `lingbot_act`, check the observed `eval_success` flag and
  the state after each, and end the cell to look at the images only when the
  state does not tell you why a chunk was unproductive.
- Prefer the VLA for grasp and re-grasp, the receiving arm of a handover,
  bimanual coordination, insertion, hanging, tool use and contact-rich
  motion.  Use the primitives after a verified state for measured free-space
  transport, staging, retreat, release or one small geometric correction.
  Never call a primitive just to see whether it helps, and never transport
  because a gripper merely looks closed.
- When the VLA has correct contact and visible progress, let it continue
  rather than interrupting it with speculative primitives.  After an
  unproductive chunk, look at the images and decide: continue, shorten the
  next chunk, or improve visibility / physical staging first.
- Physical-state shaping: when the VLA is bound to the right target but keeps
  failing because of orientation, occlusion, reach or an unsafe height, one
  observed primitive may fix one variable (lift a verified hold to clearance,
  one small safe-height rotation, move a held object to an unobstructed
  staging pose), then hand control back with one chunk near contact.  Do not
  pre-position an empty gripper or disturb a near-success state to test
  whether shaping helps.
"""

PERCEPTION = r"""
- Three cameras, 320x240 each.  `head` is the semantic authority: identity,
  distractors, destinations, the relation the language asks for, global
  progress.  The wrist views refine the geometry of the SAME candidate the
  head view selected; do not let a wrist view silently switch to a
  look-alike.
- World maps are `[row, col] -> [x, y, z]` metres and may contain NaN.
  `sample_world_xyz` gives the median of a small window;
  `query_world_map` gives extent and median of a box.  Pair pixels with the world map of
  the same step and view.
- A visible surface point is not an object centre: sample several interior
  pixels and take robust statistics; pixels on edges, shadows or the gap to
  the table return background points.  Flat objects at table height need the
  RGB image, not depth, to be told apart.
- Re-localise after occlusion, contact or substantial arm / object motion;
  call `render` first when time has passed without a motion call, since
  images belong to recorded steps.
"""

OBSERVABLE_GATES = r"""
- Grasp: the target leaves its source and moves with the TCP.
- Transport: the hold stays stable through a clearance waypoint and the
  lateral move.
- Placement on a support: the object rests on the correct support before
  release, then stays put and separated while the arm withdraws.
- Container: the object body crosses the opening and remains inside after
  release; rim or nearby placement is incomplete.  A pad, plate, scale,
  skillet or stand is not a container.
- Short contact (button, switch, bell): the intended control visibly changes
  after one guarded contact; distinguish the control from printed markings.
- Articulation (lid, door, hinge, knob): contact is retained while the part
  moves in the requested direction; verify the state change before letting
  go.  A momentary press is not an articulation.
- Handover: the receiving arm's hold is verified before the giving arm
  releases.  A task name alone does not prove a handover is required; the
  instruction does.
- Ranking / stacking: follow the order the language gives; each correct
  relation is protected from later paths and actions.  Ranking does not
  imply stacking.
- Orientation / hold: the requested orientation is visible while control is
  retained; do not release when the instruction asks to hold, lift or shake.
- Only the native `episode_status.eval_success` flag proves task success; stop
  acting as soon as it fires.  Primitive success is not task success.
"""

RECOVERY_AND_BUDGET = r"""
- After a failed phase name the first unmet gate and the blocker: wrong
  identity or destination, missed grasp, lost hold, planning / collision,
  insufficient contact, premature release, unstable placement, incomplete
  relation.  Change one meaningful variable and verify; do not repeat the
  same primitive target or the same hand-written recovery twice.  Repeated
  VLA chunks are allowed when contact and the physical state support it.
- Near success, repair only the remaining blocker; do not restart the task
  or disturb objects that are already correct.
- Work in phases with in-code checks (the move result’s `plan_status` / `final_dist_m`, gripper
  value, world samples before and after, the observed `eval_success` flag), and look at an image
  only when a decision depends on it.
- There is no reset.  Recover in place: re-observe, re-localise, re-grasp.
"""


def system_prompt() -> dict[str, PromptNode]:
    """Assemble code execution rules and robotwin operating knowledge."""
    return {
        **common.system_prompt(),
        "Observation and task success": OBSERVATION_AND_TASK_SUCCESS,
        "Frames, units, robot": FRAMES_UNITS_ROBOT,
        "Gripper": GRIPPER,
        "Motion primitives": MOTION_PRIMITIVES,
        "The VLA": VLA,
        "Perception": PERCEPTION,
        "Observable gates": OBSERVABLE_GATES,
        "Recovery and budget": RECOVERY_AND_BUDGET,
        "API reference": "{{code_policy_api}}",
    }
