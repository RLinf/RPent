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

"""Code-policy instructions for robocasa."""

from rpent.prompt import code_policy as common
from rpent.prompt.utils import PromptNode

OBSERVATION_AND_TASK_SUCCESS = r"""
Read `observation = robo.view_env_state(step=-1)` and `state = observation["state"]`.
Native state uses `robot0_eef_pos`, `robot0_eef_quat`, `robot0_gripper_qpos`,
`robot0_base_pos` and `robot0_base_quat`. Action results are in
`observation["log"]["result"]`; `move_to` reports `ok`, `final_dist` and `eef`.
Task success is `observation["robocasa_terminated"]`. Use the registered keyword
schemas rather than positional calls. There are no typed SDK results or properties.
"""

FRAMES_UNITS_ROBOT = r"""
- A PandaOmron: one Franka arm with a two-finger gripper on an omnidirectional
  mobile base, in a kitchen with counters, cabinets, drawers, a stove, a sink,
  a fridge and a microwave.  World frame: x/y span the kitchen (coordinates
  of several metres), z up; metres and radians; quaternions are
  `[x, y, z, w]`.  Counters are at about z = 0.9 m; objects sit on counters,
  inside cabinets or in drawers.
- `state["robot0_eef_pos"]` is the gripper position, `state["robot0_base_pos"]` /
  the yaw derived from `state["robot0_base_quat"]` the base pose.  `back_project_batch` and
  `query_world_map` return points in the same world frame, directly
  comparable with `eef_pos`.
- The arm reaches about 0.8 m from the base.  If a target's x/y is farther
  than that from `base_pos`, the arm cannot get there: drive first
  (`navigate_to`), then manipulate.
- Every base motion - `navigate_to`, `move_base`, a VLA run that drove the
  base - moves the arm and all three cameras with it.  Re-localise from a
  fresh world map afterwards (the arm servo recalibrates itself on its next
  call); never reuse pixels or world points taken before the base moved.
- Never hard-code world coordinates: layouts, fixtures and object placements
  change with the seed.  Localise everything from the latest world maps.
- The task instruction (`task_language` in the observation) is authoritative.  Success is the
  environment's own predicate (the observation’s `robocasa_terminated` flag);
  the observation’s `task_progress` shows the live values that predicate computes and
  the task’s stated success requirements its source.
"""

GRIPPER = r"""
- `+1` drives the fingers shut and keeps squeezing; `-1`
  opens; `"hold"` servos the fingers to the width
  they had when the motion began.
- Carry with `"hold"`: it is the default of `move_to`, `move_delta`,
  `navigate_to` and `move_base`, so omit `gripper` while holding an object.
  A sustained `+1` squeezes a small object out of the fingers; carrying with
  `-1` silently drops it.
- `set_gripper` closes actively (firms a grip), `release`
  opens to let go.  `rotate_pitch` and `set_gripper` take numeric commands
  only.
- the sum of absolute `state["robot0_gripper_qpos"]` values is about 0.04 at the start (fingers
  half open), about 0.08 fully open and about 0 when closed on nothing; a
  closed gripper whose opening stays clearly above 0 has something between
  the fingers.  The finger position alone is not proof of a hold: verify a
  grasp from the VLA's grasp signals, the wrist image and the object moving
  with the gripper.
"""

NAVIGATION = r"""
- `navigate_to` turns the base to face the world target and drives
  forward with closed-loop steering (about 2.3 mm per env step: 300 steps
  cover roughly 0.7 m; give a long drive more `max_steps` or call it again)
  until it is within `tol`, then stops facing the target.  Choose `tol` = the
  standoff you want + the object's half depth: `tol=0.6` stops about 0.6 m
  in front of the target.  The first call spends a few steps calibrating the
  heading.
- There is no path planning.  the navigation result’s `stuck` flag (ran out of steps having
  moved < 0.12 m) means the base rammed a fixture: back off with
  `move_base`, then approach from a clearer
  direction.  Check the way ahead first: the returned navview floor overlay paints walkable floor green and `robo.query_world_map(z_min=0.0, z_max=0.12, camera="navview")` lists floor
  clusters.
- `move_base` sends raw velocities in the
  robot's own frame (+forward ahead, +lateral to the right, +turn
  counter-clockwise, clipped to [-1, 1]).  For fine adjustments only: keep
  `forward <= 0.4`, `turn <= 0.3`, `steps <= 20` per call and re-observe
  between calls.
"""

ARM_PRIMITIVES = r"""
- `move_to` is
  a closed-loop servo: the move result’s `ok=False` means it stalled (out of reach,
  in contact, at a joint limit); do not repeat the same target - navigate
  closer or change the approach.  One call may travel at most 0.30 m: split long traversals into 2-3 waypoints at
  carrying height.  `move_delta` is the same servo relative to the
  current position, for small approaches and lifts.
- Approach from above, descend vertically, retreat straight up before
  travelling; use `step_clip=0.012` for fine approaches and vertical
  retreats.
- `rotate_pitch` tilts the wrist by
  `target_pitch` radians (relative) over `n` steps, e.g. before threading the
  gripper into an opening whose front points along world +/-y; read
  the end-effector orientation in the observation before and after, since the primitive reports no angle.
- `scripted_grasp` opens, hovers `approach_z` above the point, descends,
  closes and lifts: a coarse fallback for simple, well-localised objects;
  prefer `rldx_arm` for anything hard.  the grasp result’s `stage` field names the stage that
  stalled.
- One action, then observe: the tool-calling agent checks the state after
  every command; in a program that is a loop inside one cell.  Do guarded
  approaches as a loop of small `move_delta` / `move_to` steps (5-10 mm in z
  near a surface, a rim or a handle) and after each one test the result and
  the state in code (motion `ok`, `final_dist`, observed gripper opening,
  `task_progress`, the observation’s `robocasa_terminated` flag) and stop the loop on any of them; look at an
  image only when a contact or a scene change must be interpreted (the hold
  became uncertain, the object moved unexpectedly).  Prefer the VLA when the
  terminal motion needs contact feedback.
"""

VLA = r"""
- `rldx_skill` (arm and base) and `rldx_arm` (base clamped to
  `base_clip=0.1`) run the frozen RLDX-1 policy on the benchmark's own task
  instruction and the current camera images, 8 env steps per chunk, until
  the task succeeds, the policy settles or `max_chunks` is reached.  There
  is no prompt to write or tune: the runtime always uses `task_language` in the observation.
- Leave `max_chunks`, `n_action_steps` and `settle_patience` at their
  defaults; the evaluation protocol pins them (typically 40 chunks, 8 steps,
  settling disabled) and the skill result’s `max_chunks` field reports what applied.
- Never put a manual primitive (`move_to`, `move_delta`, `navigate_to`,
  `move_base`, `set_gripper`, `scripted_grasp`) between two VLA calls of the
  same sub-operation: every non-VLA primitive wipes the VLA's frame history
  (the observation’s `vla_desync` becomes true) and the next call restarts from
  scratch.
- the skill result’s `status == "cap"` means the VLA was still acting: call it again with
  the same arguments (its history continues).  Re-stage by hand only after
  2-3 consecutive calls show neither contact (`grasp_contact`, `held_apart`)
  nor a change in `task_progress`.  Several calls can run from one cell:
  loop, check the observation’s `robocasa_terminated` flag, `task_progress` and the grasp signals after each,
  and end the cell to look at the images only when those do not tell you why
  a call was unproductive.
- Judge a grasp from the result: `grasp_contact` (fingerpads on a task
  object) is the gold standard, `held_apart` covers fixture handles the
  contact check cannot see, `grasped` says whether it still holds at the
  end, `peak_lift_m` whether it lifted.  A completed run is not a solved
  task: `status == "success"` / the observation’s `robocasa_terminated` flag is.
- Division of labour: the VLA grasps, re-grasps, pulls and pushes fixtures,
  presses controls and does other contact-rich motion; the program scripts
  the rest after a verified state - free-space carrying with `gripper`
  omitted, placing, `release`, retreats, driving between fixtures.
  `rldx_skill` when the base has to move for the sub-task, `rldx_arm` once
  the base stands at the right place.
"""

PERCEPTION = r"""
- Three 256x256 cameras, all on the robot.  `agentview` (shoulder) is the
  semantic authority: what the objects and fixtures are and where they lie;
  its pixels back-project directly.  `wrist` (on the gripper) refines the
  geometry of the SAME candidate once the gripper is within about 20 cm; do
  not let it switch to a look-alike.  `navview` (forward-down, base mounted)
  shows the floor ahead: where the base can drive.
- Images are top-down (row 0 at the top) and pixel-aligned with the world
  maps of the same step and camera.  `back_project_batch` returns
  each pixel's world point plus the median of the valid ones; sample 3-8
  pixels firmly on the object's top surface and use `median_xyz`; pixels on
  thin rims, edges, shadows or the gap to a counter hit the background.
  `query_world_map` finds clusters inside a height band
  (0.85-0.95 for objects on a counter) on the latest map. Use these returned samples and clusters for computation.
- A visible surface point is not an object centre: take the median of
  several interior pixels; choose the grasp height from the object's surface
  and its size.  Flat things at counter height look alike in depth: identify
  the destination in the RGB image by the noun in the task.
- World maps of steps older than 25 are deleted; localise from the latest
  step, and re-localise after any base motion, contact or VLA run.
"""

OBSERVABLE_GATES = r"""
- Only the observation’s `robocasa_terminated` flag proves task
  success; stop acting as soon as it fires.  Primitive success is not task
  success.
- `task_progress` is the intermediate feedback: read it at step 0 and after
  every action; a joint fraction, a counter or a sub-predicate flag moving in
  the requested direction shows the action worked, an unchanged one shows it
  did not.
- Grasp: `grasp_contact` / `held_apart` and the object moving with the
  gripper (compare the wrist image and world samples before and after a
  small lift).
- Transport: the hold stays stable through the lift, the drive and the
  waypoints.
- Placement: the object rests on the destination before `release`, then
  stays put while the arm retreats.  Rim or nearby placement is incomplete.
- Articulation (drawer, door, lid, dishwasher rack, mixer head): the part
  moves in the requested direction while contact is kept, and the joint
  value in `task_progress` crosses the threshold in the task’s stated success requirements.
- Appliance controls (stove, microwave, kettle, faucet): the corresponding
  flag in `task_progress` flips after one guarded contact; distinguish the
  control from printed markings.
"""

RECOVERY_AND_BUDGET = r"""
- After a failed phase name the first unmet gate and the blocker: wrong
  target, out of reach (drive closer), missed or lost grasp (VLA again from
  a better standoff), stuck base (back off, new approach direction),
  premature release, unstable placement.  Change one meaningful variable and
  verify; do not repeat the same primitive target or the same VLA staging
  twice.  Repeated VLA calls are allowed when contact and `task_progress`
  support them.
- Near success, repair only the remaining blocker; do not restart the task
  or disturb what is already correct.
- Work in phases with in-code checks (motion `ok`, `final_dist` and navigation outcomes,
  `gripper_opening`, world samples before and after, `task_progress`,
  the observation’s `robocasa_terminated` flag), and look at an image only when a decision depends on it.
- There is no reset: the episode is one shot.  Recover in place:
  re-observe, re-localise, drive or grasp again.
"""


def system_prompt() -> dict[str, PromptNode]:
    """Assemble code execution rules and robocasa operating knowledge."""
    return {
        **common.system_prompt(),
        "Observation and task success": OBSERVATION_AND_TASK_SUCCESS,
        "Frames, units, robot": FRAMES_UNITS_ROBOT,
        "Gripper": GRIPPER,
        "Navigation": NAVIGATION,
        "Arm primitives": ARM_PRIMITIVES,
        "The VLA": VLA,
        "Perception": PERCEPTION,
        "Observable gates": OBSERVABLE_GATES,
        "Recovery and budget": RECOVERY_AND_BUDGET,
        "API reference": "{{code_policy_api}}",
    }
