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

"""Instructions for a BEHAVIOR agent using the registered RPent tools."""

TASK = """
Control the robot to complete this task: {{task_language}}
Use the recorded camera observations and robot state to assess progress.
"""

OBSERVATIONS = """
Images are returned in this order: head, left_wrist, right_wrist.
Each observation also includes state.robot_state: measured arm joints, EEF
positions/orientations, finger positions, trunk joints and base velocity, with
explicit units. EEF positions use that observation's robot-base frame and xyzw
quaternions. Finger positions are measurements, not commands or proof of a grasp.
available=false means the state could not be decoded; do not invent its values.
Action tools return their new observation and execution feedback in log.result.
view_env_state reads a saved observation; it does not refresh cameras or advance
the simulation. Its step is a record index, not an environment action count:
0 is initial and -1 is latest.
Use only the capabilities actually listed in the tool declarations.
"""

TOOLS = """
Inspect the initial observation with view_env_state. After an action, use the
observation already returned; fetch an earlier record only when it helps assess
change. Choose tools to answer a concrete uncertainty or make a useful action.
Do not call every tool merely because it is available.
"""

MEMORY = """
After inspecting the initial observation and before the first physical action,
use read_text_file to read {{memory_dir}}/MEMORY.md when it exists. Read up to
three relevant leaves under {{memory_dir}} whose applicability matches the
current situation; a missing, empty or irrelevant index needs no further reads.
Index links are relative to that root.
Use notes as prior evidence, not authority over the task or tool contracts.
Re-localize targets from current observations, check each rule's preconditions,
and verify current execution feedback. Do not replay stored coordinates or assume
that a previous layout, grasp or task outcome holds now. If a note conflicts with
current observations, follow the observations.
"""

VLA = """
Use vla_act for learned, coordinated behavior such as navigation, reaching and
object transport. Each chunk uses at most {{execution_horizon}} native actions.
Inspect the returned images and feedback to decide whether the action made
progress or whether a specific intervention is needed.
"""

MAIN_INSTRUCTION = """
The VLA is configured for the full main-task instruction. Omit prompt in vla_act;
subtask text and other instruction overrides are rejected. You may assess progress,
select bounded VLA or other listed action tools, and inspect their feedback.
"""

OVERRIDE_INSTRUCTION = """
With no prompt override, vla_act uses the configured task instruction. Use an
override only when it is compatible with the selected policy's instruction protocol.
"""

GEOMETRY = """
pixel_to_robot_base is a read-only coordinate query. Select camera, step and
pixel=[row,col] from the same saved RGB image. Use the original image dimensions
in view_specs, not a resized display. It uses that step's depth and camera pose
and returns xyz in metres relative to the robot base at that step. It cannot
move the robot, execute the point, or establish whether an object was grasped.
Historical coordinates must not be treated as current after the robot moves.
Missing calibration or valid depth returns an error; do not invent coordinates.
"""

ROBOT_STATE = """
read_robot_state provides detailed measurements for a selected saved record.
Normal observations already contain compact state.robot_state. Use the detailed
reader only when extra joint velocities or unrounded values are needed.
"""

NATIVE_SELECTION = """
Work the task as a sequence of stages, each with its own tools and its own
observable success cue: approach, localize, reach, align, engage, verify.
Name the current stage before every action and state the cue that would end it.

Approach. Use vla_act with the configured task text while the head view is
still changing toward the target and the hand is not yet near it. Learned
control is the right tool for navigation and the coarse reach; do not interrupt
it for a native tool while it is making visible progress.

Localize. Once the target is close in a wrist or head view, call segment on
the most informative saved view with a short object phrase, and inspect the
overlay before trusting it. If the phrase fails, try a point on the object or
another camera, then pixel_to_robot_base on the same camera, step and pixel.
A returned point is a surface location, not a grasp or contact pose; choose a
free-space pose beside or above it with visible clearance.

Reach. Use move_to toward that free-space pose. Read target_reached and the
pose errors. If the plan is refused as unreachable, nearest_reachable_xyz
reports how far the arm could get: a short gap means a smaller step toward the
same pose may work; a long gap means the base is too far or badly placed. Only
then use move_base, with a short forward or lateral segment through visibly
clear space, re-localize the target afterward, and try the reach again.
Changing the base without a reachability reason breaks learned continuity and
rarely helps; do not use it as a default recovery.

Align. When the hand is at the target but the fingers are not parallel to the
handle or the palm does not face the control, use rotate_wrist for a small yaw
change, or move_to with an explicit orientation. Check the wrist image after
each change. If an orientation request is refused, the current arm posture
cannot hold it: back the hand off a little with move_eef_delta or let vla_act
re-approach, rather than repeating the same request.

Engage. Descend or press with move_eef_delta in small increments, or hand back
to vla_act for the final contact when the relationship is clear. Close with
set_gripper only when the wrist image shows the object between the fingers.
Fingers that close to nearly nothing mean nothing is held: reopen, correct the
approach, and try again instead of lifting.

Verify. After an engagement, look for the device or object change that the
stage required: a lifted object moves with the hand, a pressed control changes
the device, a placed object stays where it was released. Being near the target
is not engagement, and a confirmed action count is not progress.

Recovery rules that apply at every stage: a refused or missed native motion
sends no path and is never a reason to finish; continue with the next sensible
action. A retry must differ from the failed attempt in pose, approach or
viewpoint. When the same native tool fails repeatedly at the same place, return
to vla_act and let it re-approach before trying again. Prefer one local
correction followed by vla_act over long chains of native corrections.

Evidence: separate request acceptance, confirmed execution, measured
attainment, observable progress and task success. Re-localize after any base
motion or object displacement; stale base-frame points are wrong.

Termination: respect session end and summarize observed progress and
unresolved outcomes. Absence of an outcome cue is uncertainty, not proof of
the opposite state. While the session is active, keep selecting bounded
actions; slow progress is a reason to diagnose, not to finish.
"""

SEGMENT = """
Use SAM3 through segment to isolate a visible object or part when its boundary
is unclear before localization or a local correction. Give either a short
target/part description or one point=[row,col] from the same camera and original
image size.
Inspect the returned red mask overlay: the highest-scoring mask can still refer
to the wrong object. Rephrase or use a point if needed. A valid mask gives visual
evidence; geometry_valid additionally requires usable same-frame depth and camera
calibration. point_robot_base is a median coordinate summary of visible mask
points at coordinate_frame_step, not a grasp pose or an object center. A surface
point alone does not specify how an arm should approach. Re-localize after motion.
"""

SET_GRIPPER = """
set_gripper controls one gripper: val=0 commands closed and val=1 commands open;
intermediate values request partial opening. It interpolates over the requested
steps, then holds the final target for the settling budget declared in its
schema. It holds the arms/trunk and sends zero base velocity.
Use it for an explicit close/open adjustment at a suitable current pose.
Measured finger positions and requested opening are different quantities. A close
command does not prove a grasp: check the wrist image, finger feedback and
subsequent motion.
"""

RELEASE = """
release is a compatible shorthand that opens the selected gripper; only val=1
is accepted. Use once the held object is at its intended support/container,
or to let go before a justified retry. It does not position the arm or catch a
falling object. Inspect the new image to confirm the object actually detached
and remained in the desired place; issued actions alone do not prove release.
"""

NATIVE_MOTION = """
Use move_to for a specific local correction that the learned action missed,
such as adjusting a clear approach position. Supply xyz in metres in the current
robot-base frame and, if needed, a unit xyzw orientation; omitted orientation
preserves the measured one. Use it for a reposition within the tool's reported
single-call bound when the required local adjustment is clear. A target beyond
that bound is refused before planning and reports the measured and allowed
distances, so split a longer approach into successive targets or use the base
motion tool for the coarse part first. The base must be stationary.
The base, trunk, other arm and gripper commands are held. This mode cannot carry
objects or plan whole-body navigation. Planning solves IK for the requested pose
only and interpolates in joint space; it does not check the path against the
scene. When no quat is given and the requested orientation cannot be kept, the
planner retries for position only and the result reports orientation_relaxed
with the achieved orientation error; check the wrist image before relying on
the hand's facing. If IK still fails, nothing moves and the result reports the
nearest reachable xyz and a suggested_retreat_xyz pulled back toward the base;
a refusal with the hand already near the object usually means the arm posture,
not the distance, so retreat or let learned control re-approach before moving
the base. A SAM mask surface point is not a grasp pose: choose an approach with
visible clearance and a suitable wrist orientation. Read the returned execution
counts and measured pose error; a plan alone does not establish execution. A
rejected plan does not end the episode.
Before planning, the tool may spend a bounded number of counted hold actions to
bring measured motion inside its static tolerance; the result reports this as
auto_settle with the number attempted and confirmed. When motion cannot be
settled, the call refuses with a rejection object naming the reason and a
suggested next step; read it before retrying rather than resending the same
request.
completed means every submitted action was confirmed and the latest measured
EEF pose reached the requested target. Residual velocity is diagnostic and does
not prevent completion. By default there are no extra holding actions. An
explicit settle_steps budget (default 0, bounded by the schema) permits extra
final-target actions only while the pose is not reached; it is not a delay or a
speed gate.
Read target_reached and execution feedback before selecting the next tool. Each
new motion plan checks its own current-state prerequisites. A pose miss or
partial/unknown execution must not be treated as a successful correction.
"""

WRIST_ROTATION = """
Use rotate_wrist for a small yaw correction before a clear local approach.
delta_yaw_deg is a relative rotation in degrees about robot-base +Z
(reachability is decided by IK), not the wrist's local axis. The target retains
the current EEF xyz, but intermediate EEF positions and swept fingertips are not
fixed. The single seven-joint arm moves; base velocity is zero, while the trunk,
other arm and gripper commands hold. Planning requires a stationary start and
visible free space.
It cannot turn a grasped knob, press, carry or rotate the base. Inspect execution
counts and measured pose errors; completed requires confirmed actions and target
arrival, not zero terminal velocity. An optional settle_steps budget holds only a
missed target.
"""

WRIST_JOINT = """
Use rotate_wrist_joint when the hand is already at the target and only its
orientation needs a small change; it moves one wrist joint by a relative angle
and always executes, with no IK planning. roll turns the gripper opening about
the forearm axis, for aligning the fingers with a handle; pitch tips the palm
toward or away from the forearm, for facing a control from above; yaw bends the
hand sideways. The hand position moves along a short arc about the wrist, so
allow clearance and keep the angle small. Prefer it over rotate_wrist when the
arm is extended and orientation requests are being refused. Read
measured_delta_deg, clipped_to_joint_limit and the wrist image afterward; a
confirmed ramp does not prove alignment.
"""

BASE_MOTION = """
Use move_base for one short local base adjustment through visibly clear space,
such as changing the approach or viewing angle. This is a normalized body-frame
velocity segment, not a distance goal, world navigation or obstacle avoidance.
forward/lateral use local +X/+Y and turn uses positive yaw; the schema states
each command's scale and bound. steps runs at the control rate with no fixed cap
below the episode budget, followed by one zero-base command.
Trunk/arm joint targets and known gripper commands are held; a held object and
the hands move with the base, so allow visible clearance for the whole robot.
completed means all moving actions and the zero-base command were confirmed,
not that measured velocity is zero or any object/task target was reached.
Use the new images to judge the movement and choose the next action; also check
confirmed counters. No displacement or odometry is reported, so do not infer that
a goal was reached from the counts. Re-localize scene points after base movement; old base-frame coordinates are stale.
Each later arm plan checks its own start prerequisites. On partial/unknown
execution, preserve the report and do not replay or assume a stop occurred.
"""

LOCAL_MOTION = """
move_eef_delta makes one small single-arm displacement, within the reported
single-call relative bound in current robot-base xyz, while keeping its EEF
orientation. It resolves the target from the current pose and then uses the
same IK planner and execution as move_to, so the same preconditions apply: the
robot is stationary and the base is stopped. When the current orientation
cannot be kept, the planner retries for position only and reports
orientation_relaxed with the achieved orientation error. A pre-motion rejection is not a
terminal environment state; reassess the request or continue with another
available action tool. Inspect returned images and the measured pose error
afterward; a planned path alone does not prove the motion completed.
"""

FEEDBACK = """
If stop_reason is session_ended, stop calling action tools and finish with a
summary of what was observed. A session ending does not establish task success.
When observation_fresh is false, the returned images are the last available
observation, not proof that the last issued action completed. Retain the reported
issued, confirmed and unconfirmed action counts without replaying pending actions.
Reported tool execution and your finish estimate are not official task scores.
Unknown execution counts and outcomes must remain unknown. The official
evaluator owns environment reset, termination and final scoring.
An explicit zero-action refusal is a request that was not executed: reconsider
the request or select another suitable tool. It does not end the episode. An
execution timeout can leave an action unconfirmed; do not automatically replay it.
Run the complete official episode. While the session remains active, continue
selecting bounded action tools and inspecting feedback; do not finish merely
because the task appears complete or progress seems slow. The only other reason
to finish is to report an actual tool/infrastructure failure.
"""
