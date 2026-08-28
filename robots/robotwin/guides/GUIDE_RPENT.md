# RoboTwin RPent Registered-Tool Guide

This guide is the operational reference for the package-first registered-tool
runtime. The system prompt owns strategy; this guide only defines the tools,
geometry, and compact execution rules needed to apply it.

## Registered tools

- Resources: `list_dir`, `read_text_file`
- Observation: `view_env_state`, `render`, `sample_world_xyz`, `query_world_map`
- Control: `lingbot_act`, `move_to`, `rotate_wrist`, `set_gripper`, `release`
- Terminal: `finish`

Use no legacy `command.json` protocol or direct Env/VLA client. Tool schemas are
authoritative for arguments. Issue one mutation at a time and inspect its fresh
result before the next mutation.

## Observation and geometry

Start with `view_env_state(step=0)`. Its complete task language is
authoritative. Head views identify objects, distractors, destinations, global
relations, and completed subgoals. Wrist views refine grasp, contact, insertion,
and release geometry for the same head-selected candidate.

World maps use `[row,col]`, contain world-frame `[x,y,z]` metres, and may contain
NaN. Query the exact step/view/resolution whose RGB supplied the pixels. Sample
several interior pixels and use robust geometry; an exposed surface point is not
necessarily an object center. Re-observe after occlusion or physical change.

The aloha-agilex state is qpos14: six joints plus one normalized gripper value
per arm. Gripper 0 is closed and 1 is open. `move_to` targets EEF pose; EEF and
TCP differ, so do not send a raw object surface point as an EEF contact target.
Coordinates are world-frame metres and quaternions are `[qw,qx,qy,qz]`.

## VLA and primitives

Every `lingbot_act` must use the full current task language and `use_length=50`.
Use the same-task successful recipe's chunk cadence as a prior. Shorten to one
chunk near contact, instability, or completion. Preserve useful continuous VLA
behavior for bimanual, articulated, hanging, insertion, and tool phases.

Use `move_to` for verified free-space transport, staging, retreat, or one small
correction. Preserve the gripper and orientation while holding unless a change
is intentional. A planned motion, closed gripper, or completed VLA call is not
proof that the semantic subgoal succeeded.

## Analytic execution safeguards

### Planner outcome and residual motion

A successful `move_to` tool call or a generated plan does not prove that the EEF
reached the requested pose. Compare the requested and achieved poses, including
reported residual distance and visible scene change. If planning fails, the
achieved pose makes no useful progress, or the residual remains material, do not
repeat the unchanged target. Retreat or return to a safe height, then inspect
table clearance, the other arm, held-object clearance, perception, and
orientation. Change one supported variable such as approach, waypoint, height,
or orientation before trying again.

### Guarded low approaches

Near the table, a container rim, button, hinge, stacked object, or the other
arm, never queue several unobserved low waypoints. Base every next target on the
pose actually achieved and on fresh images, not on the previously planned pose.
When geometry is clear and only a small vertical correction is needed, preserve
the achieved x/y, orientation, and gripper state and change only z by a typical
0.005-0.010 m increment with at most 8 planner substeps. Execute one increment,
then re-observe. Stop that approach if planning fails, z makes no useful
progress, x/y drifts materially, the hold becomes uncertain, or contact cannot
be interpreted. Prefer one short VLA chunk when terminal motion needs contact
feedback or the correct EEF height is uncertain.

### Wrist rotation and swept volume

`rotate_wrist` keeps EEF xyz fixed, but it does not keep the TCP or a held object
fixed. The EEF-to-TCP offset makes them sweep an arc through the scene. Rotate
only after a verified hold, with surrounding clearance, preferably at a safe
transport height and in small increments. After each rotation, re-observe and
verify the hold, actual object orientation, and target location. Do not assume
that EEF yaw change equals object yaw change, and do not rotate near contact or
inside a constrained opening unless current evidence supports it.

### Physical state shaping before VLA

If VLA has the right task binding but repeatedly cannot advance because of
object orientation, occlusion, reach, or unsafe height, one observed primitive
may shape one major physical variable before returning control to VLA. Examples
include lifting a verified hold to safe clearance, one small safe-height wrist
rotation, or moving a held object to an unobstructed staging pose. Re-observe
after the primitive and hand back with the same complete task language, normally
using one chunk near contact. Do not use empty-gripper pre-positioning or disturb
a near-success state merely to test whether shaping helps.

## Observable gates

- Grasp: target leaves its source and moves with the TCP; gripper closure alone
  is insufficient.
- Transport: hold remains stable through a clearance waypoint and lateral move.
- Support placement: object is on the correct support before release, then stays
  stable and separated while the arm withdraws.
- Container placement: object body crosses the opening and remains internally
  supported after release; rim or nearby placement is incomplete.
- Short contact: the intended button/control visibly changes after one guarded
  contact.
- Articulation: contact is retained while the lid, door, hinge, or knob moves in
  the requested direction.
- Handover: receiver hold is verified before giver release.
- Ranking/stacking: each correct relation is protected from later paths/actions.
- Orientation/hold: requested pose is visible while control is retained; do not
  release if the language requires holding, lifting, or shaking.

Apply only gates that match the current language. A pad, plate, scale, skillet,
or stand is not a container. A task name containing `handover` does not override
an instruction that only requests placement.

## Recovery and budget

After failure, identify the first unmet gate and classify the blocker: wrong
identity/destination, missed grasp, lost hold, planning/collision, insufficient
contact, premature release, unstable placement, or incomplete relation. Change
one meaningful variable and verify. Do not repeat the same ineffective analytic
target or hand-written recovery twice. This limit does not cap `lingbot_act`:
VLA may be called repeatedly when the recipe, current contact, and recoverable
physical state support continuation. Near success, fix only the remaining
blocker instead of replaying the task.

Track both native steps and Planner turns. The runtime step limit is a ceiling;
the same-task recipe and its phase count indicate expected complexity. Preserve
enough budget for remaining phases, verification, one
evidence-based recovery, and `finish`.

Only fresh native `eval_success=true` proves task success. Stop mutations then
and call `finish` exactly once. If no safe meaningful recovery remains, check
fresh status and finish with an honest failure summary.
