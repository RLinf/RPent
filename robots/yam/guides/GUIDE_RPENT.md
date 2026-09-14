# YAM RPent Guide

YAM uses three views: `top`, `left`, and `right`. The left and right cameras are
wrist cameras. The state and action layout is absolute `qpos14`:
`[left_q0..q5, left_gripper, right_q0..q5, right_gripper]`, with grippers
normalized as `0=closed, 1=open`.

Use `view_env_state` first. The top image is the global semantic view. Wrist
images refine grasp geometry for the same selected object; do not silently switch
to a look-alike visible in a wrist camera. `sample_world_xyz` and
`query_world_map` read persisted same-frame depth projections in the YAM
left-base frame.

Without a VLA endpoint, both evaluation and Explore use primitives only;
`pi05_act` is absent from the available tools. Once a trained endpoint is supplied,
`pi05_act(chunks=1, use_length=30, prompt=None)` runs that policy. The server
returns 30 absolute joint targets; execution alone chooses 1–30 steps. Omitting
`prompt` uses the full task language. Arbitrary subtask prompts are unvalidated.
`move_to` delegates reachability,
IK, table protection, and waypoint generation to the env server, then executes
the full returned waypoint list.
The site may enable bounded outer joint PI after that path. Its `servo` result
reports actual convergence, correction bias, timeout or no-progress stopping.
`success=false, recoverable=true` means a stationary residual within 15 mm,
with bounded joint and rotation errors and live feedback; correction stopped
without latching an episode stop. It does not prove free space or absence of
contact. Observe again before any action. For a clear free-space waypoint,
use the measured pose to plan the next segment without requiring exact arrival.
Compare actual before/after displacement with the request. Do not repeat 5 mm
translations when observed tracking error is centimetre-scale. Use meaningful
centimetre-scale free-space segments with verified arm/payload clearance, then
replan from measured pose. Measure tracking error for each direction, posture and load rather than
assuming a fixed compensation factor. Near objects, if tracking
uncertainty exceeds clearance, change route/orientation or use a supported policy
segment instead of increasing the target into contact. Never enlarge a target to
force through an obstacle or change site limits. Other servo failures still stop.
Reaching the target during PI does not prove it remains there after a stop:
the stop handler switches to measured-pose hold and discards that correction.

The dual-arm names follow RoboTwin: `move_to(arm="left"|"right", xyz=...,
quat=..., gripper=None, substeps=25)`,
`rotate_wrist(arm=..., delta_yaw_deg=..., gripper=None, substeps=25)`,
`set_gripper(arm=..., val=..., steps=10)`, and `release(arm=..., val=1.0, steps=10)`.
Open with `val=1` (or `release`); close with `val=0`. There are no separate
`open`/`close` aliases. LIBERO's signed controller gripper actions are not YAM
gripper values. `gripper_val` reports measured position; primitive execution
success does not prove a secure grasp.

Each geometric action selects one arm and preserves the other arm's previous
accepted target and gripper. The control machine executes all
waypoints at 30 Hz with per-step feedback and stop handling. This differs from
RoboTwin's per-waypoint Agent RPC. `substeps` can add samples but never removes
server-planned safety waypoints; use the returned executed step count. Two arms
can work in observed alternating phases, but there
is no simultaneous dual-arm Cartesian primitive or coordinated collision planner.

All xyz targets use `left_base` as the shared world frame, in metres. Pose
quaternions are **wxyz**. Each arm's native FK has its own base; the server
converts right-arm poses through the calibrated base-to-base transform.
The installed `yam + flexible_4310` MuJoCo model places `grasp_site` 100 mm
along +Z of `tcp_site`, with the same orientation. In the source gripper XML
this is `pos="0 0 -0.1", quat="0 1 0 0"`; the combined model also applies
the gripper mounting transform. Do not interpret the source offset as a
translation along an arm-base axis. Finger slides run along opposite Y_site
directions. The modeled soft tips extend toward +Z_site from the mount;
therefore +Z_site is the model's forward direction toward the tips, not
-Z_site or -X_site. This is an offline geometry contract, not a verified
physical approach or contact point. Confirm the installed fingers, TCP,
calibration, and clear approach direction before contact. Choose an orientation
supported by observed reachability; strict vertical top-down grasping is not
guaranteed to be reachable.
Do not invent a new TCP offset on the Agent side. The supplied quaternion is
an exact request: the planner may fail instead of silently changing orientation.

When enabled in the site configuration, the model guard checks sampled arm link
self/two-arm collisions and the configured finite table. Held objects, wrist
cameras, cables, fixtures, and contact forces remain outside that model. Start with one arm in a cleared
workspace and keep the other arm in its operator-confirmed staging area.
When a view reports `world_xyz_limitation`, inspect it before attempting pixel
localization. Invalid depth or an unaligned wrist frame is not a usable target.
Calibration files must match the current camera mounts and TCP model. A saved
matrix or a passing geometry test does not establish current physical accuracy.

The operator's first environment observation starts hardware. The Dashboard and
diagnostic consoles check `env.is_started` before observing; they refuse to
start hardware themselves. With
`gripper_limits` unset, the installed SDK drives each gripper in both directions
to detect its stops before normal operation; this is not a motion-free hold.
Hardware takeover requires the operator's confirmed workspace and startup
authorization, including empty, unobstructed grippers for this calibration.

For free-space manipulation, verify a hold before transport. For contact-rich
grasping, re-grasp, insertion, tool use, or bimanual coordination, proceed only
within demonstrated primitive capabilities or a connected trained VLA's validated
capabilities. An untrained or absent VLA is not a fallback. Re-observe
after every motion and protect already achieved task relations.

Success is not inferred from an agent statement or primitive return. `finish`
checks the latest env `eval_success` flag. During early real-robot exploration,
that flag is expected to come from an operator receipt or a station-specific
success checker exposed by the env server.

Evaluation has one operator-prepared attempt and read-only memory. Exploration
can request reset only after an operator has restored the scene and written a
fresh ready receipt for the advertised episode ID. The operator's `start` command
is for the first episode before Agent recording; `ready` only writes the receipt
for the Agent's retry reset. Reset does not home, fold,
clear motor faults, or turn torque off. Never write the operator receipt file,
call a hidden API, or substitute an Agent judgement for an operator verdict.
Archive failed attempts; export only the current successful attempt's recipe.

During exploration, use `reset` to consume ready for both the initial attempt and
retries. Waiting for ready or a verdict lasts at most 20 seconds per tool call;
`pending` leaves the session and attempt count unchanged. An operator failure
allows another prepared attempt within budget; abort means finish failure now.
The operator's separate `reset_pose` command moves to the recorded start pose.
It is a motion, not a scene reset or an Agent tool.

Planner session close requests hold only. The environment service's `shutdown`
or Ctrl+C moves to configured `park_on_close` home, verifies convergence, then
closes motor output. A failed home keeps the service/runtime alive for an operator
retry. Abrupt process termination or power loss cannot run this sequence.

Use one official MemoryManager corpus. A/B bag rules belong to their distinct
task IDs; shared perception and motion lessons must not imply a universal bag
mapping. A successful run contributes one independent piece of evidence. The
runner filters recipes to that successful episode, merges the inbox with the
official conflict archive rules, and rebuilds the index for the next run.

### Local exploration recovery

A zero-execution planning rejection or explicitly recoverable residual does not
end the episode. Reobserve, preserve progress and change the next target or
approach using measured state. Visual uncertainty pauses the uncertain action,
not exploration: compare fresh top and both wrist views, identify missing
clearance evidence, and attempt a supported observation or small recovery before
asking for routine operator judgement. A retreat is not automatically safe when
a handle may be caught. Do not push through unknown contact or release without
supported destination clearance. Use at most three distinct local recovery cycles
per unresolved obstacle, recording hypotheses and results in the existing inbox.
Escalate when no supported next action exists, ambiguity persists after those
cycles, or a real fault/stop requires the operator. Local cycles do not increment
episode attempts, change guards, or replace authoritative success receipts.

Use `status()` for fresh episode rules and action counters. `can_continue` only
checks program state; it does not establish visual clearance. `move_to` accepts
optional `xyz_bounds=[lower_xyz, upper_xyz]` containing the target. A bounded
set of candidates is checked, not every point in a continuous region; only a
validated path is executed. Bounds must respect payload and destination margins.
