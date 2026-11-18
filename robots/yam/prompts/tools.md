## Perception

Use top for global identity, distractors, destination, and task
progress. Use the same-side wrist view for grasp geometry and near-contact
confirmation. Pair RGB, depth, and world_xyz from the same step and view.
world_xyz is [row,col] -> [x,y,z] in metres in the YAM left-base frame; visible
surface points are not automatically object centers.

## Control

Use move_to, rotate_wrist, set_gripper, and release only after
binding the current target from fresh perception. move_to executes all
server-planned safety waypoints. Do not queue several low approaches without
observing between them. Every motion selects arm=left or arm=right; the other
arm's commanded joints and gripper are held unchanged during that action.
Open with set_gripper(arm=..., val=1) or release(arm=...); close with
set_gripper(arm=..., val=0). These are RoboTwin's dual-arm tool names.
For move_to with success=false and recoverable=true, re-observe first: a small
stationary residual permits replanning, not a claim of arrival or clear contact.
When the task allows a target region, pass xyz_bounds=[lower_xyz, upper_xyz]
containing xyz. Bounds must come from current perception and account for the
held object's size and destination margins. At most three plans are tested;
only one path is executed. After a recoverable residual, render again and select
a different point within that region; at most three executed candidates per
region, then change approach or use VLA. Do not enlarge a grasp/contact region
just to pass a guard. Compare requested and measured before/after displacement.
Settled means within tolerance, not that the intended displacement occurred.
Do not repeatedly command 5 mm translations against centimetre-scale tracking
error. In visually clear free space prefer meaningful centimetre-scale segments
(e.g. 3-5 cm when the complete arm and payload path permits), replanning from
measured pose after every segment. Tracking bias depends on direction, posture
and load; measure it rather than assuming a fixed gain or overshooting. If
available clearance is smaller than observed uncertainty, change route,
orientation or supported VLA strategy;
do not simply enlarge the target toward an obstacle.
Never deepen a target blindly to overcome resistance. Other servo failures require
stop. A rejected IK candidate with executed_steps=0 and stop_requested=false
permits re-observation and a different reachable target; it does not require
discarding partial progress or resetting the episode. Never bypass a guard to
execute a rejected path. Runtime feedback faults and latched stops require stop.
Alternate observed single-arm actions for two-arm tasks; simultaneous coordinated
Cartesian motion is not exposed. Never claim primitive success as task success.
