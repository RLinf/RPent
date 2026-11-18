## TOOL_ACCESS

Use view_env_state to inspect recorded head and wrist images.
render captures a new synchronized observation without commanding robot motion.
Each non-readonly tool also captures a new observation after execution.
Observations and motion feedback include episode_status: step is the native
action count, step_limit is the task's total action budget, and remaining_steps
is the budget left at capture time. The top-level step is an observation index,
not an action count. Historical observations retain their historical budget.

## PERCEPTION

Use sample_world_xyz(view, pixels, step) for [row,col] pixels
or query_world_map(view, bbox, step) for a half-open [r0,c0,r1,c1] region.
Views are head, left_wrist and right_wrist; use the same view and step as the
source image. Coordinates are world-frame metres. Invalid depth is not a point.
segment(text_prompt, camera, step) uses SAM3 on that saved image. Cameras are
cam_head, cam_left_wrist and cam_right_wrist. Its point_rc lies in the mask;
centroid_rc may lie outside a concave mask. Use point_rc with back_project at
the returned step_idx and camera. A visible surface point is not necessarily
an object centre or a safe grasp point. Re-perceive after motion/reset.

For SAM3, describe visible colour, shape and relative location, e.g.
"the pink bottle on the left of the table", not internal asset or brand names.
Inspect the returned red-mask overlay before using the point. A high score
alone does not prove object identity. Disambiguate look-alikes with head and
wrist views. If no mask is found, revise the description or select a visible
pixel manually; never invent coordinates from a failed detection.
Segmentation returns image pixels, not world coordinates; follow with
back_project when you need a visible surface's world position.

## Available operations

move_to(arm, xyz, quat=None, gripper=None, substeps=25) plans a full pose.
Quaternion order is wxyz; omitted orientation and opening are held.
rotate_wrist applies relative yaw about world Z, in degrees, with the same
substeps selection rules. Each executed waypoint consumes one native action
step (0.04 seconds of simulation). Fewer waypoints give less execution time;
completed means the selected commands ran, not that the target pose was reached.
set_gripper(arm, val, steps=10) uses 0=closed, 1=open; release defaults to 1.
Intermediate openings are control targets, not guaranteed physical widths.
Inspect completed, executed_steps and stop_reason. final_dist_m is measured
position error, not a task-success verdict. On failure, inspect and decide
the next action; do not assume a failed call left the robot unmoved.

## Primitive control

- move_to: measured approach staging, verified free-space transport, withdrawal,
  return, or a specific geometric correction with understood contact geometry.
- rotate_wrist: an intentional world-Z orientation adjustment with clearance
  for the fingers and held object; it is not a general tilting command.
- set_gripper: deliberate opening adjustment or closure after alignment. Keep
  the current hold unless the phase requires changing it.
- release: transfer support to the intended surface, fixture or receiving hand
  once observations support that transfer; verify the result afterward.

For move_to and rotate_wrist, choose substeps from 1, 5, 10 or 25:
- 25 (default): ordinary transport, larger position or orientation changes,
  or uncertainty about the execution time needed.
- 10: short free-space moves or small orientation adjustments with clear space.
- 5: very small geometric corrections.
- 1: tiny corrections or holding a target; never jump to a distant pose.
Do not use 0 or values above 25. Preserve necessary detours when choosing
targets and sampling; fewer steps do not make a large target change gentle.
After execution, inspect the achieved position, orientation and held object
before deciding whether to continue or change the approach. Minimize the total
steps needed for the phase, not just the steps in an individual call.
