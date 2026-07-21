---
name: thin-handle-drawer-closure
description: Close a shallow slide drawer by verifying object fit, maintaining low front-panel contact, and using short capped motions instead of one sustained push.
metadata:
  node_type: memory
  type: feedback
---

# Thin-handle drawer closure

## Diagnose before pushing

Separate insertion from closure. An object can be inside the visible cavity yet
still block the drawer from seating. Verify that the object lies below the rim,
does not bridge the cavity, and is aligned with the drawer's longer interior
axis. Long objects may need to be rotated horizontally before insertion.

Do not declare a drawer unreachable from visual flush or from the fixture body
center. Re-localize the current cavity, front panel, and handle, and reason about
the actual direction of drawer travel.

## Closure pattern

1. Place the object near the cavity center with enough clearance for the front
   panel to pass it.
2. Approach the front panel from outside the cabinet and below any overhanging
   rim when geometry permits.
3. Maintain panel or handle contact with a short capped motion. For the known
   contact-sensitive pattern, `max_steps` around `80-120` and `step_clip` around
   `0.010-0.018` are safer than one long forceful push.
4. Back out, inspect the current images, re-acquire contact, and continue only
   while the simulation remains stable. A visually flush drawer may still need
   a small final seating motion.

If a vertical approach walls above the handle, use the orientation-aware
`move_pose` or low tilted-hook strategy documented by the matching drawer
recipe. For a horizontal handle grasp, roll the gripper so the finger pads
straddle the bar across its axis. Never compensate for missed contact by
holding one sustained OSC push against the cabinet.

## Related memory

[[feedback_osc_push_mujoco_nan]] [[feedback_handle_bar_grasp_orientation]]
[[feedback_move_pose_covarying_reach]]
[[horizontal_bottle_drawer_insertion_and_close]]
[[single_rollout_drawer_insert_and_close]]
