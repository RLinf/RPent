---
name: pi0-false-positive-lift
description: pi0_pick reports success=True on eef-rise alone — bowl may never have been grasped; set lift_thresh=0.5 + track_obj_lift_thresh=0.08 to force track-object validation
metadata: 
  node_type: memory
  type: feedback
  originSessionId: cf2f87e2-37a3-4480-8b9a-4e5666fbf61f
---

**Rule**: For pi0_pick calls in unfamiliar layouts (PRO swap scenes, novel scene compositions), set `lift_thresh=0.5` (effectively disable the eef-ascent shortcut) and rely on `track_obj_lift_thresh=0.05..0.08` (object actually lifted). Then always read the diagnostics `track_obj_init_z` and `track_obj_final_z` and only treat `track_obj_final_z - track_obj_init_z > 0.05` as a real grasp — `success: True` is unreliable.

**Why**: `pi0_pick`'s exit condition is `descent_done AND ascended AND closed`, where `ascended = post_min_eef_z - min_eef_z >= lift_thresh`. This measures eef ascent ONLY. If Pi0 closes the gripper around empty space then lifts the empty gripper, `ascended` triggers and `success: True` is returned even though `track_obj_init_z == track_obj_final_z`. In familiar (base) layouts Pi0's visual policy lines up so this rarely happens; in swap layouts where the object is in a new place, Pi0 frequently misses and false-positives. Spent ~1 hour debugging goal_swap t1 before realizing the bowl never moved despite repeated "success".

**How to apply**:
- For swap scenes: `track_obj_lift_thresh: 0.08, lift_thresh: 0.5, gripper_closed_thresh: 0.01`
- Always check the pi0_pick step entry's `result.diagnostics.track_obj_final_z` in `states.json` after pi0_pick before continuing — gating on `success` flag alone causes downstream releases to drop nothing onto the target site.

**Related**: [[pi0-pick-full-prompt]] [[pi0-delivery-service]]
