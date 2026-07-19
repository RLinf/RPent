---
name: staged-held-object-transport
description: Carry a verified held object with lift, horizontal transport, and descent stages while compensating the observed object-to-EEF offset.
metadata:
  node_type: memory
  type: feedback
---

# Staged held-object transport

## Rule

After a grasp, verify that the object is actually held before planning the
carry. Move in three stages: lift to scene-derived clearance, translate through
clear horizontal waypoints, then descend over the target. Large diagonal moves
with a held object are more likely to stall, slip, or sweep nearby objects.

## Held-object offset

Measure the held-object offset from the current images whenever placement
tolerance is tight:

`held_offset_xy = object_xy - eef_xy`

`target_eef_xy = target_object_xy - held_offset_xy`

Do not assume that the object center is directly below the EEF. Rim grasps,
handle grasps, and asymmetric finger contact can produce several centimeters of
offset.

## Execution

1. Confirm the grasp with EEF/gripper diagnostics and the latest wrist and
   agentview images.
2. Lift vertically until the held object, not only the EEF, clears nearby
   fixtures and distractors.
3. Translate through one or more high-clearance waypoints. For objects that
   slip during travel, use shorter segments and a conservative `step_clip`
   around `0.012-0.020`, subject to the object-specific recipe.
4. Re-read the current images before the final descent; earlier arm motion may
   have disturbed the target or nearby objects.
5. Descend until the object is supported by the target surface or the
   controller reaches the expected contact stall, then release. Releasing from
   unnecessary height adds lateral drift.

Do not override a policy-owned grasp with `set_gripper` when the selected
object-specific recipe explicitly forbids it. Use current scene geometry rather
than recorded world coordinates.

## Related memory

[[feedback_pi0_false_positive_lift]] [[feedback_bowl_eef_y_offset]]
[[feedback_read_image_before_decide]] [[feedback_move_pose_covarying_reach]]
