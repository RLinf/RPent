---
name: rotate-pitch-orientation-control
description: Use rotate_pitch to tilt the gripper around world X for low-clearance insertion, then co-vary position and pitch with move_pose through the opening.
metadata:
  node_type: memory
  type: feedback
---

# Rotate-pitch orientation control

## Primitive meaning

`rotate_pitch` tilts the gripper around the world X axis while holding its
position and yaw approximately fixed. Use the exposed primitive rather than
writing raw action dimensions.

- `target_pitch` sets an absolute pitch.
- `delta_pitch` applies a relative change.
- `pitch = 0` points the gripper down in the canonical pose.
- Positive pitch leans the gripper axis toward world `+y`.
- Negative pitch leans it toward world `-y`.

The implementation derives pitch as `atan2(R[1,2], -R[2,2])`. A positive pitch
near `0.9` rad has been sufficient to move the wrist body under a low ceiling;
near-horizontal insertion recipes may require their documented larger value.

## Narrow-opening pattern

1. Verify the object is held and move to a clearance pose outside the opening.
2. Rotate toward the orientation required by the opening.
3. Use `move_pose` to co-vary xyz and pitch while entering. A decoupled
   `move_to` followed by rotation can return to the same singular approach.
4. Release only after the object is supported inside the destination.
5. Retreat clear of the opening before restoring `target_pitch=0.0`.

Use scene-relative opening geometry and the object-specific recipe. Do not copy
recorded cavity coordinates into a different layout.

## Related memory

[[feedback_move_pose_covarying_reach]] [[feedback_rotate_wrist_yaw_sign]]
[[feedback_no_pi0_end_to_end]] [[horizontal_bottle_drawer_insertion_and_close]]
