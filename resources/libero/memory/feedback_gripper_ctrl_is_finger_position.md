---
name: gripper-ctrl-is-finger-position
description: "RPent high-level primitives use gripper=+1 to close or hold and gripper=-1 to open; verify the grasp with supported gripper telemetry and current images."
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 173e389d-6fb4-4726-99ec-b7c6d73b9da0
---

# RPent gripper command convention

During evaluation, use the signed gripper commands exposed by RPent's high-level
primitives.

| Action | Command | Applies to |
| --- | --- | --- |
| Close or maintain a hold | `gripper=+1` | `set_gripper`, `move_to`, `move_pose`, `rotate_wrist`, and `rotate_pitch` |
| Open | `gripper=-1` or `release()` | Placement, retreat, and approach without a held object |

Keep `gripper=+1` on every transport or orientation stage while carrying an object.
After closing, treat gripper closure as provisional: confirm the hold from the
supported gripper-opening telemetry and the latest wrist and agent-view images. A
closed gripper without corresponding object motion may still be an empty grasp.

See [[feedback_handle_bar_grasp_orientation]].
