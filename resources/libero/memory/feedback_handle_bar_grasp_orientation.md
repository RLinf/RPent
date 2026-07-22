---
name: handle-bar-grasp-orientation
description: "For a horizontal drawer-handle bar, roll the gripper so its pads close across the short axis of the bar, then verify the grasp from images and gripper opening."
metadata:
  node_type: memory
  type: feedback
  originSessionId: 173e389d-6fb4-4726-99ec-b7c6d73b9da0
---

A drawer handle is a horizontal bar. A parallel-jaw gripper must place one pad on each side of the bar's short axis.

- **Wrong:** the closing direction runs along the length of the bar. The fingers approach different parts of the bar length and can close without pinching it.
- **Right:** roll the wrist by about 90 degrees around the approach direction so the pads straddle the bar from above and below, then close across its thickness.

**How to apply**:
1. Use the current image to localize the bar and estimate its long axis.
2. Approach the handle while keeping the gripper opening perpendicular to that long axis. For a horizontal bar, this normally requires the rolled wrist orientation.
3. Inspect the wrist and agent views while closing. Both fingers should surround handle material rather than meeting in empty space.
4. Use the reported gripper opening only as supporting evidence: a nearly fully closed gripper usually indicates an empty close, while a retained opening consistent with the bar thickness supports a grasp.
5. Apply a small pull in the drawer's observed travel direction and verify that the drawer moves with the EEF before continuing.

**Related**: [[feedback_gripper_ctrl_is_finger_position]] [[feedback_move_pose_covarying_reach]] [[feedback_thin_handle_drawer_closure]]
