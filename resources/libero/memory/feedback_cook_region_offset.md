---
name: cook-region-offset
description: "Visually localize the usable burner surface instead of aiming at the center of the whole stove platform or a remembered coordinate."
metadata:
  node_type: memory
  type: feedback
  originSessionId: cf2f87e2-37a3-4480-8b9a-4e5666fbf61f
---

When placing an object on a stove, treat the visible burner as the target surface rather than the center of the entire stove fixture.

**Rule**: Localize the burner from the current image. Do not derive its position from a stored stove origin, a fixed offset, or an internal scene description.

**Why**: The stove base and the usable burner can have different visual centers. A target on the outer platform may look close while leaving the object unsupported by the burner.

**How to apply**:
1. Inspect the high-resolution view and identify the round burner or coil requested by the task.
2. Select several interior burner pixels, back-project them, and use a robust center of the valid surface points. Avoid the stove housing and outer boundary.
3. Estimate the held object's current offset from the EEF. For a matching bowl rim-hook grasp, a small positive y compensation can be used as an initial estimate, but verify it from the current images.
4. Approach from above, descend until the object is visibly supported by the burner, release, and retreat with the gripper open.
5. Reinspect the scene and distinguish stable burner support from contact with the surrounding stove platform or rim.

**Related**: [[feedback_bowl_eef_y_offset]] [[feedback_staged_held_object_transport]] [[feedback_read_image_before_decide]]
