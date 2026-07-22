---
name: bowl-eef-y-offset
description: "A rim-hook bowl grasp can leave the bowl center behind the EEF; estimate the current held-object offset and compensate relative to the visually localized target."
metadata:
  node_type: memory
  type: feedback
  originSessionId: cf2f87e2-37a3-4480-8b9a-4e5666fbf61f
---

**Rule**: For a rim-hook bowl grasp, place according to the observed bowl center rather than assuming that the bowl is centered under the EEF.

Use the relationship:

`target_eef_xy = visually_localized_target_xy - held_object_offset_xy`

**Why**: A small bowl is often held from its rim, leaving its center several centimeters to one side of the EEF. Sending the EEF directly to the target center can therefore leave the bowl short of the usable surface.

**How to apply**:
1. Confirm from the wrist and agent views that the bowl is held by a rim-hook grasp.
2. Estimate the current bowl-center-to-EEF offset using visible geometry and back-projected points from the current images.
3. A positive y compensation of roughly 0.04-0.045 m is a useful starting reference only when the current grasp matches this pattern; refine it from what is visible.
4. Visually localize the requested burner, container interior, or other support surface, then apply the measured held-object offset.
5. Do not apply the bowl offset blindly to boxes, cylinders, plates, or centered grasps.

**Related**: [[feedback_cook_region_offset]] [[feedback_pi0_false_positive_lift]] [[feedback_staged_held_object_transport]]
