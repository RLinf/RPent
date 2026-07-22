---
name: swap-perturbs-fixtures
description: "Layout changes can relocate whole fixtures; identify the requested fixture and localize its placement surface from the current images."
metadata:
  node_type: memory
  type: feedback
  originSessionId: cf2f87e2-37a3-4480-8b9a-4e5666fbf61f
---

**Rule**: A changed layout may move the stove, cabinet, rack, or other fixture to a different part of the workspace. Never reuse a remembered fixture coordinate. Resolve the requested fixture from the task language, then localize its visible placement surface in the current images.

**Why**: A grasp can succeed while the later placement fails because the robot carries the object toward the old location of the requested fixture. The reusable lesson is to re-localize the fixture, not to memorize a layout-specific coordinate.

**How to apply**:
1. Identify the requested fixture and surface from the task language, such as a burner, cabinet top, rack shelf, or drawer interior.
2. Inspect the current high-resolution view and distinguish that surface from nearby fixtures with a different appearance or function.
3. Select several pixels inside the visible usable surface and back-project them. Use a robust center of the valid points instead of a boundary pixel.
4. After grasping, estimate the held object's offset from the EEF using the current images and compensate the placement target when needed.
5. Carry above visible obstacles, descend toward the perceived surface, release, and retreat with the gripper open.
6. Inspect the resulting image to confirm that the object is supported by the requested fixture rather than a neighboring surface.

**Related**: [[feedback_cook_region_offset]] [[feedback_bowl_eef_y_offset]] [[feedback_region_ranges_table_frame]] [[feedback_read_image_before_decide]]
