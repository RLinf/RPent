---
name: stove-turnoff-strict
description: "Stove on/off tasks require a real, visible knob rotation; do not infer completion from the initial appearance or target an internal joint value."
metadata:
  node_type: memory
  type: feedback
  originSessionId: cf2f87e2-37a3-4480-8b9a-4e5666fbf61f
---

**Rule**: Physically rotate the stove knob in the direction requested by the task. Verify a visible state change from images and the official task signal; do not read or target an internal fixture joint value.

**Why**: A stove may initially look off without the robot having performed the required interaction. Likewise, touching the knob without producing a visible rotation is not reliable evidence of completion.

**How to apply**:
- Use a task-matching high-level contact action such as `pi0_doubled("turn on the stove")` or `pi0_doubled("turn off the stove")`, with the budget specified by the matching recipe.
- Compare the knob orientation before and after the action. The wrist or agent view should show deliberate contact followed by physical rotation.
- For turn-on tasks, visible burner illumination is useful supporting evidence when available.
- For turn-off tasks, verify that the knob was deliberately moved toward the off state and that visible burner illumination is gone when applicable.
- If the official task signal remains false, also check whether another requested subgoal is incomplete; do not replace visual verification with hidden fixture state.

**Related**: [[feedback_cook_region_offset]] [[feedback_read_image_before_decide]]
