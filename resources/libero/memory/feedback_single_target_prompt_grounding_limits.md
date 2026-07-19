---
name: single-target-prompt-grounding-limits
description: Similar motion under different prompts in a single-target scene does not prove language grounding; keep exact prompts and verify the selected object in clutter.
metadata:
  node_type: memory
  type: feedback
---

# Single-target prompt grounding limits

## Rule

Do not infer strong prompt following from a scene that contains only one
visually dominant graspable object. In an observed single-target tabletop run,
the full instruction, a pick-only instruction, and an instruction naming an
absent object produced nearly indistinguishable pick motions. The visual prior
was sufficient to choose the canonical object, so that comparison could not
identify whether language controlled the selection.

## How to apply

- Still pass the exact active instruction or the object-specific prompt required
  by the selected recipe.
- In cluttered or multi-object scenes, verify the selected object in the latest
  wrist and agentview images before carrying it.
- Treat prompt robustness measured in a single-target scene as local evidence,
  not permission to ignore language in a scene with distractors.
- When testing prompt sensitivity, use a scene with multiple plausible objects
  and compare the physical target actually approached or grasped.

## Related memory

[[feedback_pi0_delivery_service]] [[feedback_pi0_false_positive_lift]]
[[feedback_read_image_before_decide]]
