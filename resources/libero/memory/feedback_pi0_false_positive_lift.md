---
name: pi0-false-positive-lift
description: Treat pi0_pick success as provisional because EEF ascent plus gripper closure can still be an empty grasp; verify with supported diagnostics and current images.
metadata: 
  node_type: memory
  type: feedback
  originSessionId: cf2f87e2-37a3-4480-8b9a-4e5666fbf61f
---

**Rule**: For `pi0_pick` calls in unfamiliar layouts, treat `success: True` as a provisional primitive result rather than proof that the object is held. Confirm the grasp using the supported EEF/gripper diagnostics together with the latest wrist and agentview images before starting the carry.

**Why**: `pi0_pick` succeeds after a descend-close-ascend pattern. The ascent signal measures EEF motion, not object motion, so an empty close followed by an EEF lift can look successful in the numeric result. Unfamiliar layouts make this failure more likely because the visual policy may close beside the intended object.

**How to apply**:
- Use only the parameters exposed by the current `pi0_pick` schema. Do not inflate `lift_thresh` to suppress the normal EEF-ascent terminator.
- Read `post_min_ascent_m` / `peak_lift_m` and `final_gripper_opening` as supporting signals, not proof that the object is held.
- Inspect the latest wrist image for material between the fingers and the latest agentview image for displacement from the source location. If the views disagree with `success`, re-localize and recover within the current episode instead of carrying an assumed grasp.

**Related**: [[feedback_pi0_pick_full_prompt]] [[feedback_pi0_delivery_service]]
