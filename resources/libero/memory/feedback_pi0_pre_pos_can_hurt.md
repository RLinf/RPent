---
name: pi0-pre-pos-can-hurt
description: "Choose Pi0's start pose from the current scene: use the default home pose for isolated familiar objects, and low object-relative pre-positioning only for clutter-driven grounding ambiguity."
metadata: 
  node_type: memory
  type: feedback
  originSessionId: cf2f87e2-37a3-4480-8b9a-4e5666fbf61f
---

**Rule**: For some PRO task variants where Pi0 fails to pick a normally-graspable object, **try `pi0_pick` from the default home pose with NO pre-pos `move_to`**. Pi0 has its own internal approach trajectory; pre-positioning the gripper near the object can break it.

**Why**: libero_goal t4 task ("Put the plate on the top of the drawer") was failing with both sub-instr "pick up the plate" and full official task instruction when preceded by a pre-pos move_to (gripper centered above plate). After 3 strategies and image inspection, removing the pre-pos and letting Pi0 plan from the default eef home (-0.21, 0, 1.17) succeeded on the very next attempt; the post-action images showed the plate held and displaced with the gripper.

The hypothesis: Pi0's policy was trained on episodes that START from a specific pose distribution. A pre-pos move_to puts the eef close to the object but in a config Pi0 doesn't know how to act from — its policy outputs garbage or no-op. Starting from the trained home pose lets Pi0 use its learned approach trajectory.

**How to apply**:
- For objects Pi0 SHOULD know how to grasp (training-distribution objects: bowl, plate, cream_cheese, wine_bottle, mug, moka_pot in libero_goal/object scenes), try `pi0_pick` from default home pose FIRST.
- In a cluttered or occluded current scene where the named target is not visually dominant, use a low object-relative pre-position as the exception: center on the currently localized object and set `z = object_top_z + 0.18`, with `tol=0.006-0.008` and a tight `step_clip`. Reference heights were about `z=0.65` for table-level cans and `z=1.00` for books on a study table; derive the value from the current object top rather than copying either constant.
- For swap/task perturbations where pre-pos previously helped: still try home-pose first; pre-pos may now hurt because Pi0's internal trajectory expects to start from home.

**Related**: [[feedback_pi0_delivery_service]] [[feedback_pi0_pick_full_prompt]] [[feedback_scripted_pick_limits]]
