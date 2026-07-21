---
name: failure-forensics-render-images
description: "After an action sequence fails, inspect the relevant view_driver_state images and compare what happened with what was expected; do not tune numbers blindly."
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 23bda317-5513-4ade-ae34-7d3f3988aeef
---

When a strategy fails twice, the next action must be **`Read` the failed-step PNGs**, not "tune step_clip/tol/qpos one more time." The image is the debugger.

**Mandatory diagnostic render set after any failed (suite, task, scene) attempt:**

1. `view_driver_state({"step":0})` — inspect `image_cam_hi_path` and verify the initial scene matches the state in `states.json[0]` (catches unexpected distractors, unexpected fixtures, perturbation effects).
2. Image right after each `pi0_pick` — confirm object is actually held (gripper around it, lifted) vs hanging off a finger or untouched. `chunks_used` reports budget consumption, not grasp truth.
3. Image right before every `release` — confirm object is over target xy zone, not 5 cm off due to stalled `move_to`.
4. Image right after every `release` — confirm landed where physics predicted, not bounced/rolled/tipped.
5. For articulation / contact tasks: image immediately before AND after each push or close — physical-contact failures (the door swing sweeping the object out of the cavity, a drawer that didn't actually move because the push lost contact) leave visually obvious evidence that JSON only hints at via a z drop.

**Write the diagnosis in words.** For each image: one sentence on what you see, one sentence on what you expected. The mismatch is the bug.

**Why:** Numerical parameter tuning without image inspection is the most reliable way to waste session-hours on the wrong abstraction level. The libero_10 t3 drawer-with-bowl bug took multiple sessions of "try smaller step_clip, try different drop xy" before rendering the post-close image revealed the drawer floor had moved without carrying the bowl (object left behind on the table) — telling us the close needs a *continuous* push that keeps the drawer wall in contact with the object. Two minutes of visual reasoning would have caught this on attempt 2.

**How to apply:** the iteration heuristic ladder (robots/libero/guides/strict_hybrid_guide.md §"Iteration heuristics") now includes step 8a — after two failed retries, *stop tuning numbers* and render the failed-step PNGs. Describe what's on screen. If the visual layout disagrees with your mental model of the task (e.g. drawer slides in -y not +y, plate is in front not back, mug is held by rim not body), the disagreement is the fix — not a different parameter.

Codified in robots/libero/guides/strict_hybrid_guide.md §Rule 0 "Failure forensics" subsection.

## Related memory

[[feedback_read_image_before_decide]] [[feedback_osc_push_mujoco_nan]]
