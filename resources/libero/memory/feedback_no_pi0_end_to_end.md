---
name: no-pi0-end-to-end
description: "STRICT_HYBRID_GUIDE Rule 1 forbids pi0_end_to_end — Pi0 only does the gripper grasp; LLM must do all motion planning and release. Recovery must remain inside the current episode."
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 0074d341-bdcd-41b6-b40e-036b75648dc3
---

User instruction (2026-05-19): **Pi0 may ONLY be used for the gripper grasp
(`pick`). LLM must script every motion (`move_to`) and every `release`.**
The `pi0_end_to_end` regime is explicitly forbidden.

**Why:** the whole point of the strict-hybrid benchmark is to demonstrate
that an LLM + a VLA-as-pick-primitive can solve manipulation tasks; if Pi0
also handles the place, we've measured Pi0, not the hybrid. The user wants
the audit corpus to reflect LLM-place performance honestly.

**How to apply:**
- Encode as Rule 1 at the top of
  [strict hybrid guide](../../../robots/libero/guides/strict_hybrid_guide.md).
  Use current images, re-localization, and in-episode retries when a placement
  misses.
- Never call `pi0_pick` with `lift_thresh=99 gripper_closed_thresh=99` (the
  pattern that lets Pi0 run the full task chunked) for the place step. That
  pattern remains valid only for `pi0_doubled` non-pick VLA skills (knob
  turn in t2, drawer close in t3) where OSC scripting genuinely cannot
  execute the skill.
- If a strict placement fails, inspect the current images and recover from the
  current physical state rather than handing the place to Pi0. If the state is
  unrecoverable, document the strict failure instead of escalating.
- Existing `pi0_end_to_end` audit JSONs (t5 from 2026-05-19, prior t9) are
  grandfathered but new attempts must NOT produce that regime. The t5
  audit should be redone strict before treating the libero_10 corpus as
  Rule-1-compliant.

Use [[feedback_staged_held_object_transport]] for the scripted carry and release
that follows a verified policy grasp.
