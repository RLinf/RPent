---
name: no-pi0-end-to-end
description: "STRICT_HYBRID_GUIDE Rule 1 forbids pi0_end_to_end — Pi0 only does the gripper grasp; LLM must do all motion planning and release. Multi-episode iteration (reset + retry) is the permitted escalation, not Pi0 fallback."
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
  [STRICT_HYBRID_GUIDE.md](../../../mnt/public2/zhangyixian/RLinf_agentic/examples/embodiment/primitives/STRICT_HYBRID_GUIDE.md);
  also Rule 2 there allows multi-episode `reset`-and-retry iteration, which
  is the *permitted* escalation when one episode's placement misses.
- Never call `pi0_pick` with `lift_thresh=99 gripper_closed_thresh=99` (the
  pattern that lets Pi0 run the full task chunked) for the place step. That
  pattern remains valid only for `pi0_doubled` non-pick VLA skills (knob
  turn in t2, drawer close in t3) where OSC scripting genuinely cannot
  execute the skill.
- If a strict placement fails, `reset` and try a fresh episode with a
  different plan rather than handing the place to Pi0. After honest
  exploration, document strict failures as such instead of escalating.
- Existing `pi0_end_to_end` audit JSONs (t5 from 2026-05-19, prior t9) are
  grandfathered but new attempts must NOT produce that regime. The t5
  audit should be redone strict before treating the libero_10 corpus as
  Rule-1-compliant.

Cross-ref [[libero-10-t0-t5-done]] for the audit table that includes the
one current `pi0_end_to_end` entry (t5_s0) needing replacement.
