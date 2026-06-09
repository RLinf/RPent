---
name: project-pi05-libero-prompt-blind
description: pi05_libero130_fullshot/30000 ignores language prompt on libero_spatial — picks the canonical scene object regardless of prompt (incl. negative control naming an absent object)
metadata: 
  node_type: memory
  type: project
  originSessionId: 9df802c0-b380-4d01-8d48-706e324854e2
---

`pi05_libero130_fullshot/30000` on libero_spatial tasks 0/1/2 (3 seeds each, 27 rollouts total via `physicalagent/primitives/test_pick_generalization.py`):

- `full` prompt: 9/9 libero_term=True (baseline works)
- `pick_only` ("pick up the {OBJ}" — sub-instruction): 9/9 pick stroke 0.248–0.273 m, gripper closed
- `pick_negctrl` ("pick up the alphabet soup" — not in libero_spatial scenes): **9/9 pick stroke 0.245–0.269 m, gripper closed — INDISTINGUISHABLE from `pick_only`**

**Conclusion:** the checkpoint is effectively vision-driven on libero_spatial — the language channel has near-zero effect because the scene only contains one canonical target (a black bowl) and SFT downweighted language attention.

**Why:** confirmed experimentally on 2026-05-18 by adding a negative control prompt naming an object guaranteed absent from libero_spatial.

**How to apply:**
- For libero_spatial: `driver.pick(...)` works reliably regardless of `object_text` — usable as a "trigger a pick" executor in primitive-DSL plans, but do not rely on prompt to disambiguate.
- To test true prompt-following: switch to `libero_object` (8 distinct items in one scene), where vision alone cannot solve the task. Same driver, just `task_suite_name="libero_object"`.
- Driver lives at `/mnt/public/jxqiu/physicalagent/physicalagent/primitives/`.
