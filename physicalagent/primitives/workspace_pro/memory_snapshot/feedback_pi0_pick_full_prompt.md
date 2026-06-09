---
name: pi0-pick-full-prompt
description: "For elevated picks under LIBERO-Pro (stove z≈0.93, cabinet-top z≈1.13, drawer interior), use the FULL perturbed task language as pi0_pick.prompt + raise track_obj_lift_thresh to 0.08."
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 329477e9-3629-4a3d-8360-884826ef59f0
---

For elevated/non-table-level pi0_pick under LIBERO-Pro, always set
`prompt` to the full perturbed BDDL `:language` (or paraphrased lan
text) and raise `track_obj_lift_thresh` from 0.05 → 0.08.

**Why:** The generic prompt `"pick up the black bowl"` consistently
fails on stove (z≈0.929) and cabinet-top (z≈1.126) picks — pi0 descends
but the gripper never closes (`final_gripper_opening` stays at 0.05+,
`peak_lift_m` is misleading because the eef itself moved). Switching
to the full task language is what the t4 drawer recipe in [[liberopro-driver-patch]]
already required; the t5–t9 PRO sweep on 2026-05-21 confirmed it on
every elevated pick (t6_task stove, t6_lan cookies-side, t7_task cabinet,
t7_swap stove, t9_task stove, t9_swap cabinet, t9_lan cabinet).

**How to apply:** When writing a hybrid recipe and the bowl init_z is
above ~0.92, set the pick step's `prompt` to the env's
`task_descriptions[0]` verbatim and `track_obj_lift_thresh: 0.08`. The
generic prompt still works for table-level picks at z≈0.90 (t6_swap
bowl_1 next-to-cookies, t8_*). See `recipe_spatial_*_t{6,7,9}_s0.jsonl`
in [[libero-spatial-t0-t9-pro]] for canonical examples.
