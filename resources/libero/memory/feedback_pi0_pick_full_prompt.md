---
name: pi0-pick-full-prompt
description: "For elevated picks under LIBERO-Pro (stove z≈0.93, cabinet-top z≈1.13, drawer interior), use the full active task language as pi0_pick.prompt and verify the grasp visually."
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 329477e9-3629-4a3d-8360-884826ef59f0
---

For elevated/non-table-level pi0_pick under LIBERO-Pro, always set
`prompt` to the full active task instruction (or its current paraphrased
language) rather than reducing it to a generic object-only prompt.

**Why:** The generic prompt `"pick up the black bowl"` consistently
fails on stove (z≈0.929) and cabinet-top (z≈1.126) picks — pi0 descends
but the gripper never closes (`final_gripper_opening` stays at 0.05+,
`peak_lift_m` is misleading because the eef itself moved). Switching
to the full task language is what the t4 drawer recipe in [[feedback_liberopro_driver_patch]]
already required; the t5–t9 PRO sweep on 2026-05-21 confirmed it on
every elevated pick (t6_task stove, t6_lan cookies-side, t7_task cabinet,
t7_swap stove, t9_task stove, t9_swap cabinet, t9_lan cabinet).

**How to apply:** When the current images show the target on an elevated
surface or inside an articulated container, set the pick step's `prompt` to
the active `task_language` verbatim. A generic prompt can still work for an
isolated table-level object. Re-localize the object in the current scene,
use only parameters exposed by the current `pi0_pick` schema, and verify the
returned grasp with supported diagnostics plus the latest images.
