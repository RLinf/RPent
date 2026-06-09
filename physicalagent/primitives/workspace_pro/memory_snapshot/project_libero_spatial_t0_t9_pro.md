---
name: libero-spatial-t0-t9-pro
description: "libero_spatial t0–t9 hybrid audit complete at seed=0 across all four PRO perturbations (base, _task, _swap, _lan) — 30/30 hybrid pass, Pi0 baseline 8/30 fails on _task/_swap (0/10 on _lan)."
metadata: 
  node_type: memory
  type: project
  originSessionId: 329477e9-3629-4a3d-8360-884826ef59f0
---

libero_spatial all 10 base tasks audited at seed=0 across all four
LIBERO-Pro perturbations on 2026-05-21. Saved in
`physicalagent/primitives/workspace_pro/results_spatial_pert/`.

**Why:** This closes the spatial axis under PRO at seed=0 — the
LIBERO-Pro paper headlines were generated from the {_task, _swap}
cells, and we now have a full grid of hybrid vs Pi0 fullshot on those.

**How to apply:** When extending to libero_object / libero_goal /
libero_10 under PRO, port the elevated-pick recipe pattern
(see [[pi0-pick-full-prompt]] + offset-compensated place) from
`recipe_spatial_*_t{5,6,7,9}_s0.jsonl`. Stove, cabinet-top, drawer
patterns reused across spatial; expect them to port directly.

**Headline:** Hybrid 30/30. Pi0 fullshot per-axis (10 tasks each):
- P1 Task: 4/10 pass
- P2 Position: 4/10 pass
- Semantic: 10/10 pass
Of the 8 cells where Pi0 fullshot fails (in {_task, _swap}), hybrid
succeeds on 8/8.

Reports: `REPORT_spatial_t0.md`, `REPORT_spatial_t1.md`,
`REPORT_spatial_t2_t3_t4.md`, `REPORT_spatial_t5_t6_t7_t8_t9.md`.
