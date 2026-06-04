---
name: libero-goal-pro-progress
description: "libero_goal PRO 4-cell — 40 Pi0 baselines done, 10/40 hybrid done (t0/t1/t2-base+task); cook_region offset documented"
metadata: 
  node_type: memory
  type: project
  originSessionId: cf2f87e2-37a3-4480-8b9a-4e5666fbf61f
---

**Status as of 2026-05-21 (re-run)**: workspace_pro/results_goal_pert/ populated with:
- 40/40 Pi0 fullshot baselines across {base, task, swap, lan} × {t0..t9}, seed=0
- 16/40 hybrid runs done with full schema: t0 4/4 PASS, t1 3/4 PASS (swap FAIL), t2 3/4 PASS (swap FAIL), t3 4/4 PASS
- Conditional hybrid win on Pi0-fails (t0-t3): 6/7 = 86%
- summary.csv, REPORT_goal_progress.md updated; per-cell recipe + full-schema audit JSONs
- Reusable harness: /tmp/run_cell_dynamic.py + /tmp/strategies/t{N}_*.py

**Headline Pi0 numbers**: base 8/10, task (P1) 4/10, swap (P2) 4/10, lan 9/10. P1 and P2 are the hybrid-wins axes.

**Why**: user asked to set up libero_goal hybrid runs + Pi0 baselines following PRO_HYBRID_GUIDE workflow on 2026-05-21.

**How to apply**:
- To continue, work through t2 swap/lan + t3..t9 × 4 cells (30 remaining). Each follows base recipe with target-object swap for `_task`, position lookup from state_00 for `_swap`, paraphrased prompt for `_lan`.
- Stove tasks need cook_region offset trick — see [[cook-region-offset]].
- Cabinet top tasks (t2 base, t4): top_side site at world (0.026, -0.256, table_z+0.22).
- Drawer open/close (t0, t3): NO teleport — open/close `wooden_cabinet_1_{top,middle,bottom}_level`
  physically via `pi0_doubled` ("open/close the drawer") or scripted OSC pull/push contact.
- Knob turn (t7): NO teleport — turn `flat_stove_1_button` physically via `pi0_doubled`
  ("turn on the stove"); see [[stove-turnoff-strict]] + the libero_10 t2 physics-only redo.
- swap-scene Pi0 picks: use full BDDL prompt + `track_obj_lift_thresh=0.08, lift_thresh=0.5` to force track-object validation (otherwise Pi0 false-positives via eef-ascent without grasping).

**Related**: [[libero-10-t0-t5-pro]] [[libero-spatial-t0-t9-pro]] [[libero-object-pro-done]] [[cook-region-offset]] [[liberopro-driver-patch]]
