---
name: project_pi0_baseline_grid
description: Pi0 fullshot baseline grid script + object PRO baseline numbers (the comparison ceiling for hybrid sweeps)
metadata: 
  node_type: memory
  type: project
  originSessionId: 1d2ff8f3-2dd8-4212-b752-4547e53239bb
---

`scripts/libero/run_pi0_baseline_grid.sh` runs the Pi0
fullshot baseline as an 8-GPU parallel grid: SUITES × TASKS × SEEDS, one
`pi0_baseline.py` cell per GPU, skip-if-output-exists, 900s/cell timeout,
`LIBERO_TYPE=pro`. No claude -p → no subscription quota. ~2 min/cell, 300
cells in ~94 min. Output: `multi_seed_exp/<name>/baseline_<suite>_t<task>_s<seed>.json`.

**libero_object PRO baseline (2026-05-23, seeds 0-9, 300 cells, 0 errors):**
- object_task: 56/100 (56%)
- object_lan:  96/100 (96%)  ← lan barely dents Pi0 (vision-driven, see [[project_pi05_libero_prompt_blind]])
- object_swap: 76/100 (76%)
- overall 228/300 (76%)

`task` perturbation is the hardest axis (56%) = where the hybrid LLM agent
should win. This is the apples-to-apples ceiling for the hybrid claude -p
sweeps in `multi_seed_exp/`. Spatial hybrid sweep lives in
[[project_spatial_multiseed_sweep]].
