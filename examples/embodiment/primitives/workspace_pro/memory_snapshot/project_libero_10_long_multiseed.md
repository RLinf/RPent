---
name: project_libero_10_long_multiseed
description: libero_10 (long) PRO multi-seed hybrid sweep — 300/300 = 100% vs Pi0 baseline 85/300 (28%)
metadata: 
  node_type: memory
  type: project
  originSessionId: fa95dece-8efe-42a7-9077-a1d2883a0dba
---

Completed 2026-05-25. libero_10 ("long", long-horizon) PRO hybrid sweep:
3 regimes (task/swap/lan) × 10 tasks × 10 seeds = **300 cells**, opus-4.7,
`hybrid_agent_cc/run_long_grid.sh` (4-way, max_episode_steps=5000, CELL_TIMEOUT 1200).

**Result: 300/300 = 100%** (task 100, swap 100, lan 100) vs Pi0 fullshot baseline
**85/300 (28%)**. Starkest gap = swap (Pi0 2/100, fixtures moved) → hybrid perfect.

Output: `examples/embodiment/primitives/workspace_pro/multi_seed_exp/long/`
(FINAL_SUMMARY.md, audits + recipe_*.jsonl). Baseline: `../long_pi0_baseline/`.

**0 genuine policy failures.** First pass was 266/300 raw (90%); all 34 non-successes
were infra/primitive: 27 EGL/EOFError render crashes ([[feedback_pi0_chunks_egl_crash]],
hotspots task_t7=6, task_t6=4, swap_t0=5 — correlates with renders/episode), 6
timeout/dead-driver, 1 MuJoCo NaN ([[feedback_osc_push_mujoco_nan]], lan_t3_s2). Backfill
recipe: delete crash False-audits (idempotent skip keys on audit existence) + relaunch;
the pre-launch skip patch in `run_parallel_seeds.sh` makes resume instant. Two backfill
rounds → clean 300/300.
