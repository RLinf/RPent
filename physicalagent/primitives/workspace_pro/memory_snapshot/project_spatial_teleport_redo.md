---
name: project_spatial_teleport_redo
description: "libero_spatial teleport-redo (multi_seed_exp) physics-only result — 30/41 solved, 11 honest-blocked, zero teleport"
metadata: 
  node_type: memory
  type: project
  originSessionId: 736c7752-a7a7-4d37-93a7-98a7af2c85dd
---

2026-05-27: redid all 41 teleport-contaminated **libero_spatial** multi_seed_exp
cells (swap+task, from [[feedback_no_teleport_rule]] / TELEPORT_REDO_CELLS.md)
physics-only via the `claude -p` harness. Results in
`multi_seed_exp/redo_spatial_<regime>_t<N>/` (originals in `multi_seed_exp/spatial/`
untouched). Orchestrator: `multi_seed_exp/run_spatial_teleport_redo.sh`.

**Outcome: 30/41 solved (term=True), 11 honest term=False, ZERO teleport in any
recipe.** Verified all `recipe_*.jsonl` contain none of set_object_pose/
articulate_to/js_move_to/carry_object.

Clean groups: swap t3 6/6, t5 3/3, t7 1/1; task t1 6/6, t3 2/2, t7 1/1, t9 1/1.

**The 11 genuine physical dead-ends are exactly the cells whose originals used
carry/js_move_to teleport — don't burn time re-attempting blindly:**
- **swap t6** s4,s6,s7,s8,s9 — carry bowl to plate ELEVATED ON THE STOVE (back-left)
  across the OSC front singularity.
- **swap t9** s0,s1,s4,s8 — cabinet-top bowl → far-left table plate, same x~0
  singularity crossing (t9_s0 audit has a detailed thread-the-singularity analysis;
  see [[move_pose_covarying_reach]] as the only lever that might help).
- **swap t4 s6 / task t4 s6** — bowl on cabinet-top back-RIGHT corner; Pi0 won't grasp.

**Process lessons:** needed CELL_TIMEOUT_S=1200 (see
[[feedback_redo_cell_timeout_1200]]); first pass had 6 EGL crashes (no audit) — a
backfill re-run at LOWER concurrency (GPUS="0 1 2", stagger 75) cleared all of them
(skip-if-audit-exists only re-attempts crashed/deleted cells). Re-synced the stale
memory_snapshot to live first ([[feedback_worker_reads_memory_snapshot]]).
Remaining teleport-redo suites still TODO: libero_10 (121), libero_goal (87),
libero_object (11) swap+task, plus all the lan-regime cells (TELEPORT_REDO_CELLS_LAN.md).
