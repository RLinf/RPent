---
name: project_spatial_multiseed_sweep
description: "Spatial PRO hybrid claude -p multi-seed sweep — orchestrator, 4-GPU config, EGL-crash mitigation"
metadata: 
  node_type: memory
  type: project
  originSessionId: 1d2ff8f3-2dd8-4212-b752-4547e53239bb
---

`scripts/libero/run_pro_sweep.sh` with `ENV_BASE=libero_spatial` runs the hybrid
claude -p sweep over libero_spatial PRO: 3 regimes (task/lan/swap) × 10
tasks × 10 seeds = 300 cells → `multi_seed_exp/spatial/`. Defaults:
`GPUS="0 1 2 3"` (4-GPU, 1 cell/GPU), `STAGGER_S=60`, `MODEL=claude-opus-4-7`,
`LIBERO_TYPE=pro`. Uses opus, so it consumes Claude Code subscription quota.

**Why 4 GPUs not 8:** 8 simultaneous Pi0/EGL inits (1 cell/GPU on 8-GPU
host) crash the driver with EOFError/EGL_NOT_INITIALIZED — see
[[feedback_pi0_chunks_egl_crash]]. On 2026-05-23 this hung 5/10 cells for
6.5h in `until [ -f states.json ]` poll loops. Fixes applied:
- `run_one_cell.sh` now wraps `claude -p` in `timeout 600` (`CELL_TIMEOUT_S`)
  so a stuck worker can't block the orchestrator forever.
- `run_one_cell.sh` skips a cell if `$OUTPUT_DIR/<tag>.json` exists →
  orchestrator is re-launchable idempotently (resume only does missing cells).
  Skips are now instant ("no slot, no stagger") — the earlier stagger-on-skip
  waste is gone, so resuming a mostly-done suite is fast.
- Crash rate at 4-GPU is still ~5-10%/cell; failed cells get NO AUDIT and
  are retried on the next relaunch.

Status 2026-05-23 (PAUSED for quota): 64/300, all True (task t0-t6 mostly,
lan+swap not started). **RESUMED 2026-05-25** (PID-detached, /tmp/spatial_grid_outer.log):
existing 64 task cells all 64/64 True; remaining ~236 = 36 task-backfill
(t1·s1, t4·s6, t6·s6-9, t7-t9 all) + 100 lan + 100 swap. No spatial Pi0
baseline dir exists yet (siblings object/long = 300/300, goal 62/62).
Compare against Pi0 baseline ceiling [[project_pi0_baseline_grid]].
