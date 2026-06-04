# Onboarding for a fresh Claude — PERCEPTION-ISOLATED multi-seed runs

You are a fresh Claude (the MONITOR) asked to run LIBERO-PRO experiments in
**perception-isolated mode**: the worker agents get **no object world
coordinates** — they must localize objects themselves from the camera image +
depth + calibration. This file teaches the whole stack in one read. It is the
perception-mode sibling of `ONBOARDING_FRESH_AGENT.md` (the oracle-mode doc); if
you only read that one you will accidentally run in ORACLE mode (with coords).

> **TL;DR**: a `claude -p` worker drives a Pi0.5 + LIBERO REPL, but its
> `state_NN.json` has only robot proprioception + object NAMES (no positions).
> It localizes objects by reading `image_cam_NN.png` + `depth_NN.npy` and
> back-projecting with `camera_meta.json`, then scripts move_to/release itself.
> Your job: launch + monitor + summarize the sweep.

---

## 0. What's different from oracle mode (read this first)

| | oracle (`ONBOARDING_FRESH_AGENT.md`) | **perception (this doc)** |
|---|---|---|
| worker prompt | `agent_task_prompt.md` | **`agent_task_prompt_perception.md`** |
| `state_NN.json` objects | full `objects:{name:[x,y,z]}` | **`object_names:[…]` only, NO coords** |
| extra obs files | image_NN.png | **+ `image_cam_NN.png`, `depth_NN.npy`, `camera_meta.json`** |
| driver flags | (none) | **`--hide_object_coords --always_render`** |
| cell timeout | 600 | **1200** (localization + manipulation is slower) |
| audit `regime` | `strict` | `strict_perception` |

How the worker localizes (it's all in the prompt, but so you understand the
data): find the object's pixel `(row,col)` in `image_cam_NN.png`, read
`z = depth_NN.npy[row,col]` (meters), then
`P_world = extrinsic_cam2world @ [col*z, row*z, z, 1]` (matrices in
`camera_meta.json`). **VERIFIED**: projecting GT object poses through this lands
on the right pixel/depth 5/5 (plate Δ6 mm), so the calibration is trustworthy.

The seed-0 recipe library (`results_*_pert/`, `results_all_object_new/`) is
still given to the worker as a **strategy prior** (which object, prompt ladder,
sequence, offsets) — but its coords are for a different scene, so the worker
re-derives positions by perception. (See prompt §3.)

---

## 1. Repo + key files

Repo root: `/mnt/public2/zhangyixian/RLinf_agentic`.
Working dir for these tools:
`examples/embodiment/primitives/workspace_pro/hybrid_agent_cc/`

```
agent_task_prompt_perception.md   # the per-cell worker prompt (READ this)
run_perception_grid.sh            # the one-command multi-seed launcher
run_parallel_seeds.sh             # per-(suite,task) parallel driver (called by the grid)
run_one_cell.sh                   # per-cell: starts driver + claude -p
status.sh                         # progress printer
```
Driver: `examples/embodiment/primitives/interactive_driver.py`
(`--hide_object_coords` strips coords; `--always_render` keeps depth fresh
every step — the render-skip optimization leaves depth stale after OSC moves).

---

## 2. Launch a specified multi-seed sweep (one command)

`run_perception_grid.sh` bakes in the perception env (prompt + flags + timeout)
and loops `REGIMES × TASKS`, each parallelized across `GPUS`. Override via env:

```bash
cd /mnt/public2/zhangyixian/RLinf_agentic
PERT=examples/embodiment/primitives/workspace_pro

# object swap+task, all tasks t0-t9, seeds 0-9, 4 GPUs:
ENV_BASE=libero_object REGIMES="swap task" \
TASKS="0 1 2 3 4 5 6 7 8 9" SEEDS="0 1 2 3 4 5 6 7 8 9" \
GPUS="0 1 2 3" \
    bash $PERT/hybrid_agent_cc/run_perception_grid.sh \
    > /tmp/percep_object_outer.log 2>&1 &

# pilot (one task, few seeds):
ENV_BASE=libero_object REGIMES="swap" TASKS="2" SEEDS="0 1 2" GPUS="0 1 2" \
    bash $PERT/hybrid_agent_cc/run_perception_grid.sh > /tmp/percep_pilot.log 2>&1 &
```

- `ENV_BASE` ∈ `libero_object | libero_spatial | libero_goal | libero_10`.
- Results → NEW dirs `multi_seed_exp/percep_<env>_<regime>_t<N>/` (originals
  untouched). Re-runnable: cells with an audit are skipped, so a backfill pass
  re-attempts only crashed cells.
- Defaults: 4 GPUs, stagger 60s, CELL_TIMEOUT 1200, MAX_EPISODE_STEPS 2000
  (5000 for libero_10), LIBERO_TYPE=pro.

> **Concurrency / EGL:** `--always_render` renders every env.step, so long
> episodes accumulate more EGL load than oracle mode. Keep ≤4 cells/GPU-set at
> 1/GPU + stagger 60. Crashes show as MISSING_AUDIT → re-run the grid to backfill.

---

## 3. Monitor

```bash
# per-group progress:
SUITE=libero_object_swap TASK=2 \
OUTPUT_DIR=examples/embodiment/primitives/workspace_pro/multi_seed_exp/percep_object_swap_t2 \
    bash $PERT/hybrid_agent_cc/status.sh
# outer log (group boundaries + per-group FINAL X/N):
tail -f /tmp/percep_object_outer.log
```
After each group, check the audits:
```bash
for f in $OUTPUT_DIR/*.json; do
  jq -r '"\(input_filename)  term=\(.libero_terminated)  regime=\(.regime)"' "$f"
done
```
Outcomes per cell: `term=True` (perception solve), `term=False` (honest fail —
couldn't localize/reach), MISSING_AUDIT (driver/EGL crash → backfill by re-run).
If 3+ cells in a row fail to LOCALIZE (not just place), the bug is likely the
prompt's back-projection step — raise it before burning the whole sweep.

---

## 4. Per-cell artifacts (`<tag>` = `<env>_<regime>_t<N>_s<M>`)

| File | Contents |
|---|---|
| `$OUTPUT_DIR/<tag>.json` | audit (libero_terminated, regime:`strict_perception`, strategy_notes, pick_result, final_state) |
| `$OUTPUT_DIR/recipe_<tag>.jsonl` | the command sequence that worked |
| `$OUTPUT_DIR/claude_<tag>.txt` | full claude -p stdout (only flushed at exit; empty if timed out) |
| `/tmp/hybrid_repl_<tag>/state_NN.json` | proprioception + object_names (NO coords) |
| `/tmp/hybrid_repl_<tag>/image_cam_NN.png` | calibration-frame RGB (pixel-pick here) |
| `/tmp/hybrid_repl_<tag>/depth_NN.npy` | metric depth (meters) |
| `/tmp/hybrid_repl_<tag>/camera_meta.json` | K, cam→world extrinsic, near/far, projection recipe |
| `/tmp/cc_driver_<tag>.log` | Pi0 load + driver stderr |

**Verify a sweep is genuinely perception-isolated** (no coord leak): every
`state_*.json` must have `object_names` and NO `objects` block; every
`recipe_*.jsonl` must contain none of set_object_pose/articulate_to/js_move_to/
carry_object. A quick check:
```bash
grep -l '"objects"' /tmp/hybrid_repl_<tag>/state_*.json   # should print nothing
```

---

## 5. Summarize for the user

When all groups finish, report per (regime,task): `X/N term=True`, count of
honest fails vs MISSING_AUDIT crashes, then run a backfill (`re-run the same grid
command`; skip-if-audit-exists re-attempts only crashes). Note whether the
failures are LOCALIZATION failures (perception couldn't find/place the object —
the interesting signal) vs reach/kinematic dead-ends. Keep updates to 2-3 lines;
full table on completion. One commit per sweep.

That's the loop. The worker prompt (`agent_task_prompt_perception.md`) has the
full localization recipe — `Read` it when you need the worker's exact contract.
```
