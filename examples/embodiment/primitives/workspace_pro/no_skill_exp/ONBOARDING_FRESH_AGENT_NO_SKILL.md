# Onboarding for a fresh Claude — NO-SKILL-LIBRARY (× perception-isolated) runs

You are a fresh Claude (the MONITOR) asked to run a **skill-ablation** of the
LIBERO-PRO hybrid agent. The worker agents run in the perception-isolated mode
(no object coords — they localize from depth) AND with the **recipe/skill
library taken away**: they may NOT read any prior worked solution. The point is
to measure what the library of past `recipe_*.jsonl` solves was buying us —
compare these results against the matching `percep_*` cells, which DID have the
recipe prior. This doc is the no-skill sibling of
`../hybrid_agent_cc/ONBOARDING_FRESH_AGENT_PERCEPTION.md`; read that first if you
want the full perception stack — this file only describes the delta.

> **TL;DR**: identical to the perception harness (a `claude -p` worker drives a
> Pi0.5 + LIBERO REPL, no coords, localizes from depth), with ONE change: the
> worker prompt forbids reading any `recipe_*.jsonl` or prior audit `*.json`. It
> must solve each cell from scratch using only the image + the operating memory
> (`memory_snapshot/`) + the generic guides. Your job: launch + monitor +
> verify the isolation + summarize the A/B.

---

## 0. What's different — three nested conditions

| | oracle | perception | **no-skill (this doc)** |
|---|---|---|---|
| worker prompt | `agent_task_prompt.md` | `agent_task_prompt_perception.md` | **`agent_task_prompt_no_skill.md`** |
| object coords | full `objects:{}` | none (localize from depth) | **none (localize from depth)** |
| recipe/skill prior | recipes pointed-to | recipes pointed-to | **FORBIDDEN — no worked solutions** |
| curated `memory_snapshot/` | yes | yes | **yes (kept — distilled wisdom, not a per-cell recipe)** |
| CC project auto-memory | injected | injected | **DISABLED (`CLAUDE_CODE_DISABLE_AUTO_MEMORY=1`) — it held per-cell solves** |
| audit `regime` | `strict` | `strict_perception` | `strict_perception_noskill` |
| results dir | `multi_seed_exp/` | `multi_seed_exp/percep_*` | **`no_skill_exp/results/noskill_*`** |

**What "no skill" removes (per the experiment design):**
  1. the per-cell solved recipes + audits — `recipe_*.jsonl`, the result/audit
     `*.json` under `results_*_pert/`, `results_all_object_new/`,
     `multi_seed_exp/`, `results_claude_p_*/`, `baseline_pi0_*.json` (Rule 5);
  2. the Claude Code **project auto-memory** at
     `~/.claude_local/projects/-…-RLinf-agentic/memory/`. ⚠ This was the SUBTLE
     one. Claude Code auto-injects that git-repo's `MEMORY.md` index into every
     `claude -p` worker's context, and that store had accumulated ~96 per-CELL
     solved-experience notes from prior runs (e.g. "object_swap t0 needs full
     BDDL prompt", "tall bottle slip+topple recovery", and a percep
     back-projection fix literally tagged "confirmed on object_swap t2 s0"). The
     first pilot leaked through this channel before it was closed; the grid now
     exports `CLAUDE_CODE_DISABLE_AUTO_MEMORY=1` to suppress it, **verified** by
     `hard_audit_no_skill.sh` (0 auto-memory reads after the fix).

**What it KEEPS:** the CURATED `memory_snapshot/` (MEMORY.md + feedback_*.md +
project_*.md — general operating wisdom + magic numbers; verified clean of
per-cell solved recipes) and the generic guides (STRICT_HYBRID_GUIDE,
PRO_HYBRID_GUIDE, env_calibration). So this is a *recipe + auto-memory* ablation,
NOT a curated-memory ablation. (Note: the kept `memory_snapshot/` still contains
e.g. `feedback_percep_backproject_formula_wrong.md`, so the localization fix is
still available — that is a deliberate keep, not a leak.)

Everything else — driver, the parallel runner, the per-cell runner, the
localization recipe, the no-teleport/single-episode rules — is shared with
`../hybrid_agent_cc/`. We only swap the prompt + the output dir.

---

## 1. Files (this directory)

Working dir: `examples/embodiment/primitives/workspace_pro/no_skill_exp/`

```
agent_task_prompt_no_skill.md   # the per-cell worker prompt (READ this — Rule 5 is the ablation)
run_no_skill_grid.sh            # one-command multi-seed launcher (perception flags + no-skill prompt)
verify_no_skill.sh              # post-hoc: did perception + no-skill isolation hold?
results/                        # noskill_<env>_<regime>_t<N>/ output dirs land here (created at runtime)
```
Shared (reused, NOT copied): `../hybrid_agent_cc/run_parallel_seeds.sh`,
`../hybrid_agent_cc/run_one_cell.sh`, `../hybrid_agent_cc/status.sh`; driver
`../../interactive_driver.py`; memory `../memory_snapshot/`.

---

## 2. Launch

```bash
cd /mnt/public2/zhangyixian/RLinf_agentic
NS=examples/embodiment/primitives/workspace_pro/no_skill_exp

# PILOT smoke test (object swap, task 2, seeds 0-2, 3 GPUs):
ENV_BASE=libero_object REGIMES="swap" TASKS="2" SEEDS="0 1 2" GPUS="0 1 2" \
    bash $NS/run_no_skill_grid.sh > /tmp/noskill_pilot.log 2>&1 &

# Full A/B vs percep_object (swap+task, all tasks, seeds 0-9, 4 GPUs):
ENV_BASE=libero_object REGIMES="swap task" \
TASKS="0 1 2 3 4 5 6 7 8 9" SEEDS="0 1 2 3 4 5 6 7 8 9" GPUS="0 1 2 3" \
    bash $NS/run_no_skill_grid.sh > /tmp/noskill_object_outer.log 2>&1 &
```

- `ENV_BASE` ∈ `libero_object | libero_spatial | libero_goal | libero_10`.
- Results → `no_skill_exp/results/noskill_<env>_<regime>_t<N>/`. Re-runnable:
  cells with an audit are skipped, so a backfill pass re-attempts only crashes.
- Defaults: 3 GPUs (pilot), stagger 60s, CELL_TIMEOUT 1200, MAX_EPISODE_STEPS
  2000 (5000 for libero_10), LIBERO_TYPE=pro, MODEL=claude-opus-4-7.
- Same EGL caution as perception mode: `--always_render` is on; keep ≤4
  cells/GPU-set at 1/GPU + stagger 60. Crashes show as MISSING_AUDIT → re-run to backfill.

---

## 3. Monitor

```bash
NS=examples/embodiment/primitives/workspace_pro/no_skill_exp
# per-group progress (status.sh is shared; just point OUTPUT_DIR at the noskill dir):
SUITE=libero_object_swap TASK=2 \
OUTPUT_DIR=$NS/results/noskill_object_swap_t2 \
    bash examples/embodiment/primitives/workspace_pro/hybrid_agent_cc/status.sh
tail -f /tmp/noskill_pilot.log
```
Per cell: `term=True` (no-skill solve — the interesting positive), `term=False`
(honest fail — couldn't solve without a recipe), MISSING_AUDIT (driver/EGL crash
→ backfill by re-run).

---

## 4. Verify the isolation held (run this before trusting numbers)

```bash
bash examples/embodiment/primitives/workspace_pro/no_skill_exp/verify_no_skill.sh
```
It reports, per cell: solve status, the **perception coord-leak check** (SOLID —
every `state_*.json` must have `object_names` and no `objects` block) and a
**best-effort no-skill prose scan** of the worker transcript for references to
forbidden artifacts.

> ⚠ **Honest limitation:** `run_one_cell.sh` runs the worker with
> `--output-format text`, which captures only the worker's final narration, not
> a full tool-call trace. So the no-skill scan is best-effort, not airtight — a
> clean scan is reassuring but does not *prove* the worker never `cat`-ed a
> recipe (it runs with the repo as cwd and Bash enabled, so it physically
> could). The prompt's Rule 5 forbids it explicitly. If you need certainty,
> re-run a cell with `--output-format stream-json --verbose` and grep that trace
> (the recipe at the bottom of `verify_no_skill.sh`), or add a Claude Code
> `permissions.deny` settings rule for `Read`/`Bash` on the recipe dirs.

---

## 5. Summarize for the user

When the sweep finishes, report per (regime,task): `X/N term=True` under no-skill
vs the matching `percep_*` X/N (with skill), the delta, and whether the no-skill
failures are LOCALIZATION failures vs STRATEGY failures (the worker localized
fine but picked the wrong object / sequence / offset because it had no recipe to
copy — that gap IS the value of the skill library, the headline result). Keep
updates to 2-3 lines; full A/B table on completion. One commit per sweep.

The worker prompt (`agent_task_prompt_no_skill.md`, Rule 5 + §3) has the exact
no-skill contract — `Read` it when you need the worker's precise constraints.
