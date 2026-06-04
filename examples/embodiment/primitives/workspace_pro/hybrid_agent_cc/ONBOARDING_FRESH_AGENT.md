# Onboarding for a fresh Claude (monitor + serial claude-p workflow)

You are a fresh Claude session asked to help a user run LIBERO PRO
hybrid experiments via `claude -p` (Claude Code subscription, not the
metered API). This file teaches you the whole stack in one read.

> **TL;DR**: the user has a Python REPL "driver" that loads Pi0.5 and a
> LIBERO sim. A `claude -p` invocation (a separate Claude Code) drives
> the REPL through Bash/Read/Write tools, picking + placing one object
> per episode. Each cell is one fresh `claude -p`; runs are serial.
> Your job is to launch + monitor + summarize the sweep.

---

## 0. What lives where

Repo root: `/mnt/public2/zhangyixian/RLinf_agentic`.

Two sibling directories under
`examples/embodiment/primitives/workspace_pro/`:

| Dir | Purpose |
|---|---|
| `hybrid_agent/` | **API variant**. Anthropic-SDK-driven agent (`runner.py`, `tools.py`, `prompts.py`, `parallel_launch.py`). Run via Python; parallelizes across GPUs; metered by tokens. |
| `hybrid_agent_cc/` | **Claude Code variant** (you are here). Bash + `claude -p`; sequential; subscription quota. THIS file lives here. |

Read these once before doing anything:

```
hybrid_agent_cc/README.md            # overview + quickstart
hybrid_agent_cc/agent_task_prompt.md # the prompt template that claude -p sees
hybrid_agent_cc/run_one_cell.sh      # per-cell driver + claude -p invocation
hybrid_agent_cc/run_all_seeds.sh     # outer sequential loop
hybrid_agent_cc/status.sh            # progress printer (the eyes for monitoring)
```

Background guides (read them when you need to understand a primitive):

```
examples/embodiment/primitives/STRICT_HYBRID_GUIDE.md   # operating manual
examples/embodiment/primitives/workspace_pro/PRO_HYBRID_GUIDE.md
examples/embodiment/primitives/workspace_pro/env_calibration.md
```

The user's accumulated wisdom (you should also browse this — it has
"magic numbers" not in the guides):

```
/root/.claude/projects/-mnt-public2-zhangyixian/memory/MEMORY.md  # ← index
/root/.claude/projects/-mnt-public2-zhangyixian/memory/feedback_*.md
```

---

## 1. The two-layer model

There are TWO Claude instances in this workflow. Don't confuse them:

```
   ┌──────────────────────────────────────┐
   │ Layer 1: MONITOR (you, this session) │
   │ - reads logs, runs status.sh         │
   │ - summarizes for the human user      │
   │ - never touches /tmp/hybrid_repl     │
   └──────────────────────────────────────┘
                    ↓ spawns
   ┌──────────────────────────────────────┐
   │ Layer 2: WORKER (claude -p, one per  │
   │ cell, ephemeral)                     │
   │ - reads agent_task_prompt.md         │
   │ - uses Bash to drive the REPL driver │
   │ - uses Read on state/log/image PNG   │
   │ - exits when libero_terminated=True  │
   └──────────────────────────────────────┘
                    ↓ writes JSON commands
   ┌──────────────────────────────────────┐
   │ Layer 3: DRIVER (Python subprocess)  │
   │ - interactive_driver.py + Pi0 + sim  │
   │ - per-cell workdir /tmp/hybrid_repl_<tag> │
   └──────────────────────────────────────┘
```

YOU are layer 1. You orchestrate, observe, summarize. The user is
busy and wants you to tell them whether things are going well, where
they're stuck, and what was learned. **Do not** spawn worker Claudes
in your own context (don't `claude -p ...` from yourself); you
launch the wrapper sh which manages workers properly.

---

## 2. The two sh entrypoints

### Launching a sweep

Two flavours: **sequential** (one cell at a time on one GPU) or
**parallel** (N cells concurrently across GPUs). Default to parallel
unless you have only one GPU.

#### A. SEQUENTIAL — `run_all_seeds.sh`

Single GPU, one cell at a time. Use this on single-card boxes or
when API quota is constrained.

```bash
# Defaults: libero_spatial_lan task 0 seeds 0..9 on GPU 0, sonnet
bash hybrid_agent_cc/run_all_seeds.sh

# Override:
SUITE=libero_spatial_task TASK=0 SEEDS="0 1 2 3 4 5 6 7 8 9" \
MODEL=claude-opus-4-7 CUDA_DEVICE=0 \
OUTPUT_DIR=examples/embodiment/primitives/workspace_pro/results_claude_p_runs_task_t0 \
MASTER_LOG=/tmp/claude_p_master_task_t0.log \
    bash hybrid_agent_cc/run_all_seeds.sh \
    > /tmp/claude_p_outer_task_t0.log 2>&1 &
```

#### B. PARALLEL — `run_parallel_seeds.sh`  ← recommended for ≥2 GPUs

Concurrent claude -p sessions, one per GPU slot. `GPUS=` is a
space-separated list — its **length = concurrency**. Seeds beyond
the slot count queue and reuse the first freed GPU (round-robin).

> **2026-05-23 incident — 8-way (1/GPU on 8-GPU host) is NOT stable.**
> 8 simultaneous Pi0 + EGL context inits trigger
> `EGL_NOT_INITIALIZED`/`EOFError` in the driver subprocess (see memory
> `feedback_pi0_chunks_egl_crash`). 5/10 cells of `libero_spatial_task
> t1` crashed and the workers hung forever in `until [ -f done_NN.flag
> ]; sleep 1; done` poll loops, blocking the orchestrator for 6.5h.
> **Default concurrency is now 4 GPUs at 1/GPU + STAGGER_S=60**.
> `run_one_cell.sh` now wraps `claude -p` in `timeout 600` so a stuck
> worker can't block forever even if it happens again.

```bash
# RECOMMENDED — 4 GPUs, 1/GPU, stagger 60s, 10 seeds queue 6:
SUITE=libero_spatial_swap TASK=0 \
SEEDS="0 1 2 3 4 5 6 7 8 9" \
GPUS="0 1 2 3" \
MODEL=claude-opus-4-7 \
STAGGER_S=60 \
MAX_BUDGET_USD=10 \
OUTPUT_DIR=examples/embodiment/primitives/workspace_pro/results_claude_p_parallel \
MASTER_LOG=/tmp/claude_p_parallel.log \
    bash hybrid_agent_cc/run_parallel_seeds.sh \
    > /tmp/claude_p_outer.log 2>&1 &

# 8 seeds on 4 GPUs (2/GPU pattern — also previously validated):
GPUS="0 1 2 3 0 1 2 3"  SEEDS="0 1 2 3 4 5 6 7"   # rest unchanged

# 10 seeds on 4 GPUs, queued (each GPU takes ~3 seeds round-robin):
GPUS="0 1 2 3"  SEEDS="0 1 2 3 4 5 6 7 8 9"  ...
```

##### Two patterns for `GPUS`

| Want | `GPUS=` value | Concurrency | What happens |
|---|---|---|---|
| N seeds on N GPUs (1 cell/GPU) | `"0 1 2 ... N-1"` | N | clean parallel |
| 2N seeds on N GPUs (2 cells/GPU) | `"0 1 2 ... N-1 0 1 2 ... N-1"` | 2N | 2 Pi0 drivers per GPU (~16GB each, fits in 80GB) |
| K seeds on M GPUs, queue overflow | `"0 1 ... M-1"` (len M), seeds K>M | M | first M start, the rest queue |

Each Pi0 driver uses ~8GB VRAM, so 2-3 cells per 80GB GPU is safe.
Don't push beyond ~3/GPU without checking; `nvidia-smi --query-gpu=memory.used`.

##### Validated numbers (opus 4.7, libero_spatial_swap t0)

| Concurrency | GPUs | Wall time | Per-cell avg | Strict | Notes |
|---|---|---|---|---|---|
| 1 (sequential extrapolated) | 1 | ~26 min | ~200s | n/a | n/a |
| 4 | 4 (1/GPU) | 8.3 min | 183s | 4/4 | **current default — stable** |
| 8 | 4 (2/GPU) | 9.0 min | 163s | 8/8 | OK when 2/GPU; do NOT do 1/GPU x8 |
| **8 (1/GPU)** | **8 (1/GPU)** | **N/A — crash** | n/a | **5/10 driver-crash on t1** | EGL_NOT_INITIALIZED, do not use |

Claude Code subscription tolerates 4-8 concurrent claude -p sessions
on one host. The dangerous knob is **simultaneous Pi0/EGL inits across
N distinct GPUs** — 8 separate EGL contexts coming up at the same
moment trips the multiprocessing connection. 2-per-GPU at 4 GPUs is
fine because the second cell on each GPU comes up 30s later and the
EGL context is already initialised.

##### Things to consider before parallelising

- **API rate**: empirically 8 concurrent on one subscription is fine.
  Higher concurrency may eventually hit a 429 — if it does, the cell's
  `claude_<tag>.txt` will say so, and you should reduce `len(GPUS)`
  or raise `STAGGER_S`.
- **GPU memory**: 2 Pi0 drivers per GPU = ~16GB; 3 = ~24GB. On
  80GB A800/A100/H100 that's comfortable. Don't put 4+ on a 24GB
  consumer card.
- **Disk IO**: every cell's driver loads the ~5GB Pi0 checkpoint at
  startup. `STAGGER_S=30` is the empirically-validated minimum to
  avoid filesystem contention. On NFS/network storage bump to 60.
- **Budget**: each cell can spend up to `MAX_BUDGET_USD` (default 10).
  8 cells × $10 = $80 worst case; typical opus-4.7 cell costs $2-3.
- **Workdir collisions**: each cell uses `/tmp/hybrid_repl_<tag>/`
  where tag is `<suite>_t<task>_s<seed>`, so different (suite,task,seed)
  triples never collide. If you re-run the same triple, the previous
  workdir is wiped automatically by `run_one_cell.sh`.

Defaults if you omit env vars:
- `SUITE`=libero_spatial_lan, `TASK`=0, `SEEDS`="0..9", `MODEL`=sonnet,
  `CUDA_DEVICE`=0
- For `run_parallel_seeds.sh`: `GPUS="0 1 2 3"`, `STAGGER_S=60`
- For `run_spatial_grid.sh`: `MODEL=claude-opus-4-7`, `LIBERO_TYPE=pro`,
  same GPU/stagger defaults
- `OUTPUT_DIR`=`workspace_pro/results_claude_p_runs/`
- `MASTER_LOG`=`/tmp/claude_p_master_<suite>_t<task>.log`
- `MAX_BUDGET_USD`=10 (claude -p's per-invocation USD cap; if you
  see "Error: Exceeded USD budget (N)" in `claude_<tag>.txt`, bump
  this. Sonnet cells almost never hit $5; opus 4.7 edge cases that
  need recovery loops can reach $4–6, so 10 is the safe default.)
- `MAX_EPISODE_STEPS` — **suite-dependent, do NOT use one value for all.**
  This is the `interactive_driver --max_episode_steps` cap, i.e. how many
  cumulative robosuite env.steps the episode may take before it self-
  terminates. Each primitive does many internal steps (move_to ≤80–300,
  pi0_pick = chunks×5 ≈ 75–150, articulate settle ≈ 40), so the budget is
  consumed FAST. Once the cap is hit, the next env.step raises
  `ValueError("executing action in terminated episode")`, which kills the
  worker mid-recipe and produces NO audit (looks like an EGL crash but is
  not — see memory `max-episode-steps-libero`).
    - `libero_spatial`, `libero_object`, `libero_goal` (short, 1 pick +
      scripted carry/place ≈ 7–15 primitives): **600 is fine** (the
      default). The 300-cell object sweep ran clean at 600.
    - **`libero_10` (long-horizon, ~15 primitives incl. carry +
      articulate + set_object_pose chains, 1000+ cumulative env.steps):
      use `MAX_EPISODE_STEPS=5000`** — the same value used during basic
      interactive exploration. At 600 these cells crash mid-recipe with
      the ValueError above. `run_one_cell.sh` now auto-bumps to 5000 when
      the suite name contains `libero_10`, but pass it explicitly in any
      libero_10 grid launch so it's visible.
- `CELL_TIMEOUT_S`=600 (added 2026-05-23 to `run_one_cell.sh`): hard
  wall-clock cap on `claude -p`. If a worker hangs polling for a
  `done_NN.flag` after the driver crashed, the cell will be killed
  at this limit and the orchestrator moves on. **Do not remove this
  unless you also fix the worker-side death-detection.** Note this was
  tuned for the ~183s/cell spatial/object sweeps; `libero_10` cells run
  longer (more primitives + more Pi0 chunks + more image reads per turn),
  so a few legitimately-progressing cells WILL get killed at 600s and
  show up as MISSING_AUDIT. For libero_10 bump `CELL_TIMEOUT_S=1200` and
  re-launch the grid afterward to backfill any that still time out (the
  skip-if-audit-exists logic makes this safe).

`run_one_cell.sh` also skips if `$OUTPUT_DIR/<tag>.json` already
exists, so the orchestrator can be re-launched idempotently after a
partial sweep — only missing cells are run.

Always run with `> some_outer.log 2>&1 &` so it survives terminal
disconnects; the master log is structured progress, the outer log is
just stdout/stderr capture.

### Monitoring a sweep

```bash
# One-shot status (run any time):
bash hybrid_agent_cc/status.sh
# or for a non-default cell:
SUITE=libero_spatial_task TASK=0 \
OUTPUT_DIR=examples/embodiment/primitives/workspace_pro/results_claude_p_runs_task_t0 \
    bash hybrid_agent_cc/status.sh

# Watch (refresh every 30s):
watch -n 30 'SUITE=libero_spatial_task TASK=0 \
    OUTPUT_DIR=examples/embodiment/primitives/workspace_pro/results_claude_p_runs_task_t0 \
    bash hybrid_agent_cc/status.sh'

# Tail the master log:
tail -f /tmp/claude_p_master_task_t0.log
```

`status.sh` prints a table with per-seed state: `DONE / RUNNING / pending`,
and totals at the bottom. It's idempotent and fast.

### What "monitoring" actually means here

You're not just printing; you're **interpreting**. Each cell may:
- finish cleanly with `libero_terminated=True` and a valid audit JSON
- finish with `libero_terminated=False` and the agent's "I'm stuck"
  audit
- die mid-run (claude -p exits non-zero, no audit written → master log
  has `[end] ... rc=N`)
- hang (no new `done_NN.flag` appearing for >5 min → likely the worker
  Claude got into a sleep loop or an OSC stall the driver can't escape)

After each cell finishes, check:
```bash
cat $OUTPUT_DIR/<tag>.json | jq '.libero_terminated, .strategy_notes'
```
and summarize for the user. If you see 3+ consecutive failures with
the same symptom, raise it — the user can interrupt and refine the
prompt template or the memory pointers.

---

## 3. The one-cell flow (what `run_one_cell.sh` actually does)

```
1. clean /tmp/hybrid_repl_<tag>/
2. spawn   /opt/venv/openpi/bin/python interactive_driver.py
                --suite ... --task ... --seed ...
                --workdir /tmp/hybrid_repl_<tag>
   (writes driver.log to /tmp/cc_driver_<tag>.log)
3. wait until <workdir>/state_00.json appears (~80–100s, Pi0 load)
4. substitute placeholders in agent_task_prompt.md → /tmp/cc_prompt_<tag>.md
5. invoke:
     claude -p "$(cat /tmp/cc_prompt_<tag>.md)" \
       --model $MODEL \
       --add-dir <workdir> \
       --add-dir /root/.claude/.../memory \
       --allowedTools "Bash Read Write Glob Grep" \
       --max-budget-usd 2 \
       > $OUTPUT_DIR/claude_<tag>.txt 2>&1
6. send {"action":"exit"} to driver; kill it
7. report libero_terminated to caller
```

The two `--add-dir` are essential:
- workdir: lets the worker Read state_*.json + image_*.png
- memory: lets the worker Read the magic-number feedback notes

You should not need to modify this script unless the user wants
different paths or env wiring. Read it once so you know what files
each cell touches.

---

## 4. The agent_task_prompt.md (what the worker Claude sees)

Don't paraphrase it from memory — `Read` it when you need to know
what the worker is told. Key facts you'll often need:

- Rule 0: USE IMAGES (worker reads PNG via Read tool).
- Rule 1: Pi0 only for the grasp (`pi0_pick` with `track_obj` cut).
- Rule 2: inspect THEN act.
- Rule 3: Pi0 prompt ladder — sub-instr → full BDDL → spatial qualifier
  → re-pre-pos, before scripted pick.
- Rule 4: **NO** `reset` / `exit` mid-run. Single episode.
- It tells the worker to read MEMORY.md first, then the guides.

If you find the worker repeatedly making the same mistake across
seeds (e.g. forgetting bowl-eef +0.045 offset), the right fix is to
strengthen `agent_task_prompt.md` — not to rewrite the worker's
intermediate output. Edit, commit, re-run.

---

## 5. Per-cell artefacts (what to read when summarizing)

For each cell `<tag>` = `<suite-without-libero_>_t<N>_s<M>`:

| File | What's in it |
|---|---|
| `$OUTPUT_DIR/<tag>.json` | Audit (libero_terminated, regime, strategy_notes, pick_result, final_state) |
| `$OUTPUT_DIR/recipe_<tag>.jsonl` | The actual command sequence that worked |
| `$OUTPUT_DIR/claude_<tag>.txt` | Full claude -p stdout — read this if a cell failed |
| `/tmp/hybrid_repl_<tag>/log_NN.json` | Per-primitive log: command + result + elapsed_s |
| `/tmp/hybrid_repl_<tag>/state_NN.json` | Sim state after step NN |
| `/tmp/hybrid_repl_<tag>/image_NN.png` | Agentview RGB after step NN |
| `/tmp/cc_driver_<tag>.log` | Pi0 load + per-command driver stderr |
| `/tmp/cc_prompt_<tag>.md` | What was actually fed to claude -p (placeholder-substituted) |

Diagnostic recipes you'll keep using:

```bash
# Rule-1 audit: did Pi0 do the place?
for log in /tmp/hybrid_repl_<tag>/log_*.json; do
  /opt/venv/openpi/bin/python -c "
import json
d=json.load(open('$log'))
c=d['command']; r=d.get('result',{})
if c.get('action') in ('pi0_pick','release','set_object_pose'):
    print(c['action'], 'libero_term=', r.get('libero_terminated'))
"
done

# Comparison vs Pi0 fullshot baseline (if you ran one):
diff <(jq '.libero_terminated' $OUTPUT_DIR/<tag>.json) \
     <(jq '.libero_terminated' results_pi0_baseline_seeds_*/baseline_<suite>_t<N>_s<M>.json)
```

---

## 6. Pi0 fullshot baseline (the comparison)

This is the apples-to-apples baseline: same env, same seeds, but
Pi0.5 drives end-to-end (no LLM, no scripting).

```bash
SUITE=libero_spatial_task TASK=0 SEEDS="0 1 2 3 4 5 6 7 8 9" \
OUTDIR=examples/embodiment/primitives/workspace_pro/results_pi0_baseline_seeds_task_t0 \
    bash examples/embodiment/primitives/workspace_pro/run_pi0_baseline_seeds.sh \
    > /tmp/pi0_baseline_task_t0.log 2>&1 &
```

This sweep is fully parallel (4 GPUs, ~1-2 cells per GPU, total ~5 min
end-to-end). The output is per-seed JSON with `libero_terminated` and
`chunks_used`. Score is `sum(libero_terminated)/N`.

Pi0 baseline = ceiling for what raw VLA can do. Hybrid agent (claude -p
or API) should beat it on P1 task perturbations (instruction-sensitive)
and on P2 swaps. If hybrid loses to baseline, something is broken in
the worker prompt or memory.

---

## 7. Tips for a fresh you

- **Don't context-bloat your own session reading the worker's PNGs**.
  The worker reads images for spatial decisions; you don't need to
  see them. Read state JSONs only.
- **Don't paraphrase the prompt template from memory**. Always `Read`
  `agent_task_prompt.md` when explaining to the user what the worker
  was told.
- **Don't `reset` the driver yourself**. Rule 4 binds workers; for
  monitoring you have no business writing to `/tmp/hybrid_repl_*/`.
- **Don't re-launch a running sweep**. Check `ps -ef | grep -E
  "run_all_seeds|claude -p|interactive_driver"` first.
- **Use the `loop` skill if you want auto-monitor**. From a Claude Code
  prompt: `/loop 5m bash hybrid_agent_cc/status.sh && tail -10 $MASTER_LOG`
  — Claude Code will check every 5 min and summarize.
- **Be parsimonious with the user**. Two-three lines per status
  update; full report on completion.
- **One commit per sweep**. After all seeds finish + you've audited
  results, stage the output_dir and master log path, write a commit
  message summarizing (a) cell, (b) success rate vs Pi0 baseline,
  (c) Rule-1 compliance, (d) notable cells (stuck / odd recovery).

## 8. End-to-end worked example: 8-seed parallel sweep + Pi0 baseline

Mentally rehearse this flow so you can execute it without thinking.
Assume the user asks: "run libero_spatial_task t0 seeds 0-7 against
Pi0 fullshot baseline, opus 4.7".

```bash
cd /mnt/public2/zhangyixian/RLinf_agentic
REPO_PERT_DIR=examples/embodiment/primitives/workspace_pro

# === 1. Kick off Pi0 fullshot baseline in background (separate GPUs/logs) ===
SUITE=libero_spatial_task TASK=0 SEEDS="0 1 2 3 4 5 6 7" \
OUTDIR=$REPO_PERT_DIR/results_pi0_baseline_seeds_task_t0 \
    bash $REPO_PERT_DIR/run_pi0_baseline_seeds.sh \
    > /tmp/pi0_baseline_task_t0.log 2>&1 &

# === 2. Kick off claude -p parallel sweep ===
# 8 cells on 4 GPUs (2 per GPU) — same trick as the validated run.
SUITE=libero_spatial_task TASK=0 \
SEEDS="0 1 2 3 4 5 6 7" \
GPUS="0 1 2 3 0 1 2 3" \
MODEL=claude-opus-4-7 \
STAGGER_S=30 \
MAX_BUDGET_USD=10 \
OUTPUT_DIR=$REPO_PERT_DIR/results_claude_p_runs_task_t0 \
MASTER_LOG=/tmp/claude_p_parallel_task_t0.log \
    bash $REPO_PERT_DIR/hybrid_agent_cc/run_parallel_seeds.sh \
    > /tmp/claude_p_outer_task_t0.log 2>&1 &

# === 3. Monitor — `watch` in another terminal, or periodic status.sh ===
SUITE=libero_spatial_task TASK=0 \
OUTPUT_DIR=$REPO_PERT_DIR/results_claude_p_runs_task_t0 \
    bash $REPO_PERT_DIR/hybrid_agent_cc/status.sh
tail -20 /tmp/claude_p_parallel_task_t0.log

# === 4. When done, summarize ===
for s in 0 1 2 3 4 5 6 7; do
    audit=$REPO_PERT_DIR/results_claude_p_runs_task_t0/spatial_task_t0_s${s}.json
    base=$REPO_PERT_DIR/results_pi0_baseline_seeds_task_t0/baseline_libero_spatial_task_t0_s${s}.json
    [ -f $audit ] && jq -r --arg s $s '"s\($s)  hybrid_term=\(.libero_terminated)  regime=\(.regime)"' $audit
    [ -f $base ]  && jq -r --arg s $s '"      pi0_term=\(.libero_terminated)  chunks=\(.result.chunks_used)"' $base
done

# === 5. Rule-1 audit (did Pi0 silently finish any place?) ===
for s in 0 1 2 3 4 5 6 7; do
  wd=/tmp/hybrid_repl_spatial_task_t0_s$s
  echo "===== s$s ====="
  for log in $wd/log_*.json; do
    /opt/venv/openpi/bin/python -c "
import json
d=json.load(open('$log')); c=d.get('command',{}); r=d.get('result',{})
a=c.get('action')
if a in ('pi0_pick','release','set_object_pose'):
    print(f'  {a}: libero_term={r.get(\"libero_terminated\")}')"
  done
done
```

Then post a 5-line summary to the user:
- Hybrid (claude -p / opus 4.7): X/N libero_term, Z/N strict Rule-1
- Pi0 fullshot baseline: Y/N libero_term
- Total wall time: M min (claude -p), B min (Pi0 baseline)
- Notable cells: <list any stuck / odd recovery / hit set_object_pose>
- Cost (claude -p): approx N × $2-3 with opus 4.7

That's the loop. Have fun.
