# Reproducing hybrid_agent / hybrid_agent_cc on a fresh machine

This file is the checklist for someone who cloned the repo and wants to
run the same experiments. It enumerates everything that LIVES OUTSIDE
the repo (and therefore must be set up locally) plus every env var
that can override a hard-coded default.

## 0. What's in the repo

After clone you have:
- `physicalagent/primitives/` — driver code (`interactive_driver.py`,
  `primitives.py`, `pi0_baseline.py`) and the three guides
  (`STRICT_HYBRID_GUIDE.md`, `workspace_pro/PRO_HYBRID_GUIDE.md`,
  `workspace_pro/env_calibration.md`).
- `physicalagent/primitives/workspace_pro/hybrid_agent/` —
  API-driven agent (Anthropic SDK, parallel).
- `.../hybrid_agent_cc/` — `claude -p` driven agent (Claude Code,
  sequential). Read `hybrid_agent_cc/README.md` and
  `hybrid_agent_cc/ONBOARDING_FRESH_AGENT.md`.
- `.../memory_snapshot/` — frozen copy of the operating wisdom that
  both agents read at start (see `memory_snapshot/README.md`).
- `.../results_*/` — past runs as JSONL recipes + JSON audits (the
  in-repo "examples library" the agents browse for prior art).

## 1. External dependencies (NOT in repo)

### 1.1 Pi0.5 checkpoint

`primitives.py:CHECKPOINT_PATH` points to:

```
/mnt/public2/data_move/16T_5_slz_zyx/zhangyixian/zhangyixian/pi05_libero130_fullshot/30000
```

This is the `pi05_libero130_fullshot` checkpoint at step 30000 (SFT
on all 130 libero tasks). Download from your usual model store and
either:
- place it at the same path on your machine, OR
- `git apply` a one-line patch to set `CHECKPOINT_PATH` to your local
  path.

### 1.2 Python env

The driver runs in `openpi`'s Python (Pi0.5 + JAX/Torch + LIBERO).
On the origin machine: `/opt/venv/openpi/bin/python`. Set
`PYTHON_BIN=/path/to/your/openpi/bin/python` to override in
`run_one_cell.sh` / `run_baselines_*.sh`. (Default is still
`/opt/venv/openpi/bin/python`.)

For the agent side (Anthropic SDK only — no Pi0/sim deps): any
Python 3.9+ with `pip install anthropic`. The CC variant just needs
`claude` CLI on PATH.

### 1.3 LIBERO + LIBERO-PRO packages

See `PRO_HYBRID_GUIDE.md` §2 (Setup) for the full chain:
- LIBERO core (installed in openpi venv)
- LIBERO-PRO at `/opt/venv/openpi/libero_pro/` from
  `https://github.com/RLinf/LIBERO-PRO.git` commit `0bcf736`
- Apply `liberopro_register_perturbations.patch` from this repo
- Persist the perturbation BDDLs from `zhouxueyang/LIBERO-Pro` HF
  dataset at `/mnt/public2/zhangyixian/datasets/liberopro_hf/`,
  then run the sync script that overlays them into the liberopro
  install.

`PRO_HYBRID_GUIDE.md` has the exact bash. Don't paraphrase.

### 1.4 Claude credentials

- `hybrid_agent/` (API): set `ANTHROPIC_API_KEY` (and optionally
  `ANTHROPIC_BASE_URL` if you proxy).
- `hybrid_agent_cc/` (CC): `claude` CLI must already be authed —
  `claude setup-token` once, or have an active subscription session.

## 2. Env vars the scripts honour

| Var | Default | Where |
|---|---|---|
| `PYTHON_BIN` | `/opt/venv/openpi/bin/python` | `hybrid_agent_cc/run_one_cell.sh`, baselines |
| `REPO` | (auto: 5 dirs up from script) | `hybrid_agent_cc/run_one_cell.sh` |
| `MEMORY_DIR` | `<repo>/.../memory_snapshot/` | `hybrid_agent_cc/run_one_cell.sh` |
| `MEMORY_LIVE` | `/root/.claude/projects/-mnt-public2-zhangyixian/memory` | `sync_memory.sh` |
| `WORKDIR_ROOT` | `/tmp` | `hybrid_agent_cc/run_one_cell.sh` |
| `OUTPUT_DIR` | `<repo>/.../results_claude_p_runs` | `run_all_seeds.sh`, `run_one_cell.sh` |
| `MASTER_LOG` | `/tmp/claude_p_master_<suite>_t<task>.log` | `run_all_seeds.sh` |
| `MODEL` | `sonnet` | `run_all_seeds.sh`. Use `claude-opus-4-7` for opus. |
| `MAX_BUDGET_USD` | `10` | `run_one_cell.sh`. claude -p hard-terminates at this cap. Sonnet cells rarely exceed $1; opus 4.7 cells with image-heavy turns and recovery loops can reach $4–6 — keep the default at $10 so edge-case cells like extreme-layout seeds aren't cut off mid-thought. |
| `MAX_TURNS` | `60` | `run_one_cell.sh` |
| `MAX_EPISODE_STEPS` | `600` | `run_one_cell.sh` |
| `CUDA_DEVICE` | `0` | `run_one_cell.sh`, `run_baselines_*.sh` |
| `GPUS` | `"0 1 2 3"` | `run_parallel_seeds.sh` — space-separated GPU indices; len() = concurrency. |
| `STAGGER_S` | `30` | `run_parallel_seeds.sh` — seconds between successive launches; avoids simultaneous Pi0 model-load disk thrash + API TLS handshake burst. |
| `SUITE` `TASK` `SEEDS` | spatial_lan, 0, 0..9 | `run_all_seeds.sh`, `run_parallel_seeds.sh`, `run_baselines_*.sh` |
| `ANTHROPIC_API_KEY` `ANTHROPIC_BASE_URL` | env | `hybrid_agent/runner.py`, `parallel_launch.py` |

## 3. End-to-end smoke test

```bash
# 1. clone, set env
git clone <repo-url>
cd <repo>
export ANTHROPIC_API_KEY=...          # for API variant
# OR have `claude` CLI authed         # for CC variant

# 2. install Pi0 / LIBERO / LIBERO-PRO per PRO_HYBRID_GUIDE.md §2

# 3. confirm CHECKPOINT_PATH resolves (or patch it)
python -c "from physicalagent.primitives.primitives import CHECKPOINT_PATH; import os; print('Pi0 ckpt OK?', os.path.isdir(CHECKPOINT_PATH))"

# 4. (CC variant) one-cell smoke
bash physicalagent/primitives/workspace_pro/hybrid_agent_cc/run_one_cell.sh \
    libero_spatial_lan 0 0
# expected: ~5 min wall, audit file with libero_terminated=True

# 5. (API variant) one-cell smoke
cd physicalagent/primitives/workspace_pro/hybrid_agent
python runner.py --suite libero_spatial_lan --task 0 --seed 0 --model claude-sonnet-4-5
# expected: ~3 min wall when alone, similar result

# 6. monitor running sweep (any time)
bash physicalagent/primitives/workspace_pro/hybrid_agent_cc/status.sh
```

## 4. If something doesn't work

In approximate order of probability:

1. **`/opt/venv/openpi/bin/python: not found`** → set `PYTHON_BIN` to
   your openpi venv path.
2. **`Pi0 model load fails`** → CHECKPOINT_PATH points at a directory
   that doesn't exist; download or patch.
3. **`LIBERO_TYPE=pro env unknown`** → didn't apply
   `liberopro_register_perturbations.patch`. See PRO_HYBRID_GUIDE.md §2.2.
4. **`state_00.json never appears`** → driver crashed at boot. Read
   `/tmp/cc_driver_<tag>.log` (or `/tmp/hybrid_agent_driver.log` for
   the API variant).
5. **`claude -p` says "memory not found"** → on a fresh clone, you
   need the snapshot. Both variants now point at
   `workspace_pro/memory_snapshot/` instead of the live `/root/.claude/`
   path; ensure it's committed.
6. **API agent: connection refused / 403** → check API key, base URL,
   balance / quota (for proxy endpoints).

## 5. What to commit back

When you accumulate new wisdom on a fresh fork:

- New recipes / audits → drop them under `results_<suite>_pert/` or
  `results_agent_runs_parallel/`. The agents will browse them next time.
- New memory entries → write a new `feedback_*.md` in your local
  `/root/.claude/.../memory/` (or copy directly into
  `workspace_pro/memory_snapshot/`), then run `sync_memory.sh` and
  commit the snapshot.
- New magic numbers / failure modes that should reach future agents →
  also update `STRICT_HYBRID_GUIDE.md` or `PRO_HYBRID_GUIDE.md` so they
  appear in the guides the agent reads.
