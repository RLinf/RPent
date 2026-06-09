# LIBERO Hybrid Driver (perception-isolated, no teleport)

A `claude -p` worker drives a Pi0.5 + LIBERO REPL via JSON commands. **Pi0.5
does only the grasp; the LLM scripts every motion, localisation, and release.**


## TL;DR — one command after setup

```bash
cd <repo-root>
bash physicalagent/primitives/workspace_pro/hybrid_agent_cc/run_perception_cell.sh \
     libero_object_swap 2 0
```

That single line:

1. Launches `interactive_driver.py` with `--hide_object_coords --always_render`.
2. Waits for `state_00.json` (Pi0.5 load, ~90 s).
3. Starts `claude -p` with the perception worker prompt
   (`agent_task_prompt_perception.md`).
4. Worker reads the RGB image + depth map + camera calibration → back-projects
   target object pixel → world `xyz` → executes pre-pos → `pi0_pick` → carry → release.
5. Writes `<tag>.json` (audit, `regime: strict_perception`) and
   `recipe_<tag>.jsonl` (the command sequence) to the output dir.
6. Prints the audit + recipe paths at the end.

**See [`STRICT_HYBRID_GUIDE_PERCEPTION.md`](STRICT_HYBRID_GUIDE_PERCEPTION.md)
for the full perception protocol** — what each per-step file contains, the
exact back-projection math, the Pi0 prompt ladder, persistence, and failure
modes. For the legacy oracle-state guide see
[`STRICT_HYBRID_GUIDE.md`](STRICT_HYBRID_GUIDE.md).

## Three things you need before that line works

1. **Pi0.5 checkpoint** at the path hard-coded in
   [`primitives.py`](primitives.py) (search for `CHECKPOINT_PATH`):
   ```
   /mnt/public2/data_move/16T_5_slz_zyx/zhangyixian/zhangyixian/pi05_libero130_fullshot/30000
   ```
   Either put the checkpoint there, or edit that one line to your local path.
   This is Pi0.5 SFT on libero_130 fullshot at 30 k steps.

2. **Python venv** with `rlinf` (editable), `robosuite`, `libero`, `liberopro`,
   `openpi` SDK, `imageio`, `mujoco`, `scipy`, etc. The launchers default to
   `/opt/venv/openpi/bin/python` — override per call:
   ```bash
   PYTHON_BIN=/your/path/python bash .../run_perception_cell.sh libero_object_swap 2 0
   ```

3. **Claude Code CLI** — `claude` executable on `$PATH` and a logged-in
   subscription (the worker is invoked as `claude -p`).

See [`workspace_pro/REPRODUCE.md`](workspace_pro/REPRODUCE.md) for the full
out-of-repo dependency checklist (every env var that can override a default
and every external asset).

## Two run modes

| | **perception-isolated** (default, recommended) | oracle (legacy, free-exploration) |
|---|---|---|
| state object info | `object_names: [...]` only (no coords) | full `objects: {name: [x,y,z]}` |
| extra obs files | `image_cam_NN.png`, `depth_NN.npy`, `camera_meta.json` (always dumped) | same — depth always available |
| driver flags | `--hide_object_coords --always_render` | (none) |
| worker prompt | `agent_task_prompt_perception.md` | `agent_task_prompt.md` |
| 1-cell launcher | `run_perception_cell.sh` | use `run_one_cell.sh` directly |
| audit regime | `strict_perception` | `strict` |
| guide | [`STRICT_HYBRID_GUIDE_PERCEPTION.md`](STRICT_HYBRID_GUIDE_PERCEPTION.md) | [`STRICT_HYBRID_GUIDE.md`](STRICT_HYBRID_GUIDE.md) |

> Both modes get the depth interface — `camera_depths: True` is unconditional
> in `primitives.py:build_env_cfg`. The difference is only what's *in* the
> state JSON: oracle mode also serves GT coords, perception mode strips them.

## Common launch patterns

```bash
# Single cell, PRO swap perturbation (auto-routes LIBERO_TYPE=pro):
bash .../hybrid_agent_cc/run_perception_cell.sh libero_object_swap 2 0

# Single cell, standard libero (no perturbation):
bash .../hybrid_agent_cc/run_perception_cell.sh libero_spatial 3 0

# Single cell, libero_10 task perturbation (auto MAX_EPISODE_STEPS=5000):
bash .../hybrid_agent_cc/run_perception_cell.sh libero_10_task 5 0

# Custom GPU / output dir / model:
CUDA_DEVICE=2  OUTPUT_DIR=/mnt/runs/cell  MODEL=claude-opus-4-7 \
  bash .../hybrid_agent_cc/run_perception_cell.sh libero_object_swap 2 0

# Multi-seed × multi-task grid (4 GPUs parallel):
ENV_BASE=libero_object  REGIMES="swap task" \
TASKS="0 1 2 3 4 5 6 7 8 9"  SEEDS="0 1 2 3 4 5 6 7 8 9"  GPUS="0 1 2 3" \
  bash .../hybrid_agent_cc/run_perception_grid.sh > /tmp/percep_outer.log 2>&1 &
```

For monitoring a running grid: see the per-group master logs, plus
[`hybrid_agent_cc/status.sh`](workspace_pro/hybrid_agent_cc/status.sh) and the
fresh-agent monitor walkthroughs in
[`ONBOARDING_FRESH_AGENT_PERCEPTION.md`](workspace_pro/hybrid_agent_cc/ONBOARDING_FRESH_AGENT_PERCEPTION.md)
(and `ONBOARDING_FRESH_AGENT.md` for oracle mode).

## Key files

```
physicalagent/primitives/
├── README.md                                 # this file
├── STRICT_HYBRID_GUIDE_PERCEPTION.md         # perception protocol (read THIS)
├── STRICT_HYBRID_GUIDE.md                    # oracle / free-exploration protocol
├── NEW_PRIMITIVES.md                         # rotate_pitch / move_pose / etc.
├── interactive_driver.py                     # the REPL driver (entry, never run by hand)
├── primitives.py                             # LiberoPrimitiveDriver class + CHECKPOINT_PATH
├── pi0_baseline.py                           # Pi0.5 fullshot baseline (no LLM)
└── workspace_pro/
    ├── REPRODUCE.md                          # external deps + env var checklist
    ├── PRO_HYBRID_GUIDE.md                   # LIBERO-PRO-specific gotchas
    ├── env_calibration.md                    # scene/world coords + magic numbers
    ├── memory_snapshot/                      # frozen operating wisdom (read by worker)
    │   ├── MEMORY.md                         # one-line hooks
    │   └── feedback_*.md / project_*.md      # full notes
    ├── multi_seed_exp/
    │   ├── recipe_*_s0.jsonl                 # 280 seed-0 working recipes (templates)
    │   └── TELEPORT_REDO_CELLS{,_LAN}.md     # documented physical dead-ends
    ├── results_{spatial,object,goal,10}_pert/
    │   └── recipe_*_t*_s0.jsonl              # seed-0 source-library recipes (strategy priors)
    └── hybrid_agent_cc/                      # claude -p harness
        ├── ONBOARDING_FRESH_AGENT_PERCEPTION.md  # multi-cell monitor onboarding (perception)
        ├── ONBOARDING_FRESH_AGENT.md             # multi-cell monitor onboarding (oracle)
        ├── agent_task_prompt_perception.md   # WORKER prompt: perception
        ├── agent_task_prompt.md              # WORKER prompt: oracle
        ├── run_perception_cell.sh            # 1-cell perception launcher (the TL;DR command)
        ├── run_perception_grid.sh            # N-cell perception grid launcher
        ├── run_one_cell.sh                   # generic 1-cell launcher (oracle by default)
        ├── run_parallel_seeds.sh             # generic per-(suite,task) parallel launcher
        └── status.sh                         # progress dashboard for a running grid
```

## Self-contained reproduction summary

`run_perception_cell.sh` bakes in everything that makes a cell perception-mode
(prompt, driver flags, cell timeout, suite-to-LIBERO_TYPE routing). All other
knobs are overridable via env. So **the only friction between `git clone` and
a solved cell is the three setup steps above**:

1. Pi0.5 checkpoint at `CHECKPOINT_PATH` (or edit the path).
2. A Python venv with the right deps (or `PYTHON_BIN=...`).
3. Logged-in `claude` CLI on `$PATH`.

After that, `bash run_perception_cell.sh <suite> <task> <seed>` is a one-line
solve — verified end-to-end on `libero_object_swap 2 0` (342 s,
`libero_terminated: True`, `regime: strict_perception`,
back-projection-localised pick + scripted carry + release).

---

## Lower-level: single-rollout Pi0 smoke test

For a quick "is my Pi0.5 + env install actually wired up?" check, the
`LiberoPrimitiveDriver` class also exposes a standalone CLI (no LLM, no Claude
Code — just Pi0 running its native `pick(obj_text)` sub-instruction):

```bash
CUDA_VISIBLE_DEVICES=0 /opt/venv/openpi/bin/python \
    physicalagent/primitives/primitives.py \
    --task 0 --mode pick_only --seed 0 --max_chunks_pick 24 \
    --out /tmp/primitive_smoke.json
```

`--mode` is one of `full` (original task description, Pi0 fullshot),
`pick_only` (extracted `"pick up the {OBJ}"`), or `subinstr` (`pick_only` then
scripted place). For multi-task / multi-seed sweeps of the same Pi0-only
smoke, see `test_pick_generalization.py`. These are useful for sanity-checking
the Pi0 install without involving the LLM loop.
