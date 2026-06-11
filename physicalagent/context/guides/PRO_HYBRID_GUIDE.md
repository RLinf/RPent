# LIBERO-Pro Hybrid (Pi0.5 + LLM-in-the-loop) — Handover Guide

You are picking up the **LIBERO-Pro** evaluation track from a previous session.
This guide is everything you need to continue from a cold start. Read it once
end-to-end before you launch a driver.

This document layers on top of the base hybrid playbook. Read **both**:

- [`../STRICT_HYBRID_GUIDE.md`](../STRICT_HYBRID_GUIDE.md) — primitive
  vocabulary, the three Rules (0/1/2), command JSON schemas, mental model,
  worked examples on standard LIBERO.
- **This file** — LIBERO-Pro–specific setup, the four perturbation axes,
  Pi0 baseline runner, frame split, and current audit corpus.

Whenever this guide says "see Rule N" or "the standard primitive vocabulary",
it is referring to STRICT_HYBRID_GUIDE.md. **Every rule there applies here**;
this guide only *adds* constraints and tooling.

## 1. Why LIBERO-Pro

LIBERO-Pro
([paper](https://arxiv.org/pdf/2510.03827), [repo](https://github.com/Zxy-MLlab/LIBERO-PRO))
is a robustness benchmark on top of LIBERO. Each base task gets perturbed
along five axes; the paper shows all end-to-end VLAs (OpenVLA / Pi0 / Pi0.5
/ UniVLA) collapse on the two strongest axes:

| Axis | Suffix | Paper column | What changes | Headline result |
|---|---|---|---|---|
| **Task** | `_task` | **P1** | Instruction + goal predicate inverted | All VLAs ≈ 0.0 |
| **Position** | `_swap` | **P2** | Object initial positions swapped | All VLAs 0.0–0.4 |
| Semantic | `_lan` | — | Instruction paraphrased; goal unchanged | VLAs handle (memorize visual) |
| Object | `_object` | — | Object appearance / colour / scale | VLA visual policy stressed |
| Environment | `_environment` | — | Table / scene swapped | Visual policy stressed |

The agentic LLM-in-the-loop hybrid should win on **P1 and P2** by routing
the language and spatial-state channels through the LLM. Object and
Environment perturbations enter through the Pi0 vision channel and the
hybrid inherits the VLA's weakness there — declare those upfront, don't
oversell.

**Single-seed proof point (spatial t0):**

| | base | `_task` (P1) | `_swap` (P2) | `_lan` (Semantic) |
|---|---|---|---|---|
| Pi0 fullshot | ✓ 16 chunks | **✗ FAIL — picked wrong bowl** | ✓ 18 chunks | ✓ 16 chunks |
| Hybrid | ✓ | ✓ first attempt | ✓ second attempt | ✓ first attempt |

See `results_spatial_pert/REPORT_spatial_t0.md` for the full diagnostic.

## 2. Setup (do these once on a fresh checkout)

### 2.1. LIBERO-PRO repo

Cloned at `${LIBERO_PRO_PATH:-/path/to/LIBERO-PRO}/` from
`https://github.com/RLinf/LIBERO-PRO.git` (commit `0bcf736`). Already
installed editable into the openpi venv:

```bash
python -m pip show liberopro
# Name: liberopro  Version: 0.1.0  Location: ${LIBERO_PRO_PATH:-/path/to/LIBERO-PRO}
```

### 2.2. Apply the benchmark-registration patch

The upstream `__init__.py` does **not** expose the 16 perturbation suites
through `get_benchmark()`. Our patch
[`liberopro_register_perturbations.patch`](./liberopro_register_perturbations.patch)
adds them and also overrides `Task.language` to read each BDDL's actual
`:language` tag (so the perturbed instruction reaches Pi0 / hybrid).

```bash
cd ${LIBERO_PRO_PATH:-/path/to/LIBERO-PRO}
git apply <path-to>/workspace_pro/liberopro_register_perturbations.patch
```

If the patch is already applied (likely), `git status -s` shows clean. If
you ever reinstall liberopro, you must re-apply.

### 2.3. Huggingface dataset (already persisted at `${LIBEROPRO_DATASET_PATH:-/path/to/liberopro_hf}/`)

The LIBERO-PRO git repo ships **incomplete / broken** init files for several
perturbation suites:
- `libero_spatial_swap` — 0 BDDLs in the repo
- `libero_spatial_task` t3 / t7 — `.pruned_init` files are 0 bytes (empty tensors)
- (other suites may have similar gaps; treat the git repo as unreliable for
  perturbation data)

The full, correct set lives on Huggingface
([`zhouxueyang/LIBERO-Pro`](https://huggingface.co/datasets/zhouxueyang/LIBERO-Pro)).
A clean local copy has been persisted at:

```
${LIBEROPRO_DATASET_PATH:-/path/to/liberopro_hf}/
├── bddl_files/                    16 perturbation suites, 10 BDDLs each
└── init_files/                    16 perturbation suites, 10 init files each
```

Total: 1.2 MB. Covers `{libero_spatial, libero_object, libero_goal, libero_10}
× {swap, task, lan, object}`.

**Sync this into the liberopro install** (overwrites broken upstream files):

```bash
SRC=${LIBEROPRO_DATASET_PATH:-/path/to/liberopro_hf}
DEST=${LIBERO_PRO_PATH:-/path/to/LIBERO-PRO}/liberopro/liberopro
for suite_dir in $SRC/bddl_files/*/; do
  name=$(basename $suite_dir)
  mkdir -p $DEST/bddl_files/$name
  cp -f $suite_dir/*.bddl $DEST/bddl_files/$name/
done
for suite_dir in $SRC/init_files/*/; do
  name=$(basename $suite_dir)
  mkdir -p $DEST/init_files/$name
  cp -f $suite_dir/*.pruned_init $DEST/init_files/$name/
done
```

If you ever reinstall liberopro, run this sync again. If the persistent
copy is somehow gone, re-download with:

```bash
python -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='zhouxueyang/LIBERO-Pro', repo_type='dataset',
                  local_dir='${LIBEROPRO_DATASET_PATH:-/path/to/liberopro_hf}',
                  allow_patterns=['bddl_files/**','init_files/**'])"
```

### 2.4. Verify

```bash
LIBERO_TYPE=pro python -c "
import liberopro.liberopro.benchmark as bench
for n in ['libero_spatial_task','libero_spatial_swap','libero_spatial_lan']:
    b = bench.get_benchmark(n)(); t = b.get_task(0)
    print(f'{n} t0: {t.language!r}  trials={len(b.get_task_init_states(0))}')"
```

Expected:
```
libero_spatial_task t0: 'Pick the akita black bowl not between the plate and the ramekin and place it on the plate'  trials=50
libero_spatial_swap t0: 'Pick the akita black bowl between the plate and the ramekin and place it on the plate'  trials=50
libero_spatial_lan  t0: 'lift the black bowl between the plate and ramekin and set it on the plate'  trials=50
```

## 3. PRO-specific environment gotchas

Everything in STRICT_HYBRID_GUIDE applies. The following are **additional**
constraints for LIBERO-Pro (most live in detail at
[`env_calibration.md`](./env_calibration.md)).

### 3.1. Two scene frames, picked per-task

PRO scenes use either `living_room_table` or `kitchen_table` fixtures, and
the eef home z differs by ~0.5 m.

| Fixture | eef home z | Table top z | xy reachable | Where to use |
|---|---|---|---|---|
| `living_room_table` | ≈ 0.68 | ≈ 0.43 | `(x∈±0.30, y∈±0.30)` | basket / plate / pudding tasks |
| `kitchen_table` | ≈ 1.17 | ≈ 0.90 | `(x∈±0.30, y∈±0.30)` | stove / cabinet / drawer / microwave |

**Mandatory check** at session start: read `states.json[0].state.robot0_eef_pos[2]`.
If ≈ 0.68 you're in LIVING_ROOM frame; if ≈ 1.17 you're in KITCHEN frame.
Pick `pre_pos_z`, `carry_z`, `release_z` accordingly. Sending KITCHEN-frame
coordinates while the env is in LIVING_ROOM frame will crash the env worker
(EOFError) — silent loss of state.

For `libero_spatial_*` you are always in **KITCHEN frame** (spatial tasks
all use kitchen_table). Standard heights:

```
pre_pos_z = 1.05  # ~7 cm above objects at z≈0.97
carry_z   = 1.10  # safe traversal, well under upper limit 1.15
release_z = 1.01  # ~7 cm above plate top at z≈0.90
```

### 3.2. xy single-step ±0.30 cap

OSC will flip IK branches if you command `|x|>0.30` or `|y|>0.30` in a
single move. The eef jumps to the wrong half-space and the rest of the
run is corrupt. **Never command beyond ±0.30 in a single `move_to`.**
Use waypoints. (Detail in env_calibration.md.)

### 3.3. Slow long-distance carry for swap variants

If carrying an object across the table (e.g. P2 swap moves plate from
y=0.19 back → y=0.04 front, a 15 cm y-traversal), `step_clip=0.025` lets
the object slip in the gripper. The bowl ends up centimetres off target.

Mitigation (proven on `_swap` t0):
- `carry_z = 1.15` (higher than usual 1.10)
- `step_clip = 0.020` (slower)
- Verify mid-travel that `object_xyz - eef_xyz` is unchanged from post-pick

If offset drifted >5 mm, re-pre-position and re-pick. This pattern is
baked into `results_spatial_pert/recipe_spatial_swap_t0_s0.jsonl`.

### 3.4. Task language is now from BDDL, not from filename

After our patch, `benchmark.get_task(i).language` returns the actual
`:language` tag inside each perturbed BDDL. For `_task` and `_lan` this is
the perturbed instruction. The env passes this through `task_descriptions`
to Pi0 as the prompt. **Don't override** — that's the point of the test.

## 4. The four-cell experiment per (base task, seed)

For each base task you want to claim coverage on, generate four runs:

| Suite | Variant | What we expect |
|---|---|---|
| `libero_spatial`        | base sanity | Both Pi0 and hybrid pass |
| `libero_spatial_task`   | **P1 Task** | Pi0 ✗ (picks base target), hybrid ✓ (LLM flips target from instruction) |
| `libero_spatial_swap`   | **P2 Position** | Pi0 mixed, hybrid ✓ (reads coords from states.json step 0) |
| `libero_spatial_lan`    | Semantic | Both pass (paraphrase invariant) |

Replace `spatial` with `object`, `goal`, or `10` for the other base suites.

### 4.1. Hybrid run — same as STRICT_HYBRID_GUIDE except `LIBERO_TYPE=pro`

```bash
ps -ef | grep repl_driver | grep -v grep | awk '{print $2}' | xargs -r kill
cd ${PHYSICALAGENT_REPO_ROOT:-$(pwd)}
REPL_WORKDIR="${PHYSICALAGENT_WORKDIR_PREFIX:-$(python - <<'PY'
from physicalagent.config import get_default_workdir_prefix
print(get_default_workdir_prefix())
PY
)}"
rm -rf "$REPL_WORKDIR"
LIBERO_TYPE=pro CUDA_VISIBLE_DEVICES=0 python \
  physicalagent/backends/rlinf/repl_driver.py \
  --suite libero_spatial_task --task 0 --seed 0 --max_episode_steps 600
# (run in background; wait for $REPL_WORKDIR/states.json)
```

Then issue JSON commands per STRICT_HYBRID_GUIDE §"The command vocabulary".

Save the audit at the end as:

```
workspace_pro/results_<suite_base>_pert/<perturbation>_t<N>_s<M>.json
workspace_pro/results_<suite_base>_pert/recipe_<perturbation>_t<N>_s<M>.jsonl
```

e.g. `results_spatial_pert/spatial_task_t0_s0.json`. The audit JSON schema
matches what's in `results_all_spatial/`, plus three new fields you must
populate:

```jsonc
{
  "suite": "libero_spatial_task",
  "perturbation_type": "task (P1 Task perturbation)",
  "perturbed_task_language": "<actual :language from BDDL>",
  "perturbation_semantics": "<short description of what changed>",
  "expected_baseline_behavior": "<predicted Pi0 fullshot failure mode>",
  // ... and the standard fields from STRICT_HYBRID_GUIDE
}
```

### 4.2. Pi0 fullshot baseline — `pi0_baseline.py`

For the same (suite, task, seed), run the Pi0 baseline:

```bash
cd ${PHYSICALAGENT_REPO_ROOT:-$(pwd)}
LIBERO_TYPE=pro CUDA_VISIBLE_DEVICES=0 python \
  physicalagent/primitives/pi0_baseline.py \
  --suite libero_spatial_task --task 0 --seed 0 --max_chunks 60 \
  --out physicalagent/primitives/workspace_pro/results_spatial_pert/baseline_pi0_spatial_task_t0_s0.json \
  --save_image_dir physicalagent/primitives/workspace_pro/results_spatial_pert/baseline_imgs/task
```

Runtime: ~80s model load + 10–25s rollout. Reads the perturbed
`task_descriptions` from the env and drives Pi0 end-to-end with
`driver.run_full_task(max_chunks=…)`. Outputs the same audit shape as
the hybrid run but with `"regime": "pi0_fullshot_baseline"` so they're
trivially distinguishable.

**For P1 perturbations, expect Pi0 to "succeed at the wrong task"**
(pick the base-target object, place on plate, `libero_terminated=False`
because the goal predicate names a different object). Diagnose this by
reading `final_state` and printing object xy distance to the place target.

## 5. STRICT_HYBRID_GUIDE rules — PRO clarifications

- **Rule 0 (use images).** Even more important under PRO than vanilla
  LIBERO. P2 swap can move a large object (plate, cabinet) clear across
  the table; coordinates from the *base* task are meaningless. Always
  open `images/image_00.png` and describe the scene before deciding targets.
- **Rule 1 (no `pi0_end_to_end`).** Same as STRICT_HYBRID_GUIDE: Pi0
  performs the grasp via `pi0_pick` with `track_obj` cut; the LLM does
  every motion + release. Under PRO this is doubly important — handing
  back to Pi0 means handing back to the prompt-blind / memorized-place
  habit you are *trying to falsify*.
- **Rule 2 (multi-episode iteration).** Use freely. `_swap` t0 needed 2
  episodes. Document attempts in the audit JSON's `regime_history` field.

## 6. Existing corpus

Cataloged so you don't redo:

```
workspace_pro/
├── PRO_HYBRID_GUIDE.md                 ← this file
├── README.md                           ← patch overview
├── env_calibration.md                  ← OSC frame bounds + safe altitudes
├── liberopro_register_perturbations.patch
└── results_spatial_pert/
    ├── REPORT_spatial_t0.md            ← Pi0 baseline vs hybrid table
    ├── spatial_task_t0_s0.json         ← hybrid audits
    ├── spatial_swap_t0_s0.json
    ├── spatial_lan_t0_s0.json
    ├── recipe_spatial_{task,swap,lan}_t0_s0.jsonl
    ├── baseline_pi0_spatial_{base,task,swap,lan}_t0_s0.json
    └── baseline_imgs/{base,task,swap,lan}/{initial,final}.png
```

Single-seed proof point established. **None of t1–t9 done yet, and seed 0
only for t0**. That's the work that remains.

## 7. What to do next (priority order)

1. **Extend spatial to all 10 tasks at seed=0.**
   Same four-cell (base / _task / _swap / _lan) per task. Pi0 baseline runs
   are cheap (~100 s wall each). For hybrid runs, build on existing
   `results_all_spatial/t<N>_s0.json` recipes as starting points — the
   pick step is usually identical; the place coordinates change for `_swap`,
   the target object changes for `_task`.
2. **Scale to seeds beyond 0.** Each PRO benchmark exposes 50 trials per
   task. Parameterize recipes as `commands_for(states_json_step0) -> List[dict]`
   reading object positions at runtime. Existing recipes already follow
   this data-flow pattern; codify them as Python callables in
   `workspace_pro/strategies/<suite>/<task>.py`.
3. **Replicate on `libero_object`, `libero_goal`, `libero_10`.** Each has
   `_swap`, `_task`, `_lan`, `_object` available. Frame split applies
   (LIVING_ROOM vs KITCHEN per task — read `states.json[0].state.robot0_eef_pos[2]`).
4. **Aggregate into a main table.** `(suite × perturbation × {Pi0, hybrid})`
   grid. The headline number is the **conditional**: of the seeds Pi0
   fails on, what fraction does hybrid succeed on? That is the strongest
   single statistic the agentic decomposition can claim.

## 8. Quick reference for a brand-new session

```bash
# 1. Sanity-check liberopro patch present
LIBERO_TYPE=pro python -c \
  "import liberopro.liberopro.benchmark as b; print(b.get_benchmark('libero_spatial_task')().get_task(0).language)"
# → must read 'Pick the akita black bowl not between ...' (the perturbed text)

# 2. Start a hybrid driver (background)
cd ${PHYSICALAGENT_REPO_ROOT:-$(pwd)}
LIBERO_TYPE=pro CUDA_VISIBLE_DEVICES=0 python \
  physicalagent/backends/rlinf/repl_driver.py \
  --suite libero_spatial_task --task <N> --seed 0 --max_episode_steps 600

# 3. Wait for readiness
until [ -f $REPL_WORKDIR/states.json ] && [ -s $REPL_WORKDIR/states.json ]; do sleep 5; done

# 4. Open states.json (step 0 entry) AND images/image_00.png; describe the scene; decide target
# 5. Issue JSON commands per STRICT_HYBRID_GUIDE §"The command vocabulary"
# 6. Save audit + recipe to workspace_pro/results_<base>_pert/

# 7. Run Pi0 baseline for the same (suite, task, seed)
LIBERO_TYPE=pro CUDA_VISIBLE_DEVICES=0 python \
  physicalagent/primitives/pi0_baseline.py \
  --suite libero_spatial_task --task <N> --seed 0 --max_chunks 60 \
  --out physicalagent/primitives/workspace_pro/results_spatial_pert/baseline_pi0_spatial_task_t<N>_s0.json
```

When in doubt about a primitive or a rule, the source of truth is
[`../STRICT_HYBRID_GUIDE.md`](../STRICT_HYBRID_GUIDE.md). This guide only
adds; it does not override.
