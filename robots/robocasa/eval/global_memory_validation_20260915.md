# RoboCasa global memory validation — 2026-09-15

## Scope and outcome

All functional and infrastructure checks passed. All ten focused evaluations produced valid, auditable environment results: **3/10 successes** (StirVegetables: 2/5; SteamInMicrowave: 1/5), with 0 infrastructure retries.

This is functional validation on two tasks, each at seeds 1–5, with
`--memory-policy task-global`. It does not measure a full Target50 score or
compare global memory against task-only. Valid environment failures and
timeouts are retained; only infrastructure failures may be retried, at most
three attempts per cell.

## Versions

- Code base: `e7136689dd3253b59d0d5945823816e9845e8a82` from `RLinf/RPent:main`.
- Evaluated code: `2e82ad7e4d80c3d0ccc74e4bc91b4e1f3a59ad42` and
  `77b18d09be3c9030e5ad4343ea9f9c7cff959588`; per-cell provenance is in the
  accompanying JSON. The first three StirVegetables cells used the former.
  Seed 4 used the latter's runtime files before that commit was recorded;
  its original launch HEAD and runtime-file hashes are preserved separately.
  Later cells used the latter commit.
- Dataset PR: https://huggingface.co/datasets/RLinf/RPent-memory/discussions/7; immutable revision: `d1a086d857e7fa53576d46c3851afa8e239b1b64`.
- Corpus SHA-256:
  `a7349a5b0e9fa6e92cb890983e10655dd9d7dfb53f129788d889efd80baa256b`.
- RLDX checkpoint: `RLWRLD/RLDX-1-FT-RC365` at
  `587e9ecdcc5e7184fcc17f58713908edff5af041`.
- Robosuite: `RLinf/robosuite` at
  `97cfbde4b68d8ec43dad20cf4747297866a6ca2e`.
- Python 3.10.12; Torch 2.7.0+cu126; torchvision 0.22.0+cu126;
  transformers 4.57.6; MuJoCo 3.3.1; NumPy 1.26.4;
  rlinf-rldx 1.0.1.post10; rlinf-robocasa365 1.0.1;
  openai-codex and CLI 0.147.0.
- GPU: one NVIDIA RTX 4090, driver 550.127.08.

The ten cells ran against the local staged corpus, before the data PR was
created. They correctly record `hf_revision: null`. After data publication,
the immutable PR revision was downloaded through the actual HF sync path,
compared with all staged memory hashes and read through real Codex/MCP in
both policies. The original evaluation provenance was preserved.

## Checks

- Offline package audit: 103/103 memory bodies byte-exact, 43 JSON/recipe pairs,
  expected nine Markdown removals, and a verified complete data patch.
- All 100 real task/policy combinations selected the intended files. Unit
  coverage includes cross-task rejection, global disabled, missing layers,
  corruption, stale cache entries, CLI/Dashboard parity and result isolation.
- Complete unit suite: **550 passed, 2 skipped**. The skips require the
  optional LeRobot dataset export dependency; they are unrelated to RoboCasa.
- Repository pre-commit and commit-message checks passed.
- Strict English and Chinese Sphinx builds passed.
- **Four real upstream simulator smoke tests passed**.
- Real GPU VLA OpenDrawer smoke succeeded (208 environment actions), using
  the frozen checkpoint and 40/999/8 parameters.
- Real Codex → MCP reads passed with task-global (four files) and task-only
  (three files), using the configured GPT-5.5/xhigh authentication.
- The fixed HF PR commit synchronized successfully; all 103 memory hashes
  and CORPUS.json matched staging and deleted Markdown files were absent.
  Real Codex → MCP reads passed again in both policies at that revision.
- All ten focused cells had complete selected-memory reads, exact live
  VLA task instructions, effective 40/999/8 parameters and <=100 turns.

The pre-existing LIBERO VLA and SAM3 services and their identified children
were stopped before GPU validation; all 17 recorded processes exited. The
previous RoboCasa automatic queue was paused. The checkpoint, simulator
assets and Python runtime were reused.

The noninteractive Codex loop previously logged `max_turns` without enforcing
it. The patch uses the existing Dashboard interrupt mechanism and drains
completion events normally. A real one-turn Codex smoke completed at one
turn without a planner error. All ten evaluated cells stayed within 100
actual turns; no valid task result was retried because of this correction.
SteamInMicrowave seed 2 stopped at exactly 100 turns with a valid failed
environment result; its button predicate remained false.
The memory corpus, prompt, VLA and simulator code were unchanged between the
two evaluated code commits.

## Focused environment results

| Task | Seed | Environment success | Seconds | Turns | VLA calls | Memory reads |
|---|---:|:---:|---:|---:|---:|:---:|
| StirVegetables | 1 | no | 1372.1 | 81 | 12 | 4/4 |
| StirVegetables | 2 | no | 1651.7 | 99 | 18 | 4/4 |
| StirVegetables | 3 | yes | 445.8 | 23 | 6 | 4/4 |
| StirVegetables | 4 | no | 1147.7 | 50 | 13 | 4/4 |
| StirVegetables | 5 | yes | 248.6 | 17 | 4 | 4/4 |
| SteamInMicrowave | 1 | no | 546.7 | 22 | 6 | 4/4 |
| SteamInMicrowave | 2 | no | 1707.5 | 100 | 17 | 4/4 |
| SteamInMicrowave | 3 | no | 1652.6 | 88 | 11 | 4/4 |
| SteamInMicrowave | 4 | no | 540.7 | 28 | 9 | 4/4 |
| SteamInMicrowave | 5 | yes | 327.3 | 19 | 4 | 4/4 |

Success is the simulator's `state.success`, independent of planner text.
Each cell read these four complete files before actions:

```text
task_only/<Task>_s0.json
task_only/<Task>_s0_recipe.jsonl
task_only/<Task>.md
global/GLOBAL_MEMORY.md
```

All recorded VLA calls used the complete live task language verbatim and
effective parameters `max_chunks=40`, `settle_patience=999`,
`n_action_steps=8`. The environment ran with no reset. The planning profile
was GPT-5.5/xhigh, at most 100 turns and 3600 seconds per cell.

The [machine-readable audit](global_memory_validation_20260915.json) includes
each result, final observable task progress, memory selection and read paths,
VLA call parameters, actual turn counts, code provenance and SHA-256 hashes
of the source evidence files. Complete local run directories retain
`states.json`, images, `memory_reads.jsonl`, `invocation.json`, `result.json`
and planner transcripts. Raw planner transcripts and credentials are not
included in the publication.

## Reproduce

Use the [RoboCasa setup instructions](../README.md) with the versions above.
Reuse existing checkpoint and simulator assets when available. Supply your
own planner authentication outside the repository.

```bash
export ROBOCASA_ASSETS_PATH=/path/to/existing/robocasa/assets
export MUJOCO_GL=egl
export RLDX_ATTN_IMPL=sdpa
export RLDX_MAX_CHUNKS=40
export RLDX_SETTLE_PATIENCE=999
export RLDX_ACTION_STEPS_PER_CHUNK=8
export RLDX_ALLOW_RESET=0
unset RLDX_RESET_SEED

for task in StirVegetables SteamInMicrowave; do
  for seed in 1 2 3 4 5; do
    rpent --robot robocasa --task-name "$task" --split target --seed "$seed" \
      --vla-model-path /path/to/existing/RLDX-1-FT-RC365 --cuda-device 0 \
      --planner codex --model gpt-5.5 --reasoning-effort xhigh \
      --max-turns 100 --planner-timeout-s 3600 \
      --memory-profile hf --memory-policy task-global \
      --memory-revision d1a086d857e7fa53576d46c3851afa8e239b1b64 \
      --output-dir "focused-results/composite_seen/${task}_s${seed}"
  done
done
```

For offline reproduction, use `--memory-profile local --memory-dir <corpus>`
and omit `--memory-revision`; the corpus root must contain `CORPUS.json`.
For a v2 ablation, select `--memory-policy task-only` and use a separate
output root. The frozen v1 entrypoint remains documented in the README.

Create a two-task manifest and validate only these ten expected cells:

```bash
python - <<'PY'
import json
from pathlib import Path
manifest = json.loads(Path('robots/robocasa/eval/target50_v2.json').read_text())
split = manifest['splits']['composite_seen']
split.update(tasks=['StirVegetables', 'SteamInMicrowave'], task_count=2, cell_count=10)
manifest.update(splits={'composite_seen': split}, total_tasks=2, total_cells=10)
Path('focused-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
PY
python -m robots.robocasa.eval.validate_target50 focused-results \
  --manifest focused-manifest.json
```

Run repository checks with the configured development dependencies:

```bash
env -u RLDX_MAX_CHUNKS -u RLDX_SETTLE_PATIENCE \
  -u RLDX_ACTION_STEPS_PER_CHUNK -u RLDX_ALLOW_RESET \
  -u RLDX_RESET_SEED pytest tests/unit_tests
pre-commit run --all-files
RPENT_RUN_ROBOCASA_INTEGRATION=1 pytest \
  tests/integration_tests/robots/robocasa/test_target50_runtime_smoke.py -v
sphinx-build -W --keep-going -b html docs/source-en /tmp/rpent-docs-en
sphinx-build -W --keep-going -b html docs/source-zh /tmp/rpent-docs-zh
```

The data PR must be merged before the code PR. Maintainers review and merge
both changes.
