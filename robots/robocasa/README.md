# RoboCasa in RPent

## Overview

RPent runs the RoboCasa365 kitchen benchmark with a PandaOmron mobile
manipulator and the frozen RLDX-1 policy. An agentic planner selects RPent
primitives, while RLDX-1 executes the manipulation skills. The integration is
planner-agnostic: API planners, Claude Code, Codex, and future RPent planners
all use the same RoboCasa toolkit.

The public Target50 protocol covers 50 tasks from the RoboCasa365 `target`
environment split. It is intentionally expressed as an immutable manifest and
ordinary single-task `rpent` commands; RPent does not include a benchmark batch
launcher.

## Runtime Flow

```text
rpent CLI -> task-memory sync -> environment and VLA servers
          -> planner toolkit -> RoboCasa state.success
```

Unless external endpoints are supplied, RPent starts one RoboCasa environment
server and one RLDX-1 VLA server for the run. The planner receives the current
task language and observations through the RoboCasa toolkit. The environment's
own `_check_success()` result is surfaced as `state.success` and is the only
evaluation success signal.

## Installation

RLDX-1 requires Python 3.10. From the RPent repository root, create a dedicated
environment and install the RoboCasa extra with the validated Torch pair:

```bash
uv venv --python 3.10
source .venv/bin/activate
uv pip install -e ".[robocasa]" \
  --constraint robots/robocasa/eval/target50-constraints.txt \
  --override robots/robocasa/eval/target50-overrides.txt \
  --torch-backend=cu126
uv pip check
```

The Target50-specific [constraints file](eval/target50-constraints.txt) pins
the compatibility-sensitive package versions validated for reproduction
without narrowing the shared RPent dependency ranges for LIBERO or RoboTwin.
The accompanying [override file](eval/target50-overrides.txt) resolves the
formal environment directly to the immutable Robosuite revision while the
ordinary `robocasa` extra continues to track its maintained `rpent` branch.

Download the RoboCasa kitchen assets outside `site-packages`, then export the
path printed by the command. Target50 does not use RoboCasa dataset or teleop
macros, so skip the optional private-macros setup:

```bash
robocasa-download-assets --assets-path ~/.robocasa/assets --no-macros -y
export ROBOCASA_ASSETS_PATH=~/.robocasa/assets
```

The RPent Robosuite fork provides the Omron base-mounted `navview` camera,
composed by MuJoCo as `mobilebase0_navview`. Target50 freezes Robosuite at
`97cfbde4b68d8ec43dad20cf4747297866a6ca2e`; the Target50 override in the
installation command above selects that exact revision. No installed XML file
needs to be patched manually.

## RLDX-1 Checkpoint

Download the RoboCasa365-finetuned checkpoint at the revision frozen by the
Target50 manifest:

```bash
hf download RLWRLD/RLDX-1-FT-RC365 \
  --revision 587e9ecdcc5e7184fcc17f58713908edff5af041 \
  --local-dir ./checkpoints/rldx-1-ft-rc365
```

Pass that directory to `--vla-model-path`. The checkpoint is not distributed
inside RPent.

## Task Memory

Select automatic synchronization with `--memory-profile hf` (the default).
Before every such ordinary run, RPent's shared memory manager synchronizes the
`robocasa/**` subtree from the public
[`RLinf/RPent-memory`](https://huggingface.co/datasets/RLinf/RPent-memory/tree/main/robocasa/results)
dataset. Files land under `memory/robocasa/results`, so an online ordinary
run does not require a separate memory download command.

The published RoboCasa corpus contains 111 files: 43 audit JSON files, 43
recipe JSONL files, and 25 task Markdown files. There is no global memory. For
the current task, the planner may read only:

```text
memory/robocasa/results/<Task>_s0.json
memory/robocasa/results/recipe_<Task>_s0.jsonl
memory/robocasa/results/<Task>.md  # optional
```

The Markdown files cover all 16 Composite-Seen tasks and 9 Composite-Unseen
tasks. Seven Composite-Unseen tasks have no published memory and still run from
live observations:

```text
HeatKebabSandwich, PanTransfer, PortionHotDogs, SeparateFreezerRack,
WaffleReheat, WashFruitColander, WeighIngredients
```

Memory is strategy evidence, not a trajectory to replay. The planner must not
read another task's files or reuse historical coordinates, poses, pixels, or
subtask prompts in place of the current task language and observations.

Ordinary runs synchronize the dataset's current `main`. Formal Target50 runs
instead use the immutable memory snapshot
`551fc3157b3e56b40a3d3a3b4c7ff81721ebe89b`:

```bash
hf download RLinf/RPent-memory \
  --repo-type dataset \
  --revision 551fc3157b3e56b40a3d3a3b4c7ff81721ebe89b \
  --include "robocasa/**" \
  --local-dir ./target50-memory
```

## Run One Task

HTTP RPC endpoints whose hostname is `127.0.0.1` or `localhost` are reached
directly, whether RPent starts the worker or the user supplies the endpoint.
Every other hostname and IP uses the standard proxy environment. Codex applies
the same two-host exception only to its child process for the local MCP
connection. Leave `HTTP_PROXY` and `HTTPS_PROXY` unchanged when Hugging Face,
a remote planner, or another remote service requires them; the default runtime
does not require a shell-wide `NO_PROXY` setup.

If a user-supplied local service uses another hostname or IP and should be
reached directly, add that exact value to the user's existing `NO_PROXY` and
`no_proxy` configuration.

The default HF profile synchronizes memory automatically:

```bash
rpent --robot robocasa \
  --task-name OpenDrawer \
  --split target \
  --seed 1 \
  --vla-model-path ./checkpoints/rldx-1-ft-rc365 \
  --cuda-device 0 \
  --planner claude_code \
  --model claude-opus-4-8 \
  --memory-profile hf
```

To use a reviewed local corpus, point `--memory-dir` at its RoboCasa memory
root (the directory that contains `results/`):

```bash
rpent --robot robocasa \
  --task-name OpenDrawer \
  --split target \
  --seed 1 \
  --vla-model-path ./checkpoints/rldx-1-ft-rc365 \
  --cuda-device 0 \
  --planner claude_code \
  --model claude-opus-4-8 \
  --memory-profile local \
  --memory-dir ./target50-memory/robocasa
```

Planner credentials are supplied by the user outside the repository. See the
[planner configuration guide](../../docs/source-en/rst_source/usage/configure_planner.rst)
for all supported backends.

## Harness VLA Target50 Reproduction

[`eval/target50.json`](eval/target50.json) is the canonical manifest for the
Harness VLA reproduction on RoboCasa Target50. It freezes task membership,
seeds, time limits, dependency revisions, memory scope, the success source,
and retry policy. Its protocol ID is
`robocasa-harness-vla-v1`.

| Split | Tasks | Seeds per task | Cell timeout | Cells |
|---|---:|---:|---:|---:|
| Atomic | 18 | 1-10 | 1800 s | 180 |
| Composite-Seen | 16 | 1-5 | 3600 s | 80 |
| Composite-Unseen | 16 | 1-5 | 3600 s | 80 |
| **Total** | **50** | | | **340** |

Run each manifest cell with the ordinary CLI. The reference Codex profile is
`gpt-5.5`, `xhigh`, and `max_turns=100`; this profile does not restrict the
RoboCasa runtime to Codex. The scene identity is the ordinary `--seed` value;
do not set `RLDX_RESET_SEED`. Ordinary RoboCasa uses `max_chunks=70`; Target50
alone overrides it to 40. Freeze the Target50 RLDX execution values before
running the cells:

```bash
export RLDX_MAX_CHUNKS=40
export RLDX_SETTLE_PATIENCE=999
export RLDX_ACTION_STEPS_PER_CHUNK=8
unset RLDX_RESET_SEED
```

An Atomic cell is:

```bash
rpent --robot robocasa \
  --task-name OpenDrawer \
  --split target \
  --seed 1 \
  --vla-model-path ./checkpoints/rldx-1-ft-rc365 \
  --cuda-device 0 \
  --planner codex \
  --model gpt-5.5 \
  --reasoning-effort xhigh \
  --max-turns 100 \
  --planner-timeout-s 1800 \
  --memory-profile local \
  --memory-dir ./target50-memory/robocasa \
  --output-dir ./runs/target50/atomic/OpenDrawer_s1
```

Use `--planner-timeout-s 3600` for Composite-Seen and Composite-Unseen cells.
Execute Atomic 180, Composite-Seen 80, then Composite-Unseen 80. Each cell must
own its environment and VLA worker; GPU concurrency is an execution setting and
does not change the manifest.

Each completed command atomically writes `<output-dir>/result.json`. This file
uses the final environment `state.success`, records the effective protocol
settings, and deliberately omits provider errors and credentials. After all
cells finish, validate the fixed denominator and print the task-weighted score:

```bash
python -m robots.robocasa.eval.validate_target50 ./runs/target50
```

The expected layout is
`<results-root>/<manifest-split>/<Task>_s<seed>/result.json`. A valid planner
timeout remains a failed cell; an infrastructure error is rejected for rerun.

## Success and Retry Policy

- A cell succeeds only when its final recorded environment state has
  `state.success=true`. A planner-provided `finish(status=...)` value is not an
  evaluation label.
- A valid task failure and a planner timeout remain in the fixed denominator
  and are not retried.
- An infrastructure failure may be retried only when it produced no valid
  environment result for the cell.
- All 340 cells remain in the denominator, including the seven Unseen tasks
  without task memory.

## Published Results

The published Codex reproduction reports `163/180` Atomic, `49/80`
Composite-Seen, and `12/80` Composite-Unseen successes, for a task-weighted
RoboCasa365 score of `57.00%`. See the
[complete per-task table](eval/target50_codex_results.md) for comparison with
the Harness VLA reference results and for the aggregation boundary.

## Troubleshooting

- **Assets are missing:** rerun `robocasa-download-assets` and export
  `ROBOCASA_ASSETS_PATH` in the launch shell. Private dataset and teleop macros
  are not required by Target50.
- **`mobilebase0_navview` is missing:** reinstall the frozen Robosuite revision
  above. Do not edit files under `site-packages`.
- **The RLDX server cannot load:** verify `--vla-model-path`, CUDA visibility,
  and `vla_server.log`. FlashAttention is optional; RLDX-1 can use PyTorch SDPA.
- **Task memory is missing:** check `memory/robocasa/results` for HF mode or
  the exact directory passed to `--memory-dir`. Do not substitute another
  task's files. Missing memory is expected for the seven tasks listed above.
- **A server fails to start:** inspect `env_server.log`, `vla_server.log`, and
  `run.log` inside the cell's output directory.
- **A custom endpoint unexpectedly follows an HTTP proxy:** only the exact
  `127.0.0.1` and `localhost` hostnames bypass proxies automatically. Other
  hostnames and IPs use the standard proxy environment; add the exact host to
  `NO_PROXY` and `no_proxy` only when it should be reached directly.

## Further Documentation

- [English RoboCasa usage guide](../../docs/source-en/rst_source/usage/robocasa.rst)
- [Chinese RoboCasa usage guide](../../docs/source-zh/rst_source/usage/robocasa.rst)
- [Planner configuration](../../docs/source-en/rst_source/usage/configure_planner.rst)
- [Harness VLA overview](../../docs/source-en/rst_source/awesome_works/harnessvla.rst)
