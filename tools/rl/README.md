# Adaptive VLM robot workflows

[简体中文](README_zh.md) · [Install from zero](INSTALL.md) · [Configuration and deployment reference](REFERENCE.md)

Run a VLM directly through end-effector and perception tools. All simulators
share `--robot rl`; select the simulator and workflow explicitly:

| Simulator | `--simulator` | Workflows |
| --- | --- | --- |
| LIBERO / LIBERO-PRO | `libero` | `episode`, `zap_turn`, `zap_episode`, `zap_subtask` |
| RoboCasa365 | `robocasa` | `episode` |
| RoboTwin | `robotwin` | `episode` |

All modes use the API planner, native task instructions and measured robot
state. Success comes from the native simulator. See [workflow definitions,
SAM3 pooling and fixed-panel comparison](WORKFLOWS.md) for the five modes.

## Install and configure

Follow [INSTALL.md](INSTALL.md) once: it covers system prerequisites, isolated
Python environments, model/SAM3 downloads and each simulator's assets.
Copy the shared configuration and edit its paths or endpoints as needed:

```bash
# From the checkout root.
cp tools/rl/config.example.sh .rl-env.sh
```

The examples below assume the example GPU layout: SAM3 on GPU 0, one simulator
on GPU 1, and Qwen3.8-27B TP2 on GPUs 2–3. These are device ordinals, not a
required machine layout. An existing compatible model endpoint can replace
the local model service. Size GPU memory for the actual model, scene and
concurrency; start with one episode.

## Start the shared services

Use separate terminals, each opened at the checkout root. For the model:

```bash
source .venv-sglang/bin/activate
source .rl-env.sh
SERVE_GPUS=2,3 bash tools/rl/serve.sh
```

Wait until `curl --fail http://127.0.0.1:30080/v1/models` lists
`qwen3.8-27b`. In a second terminal, start the single shared SAM3 service:

```bash
source .venv/bin/activate
source .rl-env.sh
source tools/rl/env.sh
python -m rpent.robots.components.sam3_server --cuda-device 0 --port 8112
```

Wait for its listening endpoint in the log. In a third terminal, activate the
planner and load the same configuration:

```bash
source .venv/bin/activate
source .rl-env.sh
source tools/rl/env.sh
```

`RPENT_SAM3_ENDPOINT` makes episodes reuse SAM3. Without it, each episode
starts its own service on its simulator GPU.

## Run one task

Run these examples separately. `--cuda-device` selects the simulator GPU;
the model and SAM3 use their independently configured services.

LIBERO-PRO episode:

```bash
RPENT_OUTPUT_DIR=logs/libero-episode bash tools/rl/run.sh \
  --simulator libero --workflow episode \
  --suite libero_object --task 0 --seed 1 --cuda-device 1
```

RoboCasa Target50 episode (requires the RoboCasa installation):

```bash
RPENT_OUTPUT_DIR=logs/robocasa-episode bash tools/rl/run.sh \
  --simulator robocasa --workflow episode \
  --task-name PickPlaceCounterToCabinet --split target --seed 1 --cuda-device 1
```

RoboTwin episode (requires the RoboTwin installation):

```bash
RPENT_OUTPUT_DIR=logs/robotwin-episode bash tools/rl/run.sh \
  --simulator robotwin --workflow episode \
  --task-name click_alarmclock --task-config demo_randomized \
  --seed 100000 --cuda-device 1
```

RoboTwin task/preset/seed combinations must exist in the published evaluation
list; `robots/robotwin/eval/demo_randomized.json` contains the valid seeds.
RoboCasa task names and RoboTwin examples are also listed in
`tools/rl/jobs/robocasa-robotwin100.json`; this file is a task manifest, not a
request to launch 100 environments. LIBERO indices are zero-based within a
suite. `tools/rl/jobs/panel40.txt` and `libero-dev10.json` are sample manifests.

LIBERO-PRO ZAP, using the same services and task interface:

```bash
RPENT_OUTPUT_DIR=logs/libero-zap bash tools/rl/run.sh \
  --simulator libero --workflow zap_turn \
  --suite libero_object --task 0 --seed 1 --cuda-device 1
```

Use a fresh output directory per run. The default budgets are 100 model
requests, 10000 simulator steps and a 7200-second planner timeout. The model
gets at most 32768 output tokens per request and a 262144-token context.
The batch launcher additionally enforces a 14400-second episode timeout.
See the [reference](REFERENCE.md#important-settings) for all overrides.
The launcher explicitly defaults presence penalty to 0. The example sets
repetition penalty 1.0; environment overrides are recorded in the actual requests.

## Workflow and tools

`episode` uses explicit `view_env`, `segment` and `back_project` calls. The
three LIBERO `zap_*` modes provide annotated images and depth geometry
automatically: they retain context for one turn, a full episode or a subtask.
`zap_episode` has no todo; `zap_subtask` resets conversation on each valid
structured todo update. [The workflow guide](WORKFLOWS.md) lists every tool,
initialization rule, compaction boundary and the VLM-only `--image_size` option.

RoboCasa adds `move_base` with local planar velocities and yaw rate. RoboTwin
uses anatomical `arm` selection, labelled views and optional paired targets.
All LIBERO modes share the same robot/frame/gripper instructions. Optional
`--prompt` variants add the same robot detail to each mode. Native task text
comes directly from `get_task_language()`; there are no per-task prompt files.

## Read the result

Inspect `result.json` for native success, errors, turns and token counts;
`states.json` contains measured states and native termination flags. A clean
process exit alone does not imply success. Each run also retains camera/depth
artifacts, `episode.mp4`, `rollout.json`, native transcripts and `api_trace/`
with the actual requests, image bytes and complete model responses.
For an unexpected outcome, first inspect `result.json`, `env_server.log`,
`sam3_server.log` (when episode-owned) and the last tool result in the transcript.

For audited, resumable batches start with one resident episode per GPU:

```bash
python tools/rl/run_batch.py run "$RPENT_DATA_ROOT/rollouts/libero-dev10" \
  --jobs tools/rl/jobs/libero-dev10.json --gpus 1 --per-gpu 1 \
  --max-initializing-per-gpu 1 --workflow episode
python tools/rl/run_batch.py summary "$RPENT_DATA_ROOT/rollouts/libero-dev10"
```

Use `--workflow zap_turn` with a LIBERO-only manifest for ZAP. Retry, audit, recovery,
[offline installation](REFERENCE.md#offline-installation),
[remote simulators](REFERENCE.md#planner-and-simulator-on-separate-hosts) and
[RLinf/SFT integration](REFERENCE.md#rlinf-and-sft) are in the reference.

## Code layout

| Module | Responsibility |
| --- | --- |
| `robots/rl/robot_spec.py` | CLI, configuration, service lifecycle, planner selection |
| `robots/rl/prompt_bundle.py`, `prompts/*.md` | Supported combinations and reusable robot/workflow contracts |
| `robots/rl/toolkit.py` | Tool registration, native task text and episode artifacts |
| `robots/rl/tools.py`, `geometry.py` | LIBERO primitives and shared perception geometry |
| `robots/rl/backends.py`, `server.py` | RoboCasa/RoboTwin motion adapters and isolated native servers |
| `robots/rl/workflow.py`, `snapshot.py`, `images.py` | Shared agent context policy, visible todo/perception and display coordinates |
| `robots/rl/remote.py`, `remote_manager.py` | Remote episode workers and artifact transport |
| `robots/rl/rl_episode.py` | Single-episode interface for an RL trainer |
| `tools/rl/run.sh`, `run_batch.py` | Single episode and audited batch launchers |

Native simulator facades remain in `robots/libero`, `robots/robocasa` and
`robots/robotwin`; this workflow reuses them without their VLA toolkits.
A new simulator needs a native adapter, a robot contract and explicit workflow
support, rather than another copy of the planner loop. CPU regression tests
live in `tests/unit_tests/robots/rl`.
