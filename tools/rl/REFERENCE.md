# RL workflow reference

[Run a first task](README.md) · [Installation](INSTALL.md) · [简体中文](REFERENCE_zh.md)

## Important settings

`serve.sh` accepts additional SGLang arguments after its environment settings.
Its listener is local (`127.0.0.1`). The planner and server both default to a
262144-token context.

| Service variable | Default | Meaning |
| --- | --- | --- |
| `RPENT_SGLANG_PYTHON` | `python` | Interpreter from the active SGLang environment; does not activate CUDA libraries |
| `SERVE_GPUS` | `2,3` | Visible physical GPUs; their count sets TP |
| `MODEL_PATH` | Required | Local checkpoint directory |
| `MODEL_NAME` | `qwen3.8-27b` | Served name; match `RPENT_MODEL` after `openai-chat:` |
| `SERVE_PORT` | `30080` | Local model port |
| `SERVE_MEM_FRACTION` | `0.8` | SGLang static memory fraction |
| `SERVE_CONCURRENCY` | `8` | Maximum running model requests |
| `ENABLE_MTP` | `1` | EAGLE/MTP: 3 steps, top-k 1, 4 draft tokens; set 0 for a model without MTP weights |

| Episode variable / flag | Default | Meaning |
| --- | --- | --- |
| `RPENT_MODEL`, `RPENT_BASE_URL` | Required | Served model and its `/v1` endpoint |
| `RPENT_SAM3_ENDPOINT` | Unset | Existing shared SAM3 endpoint |
| `RPENT_PROMPT` | `simple` | `simple`, `base`, `base_gripper`, `base_gripper_camera` |
| `RPENT_REASONING_EFFORT` | `xhigh` for Qwen3.8; `none` otherwise | `none` omits the effort tier while thinking stays enabled |
| `RPENT_MAX_TURNS` | `100` | Model requests per episode |
| `RPENT_MAX_EPISODE_STEPS` | `10000` | Simulator-step budget |
| `RPENT_MAX_TOKENS` | `32768` | Output-token budget per request, including thinking |
| `RPENT_CONTEXT_WINDOW` | `262144` | Planner context budget |
| `RPENT_COMPACTION_FRACTION` | `0.7` | History-compaction threshold |
| `RPENT_PLANNER_TIMEOUT_S` | `7200` | Planner wall-clock budget; set explicitly for a different limit |
| `RPENT_PRESENCE_PENALTY`, `RPENT_REPETITION_PENALTY` | Unset | Optional explicit sampling overrides |
| `RPENT_OUTPUT_DIR` | `logs/rl-adaptive-<prompt>-<time>` | Episode output directory |
| `--suite`, `--task`, `--seed` | Suite/task required; seed 0 | LIBERO task and initialization seed |
| `--cuda-device` | Unset | Simulator GPU, independent of the shared SAM3 GPU |

Thinking sampling is temperature 1.0, top-p 0.95, top-k 20. To reproduce the
SFT-checkpoint evaluation penalties, explicitly export
`RPENT_PRESENCE_PENALTY=0` and `RPENT_REPETITION_PENALTY=1.0`.
`--observation-mode adaptive` remains accepted; image/text modes are rejected.
Other arguments to `run.sh` pass through to `rpent`.

## Batch collection, saving and recovery

Use the model/SAM3 exports above, then launch one resumable queue:

```bash
python tools/rl/run_batch.py run "$RPENT_DATA_ROOT/rollouts/new-run" --jobs tools/rl/jobs/panel40.txt --seeds 3-66 --gpus 1 --per-gpu 4
```

A jobs line is `adaptive libero_object 0 0`. `panel40.txt` contains task 0–9
from each of `libero_object`, `libero_goal`, `libero_spatial`, `libero_10`.
Choose environment concurrency for the available simulator memory and startup
load; the model-service concurrency is a separate limit.

| Batch argument | Default | Meaning |
| --- | --- | --- |
| `--jobs` | Required | JSON tasks or the four-field LIBERO manifest |
| `--seeds` | Manifest seeds | Comma-separated seeds/ranges, e.g. `0,3-66` |
| `--gpus`, `--per-gpu` | GPUs required; 1 worker/GPU | Environment GPUs and worker count per GPU |
| `--prompt` | `simple` | One prompt variant for the batch |
| `--workflow` | `episode` | episode / episode_image / zap_turn / zap_episode / zap_subtask |
| `--no-thinking` | off | Disable Qwen thinking for the whole batch; recorded in its manifest |
| `--image_size` | `1024` | VLM width/height: 1024 / 512 / 256; native perception stays1024 |
| `--start-barrier` | Off | Admit model requests after all pending environments are resident |
| `--max-initializing-per-gpu` | `8` | Concurrent cold starts, separate from episode slots |
| `--endpoints` | `RPENT_BASE_URL` | Comma-separated policy `/v1` endpoints |
| `--start-interval` | `1` second | Spacing between initial worker starts |
| `--episode-timeout` | `14400` seconds | Outer process timeout per episode |
| `--infrastructure-retries` | `1` | Additional attempts after an error/audit failure per invocation |

Each job keeps independent `attempt-NNN` directories. The audit checks native
success, complete HTTP/native responses, tools and PNG SHA-256 before writing
`COMPLETED.json` and atomically updating `valid-index.json`. A normally completed
unsuccessful task remains valid data. Invalid attempts retain `INVALID.json`;
stopped attempts retain `INTERRUPTED.json`.

Normal output/context-limit and episode-timeout failures are sealed as
`EVALUATED.json`, counted once in `outcomes-index.json`, and excluded from the
training index. They are not retried as infrastructure failures. Interrupted
or corrupt attempts remain separate; an incomplete response is never labelled
a complete training output. `valid-index.json` contains only audited records.

SIGINT/SIGTERM stop owned episode process groups and preserve attempts. Repeat
the identical command to resume; sealed episodes are re-audited and skipped.
A directory lock prevents duplicate collectors, and the manifest rejects changed
tasks, model, prompt or generation settings. The read-only summary re-audits
sealed episodes without overwriting a live collector's index:

```bash
python tools/rl/run_batch.py summary "$RPENT_DATA_ROOT/rollouts/new-run"
```

Episode artifacts include `result.json`, `states.json`, transcripts, video and
`api_trace/`. The trace retains actual post-compaction requests, original PNG
bytes, native messages, complete streaming responses and compaction events.
The shared reader is `rpent.evaluation.trajectory`; dataset encoding and SFT
belong to `ms-swift/examples/rpent`.

### Compare thinking settings

Set `RPENT_REASONING_EFFORT=low` or `medium` for a Qwen3.8 batch with thinking
enabled. To disable thinking, set `RPENT_REASONING_EFFORT=none` and add
`--no-thinking` to `run_batch.py run` (or to `run.sh` for a single episode).
`none` alone omits the effort tier and keeps thinking enabled. Use a fresh
batch directory for each setting; resume rejects a changed thinking setting.
The planner keeps its sampling parameters unchanged, so disabling thinking
does not silently switch to a different sampling preset.

## Offline installation

Prepare separate **complete** wheelhouses for the two environments on a
networked Linux x86_64/Python 3.12 host. First follow the online installation
above. In each environment, build its requirements into its own wheelhouse:

```bash
source .venv/bin/activate
uv pip install pip wheel
python -m pip wheel --pre --wheel-dir wheelhouse-rpent --extra-index-url https://download.pytorch.org/whl/cu128 -r requirements/libero-rl.txt setuptools wheel
source .venv-sglang/bin/activate
uv pip install pip wheel
python -m pip wheel --pre --wheel-dir wheelhouse-sglang "sglang==0.5.23" "torchvision==0.29.2"
```

Transfer the checkout, both wheelhouses, model directories and simulator assets
to the offline host. Match OS, CPU architecture, Python ABI and CUDA drivers;
extensions without a prebuilt wheel also need their build toolchain on the
preparation host. A few cached wheels are insufficient to reconstruct a full
environment. On the offline host, create fresh venvs and install each prepared
closure separately. `--no-deps` prevents Git URL metadata from contacting the
network; all dependency wheels must already be present.

```bash
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install --no-index --no-deps wheelhouse-rpent/*.whl
uv pip install --no-index --no-deps --no-build-isolation -e .
uv venv --python 3.12 .venv-sglang
source .venv-sglang/bin/activate
uv pip install --no-index --no-deps wheelhouse-sglang/*.whl
```

The offline Python interpreter must be installed locally before `uv venv`.
Run `uv pip check` in each environment. Re-run the configuration-generation
step after relocating the simulator, since its YAML contains absolute paths.
With all assets present, export `HF_HUB_OFFLINE=1` and
`TRANSFORMERS_OFFLINE=1` in runtime terminals. Keep environment installation
separate from active training processes.

## Planner and simulator on separate hosts

Adaptive episode can place the planner beside the model servers and keep
native simulators, motion loops and shared SAM3 on a backend host. Both ends
must use identical control/tool source and workflow prompts. Install the planner
environment on both hosts; install the isolated native simulators and assets
on the backend, configuring `RPENT_ROBOCASA_PYTHON`, `RPENT_ROBOTWIN_PYTHON`,
`ROBOTWIN_ASSETS_PATH` and the RLinf adapter as described in [installation](INSTALL.md).

Start shared SAM3 on the backend, then its toolkit manager. GPU capacities
include initializing episodes; choose them from a real resident-capacity test:

```bash
python -m robots.rl.remote_manager \
  --output-dir /path/to/new-backend-run \
  --gpu-slots 0:4,1:6 --max-initializing-per-gpu 2 \
  --sam3-endpoint http://127.0.0.1:8112 --port 28090
```

The listener defaults to loopback. Establish a private route or SSH forward
before running the planner. On the planner host:

```bash
export RPENT_REMOTE_TOOLKIT_ENDPOINT=http://127.0.0.1:28090
unset RPENT_SAM3_ENDPOINT
bash tools/rl/run.sh --simulator robocasa --task-name PickPlaceCounterToCabinet \
  --split target --seed 1 --workflow episode --cuda-device 1
```

Here `--cuda-device` selects a backend GPU. The backend owns both the env and
SAM3 connections; do not also pass `--env-endpoint` or `--sam3-endpoint`.
The batch launcher inherits the same remote endpoint and distributes model
requests with `--endpoints`. This remote interface supports adaptive episode
only. The handshake rejects different source digests, tool declarations,
tasks or budgets.

Each backend episode has an owned worker and retained configuration, log,
native state, images and video. Changed artifacts travel with SHA-256 checksums
and retain their original bytes in the planner directory; `states.json` is
published after its files. `remote-runtime.json` identifies the worker/source;
`remote-transfer.jsonl` records file hashes and transferred sizes. Planner
transcripts and wire traces remain on the planner host. Transport failure
aborts the episode without replaying an action whose outcome is unknown.
A heartbeat renews the lease while the model reasons; a disconnected planner's
ready worker is released after five minutes, preserving its files.

## RLinf and SFT

Use RLinf's `examples/agent/rpent/README.md` for its own environment and
`run_train.sh` configuration. Set `tools.rpent.python` to this RPent venv's
interpreter and pass the same LIBERO/SAM3 paths to its episode subprocesses.
Those subprocesses need the RPent CUDA/EGL environment; an RL environment's
`PYTHONPATH` or Torch/NVIDIA library paths cannot substitute for it.
SGLang serving remains in its separate environment.

An RLinf configuration using this checkout must point its episode module to
`robots.rl.rl_episode`; a configuration pointing to `robots.libero_rl.rl_episode`
belongs to the older checkout. Preserve the native task fields when migrating.

For trajectory encoding and multimodal SFT, see
`ms-swift/examples/rpent/README.md`. The saved post-compaction requests,
image bytes/order, visible historical thinking and assistant loss masks must
be preserved when preparing training data.

## Relocating RoboCasa assets

For RoboCasa, the installed package owns `robocasa/models/assets`. To put
these assets on shared storage, copy its bundled files there and replace that
environment's asset directory with a symlink **before** downloading:

```bash
export ROBOCASA_ASSETS_PATH="$RPENT_DATA_ROOT/robocasa/assets"
.venv-robocasa/bin/python - <<'PY'
import importlib.util, os, shutil
from pathlib import Path
source = Path(importlib.util.find_spec('robocasa').origin).parent / 'models/assets'
target = Path(os.environ['ROBOCASA_ASSETS_PATH']).expanduser().resolve()
if source.resolve() != target:
    shutil.copytree(source, target, dirs_exist_ok=True)
    source.rename(source.with_name('assets.bundled'))
    source.symlink_to(target, target_is_directory=True)
PY
.venv-robocasa/bin/python -m robocasa.scripts.download_kitchen_assets --type all
```

The native downloader asks before downloading. No demonstration dataset is
required. Keep the existing bundled scene/configuration files when relocating
assets. `ROBOCASA_ASSETS_PATH` alone does not redirect this package version.

See the [workflow guide](WORKFLOWS.md) for all five modes, SAM3 pooling and the fixed-panel comparison.

The model launcher enables cache reporting. For eight-worker SAM3 deployment, per-worker wait metrics and long-history cache tuning, see [workflow performance configuration](WORKFLOWS.md#model-prefix-cache-diagnostics).
