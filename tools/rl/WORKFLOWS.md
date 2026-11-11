# Compare LIBERO workflows

[简体中文](WORKFLOWS_zh.md) · [Installation](INSTALL.md) · [First task](README.md)

All four modes use the API planner and the same robot/frame/gripper contract.
The differences are perception tools and how long conversation history survives.
Native environment flags determine success; a todo item or `finish` claim does
not override them. The public workflow names are the four names below.

## Tools and conversation boundaries

| Workflow | Tools | Retained conversation |
| --- | --- | --- |
| `episode` | `move_to`, `set_gripper`, `view_env`, `segment`, `back_project`, `finish` | Whole episode, with compaction |
| `zap_turn` | `move_to`, `set_gripper`, `change_query`, `update_todo`, `finish` | None between requests |
| `zap_episode` | `move_to`, `set_gripper`, `change_query`, `finish` | Whole episode, with compaction |
| `zap_subtask` | `move_to`, `set_gripper`, `change_query`, `update_todo`, `finish` | Current subtask, with compaction |

Every reply contains exactly one tool call, except `zap_subtask` initialization,
which contains `change_query` and `update_todo` exactly once each. The entire
reply is validated before any tool executes. Rejected replies consume requests
and return feedback without executing sibling actions.

`episode` starts with the task and measured state. The model calls `view_env`
for images and explicit perception tools for world coordinates. `zap_turn`
first calls `change_query`, then `update_todo`. Each following request contains
the current observation, queries, visible text todo and last raw tool result;
it carries no earlier conversation and uses no compaction.

`zap_episode` first calls `change_query`, then retains the conversation without
a todo. `zap_episode` and `zap_subtask` attach the current annotated images and
geometry to each `move_to`, `set_gripper` and `change_query` result. Perception
uses the current RGB/depth frame, not a commanded target. If no physical step
occurred, the latest recorded frame remains current.

`zap_subtask` initializes queries and a full structured todo together. A todo
item has `text` and `status` (`pending`, `in_progress`, `done`). Every successful
`update_todo` fully replaces this list and clears the context before the next
request. The new context has the same system prompt, task, current queries,
observation and updated todo. It does not replay earlier reasoning or tools.
An existing todo means initialization is already complete. The model verifies
subtask completion from observations; the controller does not use task scripts
or an oracle. Invalid updates do not reset context.

Compaction uses the existing 70%-context / 80→40-message window in accumulated
modes. It operates only within the current subtask for `zap_subtask`. Context
resets never reset the environment, global request count or deadline. Default
limits are 100 model requests, 10000 environment steps, 7200 planner seconds
and 14400 outer batch seconds.

## VLM image size

Select `--image_size 1024`, `512` or `256` (default `1024`), or export
`RPENT_IMAGE_SIZE` for `run.sh`. Native RGB, depth and SAM3 input stay at 1024;
only the VLM images change. Boxes use displayed-image coordinates, and pixel
or box tool inputs map back to native depth coordinates. World measurements
remain in metres. Annotations are drawn at the display size.

```bash
RPENT_OUTPUT_DIR=logs/libero-zap-subtask-512 bash tools/rl/run.sh \
  --simulator libero --workflow zap_subtask --image_size 512 \
  --suite libero_object --task 0 --seed 1 --cuda-device 1
```

Recorded native images/world maps remain unchanged. `api_trace/` stores the
actual model image bytes. `workflow_requests/` records context IDs, required
tools and reset boundaries; `workflow_responses/` records accepted/rejected
replies. `rollout.json` includes image size, context resets and timing counters.

## Shared SAM3 workers

The example places four independent SAM3 instances on each environment GPU.
Use the common RPent environment and load `.rl-env.sh`. Run each command in
a separate terminal; the terminal owns that service. Check readiness before
starting the pool in another terminal.

```bash
mkdir -p logs
python -m rpent.robots.components.sam3_server --cuda-device 0 --port 8114 --metrics logs/sam3-0.jsonl
python -m rpent.robots.components.sam3_server --cuda-device 0 --port 8115 --metrics logs/sam3-1.jsonl
python -m rpent.robots.components.sam3_server --cuda-device 0 --port 8116 --metrics logs/sam3-2.jsonl
python -m rpent.robots.components.sam3_server --cuda-device 0 --port 8117 --metrics logs/sam3-3.jsonl
python -m rpent.robots.components.sam3_server --cuda-device 1 --port 8118 --metrics logs/sam3-4.jsonl
python -m rpent.robots.components.sam3_server --cuda-device 1 --port 8119 --metrics logs/sam3-5.jsonl
python -m rpent.robots.components.sam3_server --cuda-device 1 --port 8120 --metrics logs/sam3-6.jsonl
python -m rpent.robots.components.sam3_server --cuda-device 1 --port 8121 --metrics logs/sam3-7.jsonl
```

```bash
python -m rpent.robots.components.sam3_pool \
  --backends http://127.0.0.1:8114 http://127.0.0.1:8115 http://127.0.0.1:8116 http://127.0.0.1:8117 http://127.0.0.1:8118 http://127.0.0.1:8119 http://127.0.0.1:8120 http://127.0.0.1:8121 \
  --port 8124 --per-backend-concurrency 1 --metrics logs/sam3-pool.jsonl
```

Use `RPENT_SAM3_ENDPOINT=http://127.0.0.1:8124` in every planner terminal.
For a ZAP observation, the two camera requests overlap and results are assembled
in their original camera/query order. Both use the same recorded observation
step. Updating only the todo reuses the existing exact-frame perception cache.

A complete image/query group stays on one worker. Its feature lock protects
that image across all queries. `--per-backend-concurrency` controls admission,
defaults to 1, and can be tested at 2; it does not remove the feature lock.
Measure observation latency and worker lock wait before raising it.
Pool metrics record queue time, backend round-trip time and errors. The backend
time includes model processing, mask serialization and transport; it is not
pure GPU time. Worker `--metrics` records `lock_wait_seconds` and
`model_lock_seconds` separately; `sam3.stats` exposes cumulative counters.
Masks use fast lossless PNG compression and decode to the same pixels.

Qualify memory with all environments resident. Eight worker processes alone do
not establish sufficient capacity or eliminate queues during an 80-request
simultaneous two-camera burst.

## Model prefix-cache diagnostics

`serve.sh` enables `--enable-cache-report` so OpenAI-compatible
`prompt_tokens_details.cached_tokens` reports actual reuse. An omitted cache
report does not establish a cache miss. Keep each episode on one model endpoint.
Check SGLang prefill logs or request-level cache details and distinguish long
KV eviction from a prefix that changed after history compaction or a context
reset. Such changed histories still follow the selected workflow.

An optional serving profile for long multimodal histories is:

```bash
export SGLANG_VLM_CACHE_SIZE_MB=2048
SERVE_GPUS=2,3 SERVE_PORT=30180 SERVE_CONCURRENCY=32 \
  SERVE_MEM_FRACTION=0.86 bash tools/rl/serve.sh \
  --mamba-full-memory-ratio 0.5 --enable-metrics
```

These settings tune cache allocation and reporting, without changing weights,
context/output limits, sampling or conversation contents. They require a
compatible SGLang version and memory headroom; test the same request workload
before applying them to other hardware. Higher cache capacity cannot reuse
tokens whose causal prefix changed. Retain model/SAM3 settings with every
performance result, and report queueing separately from generation time.

## Fixed-panel comparison

Use the serving environment to start three original Qwen3.8-27B TP2 services
on GPU pairs 2–3, 4–5 and 6–7, with ports 30180–30182. The following commands
run independently in three terminals after loading the model configuration:

```bash
SERVE_GPUS=2,3 SERVE_PORT=30180 SERVE_CONCURRENCY=32 bash tools/rl/serve.sh
SERVE_GPUS=4,5 SERVE_PORT=30181 SERVE_CONCURRENCY=32 bash tools/rl/serve.sh
SERVE_GPUS=6,7 SERVE_PORT=30182 SERVE_CONCURRENCY=32 bash tools/rl/serve.sh
```

From the common environment at the checkout root, qualify a fresh output
folder, then verify real tool/model requests and finally launch the comparison:

```bash
export RPENT_SAM3_ENDPOINT=http://127.0.0.1:8124
export MODEL_ENDPOINTS=http://127.0.0.1:30180/v1,http://127.0.0.1:30181/v1,http://127.0.0.1:30182/v1
python -m tools.rl.qualify_compare --output logs/capacity40 \
  --sam3-endpoint "$RPENT_SAM3_ENDPOINT" --gpus 0,1 --hold-seconds 300
python -m tools.rl.preflight_compare --output logs/preflight \
  --capacity logs/capacity40/verified.json --sam3-endpoint "$RPENT_SAM3_ENDPOINT" \
  --endpoints "$MODEL_ENDPOINTS" --model-path "$MODEL_PATH"
python -m tools.rl.compare_workflows --output logs/comparison \
  --qualification logs/preflight/verified.json \
  --sam3-endpoint "$RPENT_SAM3_ENDPOINT" --endpoints "$MODEL_ENDPOINTS" \
  --gpus 0,1 --per-gpu 20 --sam3-metrics logs/sam3-pool.jsonl
```

The capacity probe holds forty real environments, issues actions/SAM3 requests
and verifies cleanup. The preflight runs CPU checks unless an exact-source
`--cpu-proof` is supplied, scripted native chains for all modes, and six
bounded real-model cases at seed 1. None count toward formal task scores.
Install the repository test dependencies before these checks.

The controller runs the four modes at 1024, forty tasks each at seed 0. It
selects the highest native success count, breaking ties by batch completion
time and then declared mode order. It runs the winner at 512 and 256, reusing
its 1024 panel: 240 formal episodes total. Task-to-GPU/model assignments are
fixed, forty environments become resident before the shared start barrier,
and cold starts are limited to two per GPU. This layout needs separate native
capacity qualification on the chosen hardware.

Settings are frozen across groups: context 262144, output 32768, thinking+xhigh,
temperature 1, top-p .95, top-k 20, presence penalty explicitly 0, repetition
penalty 1. Model prefix caches are flushed between completed groups while the
services stay warm. Source and qualification hashes must match.

Read `report-<workflow>-<size>.json`, `winner.json` and `final-report.json`.
Reports retain successes, normal failures, infrastructure errors and paired
completion times for identical successful tasks. Normal/model-budget failures
do not retry; infrastructure errors have at most one additional attempt.
All forty tasks remain in the score denominator. A `STOP_REQUESTED.json` in
the comparison directory stops admission and owned collectors. Keep all output
and reuse only identical configuration when resuming an interrupted run.
