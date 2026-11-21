# Compare LIBERO workflows

[简体中文](WORKFLOWS_zh.md) · [Installation](INSTALL.md) · [First task](README.md)

All modes use the API planner and select LIBERO coordinates with `--world-cord`.
The differences are perception tools and how long conversation history survives.
Native environment flags determine success; a todo item or `finish` claim does
not override them. The public workflow names are listed below.

## Tools and conversation boundaries

| Workflow | Tools | Retained conversation |
| --- | --- | --- |
| `episode` | `move_to`, `set_gripper`, `view_env`, `segment`, `back_project`, `finish` | Whole episode, with compaction |
| `episode_image` | `move_to`, `set_gripper`, `segment`, `back_project`, `finish` | Whole episode, raw images after actions |
| `pear` | `move_to`, `set_gripper` | Whole episode, RGB/state after every action; automatic completion |
| `zap_turn` | `move_to`, `set_gripper`, `change_query`, `update_todo`, `finish` | None between requests |
| `zap_episode` | `move_to`, `set_gripper`, `change_query`, `finish` | Whole episode, with compaction |
| `zap_subtask` | `move_to`, `set_gripper`, `change_query`, `update_todo`, `finish` | Current subtask, with compaction |

Every reply contains exactly one tool call, except `zap_subtask` initialization,
which contains `change_query` and `update_todo` exactly once each. The entire
reply is validated before any tool executes. Rejected replies consume requests and return feedback without executing sibling actions in the existing workflows. In `pear`, an invalid reply immediately ends the episode as a normal failure.

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

The launcher defaults to a 70%-context / 80→10-message window in accumulated
modes. It operates only within the current subtask for `zap_subtask`. Context
resets never reset the environment, global request count or deadline. Default
limits are 80 model requests, 10000 environment steps, 7200 planner seconds
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

Launcher defaults are output 32768 tokens, 80 requests, and 10 retained tail
messages after compaction. The historical fixed-panel examples below use a
separate 32768-token/100-request profile; to reproduce that profile, explicitly
export `RPENT_MAX_TOKENS=32768`, `RPENT_MAX_TURNS=100`, and
`RPENT_COMPACTION_KEEP_MESSAGES=40` before those comparison commands.

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

## Image episodes, poses, action batches and history

LIBERO `episode_image` starts with both agentview and wrist RGB and returns the
latest pair after a physical action. It has `segment`, `back_project` and `finish`
but no `view_env`. Perception remains explicit: no automatic SAM inference.
Its perception inputs omit `step` and use the latest observation; old masks must
be re-segmented after movement. Ordinary `episode` retains historical `step`.

LIBERO `move_to(xyz?, orientation_rpy?)` requires at least one target. `xyz`
is the absolute world position of MuJoCo's grip site in metres.
`orientation_rpy=[roll,pitch,yaw]` is the same frame's absolute fixed-axis XYZ
orientation in radians: `R=Rz(yaw)@Ry(pitch)@Rx(roll)`. `[pi,0,yaw]` points
its approach axis down; the jaws close along the site's x-axis. Omitted parts
retain the value measured at that action's start. Replies report actual
`pose.xyz`, `pose.orientation_rpy`, opening and residual errors; `partial` is not
arrival. RoboCasa and RoboTwin in the base frame use their native action interfaces.

`--action-mode single` is the default. `--action-mode batch` replaces standalone
`move_to` and `set_gripper` with `execute_actions`, retaining perception and
`finish`. `--max-actions-per-call K` defaults to 3 and must be at least 1.

```json
{"actions":[
  {"action":"move_to","xyz":[0.1,0,0.3],"orientation_rpy":[3.14159265,0,0]},
  {"action":"set_gripper","opening":0}
]}
```

The complete list is validated before any movement. Each item finishes and is
measured before the next starts. Success, truncation, error or partial stops the
batch immediately. Results contain each executed item's measured pose,
`executed_count`, zero-based `stopped_at`, `stop_reason` and `final_state`.
`episode_image` sends only the final/pause camera pair; intermediate frames stay
archived. One top-level tool call is allowed per model reply. A batch uses one
request but its atomic actions share the existing environment/time budgets.

`--reasoning-history N` counts prior model replies, grouping all thinking chunks
of one reply. `--image-history N` counts image-return groups; both cameras stay
together. The initial observation counts, and single-camera view_env is one
group. Both defaults are -1:

| Value | Meaning |
|---|---|
| -1 | No extra pruning; existing history compaction still applies |
| 0 | Remove all content of that kind, including the latest images |
| N > 0 | Retain the latest N reply/observation groups |

Prune before compaction, preserving calls, ordinary assistant content and measured
text results. Raw replies and observations remain archived. Current-turn thinking
is still enabled; no fake ignore text is inserted. The original Qwen template
may emit empty thinking delimiters. Pruning old history can reduce prefix-cache
reuse, so measure actual latency and cache metrics. These options currently
support local LIBERO `episode` and `episode_image`; other combinations fail early.

After [installation](INSTALL.md), activate your environment, source the copied
configuration and start model/SAM services. Set `RPENT_OUTPUT_ROOT` to a writable
directory. All options are pinned in the manifest and checked on resume:

```bash
bash tools/rl/run.sh --simulator libero --workflow episode_image   --suite libero_object --task 0 --seed 0 --cuda-device 0   --action-mode batch --max-actions-per-call 3   --reasoning-history 0 --image-history 1

python -m tools.rl.run_batch run "$RPENT_OUTPUT_ROOT/image-batch"   --jobs tools/rl/jobs/panel40.txt --seeds 0 --gpus 0,1 --per-gpu 20   --start-barrier --workflow episode_image --action-mode batch   --max-actions-per-call 3 --reasoning-history 3 --image-history 3
```

The separate eight-panel comparison first runs both workflows and action formats
at -1/-1, then tests 3/3 and 0/1 on each workflow's selected action format.
It does not invoke `compare_workflows.py`, which runs the separate ZAP/image-size
comparison.

Successful segmentation returns a dict without `score` or `found`. Invalid query
and no detection return distinct corrective text; service errors stay separate.
`world_xyz` is the per-axis median of visible depth points; `top_z_m` is their
95th world-z percentile, `size_xyz_m` their 5th-to-95th extents, and `axis_yaw`
their elongated xy major axis or null. These are visible-surface measurements,
not complete dimensions or guaranteed grasp targets.

### Reproduce the eight-panel comparison

Run the following panels in order, one batch at a time. Every panel uses the same
LIBERO-PRO 40-task list, seed 0, image_size 1024, 100 model requests and 10000
environment steps. Keep the model, sampling and time budgets fixed.

| Panel | Workflow | Action mode | reasoning-history | image-history |
|---|---|---|---:|---:|
| 1 | episode | single | -1 | -1 |
| 2 | episode | batch, K=3 | -1 | -1 |
| 3 | episode_image | single | -1 | -1 |
| 4 | episode_image | batch, K=3 | -1 | -1 |
| 5 | episode | winner of 1/2 | 3 | 3 |
| 6 | episode | winner of 1/2 | 0 | 1 |
| 7 | episode_image | winner of 3/4 | 3 | 3 |
| 8 | episode_image | winner of 3/4 | 0 | 1 |

Select the action mode separately for each workflow: highest native success
count, then shortest complete batch wall time, then single on an exact tie.
Use the selected mode in the batch command above. This is 320 formal episodes;
short interface checks and capacity qualification are separate.

For the eight-GPU layout, place three TP2 model replicas on GPUs 2-7. GPUs 0/1
each hold 20 environments and four SAM3 workers, with one shared pool endpoint.
Qualify the final source with 40 environments and eight SAM workers for at least
300 seconds before evaluation. Hold all 40 environments ready before releasing
the first model requests. Warm services consistently and record cache resets;
KV flushing does not guarantee that the VIT feature cache is empty.
Keep every attempt; retry only infrastructure failures, at most once. Compare
matched successful tasks as well as batch wall time, so early failures do not
count as acceleration.

Qwen on SGLang uses a bounded native XML structural tag when zero-argument tools
are available. The wire uses tool_choice=auto plus an explicit at-least-one-call
format carrying the exact tool schemas; it still requires the permitted calls.
Syntactic whitespace is limited to 16 characters to avoid the upstream empty
parameter-zone loop in finish(). No placeholder arguments, dependency patches,
prompt changes or sampling changes are used. Whole-reply validation remains
mandatory. See https://github.com/mlc-ai/xgrammar/issues/802.

## Episode observation lists

`--workflow episode --perception-mode batch` replaces standalone perception
with `observe(calls=[...])`. Sources define dependencies; independent nodes run
concurrently. [CLI, examples, source IDs and outputs](OBSERVE.md).

## RoboTwin episode with shared coordinates

Run the same episode workflow with RoboTwin's shared mounting frame:

```bash
bash tools/rl/run.sh --simulator robotwin --task-name click_alarmclock --seed 100000 \
  --workflow episode --world-cord general --action-mode batch \
  --max-actions-per-call 3 --perception-mode single --prompt simple
```

The origin is the midpoint of the two arm mounts. X points forward, Y points robot-left,
and Z points up. Targets and measured finger-pad TCPs use metres and absolute fixed-axis
XYZ radians. Set `arm` to `left` or `right` on each action. Optional `other_xyz`,
`other_orientation_rpy` or `other_opening` control the other arm in the same action.
The standalone `view_env`, `segment` and `back_project` calls keep the recorded camera and step
contract, and the returned geometry uses the shared frame.

## Base frame and RGB-only control

`--world-cord base` is the default simulator-world frame and keeps the existing `grip_site`. `--world-cord general` works with every LIBERO RL workflow: the mounting base is the origin, x points forward, y left, z up, with a measured finger-pad midpoint TCP. Initial feedback gives table height above the base. The controller compensates the TCP offset; positions use metres and orientations fixed-axis XYZ radians.

All geometry, including batched observe and ZAP surfaces, uses the selected frame: `world_xyz`, `top_z_m`, `size_xyz_m`, `axis_yaw` and `z_range_m`. Raw `*_world.npz` maps remain simulator-world points; `coordinates.json` records their transform. Convert valid points before filtering height or computing extents. Images and camera/step bindings do not change.

`pear` retains the episode-image conversation and returns agentview, wrist and measured robot state initially and after every action. It uses no SAM3, depth perception, action batch or `finish`. A reply must call exactly one of `move_to` and `set_gripper`. `--reasoning` enables thinking; `--no-reasoning` disables it and requires a direct tool call without text. Both are aliases of the existing thinking flags. Native success or exhausted budgets ends the episode automatically. Invalid replies end as reward-zero task failures without correction requests; partial motion alone does not end an episode.

```bash
bash tools/rl/run.sh --simulator libero --suite libero_goal --task 0 --seed 0 \
  --workflow pear --world-cord general --action-mode single --no-reasoning
```

A pear-only remote manager can omit `--sam3-endpoint`. Existing perception workflows still require a SAM3 endpoint. Use matching RPent revisions at both RPC ends; general coordinates require native base, table and finger-pad calibration from the updated environment server.
