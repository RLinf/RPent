# RPent Unified RL Simulator Layer (`robots/rl`)

`robots/rl` provides a compact, uniform tool-calling interface, warm HTTP simulator pools, and standalone rollout/evaluation harness across all supported manipulation simulators (`libero`, `robotwin`, `robocasa`, `robodojo`, `metaworld`, and `robolab`).

## Architecture

```
RLinf AgentLoop / Standalone Rollout
   │  HTTP (JSON /open & /close, binary octet-stream /execute)
   ▼
robots/rl/server.py  (FastAPI + SimPoolManager over Unix-socket warm workers)
   ├── robots/rl/toolkit.py      (RLToolkit, action-batch execution, completion checks)
   ├── robots/rl/perception.py   (RecordedPerception, SAM3 segmentation, back-projection)
   ├── robots/rl/envs/
   │     ├── libero.py           (LocalLiberoEnv + LiberoRLPrimitives)
   │     ├── robotwin.py         (LocalRoboTwinEnv + RoboTwinRLPrimitives)
   │     ├── metaworld.py        (LocalMetaWorldEnv, MetaWorldRLPrimitives)
   │     ├── robocasa.py         (RoboCasaRLPrimitives)
   │     ├── robodojo.py         (RoboDojoRLPrimitives)
   │     └── robolab.py          (RoboLabRLPrimitives)
   ├── robots/rl/conversation.py (EpisodeConversation, validation retries, history compaction)
   ├── robots/rl/geometry.py     (camera projection, back-projection, PNG encode/overlay)
   ├── robots/rl/units.py        (centimeter/meter conversion helpers)
   └── robots/rl/prompts/        (shared & per-simulator prompt templates)
```

## Key Design & Performance Features

1. **Isolated per-service environments (`tools/rl/serve.py`)**:
   A single YAML file (`tools/rl/sim_services.example.yaml`) supervises the SAM3 GPU pool and one warm `SimPoolManager` per simulator, each running under its own isolated Python venv (`envs/sam3`, `envs/libero`, `envs/robotwin`).
2. **Full-concurrency warm worker pools**:
   Each `SimPoolManager` pre-warms `workers` processes pinned round-robin across configured GPUs, matches handshakes by explicit `--worker-id`, exposes `workers_target` / `workers_total` / `workers_idle` / `active_sessions` on `GET /health`, and guards `_replace_worker()` with an idempotent `replaced` flag so concurrent cohort aborts never leak or double-respawn workers.
3. **On-demand / deferred simulator rendering (`record_artifacts=False`)**:
   - **LIBERO**: `LocalLiberoEnv(..., record_artifacts=False)` passes `use_camera_obs=False, camera_depths=False` during physics stepping and renders RGB-D on demand only when perception tools request an observation.
   - **RoboTwin**: `LocalRoboTwinEnv(..., record_artifacts=False)` skips per-step offscreen rendering during `reset()` and `chunk_step()`, and batches all 3 cameras into a single SAPIEN `scene.update_render()` pass via `render_cameras()` / `capture_all()`.
   - Setting `record_artifacts=True` preserves full frame capture for evaluation video recording.
4. **Zero-Base64 binary `/execute` transport**:
   When a client sends `Accept: application/vnd.rpent.execute+octet-stream`, `POST /sessions/{sid}/execute` streams a compact length-prefixed binary frame (`[u32 meta_len][utf8 JSON meta][u32 num_images][u32 img_len][raw PNG bytes]...`), falling back to Base64 JSON when not requested. `EpisodeConversation` stores lightweight `rpent-image://<fid>` placeholders plus raw PNG `bytes` references without eager Base64 encoding or deep-copying across turns.
5. **Pydantic-AI–compatible conversation & history compaction**:
   `robots/rl/conversation.py` mirrors the exact validation retry and sliding-window history compaction semantics used in RLinf, enabling identical standalone evaluation and trajectory collection.

## Deploying Simulator & SAM3 Pools

Copy and edit `tools/rl/sim_services.example.yaml`, then run:

```bash
PYTHONPATH=/path/to/RPent:/path/to/RLinf \
  python tools/rl/serve.py --config /path/to/sim_services.yaml
```

Verify pool health:

```bash
curl -s http://127.0.0.1:28561/health   # LIBERO
curl -s http://127.0.0.1:28562/health   # RoboTwin
```

### MetaWorld

Install `requirements/rl-metaworld.txt` in an isolated Python 3.11 environment and enable headless EGL rendering (`MUJOCO_GL=egl`, `PYOPENGL_PLATFORM=egl`). Size `--workers` to at least your per-domain `sampling_episodes` quota (e.g., `--workers 16` for the 8-GPU GRPO config's `sampling_episodes: 16`, or use the commented `metaworld` entry in `tools/rl/sim_services.example.yaml`):

```bash
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
  python -m robots.rl.server serve \
  --env metaworld \
  --workers 16 \
  --gpus 4 5 6 7 \
  --port 28565 \
  --sam3-endpoint http://127.0.0.1:28510
```

Multi-node deployments can also front remote MetaWorld pools with `python -m robots.rl.server router --env metaworld --port 28565 --backends ...`. Open a session with an MT50 task name (for example `pick-place-v3`) and a native task index from `0` to `49` in `seed`. The fixed-wrist Sawyer enforces a strict 4-DoF `move_to` schema (`xyz`-only, rejecting `orientation_rpy`) plus `set_gripper`, reports calibrated live `pose.orientation_rpy`, provides `corner2` and `gripperPOV` views, and clamps `max_episode_steps` to MetaWorld's native 500-step horizon.

Standalone rollout example (using the task pool generated under `${RPENT_ROOT}/data/metaworld`):

```bash
python -m robots.rl.rollout \
  --model-url http://127.0.0.1:30000/v1 \
  --model-name Qwen3.5-4B \
  --sim-url http://127.0.0.1:28565 \
  --tasks-jsonl "${RPENT_ROOT:-/path/to/rpent_ppo}/data/metaworld/mt50.jsonl" \
  --output-dir /path/to/metaworld_eval_out \
  --concurrency 16 \
  --max-turns 50 \
  --record-artifacts
```

### RoboLab DROID

Use Linux x86-64, Python 3.11 and a CUDA 12.8-compatible NVIDIA driver
with Vulkan support. Have Git LFS and FFmpeg available before installing.
From the RPent checkout, install the native stack in a dedicated environment:

```bash
git clone https://github.com/NVlabs/RoboLab.git /path/to/RoboLab
git -C /path/to/RoboLab checkout b6cfcf6ccd2113c4cd1216d35b33024a7d5691da
git -C /path/to/RoboLab lfs pull
python3.11 -m venv /path/to/envs/robolab
/path/to/envs/robolab/bin/python -m pip install \
  -r requirements/rl-robolab.txt -e /path/to/RoboLab -e .
```

Install the editable RoboLab checkout so its tasks can resolve the assets
beside the source. The [DROID task catalog](../robolab/task_catalog.json)
lists 120 task IDs and their native instructions. A request to
`POST /sessions/open` can use
`"task": {"task_name": "BananaInBowlTask", "seed": 0}`.

After accepting the NVIDIA Omniverse EULA, export `OMNI_KIT_ACCEPT_EULA=Y`
in the simulator process environment. Set `robolab.python` in the service
configuration to the native interpreter above. A standalone pool can
instead be started with:

```bash
python -m robots.rl.server serve --env robolab --port 28566 \
  --sim-python /path/to/envs/robolab/bin/python --gpus 4 --workers 1 \
  --max-concurrent-spawns 1 --log-dir /path/to/robolab-worker-logs
```

Each worker keeps one Isaac application warm and builds a seeded USD scene
for each episode. Start with one worker per GPU. Use `--sleepable` for a
simulator that shares GPUs with training. Dedicated worker pools can remain
warm behind `python -m robots.rl.server router --env robolab`.
For concurrent recordings, pass a distinct `output_dir` for each session.

The general frame starts at the Franka mounting link (`panda_link0`).
The grasp TCP is the midpoint of the Robotiq contact pads. Model-facing
positions use centimetres and fixed-axis XYZ RPY uses radians. Both
cameras provide 1280×720 native RGB-D. `image_size` limits the long edge of
the returned RGB while preserving its aspect ratio. Pixel queries map back
to the synchronized native depth map. With `record_artifacts=false`, the
first perception request at a step captures the two views, and those
captured steps remain available for later queries. Request a view before
moving if its pixels will be needed later. Isaac's native rendering
cadence stays enabled for physics and sensor updates.

The native task may end before RPent's requested step limit. A blocked
gripper stops the batch, and native success takes precedence when it
coincides with the step limit.

To check native geometry and seeded scene reuse, install the RPent test
extra in the RoboLab interpreter and run:

```bash
RPENT_RUN_ROBOLAB_INTEGRATION=1 RPENT_ROBOLAB_GPU=4 \
  /path/to/envs/robolab/bin/python -m pytest -s -q tests/e2e_tests/robolab
```

## Standalone Evaluation / Data Collection

Run standalone episodes against any OpenAI-compatible serving endpoint (e.g., SGLang or vLLM) with the same conversation compaction and binary `/execute` client:

```bash
python -m robots.rl.rollout \
  --model-url http://127.0.0.1:30000/v1 \
  --model-name Qwen3.5-4B \
  --sim-url http://127.0.0.1:28561 \
  --tasks-jsonl /path/to/libero/eval.jsonl \
  --output-dir /path/to/eval_out \
  --concurrency 8 \
  --max-turns 50 \
  --record-artifacts
```

## Multi-Node & Cross-Cluster Deployment (48-GPU Topology)

For large-scale oversampled GRPO across multiple GPU nodes (and heterogeneous clusters without shared storage), `robots/rl/server.py` provides two additional primitives supervised by `tools/rl/serve.py`:

- **`SimPoolRouter` (`python -m robots.rl.server router`)**: Fronts multiple remote `SimPoolManager` HTTP backends with capacity-weighted least-loaded session placement (`_reserved / _capacity`), transparent zero-Base64 binary `/execute` passthrough, aggregated `GET /health`, and fan-out `/cohorts/{id}/close` and `/pool/{sleep,wake}` endpoints.
- **Sleepable collocated pools (`--sleepable`)**: Collocated RL+simulator nodes mark `sleepable: true` so `RLinf` can invoke `POST /pool/sleep` after rollout to release simulator GPU VRAM before Megatron backward and `POST /pool/wake` after weight sync to respawn warm workers, while dedicated simulator nodes (`sleepable: false`) remain warm across steps.
- **High-concurrency SAM3 queue (`max_pending: 2048`)**: `HttpRpcServer.request_queue_size` is raised to `2048` so 384 concurrent visual-perception episodes never hit socket listen-backlog drops.

### Reference 48-GPU Cross-Cluster Layout (5× H100 + 1× A100)

```
                      ┌──────────────────────────────────────────────────┐
                      │   Head Node (h100_1) SimPoolRouters              │
                      │   :28561 LIBERO (192)  │  :28562 RoboTwin (192)  │
                      └───────────┬────────────┴────────────┬────────────┘
             ┌────────────────────┼─────────────────────────┼────────────────────┐
             ▼                    ▼                         ▼                    ▼
  h100 (8 GPUs, dedicated)   h100_1..4 (32 GPUs, RL)   jump_j2 SSH Tunnel   a100 (8 GPUs, dedicated)
  • 32 LIBERO (non-sleep)    • 40 LIBERO/node (sleep)  -L 28572:a100:28552   • 80 RoboTwin (non-sleep)
  • 48 RoboTwin (non-sleep)  • 16 RoboTwin/node (sleep)-R 28572:h100_1:28572 • 8 SAM3 (queue=2048)
  • 16 SAM3 (queue=2048)     • 32× SGLang / Megatron
```

See `tools/rl/sim_services_48gpu.example.yaml` for a complete multi-node YAML template.

### Cross-Cluster Production Issues & Fixes

The following issues were encountered and resolved while running 48-GPU cross-cluster training (384 concurrent episodes across 5× H100 + 1× A100 bridged over a jump host):

| Symptom / Failure | Root Cause | Fix Implemented |
| :--- | :--- | :--- |
| **`EMFILE: Too many open files` / `SIGABRT`** under 384 concurrent sessions | Default container `ulimit -n 1024` is exhausted by concurrent HTTP connections, Unix sockets, and Ray/Gloo channels (`>1,400` FDs). | `robots/rl/server.py` and `tools/rl/serve.py` automatically raise `RLIMIT_NOFILE` to `min(hard, 1048576)` at startup; launch scripts and SSH tunnels also set `ulimit -n 1048576`. |
| **`httpx.RemoteProtocolError: Server disconnected without sending a response`** over cross-cluster tunnel | Default `uvicorn --timeout-keep-alive 5` raced with `httpx`'s default `keepalive_expiry=5.0` across WAN SSH tunnels, and unhandled `TransportError` surfaced as HTTP `500`. | Raised uvicorn `timeout_keep_alive=300` in `server.py`, set `SimPoolRouter` `httpx.Limits(keepalive_expiry=30.0)`, and mapped `httpx.TransportError` to HTTP `503` so clients retry cleanly. |
| **SAM3 RPC connection drops during rollout bursts** | Default `ThreadingHTTPServer.request_queue_size = 256` overflowed when 384 episodes requested segmentation simultaneously. | Raised `HttpRpcServer.request_queue_size = 2048` in `rpent/utils/rpc/http_rpc.py` and configured `sam3.max_pending: 2048` across 24 SAM3 servers (16 H100 + 8 A100). |
| **Collocated RL GPUs OOM during Megatron backward** | Peak 32-GPU Megatron backward uses `~79.2 GiB / 81.5 GiB` on H100; resident collocated sim workers (`5 LIBERO + 2 RoboTwin`/GPU) leave insufficient headroom. | Added `--sleepable` `/pool/sleep` (`~8.8s`, freeing collocated VRAM to `~5.5 GiB`) and `/pool/wake` (`~20.6s`). Fixed a race in `sleep()` by setting `_sleeping = True` before closing sessions and rejecting post-handshake spawns while sleeping so worker counts never overshoot target. |
| **Intermittent 0-byte reads on shared BeeGFS (`link7.stl is empty`, `at least 4 vertices required`, import errors)** | Concurrent opens of MuJoCo meshes and Python modules across dozens of workers trigger BeeGFS short-reads; staging ~45 GB into container `/tmp` can trigger 100 GB overlay eviction. | Stage `envs/`, `assets/`, `runtime/graphics`, `RPent`, and `RLinf` into node-local RAM tmpfs (`/dev/shm/rpent_ppo`, symlinked at `/tmp/rpent_ppo`), keeping `/` overlay at `<1.2 GiB / 100 GiB (1%–2%)` and cutting LIBERO `open()` latency from `60s+` to `~13s`. |
| **Proxy interception or silent drops across SSH tunnel** | Corporate/container `HTTP_PROXY` variables intercepted internal VPC / `127.0.0.1` traffic; idle SSH tunnels stalled without keep-alives. | Set `trust_env=False` in `SimPoolRouter` and `NO_PROXY` in service configs; run bidirectional auto-reconnecting SSH tunnels (`-o ServerAliveInterval=15 -o ServerAliveCountMax=3 -o ExitOnForwardFailure=yes`). |
| **EGL / Vulkan renderer init failures inside isolated venvs** | Heterogeneous host drivers across H100/A100 containers mismatched container-default GL/Vulkan ICD paths. | Bundle driver-matched `libEGL`/`libGLX`/`libvulkan` and `egl.json`/`vulkan.json` in `runtime/graphics` and export `LD_LIBRARY_PATH`, `__EGL_VENDOR_LIBRARY_FILENAMES`, and `VK_ICD_FILENAMES` per simulator in `serve.py` YAML. |

### Multi-Simulator Alignment & Runtime Pitfalls (5-Env Pool: LIBERO / RoboTwin / RoboDojo / RoboLab / MetaWorld)

The following issues were caught during 5-simulator concurrent rollout validation (20 concurrent episodes across all 5 environments in isolated venvs) and must be observed in every deployment:

| Symptom / Failure | Root Cause | Fix & Operational Guidance |
| :--- | :--- | :--- |
| **RoboDojo fails importing `rlinf.envs.sim.robodojo.assets` when co-deployed with LIBERO** | `libero` (`libero_type=pro`) needs the workspace `RLinf` repo on `PYTHONPATH`, whereas putting workspace `RLinf` in global `PYTHONPATH` shadows the RoboDojo venv's installed `rlinf.envs.sim.robodojo` package. | Keep global `env.PYTHONPATH` in `serve.py` YAML restricted to `RPent` only, and scope `RLinf` exclusively under `simulators.libero.env.PYTHONPATH`. |
| **Isaac Sim 5.0 / 5.1 (RoboLab / RoboDojo) crashes on NVIDIA driver `535.x` (`createDLSSContext` segfault or driver version error)** | Older `535.54.03` drivers fail Isaac Sim 5.0's version check and segfault inside `librtx.postprocessing.plugin.so` / `libgpu.foundation.plugin.so` when NGX/DLSS initializes. | Export `OMNI_KIT_ACCEPT_EULA=Y` (exact `"Y"`, not `"YES"`), `VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json`, native `LD_LIBRARY_PATH` (never prepend `/usr/local/cuda/compat` into simulator processes), and `RPENT_ISAAC_KIT_ARGS="--/rtx/verifyDriverVersion/enabled=false --/ngx/enabled=false --/rtx/post/aa/op=0 --/rtx-transient/dlssg/enabled=false"`. Both `robots/robolab/episode_server.py` and `robots/robodojo/env_server.py` forward `RPENT_ISAAC_KIT_ARGS` to Kit. |
| **RoboDojo worker fails on session open (`TypeError: missing expected_meta` or missing `curobo.yml`)** | `RoboDojoEnvClient` requires `expected_meta`, assets must be passed via `ROBODOJO_ASSETS_ROOT` (not `--source-root`), and `curobo.yml` contains absolute paths that break if `Assets/` is relocated. | `robots/rl/server.py` passes `expected_meta` and wires `ROBODOJO_ASSETS_ROOT` / `--env-cfg-type` / `init_state_id` fallback. Run `update_embodiment_config_path.py` once after moving `ROBODOJO_ASSETS_ROOT`. |
| **`httpx.ReadTimeout` in `robots/rl/rollout.py` under heavy 20-way LLM concurrency** | Large thinking models (e.g., 27B with `--disable-cuda-graph`, `max_tokens=16384`) serving 10+ concurrent multimodal turns can exceed the default `600s` `/chat/completions` client timeout even when simulators and SAM3 are healthy. | Pass `--request-timeout-s` appropriately (or enable CUDA Graph / reduce per-server concurrency) and inspect `<ep_dir>/exception.txt` to distinguish model HTTP read timeouts from simulator pool issues. |
| **IsaacSim worker cold-start crash stalls `SimPoolManager.start()` for 240s** | `_spawn_worker_impl` awaited the readiness future for `240s` without checking `proc.poll()`, so a transient Kit startup segfault hung pool startup and aborted `serve.py`. | Updated `robots/rl/server.py` (`_spawn_worker` / `_spawn_worker_impl`) to poll `proc.poll()` every `0.5s` for immediate crash detection and retry failed spawns up to 3 times. |
| **RoboDojo `seed` vs `layout` mismatch (`Eval_Layout/arx_x5/<seed>/` not found)** | In RoboDojo, `seed` selects the directory (`0`, `1`, or `2` under `Eval_Layout/RoboDojo/arx_x5/`) while `layout` selects the JSON file (`<task>_<layout>.json`, `0..19`). Swapping them causes immediate session-open exit. | Always generate RoboDojo task JSONLs with `seed ∈ {0, 1, 2}` and `layout ∈ [0..19]`, and point `--/persistent/isaac/asset_root/{default,cloud,nvidia}` at a local offline Nucleus mirror in air-gapped clusters. |
| **Single-GPU Vulkan `VkResult: ERROR_DEVICE_LOST` crashes IsaacSim pools** | A GPU with a wedged Vulkan context (while CUDA/EGL remains healthy) fails every `SimulationApp` launch with `ERROR_DEVICE_LOST` / `SIGSEGV`. | Validate per-GPU `SimulationApp` startup before pool bring-up, stagger RoboLab/RoboDojo GPU lists so each healthy GPU hosts balanced IsaacSim workers, and exclude any Vulkan-faulty GPU from Vulkan pools while retaining it for CUDA/EGL workloads (`sam3`, `libero`, `metaworld`). |

## Unit Tests

```bash
pytest tests/unit_tests/robots/rl/test_rl_core.py \
       tests/unit_tests/robots/rl/test_sim_pool_multinode.py -v
```
