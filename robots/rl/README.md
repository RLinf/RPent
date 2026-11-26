# RPent Unified RL Simulator Layer (`robots/rl`)

`robots/rl` provides a compact, uniform tool-calling interface, warm HTTP simulator pools, and standalone rollout/evaluation harness across all supported manipulation simulators (`libero`, `robotwin`, `robocasa`, and `robodojo`).

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
   │     ├── robocasa.py         (RoboCasaRLPrimitives)
   │     └── robodojo.py         (RoboDojoRLPrimitives)
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

## Unit Tests

```bash
pytest tests/unit_tests/robots/rl/test_rl_core.py -v
```
