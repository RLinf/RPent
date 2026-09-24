# RoboDojo with EmbodiedAgent

This adapter uses RPent's `EmbodiedAgent` for each model decision. Its MCP
`snapshot` tool returns the three RoboDojo camera views, robot state, execution
feedback, and a bounded RoboDawn demonstration. `submit_action` accepts one to
four discrete commands. RoboDawn's controller plans and executes the commands
in Isaac Sim; RoboDojo owns reset and native success scoring. The agent's
`finish` report is not used as a score.

The RoboDawn controller and demonstration bank are third-party MIT-licensed
data. `install_policy.py` copies the pinned revision
`9247f366cd31f278e10f2fbe5fe8469b5f1b5b94` into an isolated benchmark
workspace and retains its license. No third-party assets or API credentials are
committed to RPent.

## Prepare the experiment

The following commands use the existing `roboprobe_once` workspace preparer
provided with this RoboDojo installation. Run from the parent directory of
`RPent` and set paths to match your installation:

```bash
python roboprobe_once/prepare.py --experiment /path/to/new-experiment
git clone https://github.com/Hugo-AGI/RoboDawn.git /path/to/RoboDawn
git -C /path/to/RoboDawn checkout 9247f366cd31f278e10f2fbe5fe8469b5f1b5b94
python RPent/examples/robodojo/install_policy.py \
  --source /path/to/RoboDawn --experiment /path/to/new-experiment
python RPent/examples/robodojo/configure_parallel.py \
  --experiment /path/to/new-experiment --repo /path/to/RPent \
  --key-file /private/path/to/tokenhub-key
```

`configure_parallel.py` adapts the prepared launcher's policy name, model
endpoint, and Python path. It copies the key into
`runtime/openai_api_key` with mode `0600`; neither the key nor the experiment
directory belongs in Git. Install RPent's runtime dependencies into
`runtime/python` for the Isaac image's Python 3.11 interpreter. The launcher
adds that directory and the RPent checkout to `PYTHONPATH`.
Warp, Torch extension, and CUDA caches are isolated per shard so 16 workers
cannot race while compiling the same kernel.

```bash
python3.11 -m pip install --target /path/to/new-experiment/runtime/python \
  'pydantic-ai-slim[openai]==2.48.0' 'mcp==1.30.0' \
  'openai==3.8.0' 'imageio>=2' 'prompt-toolkit>=3'
```

For an initial bounded check, pass `--decision-limit 4` to
`configure_parallel.py` and run distinct tasks in separate shards. A full
evaluation should use a fresh experiment with `--standard-tasks-only` and no
decision limit. This selects the 42 standard tasks that have pinned
demonstrations; the preparer also lists 12 random variants. The copied
`manage.py` supports `plan --jobs 16` and `submit`; each shard gets a separate
GPU worker. This setup scores seed 0 and layout 0, so it is not directly
comparable to multi-seed published results. The exact `sco` workspace,
storage mount, resource pool, and worker spec are installation-specific.
Use RoboDojo's native `_result.json`
`details[*].success` and `score` for evaluation, and keep the decision logs
under `runtime/vlm_logs` as diagnostic evidence.

The TokenHub `gpt-6-astra/azure_L/qwb` endpoint is configured through
`ROBODOJO_LLM_MODEL` and `ROBODOJO_LLM_BASE_URL`. The adapter uses its
Responses API because this endpoint rejects tool calls with
`reasoning_effort` on Chat Completions. RPent records each failed provider
attempt in the per-decision `llm_errors.jsonl` and exposes input, output, and
cached tokens in the decision response. HTTP 400 errors are logged and
returned; transient HTTP and connection errors follow RPent's retry policy.

The adapter sets a stable `prompt_cache_key` for the model route and enables
explicit Responses caching. RPent places breakpoints after the repeated task
instruction and multimodal `snapshot` feedback, and retains at most two recent
image observation groups in the planner history. The demo and current cameras
within a single RoboDojo snapshot remain subject to `max_images=24`.
Use the provider's reported cached input tokens to measure the weighted hit
rate for one plan:

```bash
python RPent/examples/robodojo/cache_report.py \
  --log-root /path/to/experiment/runtime/vlm_logs \
  --run-prefix rp-once-your-plan-id-
```

`cache_hit_rate` is `cached_input_tokens / input_tokens` for decisions with
usage data. Failed requests without provider usage are excluded; the report
shows their count as `decisions - decisions_with_usage`. Older adapter runs did
not forward cache write tokens, so `cache_write_tokens_reported=false` means
that figure cannot be recovered from those logs. The official OpenAI
[prompt caching guide](https://developers.openai.com/api/docs/guides/prompt-caching)
describes explicit breakpoints and the response usage fields.
