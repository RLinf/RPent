# hybrid_agent — standalone LLM-in-the-loop agent

A minimal replacement for "Claude Code driving the REPL by hand". You
give it an Anthropic API key and a `(suite, task, seed)` cell; it
launches `interactive_driver.py`, reads the guides, looks at images,
issues commands, and finishes when `libero_terminated=True`.

It does **not** replicate Claude Code's harness — no skills, no shell
escape, no multi-cell orchestration. Just one cell, one episode,
LLM-in-loop with vision.

## Install

```bash
# pick a venv that has the LIBERO/openpi env already (so the driver
# spawns correctly); the agent itself only needs anthropic.
pip install -r requirements.txt
```

## Run

```bash
export ANTHROPIC_API_KEY=sk-ant-...
cd /mnt/public2/zhangyixian/RLinf_agentic

python examples/embodiment/primitives/workspace_pro/hybrid_agent/runner.py \
    --suite libero_object_task --task 0 --seed 0 \
    --model claude-sonnet-4-5 \
    --output_dir examples/embodiment/primitives/workspace_pro/results_agent_runs
```

What this does:
1. Wipes `/tmp/hybrid_repl/` and spawns `interactive_driver.py` as a
   subprocess on CUDA device 0 (Pi0.5 model load takes ~80s).
2. Once `state_00.json` exists, builds an initial user message naming
   the cell and pointing the agent at the guides.
3. Enters the Claude API tool-use loop. Each turn the model can call:
   - `read_text_file` (guides, past recipes/audits, memory)
   - `write_text_file` (save recipe.jsonl, audit.json at end)
   - `list_dir`
   - `view_repl_state(step=null)` — returns state JSON + agentview PNG
   - `send_command({...})` — issues one REPL primitive, blocks until
     the driver writes the next `done_NN.flag`, returns the new
     state+image+log
   - `finish(status, summary)` — terminates the loop
4. Saves a full transcript JSON to the output dir on exit (omits the
   base64 image bytes from images sent back to the model).

## What you should see

```
[agent] driver cmd: /opt/venv/openpi/bin/python .../interactive_driver.py --suite ...
[agent] waiting for state_00.json (Pi0 load ~80s)...
[agent] driver ready in 77.2s

[agent] === turn 1/80 ===
[claude] I'll start by reading the two guides.
[tool→] read_text_file({"path":".../STRICT_HYBRID_GUIDE.md"})
[usage] in=2103 out=89 stop=tool_use ...
[tool←] read_text_file: {"path":...,"size":48xxxx}

[agent] === turn 2/80 ===
[claude] OK, read the strict-hybrid rules. Now PRO-specific guide.
[tool→] read_text_file({"path":".../PRO_HYBRID_GUIDE.md"})
...

[agent] === turn N ===
[claude] Pi0 picked the cream_cheese. Now moving to basket...
[tool→] send_command({"command":{"action":"move_to","xyz":[-0.014,0.253,0.22],...}})
[tool←] send_command: {"step":3,"state_summary":{"eef":[-0.013,0.252,0.219],...}}
...
[agent] === turn 20 ===
[claude] libero_terminated=True. Saving recipe and audit.
[tool→] write_text_file(...)
[tool→] finish({"status":"success","summary":"..."})
[agent] FINISH called: {'_finish': True, 'status': 'success', ...}

[agent] elapsed: 132.4s
[agent] usage: in=14823 out=2451 tool_calls=11
[agent] transcript: .../transcript_object_task_t0_s0.json
```

## Files

- `runner.py`   — main entry point; subprocess + agent loop
- `tools.py`    — tool definitions + impls + content-block builders
- `prompts.py`  — system prompt + initial user template
- `__init__.py` — re-export `run_one_cell`
- `requirements.txt`

## What's intentionally NOT implemented

- **Multi-episode reset is BLOCKED by design.** The agent gets exactly
  one episode per driver. `{"action": "reset"}` and `{"action": "exit"}`
  return an error from `send_command` without reaching the driver. The
  agent must recover from failures in-episode (re-pre-position,
  escalate the Pi0 prompt-ladder, etc.) or call `finish(status="stuck")`.
  This keeps the single-attempt success metric honest.
- Skill loading / custom slash commands.
- Conversation compaction. With 80 turns + an image per
  `view_repl_state`/`send_command`, the conversation can hit ~150–200k
  tokens. Sonnet 4.5/Opus 4.5 with 200k–1M context handles this.
- Concurrent cells / dispatcher (call `run_one_cell` in a loop from
  Python if you want multiple cells).
- Cost accounting — `stats.total_input_tokens` / `total_output_tokens`
  is logged per cell; multiply by the model's per-token price yourself.

## Files the agent will typically read first

The system prompt tells the agent to read these in order; they are
not bundled — they live in the repo:

- `examples/embodiment/primitives/STRICT_HYBRID_GUIDE.md`
- `examples/embodiment/primitives/workspace_pro/PRO_HYBRID_GUIDE.md`
- `examples/embodiment/primitives/workspace_pro/env_calibration.md`

And the agent can browse past recipes:

- `workspace_pro/results_object_pert/recipe_*.jsonl` (template recipes)
- `workspace_pro/results_object_pert/object_*.json` (audit examples)

## Troubleshooting

- **Driver doesn't become ready**: check `/tmp/hybrid_agent_driver.log`.
  Usual culprits: `LIBERO_TYPE=pro` not set, missing Pi0 checkpoint,
  CUDA OOM.
- **Agent loops on the same plan**: increase `--max_turns` so it can
  reset and retry, or lower `--max_tokens` to force tighter responses.
- **Token cost blowing up**: the biggest contributor is repeated full
  state JSONs after every `send_command`. The agent already gets a
  trimmed view (state.objects is mostly small). The other contributor
  is images — every send_command attaches a fresh PNG. If you need to
  cut cost, hack `tool_result_to_content_blocks` in `tools.py` to skip
  the image except after major state changes.
- **No anthropic module**: `pip install anthropic` in whatever venv
  you launch `runner.py` from. The driver subprocess uses its own
  Pi0 venv (`/opt/venv/openpi/bin/python`), so the two don't need to
  share environments.

## Programmatic usage

```python
from hybrid_agent.runner import run_one_cell

record = run_one_cell(
    suite="libero_object_swap",
    task=2,
    seed=0,
    api_key="sk-ant-...",
    model="claude-sonnet-4-5",
    output_dir="/tmp/agent_runs",
)
print(record["finish"], record["stats"], record["elapsed_s"])
```
