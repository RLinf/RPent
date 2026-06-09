# hybrid_agent_cc — Claude Code (subscription) variant

Same goal as `../hybrid_agent/` (LLM-in-loop hybrid driver for LIBERO PRO),
but uses **Claude Code's `claude -p`** instead of direct Anthropic API.
Suitable if you have a Claude Code subscription and want to avoid API
metered billing.

Trade-offs vs the API variant:

|                              | hybrid_agent (API)              | hybrid_agent_cc (claude -p) |
|------------------------------|--------------------------------|------------------------------|
| Billing                      | API tokens                     | Subscription quota           |
| Tools                        | 6 custom tools (in tools.py)   | Claude Code built-in Bash / Read / Write |
| Multimodal images            | base64 PNG → image content block | Claude Code reads PNG via Read tool ✓ |
| Memory access                | agent reads via `read_text_file` | agent reads via `Read`        |
| Parallelism                  | concurrent subprocesses        | **sequential** (one `claude -p` per cell) |
| Rule-1 enforcement           | tool-side block (reset/exit)   | prompt-side reminder only    |
| Each cell isolation          | new Python process             | new `claude -p` (no context bleed) |

## Files

- `agent_task_prompt.md` — the prompt template with `{SUITE}`,
  `{TASK}`, `{SEED}`, `{WORKDIR}`, `{TAG}`, `{OUTPUT_DIR}` placeholders.
  Mirrors `../hybrid_agent/prompts.py` content + adds Claude-Code-style
  Bash/Read instructions.
- `run_one_cell.sh` — launches `interactive_driver.py` for one cell,
  waits for ready, invokes `claude -p`, cleans up.
- `run_all_seeds.sh` — outer loop: **sequentially** calls
  `run_one_cell.sh` for each seed (single GPU, lowest concurrency).
- `run_parallel_seeds.sh` — **parallel** sweep: one cell per GPU
  concurrently, round-robins seeds onto the GPU pool when seeds
  outnumber GPUs.
- `status.sh` — live progress printer (safe to run any time).

## Quickstart

```bash
# 1. Make sure claude CLI works
claude --version
# (you should have an active Claude Code subscription)

# 2. Default sequential sweep: libero_spatial_lan task 0, seeds 0..9 on GPU 0
bash run_all_seeds.sh

# 3. Different cell, sequential:
SUITE=libero_object_task TASK=2 SEEDS="0 1 2 3" CUDA_DEVICE=0 \
    bash run_all_seeds.sh

# 4. PARALLEL sweep on a multi-GPU box (one cell per GPU concurrently):
SUITE=libero_spatial_task TASK=0 \
SEEDS="0 1 2 3 4 5 6 7" GPUS="0 1 2 3 4 5 6 7" \
MODEL=claude-opus-4-7 \
    bash run_parallel_seeds.sh
# - Stagger between launches = STAGGER_S (default 30s) to avoid
#   simultaneous Pi0 model-load disk thrash + API TLS burst.
# - If you have more seeds than GPUs, extras queue: as soon as one
#   cell finishes, the freed GPU picks up the next pending seed.

# 5. Monitor (in another terminal):
watch -n 30 'bash status.sh'

# 6. Or have a Claude Code instance babysit (in another terminal):
#    open `claude` then type:
#       /loop 5m bash hybrid_agent_cc/status.sh && tail -20 /tmp/claude_p_master_libero_spatial_lan_t0.log
```

## Outputs

For each cell (`<tag> = <suite_without_libero_prefix>_t<N>_s<M>`):

```
results_claude_p_runs/
├── recipe_<tag>.jsonl              ← working command sequence (agent writes)
├── <tag>.json                       ← audit (libero_terminated, strategy_notes)
└── claude_<tag>.txt                 ← claude -p stdout (turns + tool calls)
```

And the master log:
```
/tmp/claude_p_master_<suite>_t<task>.log
```

## How Rule-1 is enforced

The API variant blocks `reset` / `exit` in `send_command`. With Claude
Code using raw `Bash` to write `command.json`, we can't intercept at
the tool layer.

Instead Rule 4 in the prompt explicitly states:
> DO NOT issue `{"action": "reset"}` or `{"action": "exit"}` mid-run.

Empirically this is honored. If you want a belt-and-suspenders block,
patch `interactive_driver.py:execute()` to ignore reset/exit (a 3-line
change); not done here so the driver remains general-purpose.

## Cost / quota considerations

Each cell consumes Claude Code subscription quota equivalent to a
sub-3-minute session with image-heavy tool use. 10 cells sequential
≈ 30 min wall time, ~1.5-2 hours of subscription quota (assuming
Sonnet at ~$0.20/cell from the API variant's measured cost).

`claude -p` enforces a per-invocation USD cap via `--max-budget-usd`.
The default in `run_one_cell.sh` is **10** (override with the
`MAX_BUDGET_USD` env var). Recommended values by model:

- **sonnet**: 5 is enough — typical cell costs $0.20–$1.
- **opus 4.7**: keep the default 10. Steady-state cells cost $2–3,
  but extreme-layout edge cases that need recovery loops (e.g.
  bowls near OSC reach limits, set_object_pose fallback) can hit
  $4–6 of opus output tokens. The original sweep had `MAX_BUDGET_USD=2`
  hard-coded and seed=0 was hard-terminated mid-thought; passing
  $10 the retry succeeded cleanly.

If you have an API key (paying directly), the parallel `hybrid_agent/`
variant gets the same result in ~5 min and ~$1.50.
