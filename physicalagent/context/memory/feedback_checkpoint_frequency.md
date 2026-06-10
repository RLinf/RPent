---
name: BuilderBench CRL checkpoint saving — keep frequency low
description: User wants CRL training to save few checkpoints, not one per eval step.
type: feedback
originSessionId: d8efb52a-c642-417f-8d42-9c041c8719c2
---
For BuilderBench CRL runs (`${BUILDERBENCH_ROOT:-/path/to/builderbench}/rl/impls/crl.py`), do not save a checkpoint at every eval step.

**Why:** Each `params_{es}.pkl` is ~38 MB for a small net and >100 MB for the depth=32 big net. With `num_eval_steps=200` (the watcher default) that's ~8 GB / 20+ GB per run, multiplied by many parallel runs — wasteful and clutters `runs_*` dirs. The user explicitly said "不用保存太多，也不用每个eval都保存".

**How to apply:** The script now exposes:
- `--save_every N` — only save every N evals plus the final one (default 50, so ~4 snapshots per 200-eval run).
- `--save_keep_latest_only` — overwrite `params_latest.pkl` instead of keeping snapshots.

When launching new CRL runs, prefer the defaults (or pass `--save_every 100 --save_keep_latest_only` if disk pressure is tight). Existing in-flight runs were started before this change and still save every eval — leave them alone unless asked.
