---
name: worker-reads-memory-snapshot
description: "claude -p hybrid workers read the versioned in-repo memory snapshot; re-sync it before sweeps that depend on new feedback memories."
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 82a57f3e-217b-4ac0-984d-bebcc7d3eb36
---

`physicalagent.apps.libero.runner` passes `physicalagent/context/memory/` to
`ClaudeCodeCerebrum` as an extra `--add-dir`. This is a **versioned in-repo
copy**, not a private Claude memory directory.

**Why:** keeps sweeps reproducible (workers see the same magic-number notes regardless
of what I edit live mid-sweep).

**How to apply:** any feedback/magic-number memory written AFTER the last snapshot sync
is invisible to workers. Before launching a sweep that depends on a new lesson, re-sync:
`cp -f "${PHYSICALAGENT_MEMORY_SOURCE:-/path/to/source/memory}"/*.md physicalagent/context/memory/`
(2026-05-23: snapshot was 2 files + MEMORY.md index behind source; re-synced. `cp` leaves
extra stale files — e.g. an unrelated README.md — harmless.)

Related runner-level lessons: [[max-episode-steps-libero]] (libero_10 needs
`--max_episode_steps 5000`) and `CELL_TIMEOUT_S=600` being too tight for
long-horizon cells (pass `--claude_code_timeout_s 1200`, or set
`CELL_TIMEOUT_S=1200`, then re-launch with `--skip_existing`).
