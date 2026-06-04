---
name: worker-reads-memory-snapshot
description: "claude -p hybrid workers read a FROZEN copy of memory (workspace_pro/memory_snapshot), not the live ~/.claude memory dir; re-sync before any sweep or new feedback memories silently never reach the worker."
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 82a57f3e-217b-4ac0-984d-bebcc7d3eb36
---

`run_one_cell.sh` passes `--add-dir $MEMORY_DIR` where `MEMORY_DIR` defaults to
`examples/embodiment/primitives/workspace_pro/memory_snapshot/` — a **frozen copy**,
NOT the live `/root/.claude/projects/-mnt-public2-zhangyixian/memory/`.

**Why:** keeps sweeps reproducible (workers see the same magic-number notes regardless
of what I edit live mid-sweep).

**How to apply:** any feedback/magic-number memory written AFTER the last snapshot sync
is invisible to workers. Before launching a sweep that depends on a new lesson, re-sync:
`cp -f /root/.claude/projects/-mnt-public2-zhangyixian/memory/*.md examples/embodiment/primitives/workspace_pro/memory_snapshot/`
(2026-05-23: snapshot was 2 files + MEMORY.md index behind live; re-synced. `cp` leaves
extra stale files — e.g. an unrelated README.md — harmless.)

Related script-level lessons that bypass the worker entirely (set at orchestrator level,
not via memory): [[max-episode-steps-libero]] (libero_10 needs --max_episode_steps 5000;
run_one_cell.sh now auto-bumps when suite name contains libero_10), and CELL_TIMEOUT_S=600
being too tight for long-horizon libero_10 cells (use 1200 + re-launch to backfill
MISSING_AUDIT cells). See ONBOARDING_FRESH_AGENT.md §Defaults.
