---
name: feedback_redo_cell_timeout_1200
description: "Physics-only teleport-redo cells need CELL_TIMEOUT_S=1200, not the harness default 600"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 736c7752-a7a7-4d37-93a7-98a7af2c85dd
---

When re-running the teleport-contaminated multi_seed_exp cells **physics-only**
(TELEPORT_REDO_CELLS.md), set `CELL_TIMEOUT_S=1200` even for the "short" suites
(spatial/object/goal). The harness default 600 (in `run_one_cell.sh`) was tuned
for the teleport-era recipes, which finished fast.

**Why:** physics-only solving takes much longer — more `move_to` waypoints,
re-picks, and recovery instead of one `set_object_pose`/`js_move_to` warp. First
observed 2026-05-27 on `libero_spatial_swap t3`: at 600s only s9 finished (341s),
while s0 had **actually solved the sim (`sim_libero_terminated=True`) but was
killed by `timeout -9` before it could write its audit** → MISSING_AUDIT. 4 more
hit 600s still unfinished. Relaunched the whole orchestrator with
CELL_TIMEOUT_S=1200; skip-if-audit-exists preserves cells already solved.

**How to apply:** pass `CELL_TIMEOUT_S=1200` in the orchestrator's invocation of
`run_parallel_seeds.sh` (it propagates to `run_one_cell.sh` by env inheritance —
neither script overrides it). A timed-out cell with empty `claude_*.txt` is
normal: `claude -p --output-format text` flushes its result only at exit, so a
`timeout` kill loses all stdout. Check `sim_libero_terminated` in the last
`states.json` to tell a real-but-unrecorded solve from a genuine non-finish.
Same lever ONBOARDING §2 already prescribes for libero_10. See
[[feedback_worker_reads_memory_snapshot]].
