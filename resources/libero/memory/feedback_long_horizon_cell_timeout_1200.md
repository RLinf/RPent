---
name: feedback-long-horizon-cell-timeout-1200
description: "Complex LIBERO cells need CELL_TIMEOUT_S=1200 so long recovery sequences can finish and write their audit"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 736c7752-a7a7-4d37-93a7-98a7af2c85dd
---

Set `CELL_TIMEOUT_S=1200` for complex LIBERO cells, including cells in the
spatial, object, and goal suites. The 600-second harness default can terminate
a valid long-horizon attempt before it writes its audit.

**Why:** closed-loop solving may require many `move_to` waypoints, re-picks, and
recovery actions. First observed 2026-05-27 on `libero_spatial_swap t3`: at 600s only one cell finished
(341s), while another had **actually solved the sim (`libero_terminated=True`) but was
killed by `timeout -9` before it could write its audit** → MISSING_AUDIT. 4 more
hit 600s still unfinished. A subsequent orchestrated run used
`CELL_TIMEOUT_S=1200`; skip-if-audit-exists preserved cells already solved.

**How to apply:** pass `CELL_TIMEOUT_S=1200` to the evaluation orchestrator; it
must propagate to the per-cell runner by environment inheritance. A timed-out
cell with an empty or truncated provider output file can occur because the
agent process may flush its final text only at exit. Check the latest
`view_driver_state` result, or the last entry in `states.json`, and read its
`libero_terminated` field to distinguish a real-but-unrecorded solve from a
genuine non-finish.
Same lever ONBOARDING §2 already prescribes for libero_10. See
[[feedback_worker_reads_memory_snapshot]].
