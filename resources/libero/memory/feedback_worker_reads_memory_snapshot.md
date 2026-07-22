---
name: worker-reads-memory-snapshot
description: RPent workers read the reviewed in-repository memory snapshot at resources/libero/memory; update it deliberately before evaluation.
metadata:
  node_type: memory
  type: feedback
  originSessionId: 82a57f3e-217b-4ac0-984d-bebcc7d3eb36
---

RPent points workers to `resources/libero/memory/MEMORY.md`, relative to the
repository root. This reviewed snapshot is the canonical Global Memory for a
run; external live notes are not loaded automatically.

**Why:** A frozen, reviewed copy keeps evaluations reproducible and prevents a
new unreviewed note from silently changing behavior during a running batch.

**How to apply:**

1. Run RPent commands from the repository root so memory, runtime, guide,
   script, and result paths remain portable.
2. Resolve Global Memory from `resources/libero/memory/`, the LIBERO runtime
   from `robots/libero/`, agent guides from `robots/libero/guides/`, and
   installation helpers from `scripts/`.
3. Edit and review the memory snapshot before launching an evaluation.
4. Update `MEMORY.md` whenever a leaf file is added or renamed.
5. Do not bulk-copy an external memory directory over this snapshot; merge
   useful lessons one file at a time.
6. Keep runner-only settings in the RPent launch configuration rather than in
   task recipes.

Related: [[feedback_max_episode_steps_libero]] [[feedback_long_horizon_cell_timeout_1200]]
