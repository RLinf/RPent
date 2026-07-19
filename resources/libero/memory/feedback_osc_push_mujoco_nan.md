---
name: feedback_osc_push_mujoco_nan
description: "Long OSC push (high max_steps through sustained contact) can trigger MuJoCo NaN at QACC DOF9; close drawers with short capped pushes or pi0_doubled, never one long push"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: fa95dece-8efe-42a7-9077-a1d2883a0dba
---

A sustained continuous OSC `move_to` (high `max_steps`, e.g. 300, driving through
prolonged contact — drawer wall + object inside) can blow up the MuJoCo solver:
`WARNING: Nan, Inf or huge value in QACC at DOF 9. The simulation is unstable.`
The OSC servo then spins indefinitely without converging → the cell dies at the
wall-clock `CELL_TIMEOUT`, NOT at `max_episode_steps` (that raises a clean
`ValueError("executing action in terminated episode")` instead — different signature).

**Why:** this is a **primitive bug**, not infra — the NaN is triggered by OUR command
(a long push against sustained contact at a gripper/finger DOF). It is **probabilistic**:
the identical QACC-DOF9 warning fired in `libero_10 swap t5 observed run` and there the sim
recovered and the cell *succeeded*; it only failed `libero_10 lan t3 observed run` (drawer-close,
bowl already inside, one push from done). So the warning is not always fatal.

**How to apply:** for Close(slide-drawer), do not rely on ONE long OSC push to fully shut
it. Cap each push `max_steps` low (~80–120) and issue
SEVERAL short pushes (re-acquire contact between them), or hand the drawer-close skill to
`pi0_doubled` ("close the drawer"). Lowering `step_clip` alone does NOT prevent the blow-up.
Distinct from [[feedback_max_episode_steps_libero]] (that's the clean ValueError) and
[[feedback_pi0_chunks_egl_crash]] (EGL/EOFError).
