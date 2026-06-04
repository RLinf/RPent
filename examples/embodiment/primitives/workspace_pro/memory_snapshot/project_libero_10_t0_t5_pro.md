---
name: libero-10-t0-t5-pro
description: LIBERO-Pro libero_10 (long) t0–t5 sweep done 2026-05-21 — 24 Pi0 baselines + 17 hybrid runs in workspace_pro/results_10_pert/
metadata: 
  node_type: memory
  type: project
  originSessionId: 23bda317-5513-4ade-ae34-7d3f3988aeef
---

LIBERO-Pro `libero_10` (long) tasks t0–t5 seed=0 covered 2026-05-21. Artifacts at `examples/embodiment/primitives/workspace_pro/results_10_pert/`.

**Pi0 fullshot baselines (24 runs, 4 cells × 6 tasks, minus t2_task which has 0 init_states):**

| | t0 | t1 | t2 | t3 | t4 | t5 |
|---|---|---|---|---|---|---|
| base | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ |
| _task (P1) | ✗ | ✗ | gap | ✗ | ✗ | ✓ |
| _swap (P2) | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| _lan (Sem) | ✓ | ✗ | ✓ | ✓ | ✓ | ✓ |

Pi0 P2 swap = 0/6 (matches LIBERO-Pro paper headline). P1 task = 1/5 (t5 P1 was visually unambiguous). t4 base fails — multi-mug-2-plate task is hard for Pi0 even canonically.

**Hybrid REPL (17 runs):**

| | t0 | t1 | t2 | t3 | t4 | t5 |
|---|---|---|---|---|---|---|
| _task | ✓ | ✓ | gap | ✗ (drawer) | ✗ (offset bug) | ✓ |
| _swap | ✓ | ✓ | ✓ | ✗ (drawer) | ✓ | ✓ |
| _lan | ✓ | ✓ | ✓ | **✓** (fixed via push) | ✓ | ✗ (book grasp) |

Net: hybrid solves 13/17 where Pi0 fullshot fails 12/17. Headline P2 swap: hybrid 5/6 vs Pi0 0/6.

**Known failure modes (4 hybrid fails):**

- t3 swap + task: closing the drawer with the object inside is the hard part. The
  object must be dragged along by CONTINUOUS contact — either an OSC slow-push (eef low
  in front of the drawer face, move_to +y, step_clip≈0.012, max_steps capped ~100 to dodge
  the QACC NaN, see [[feedback_osc_push_mujoco_nan]]) OR hand the close to `pi0_doubled`
  ("close the drawer") which pushes the front continuously. NO teleport finisher exists
  anymore ([[no-teleport-rule]]). **2026-05-26 caution**: a scripted descent to z≈0.95
  right at the cabinet front can crash the MuJoCo worker (EOFError) — collision blowup;
  let Pi0 control the close approach instead of ramming the cabinet with a scripted descend.
- t4 _task: handle-grasp offset_y ≈ -0.06 not compensated. Mugs landed 3cm forward of plate centers. swap/lan retries with offset compensation succeeded.
- t5 _lan: book at edge of workspace + thin object. Pi0 wouldn't grasp; vertical scripted pick also failed. Pi0 fullshot solves this in 39 chunks because it learns thinner grasp geometry. Open problem for strict.

**Data gap:** `libero_10_task t2` has 0 init_states (ZeroDivisionError in pi0_baseline). Other 39 (variant×task) combos all have 50 trials.

**Process lessons** (relevant for next session):
- Recipe pattern for cluttered libero_10: pre-pos at z=0.65 (LIVING_ROOM) or z=1.05 (KITCHEN), full BDDL task language as Pi0 prompt, track_obj_lift_thresh=0.05–0.08. See [[pi0-delivery-service]].
- Always read and apply post-pick `(target - eef)` offset to release target. Skipped on t4_task initial → 3cm landing error. Applied on t4_swap/lan → <1cm landing error.
- Drawer tasks remain hard for strict regime — the close must be a real continuous push
  (OSC capped push or pi0_doubled). Accept as a known-hard cell if no physical close works.
