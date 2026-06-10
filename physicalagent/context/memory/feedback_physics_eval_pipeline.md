---
name: physics-eval-pipeline
description: Stage29 physics eval must use z-sort placement order + grasp_yaw=1 when cube yaw=π/2 — bug masked trained cube-8 success
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 69ca7184-c9b8-4127-be4d-e47b2e4ea15e
---

When running physics eval on a Stage29 / primitive-env-trained policy via the LLM-track `CreativeCubeLanguageWrapper`'s scripted `pick_and_place`, two non-obvious requirements:

1. **Placement order must be z-sort ascending** (lowest target z first), NOT cube_id order. For dual-tower tasks, z-sort interleaves the two towers naturally (A0, B0, A1, B1, …, top). Placing in cube_id order finishes tower A first then tower B — the gripper then has to descend past the now-tall tower A to place into B, knocking A over.

2. **`grasp_yaw` must be derived from place yaw**: `grasp_yaw=1` (π/4 gripper) when the cube's place yaw is π/2 (yaw_idx=2); otherwise `grasp_yaw=0`. The gripper jaws must align with the cube's faces at the moment of release.

**Why:** these are baked into the oracle physics pipeline at `${BUILDERBENCH_ROOT:-/path/to/builderbench}/rl/impls/physics_video_oracle_yaw.py` and `compare_all_llm_vs_llm_rl.py`. Missing either turns a working trained policy into 0% physics — observed: cube-8-task1 ckpt that hit 36% in teleport eval went from `errs=[40.9, 95.2, 151.1, 213.0, 3.8, 11.8, 16.2, 143.9]mm` (cube_id order + grasp_yaw=0) to `errs=[6.2, 4.9, 5.1, 3.9, 4.6, 12.7, 17.2, 14.6]mm` (z-sort + correct grasp_yaw) — strict pass.

**How to apply:**
- Pass `--yaw_idx_per_cube` (integer indices 0/1/2) to physics_eval.py / physics_eval_video.py, NOT raw radians via `--yaw_per_cube`.
- For each placement step in z-sort order, query the policy with `cube_oh[k]=1` where `k` is the *task-local* cube id of the cube being placed at this z-sort step (NOT the step index).
- The trained primitive-env policy IS goal-conditioned by cube_oh, so it still outputs the correct xy when queried out-of-cube-id-sequence.
- Stack tasks (cube-4..7-task1) happen to have z-sort = cube_id order so they worked even with the buggy pipeline; double-tower (cube-8/9) doesn't.

Linked: [[stage29-sim2sim-results]] (now superseded — sim2sim gap was largely an eval bug).
