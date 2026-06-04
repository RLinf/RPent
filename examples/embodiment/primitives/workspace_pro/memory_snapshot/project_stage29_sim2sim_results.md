---
name: stage29-sim2sim-results
description: "Stage 29 clean (BuilderBench2) physics-eval results for c49_lin peak ckpts — stack ≈ 0 gap, dual-tower large gap"
metadata: 
  node_type: memory
  type: project
  originSessionId: 69ca7184-c9b8-4127-be4d-e47b2e4ea15e
---

c49_lin (cube-4..9-task-1, 30M, oracle yaw + linear α 1.0→0.1 + BC=5.0) → physics-eval via `physics_eval.py` on LLM-track `CreativeCubeLanguageWrapper` (16 episodes, scripted pick_and_place):

| Level | Train peak | Physics strict | mean placed |
|---|---:|---:|---|
| cube-4 stack | 100% | **94%** | 3.88/4 |
| cube-5 stack | 100% | **100%** | 5.00/5 |
| cube-6 stack | 100% | **100%** | 6.00/6 |
| cube-7 stack | 100% | **100%** | 7.00/7 |
| cube-8 double-stack | 36% | **0%** ❌ | 2.81/8 |
| cube-9 double-stack | 0% | **0%** | 0.00/9 |

**Update 2026-05-16**: the cube-8 "0%" was an eval bug, not a sim2sim gap. After fixing physics_eval to (a) place in z-sort order and (b) set grasp_yaw=1 when place yaw=π/2, results become:

| Level | Train peak | Physics FIXED |
|---|---:|---:|
| cube-4 | 100% | 81% |
| cube-5 | 100% | 69% |
| cube-6 | 100% | 87% |
| cube-7 | 100% | 94% |
| **cube-8** | 36% | **94%** |
| cube-9 | 0% | 0% (random policy) |

See [[physics-eval-pipeline]] for details.

**Why:** Stack patterns transfer cleanly because the actual pick-and-place arc just lowers a cube onto a tower — the gripper occupies space above the target only. For double-stack (cube-8/9) the policy places at positions where the 80 mm-wide gripper body, while descending to the second tower, sweeps through the 50 mm gap and knocks the first tower over. Training time we suppress this with `gripper_obb_half_perp=0.040` + `carry_pen_slack=0.015`, which allowed the gap, but the *real* gripper is still 80 mm and the slack just disabled the termination, not the geometric impact.

**How to apply:** Dual-tower/multi-column tasks won't transfer with the current teleport-only training. Options before recommending sim2sim:
- Run the actual scripted pick_and_place during training (slow but exact)
- Or shrink gripper_obb_half_perp to **match the gap** (e.g. 0.020) without slack — forces RL to avoid the gap entirely
- Or change the grasp angle so the gripper's long axis is perpendicular to the tower direction (yaw the gripper, not just the cube)

Linked: [[g8-tele-phys-gap-root-cause]] explains the analogous teleport-vs-physics gap on cube-4 task1 in the polluted Agentic-CRL repo.
