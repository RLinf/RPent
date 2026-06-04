---
name: G8 BuilderBench teleport vs physics gap — root cause = transient success
description: cube-4-task1 75% teleport metric is 1-frame transient (median sustain=1/12 steps); not a "mature" 4-stack policy
type: project
originSessionId: d8efb52a-c642-417f-8d42-9c041c8719c2
---
Diagnosis for the cube-N-task1 primitive-RL teleport-vs-physics gap (cube-4-task1: 75% strict tele → 0% physics):

**Root cause:** the teleport metric `episode_success_rate = mean(episode_success > 0)` only requires ONE timestep of geometric alignment + qvel<0.05 in a 12-step episode. Per-step diagnostics on cube-4-task1 ckpt_14 with 128 batched envs:

| task | any-step rate | per-step peak | sustain mean | sustain ≥6 |
|---|---|---|---|---|
| cube-2-task1 | 84% | 81% (step 5) | 5.79/8 | 71% |
| cube-3-task1 | 80% | 60% (step 8) | 3.33/9 | 17.6% |
| cube-4-task1 | 75% | 27% (step 11) | **1.78/12** | **0%** |

**Why:** primitive env has NO no-op action — every step writes one cube's qpos. Policy "succeeds" at step k by transient alignment, then at step k+1 picks another cube and moves it, breaking the stack. Reward signal can't distinguish "hold pose" from "rebuild" (both give max reward at the moment), so policy never learns to be still. In physics eval the alignment window is only 1 frame wide AND scripted primitive can't hit it precisely → 0%.

**Why:** transient-success collapse explains the 75pp gap better than arm-disturbance (which user observed visually does NOT happen) or spawn-distribution (override didn't help).

**How to apply:**
- Don't tout cube-4 75% as a "mature" result; cube-2 84% (sustain 5.8/8) is genuinely mature, cube-4 is not.
- To close the gap, fix the metric or the policy: e.g. require `success` for last K steps consecutively; or add a "no-op / hold" primitive; or shape reward to penalize moving an already-correct cube.
- Physics eval will only work for tasks/checkpoints whose teleport sustain ≥ ~3 steps.

**Goal-distribution OOD is the second root cause for physics_eval = 0%.** RL training goal anchor x ∈ [0.22, 0.32] (build_block.py:456); LLM-track default goal at x=0.45. Policy never trained on goals near 0.45 → commands xy scattered across (0.26, -0.18) ~ (0.55, 0.17), cubes go everywhere, max_placed=0/2 every ep.

**Verified with cube-2-task1 ckpt_14:**
- LLM default goal 0.45: 0/32 strict
- `--override_goal_x 0.27 --override_goal_y 0.0`: **22% strict / 50% easy**
- Adding `--override_spawn_x 0.10/0.15/0.20`: back to 0% — qpos override breaks scripted primitive (cubes barely move). Don't override spawn; LLM default 0.30 is what scripted IK is calibrated for.

**Residual gap (cube-2 sustained_end 72% → physics 22% with goal override) explained by:**
- Scripted primitive xy error ~15-20mm per op under load (not the 3-6mm from clean test_primitive_precision)
- Second cube drops onto under-positioned first cube → slides off → fails stack
- Spawn still mismatched (0.30 vs training 0.05-0.10), but unfix-able via override.

**Standard physics-eval invocation now:** always pass `--override_goal_x 0.27 --override_goal_y 0.0` (or task-appropriate goal in [0.22, 0.32]); never override spawn.
