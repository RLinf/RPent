---
name: build_block.py workspace_bounds also clamps action range
description: editing _workspace_bounds also changes _prim_target_hi via min(cfg, ws); breaks ckpts trained with old value
type: feedback
originSessionId: d8efb52a-c642-417f-8d42-9c041c8719c2
---
`build_block.py:_workspace_bounds[0][0]` (x_max) is used by `build_block_primitive.py:_prim_target_hi = jnp.minimum(cfg_hi, ws_hi)` — so editing the workspace x-max ALSO changes the policy's effective action range.

**Why:** during the 2026-05-09/10 cube-N-task1 sweep we set `target_anchor_x_range=(0.45, 0.45)` so policy learns to point at goal x=0.45. Workspace x_max was 0.45 at training start, so action range clamped to [0.25, 0.45] and policy learned saturated `tanh(mean[0]) ≈ 0.997` → decode to 0.45. Someone later raised workspace x_max to 0.55 (for an "OOB safety margin"). Re-importing build_block.py for offline eval gave action range [0.25, 0.55], same saturated action decoded to 0.55 → 0% physics_eval despite training-time eval showing 100%.

**How to apply:**
- For ckpts trained with old workspace x_max=0.45: pass `--target_x_max 0.45` to physics_eval / debug scripts, or revert workspace_bounds.
- Generally: don't change `_workspace_bounds` mid-sweep without also matching `cfg.primitive_target_x_max`. Better: decouple the action-range clamp from workspace OOB.
- If training and offline disagree wildly (training 100%, offline 0%), suspect a global constant (workspace, action range, action scale) was edited after training started — Python imports cache the old value.
