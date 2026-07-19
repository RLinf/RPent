---
name: dual-moka-obstacle-aware-stove-placement
description: Reusable dual-vessel stove recipe using cap-grip, tilt-extend placement, and a front-arc carry around the first placed pot.
aliases:
  - put both moka pots on the stove
  - two moka pots single burner
metadata:
  node_type: memory
  type: skill
---

# Dual-moka obstacle-aware stove placement

## Applicable pattern

Use this pattern when two tall handled vessels must share one cook surface and the second carry must avoid the first placed object.

## Reference task alias

`put both moka pots on the stove`

## Winning technique

Pick `moka_pot_1` (L) and `moka_pot_2` (R) off the table and place both
upright on the single visible burner pad (L on the back/`+y` side, R on
the front/`-y` side, ≥5 cm apart). Technique: **OSC cap-grip from
straight above for both pots**, **tilt-extend with `move_pose
target_pitch≈0.45`** to clear the OSC vertical wall and reach the
cooktop, and — *the* transfer-critical lever — **a FRONT-ARC carry for
R** that traverses via `y ≤ -0.10` and never crosses the column where
L was just placed. Place L first (simple drop), then restore wrist orientation, then
pick R, then front-arc carry, then a two-stage tilt-place ending with a
deep-punch (`target_pitch=0.5`, `step_clip=0.008`) to slip past the
`-y` OSC wall.

## Magic numbers (with ranges)

- Cap-grip descent z: `pot_z ± 0.01` (≈ 1.05); accept `qpos ∈ [0.004, 0.010]`
- L (Phase B) tilt-place pitch: `0.45`, `step_clip=0.015`, expect stall
  z ≈ 1.08 ± 0.02; call `release`, then retreat upward by `0.10 m`
- Wrist-orientation restore (Phase C): `rotate_pitch target_pitch=0.0`, then use
  `target_pitch=-0.05` on the next `move_pose` call together with its required `xyz`
- **R front-arc carry (Phase E):** keep all interior waypoints at
  `y ≤ -0.10`, `z = 1.50`. Path: lift at R → (`L_xy.x + 0.15, -0.10, 1.50`)
  → (`B_x - 0.11, -0.10, 1.50`)
- R tilt-place (F2): pitch `0.45`, `step_clip=0.015`, `max_steps=200`,
  expect stall z ≈ 1.20 ± 0.02 (-y wall is ~10 cm higher than L's wall)
- R re-clamp (F3): `set_gripper +1` for **5 steps**
- R deep-punch (F4): pitch `0.50`, `step_clip=0.008`, `max_steps=200`,
  end z ≈ 1.08–1.10; **expect `qpos` to dip to ~0.002 (pot slips from
  cap during punch — release anyway)**
- R release: call `release`, then retreat upward by `0.20 m` with `move_to`
- Burner placement offsets from `B_xy`:
  - L target = `(B_x - 0.01, B_y - 0.01)`
  - R target = `(B_x + 0.01, B_y - 0.06)`
- Pi0 max_chunks: **N/A — do not call Pi0 for placement** (see failure mode).

## Failure modes to avoid (mined from 85 failed observed run attempts)

| Failure | What happened | Fix that works |
| --- | --- | --- |
| **R carry brushes L** (att 68–72, 74, 85) | r1's high-arc carry (`y ≈ +0.10 → -0.02` at z=1.45) passes 4 cm above L's column; the swinging R pot tips L. | **Front-arc carry**: keep R at `y ≤ -0.10` for the entire traverse. |
| **Pi0 placement off-pad** (att 1–45) | `pi0_doubled` releases R at random eef positions; <1/7 land on burner; ~1/3 collide with L. | Replace Pi0 placement with manual tilt-extend + deep-punch. |
| **Body-grip slip** (att 36/38/41/47/50) | Body-grip needs the pot at z=0.97; either fragile tilt-pick or grabs through cap that slips. | **Cap-grip from above** at z=1.05. |
| **OSC vertical wall on placement** | Pure OSC descent stalls 7–13 cm above the cooktop, gripper releases L mid-air → tips. | Tilt with `move_pose target_pitch=0.45` so the wrist orientation lets OSC reach the cooktop. |
| **OSC `-y` wall stalls R 14 cm short** | F2 at `target_pitch=0.45` stalls at z≈1.21 — too high for a clean drop. | **Deep punch** F4 with smaller `step_clip=0.008` and `target_pitch=0.5`. |
| **Cap-grip qpos collapse during F4 deep-punch** | `qpos` drops 0.006 → 0.002 mid-punch; pot slipping. | F3 re-clamp `set_gripper +1 5 steps` immediately before F4; release F5 anyway — low-drop usually settles upright. |
| **Residual wrist roll after Pi0 / tilt** | `quat_x ≈ 0.37`; subsequent top-down descend on R grabs air. | Explicitly restore wrist orientation in Phase C: `rotate_pitch target_pitch=0.0`, then set `target_pitch=-0.05` on the next `move_pose` waypoint before approaching R. |
| **Cross-scene reuse of one scene's deep-punch numbers** (att 63) | -y OSC wall can differ by about 13 cm between layouts; numbers don't transfer. | Always re-derive `B_xy` from the current scene's SAM3 + `world_hi_00.npy`; use the ranges in this file, not copied literal coordinates. |
| **First cap-grip catches air** (att 5) | Cap-mask centroid off by 1–2 cm; gripper closes empty. | Lift 2 cm and check `qpos`; re-open and retry descent perturbed `(±0.012, 0)`. |

## Re-localization per-scene

- **L pot:** SAM3 `"moka pot"` on `image_cam_hi_00`; if two masks, take the
  one with smaller world-x. Fallback prompt: `"silver coffee pot"`. Use
  mask centroid via `world_hi_00.npy`; expect `L_z ≈ 1.05 ± 0.01`.
- **R pot:** SAM3 `"moka pot"` on the same frame; the mask with larger
  world-x. Same z sanity.
- **Burner center `B_xy`:** SAM3 `"red coil burner"` or
  `"red circle on cooktop"`; centroid via `world_hi_00.npy`. Sanity box:
  `B_x ∈ [0.18, 0.24]`, `B_y ∈ [0.02, 0.08]`. If outside, re-prompt
  `"red ring on stove"`; if still outside, use fallback `B_xy = (0.21, 0.05)`
  (the kitchen geometry is fixed across-layouts in this suite; only pot
  positions move).
- **SAM3 score threshold:** ≥ 0.4 acceptable; below that re-prompt.
- **L placement target** = `(B_x - 0.01, B_y - 0.01)`.
- **R placement target** = `(B_x + 0.01, B_y - 0.06)`.

## Difficulty and reliability

**86 attempts to first win.** 85 failed before the breakthrough. The wins
that almost-fired (att 51 / 62 / 66) had both pots upright inside the
burner footprint but with the predicate not firing — and replays of the
"r1 winning audit" (att 68–72, 74, 85) reproduced waypoints to mm
precision and still failed because the recipe's R carry path
stochastically brushed L. The novel front-arc carry on attempt 86 was the
single change that converted both-upright into both-upright-and-predicate-fires.
**Difficulty signal: high — expect tuning of the F4 deep-punch on new scenes
if the `-y` OSC wall sits higher than reference run's z≈1.21.**

## Related

- [[feedback_move_pose_covarying_reach]] — the underlying OSC limitation
  the tilt-extend works around
- [[feedback_no_pi0_end_to_end]] — independent confirmation that Pi0
  shouldn't drive the placement
- [[feedback_completion_sensitive_multi_object_ordering]] — choose object order from the current layout and protect later carries from the occupied target
