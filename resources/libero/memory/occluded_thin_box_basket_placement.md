---
name: occluded-thin-box-basket-placement
description: Reusable shared-basket recipe for two thin packages that begin partially occluded by distractors.
aliases:
  - put both the cream cheese box and the butter in the basket
  - cream cheese butter wicker basket
metadata:
  node_type: memory
  type: skill
---

# Occluded thin-box basket placement

## Applicable pattern

Use this pattern when two thin packages share a basket target and both sources require occlusion-aware localization before grasping.

## Reference task alias

`put both the cream cheese box and the butter in the basket`

Reference setup: cream cheese behind orange juice and butter behind a tomato-sauce can in agentview.

## Winning technique

**CREAM CHEESE FIRST via manual carry + tilt-descend; BUTTER SECOND via standalone
autonomous Pi0.** For cream cheese: wrist-scan to find its blue 'Cream Cheese'
label, low pre-pose, `pi0_pick`, carry with `gripper:1` in move_to (NEVER
`set_gripper +1` — squeezes thin box out), and release via `move_pose
target_pitch=0.5` at the basket FRONT to clear the OSC vertical-down wall (z≈0.589).
For butter: wrist-scan to find its red 'FARM FRESH BUTTER' label, pre-pose **LOW
(z≈0.55) directly over the butter top**, then `pi0_pick "pick up the butter"` and
**do nothing else** — Pi0's trained prior runs the entire pick+lift+deposit-in-
basket autonomously and terminates the episode. Critically, basket cavity center is
from a `world_hi` z-filter `[0.43, 0.47]`, NOT SAM3 (which biases to rim by 2-7 cm).

## Magic numbers (firm)

- Pre-position z before pi0_pick: **0.55** (BOTH boxes — low is critical for the
  butter's autonomous Pi0 to fire).
- `pi0_pick` cream cheese: `max_chunks=15, lift_thresh=0.05, gripper_closed_thresh=0.06`.
- `pi0_pick` butter (autonomous): `max_chunks=10, lift_thresh=0.04, gripper_closed_thresh=0.06`.
- Carry height for cream cheese: **z = 0.70 ± 0.02** (never below 0.66 over basket).
- Tilt-descend: `move_pose target_pitch=0.5`, descent z target `floor_z+0.08 ≈ 0.535`
  (actually walls at z≈0.575-0.595), `step_clip=0.015, max_steps=120`.
- OSC vertical-down wall: **z ≈ 0.589** — do NOT try vertical descent for placement.
- Cream cheese drop xy: `(bx, front_y)` where `front_y ≈ by - 0.03`.
- Park-back for clean basket localization: `(-0.30, 0.00, 0.85) gripper=-1`.
- NO `set_gripper +1` ANYWHERE between pick and place.
- After step E pi0_pick (butter), issue NO further move_to / release / set_gripper.

## Failure modes to avoid (mined from 5 failed attempts)

1. **SAM3 basket centroid as drop xy** → rim catch, bounces onto table (attempt 1,
   2). *Fix:* derive `(bx, by, floor_z)` from `world_hi` z-filter `z∈[0.43,0.47]`
   median; never use SAM3 "wicker basket" / "silver foil-wrapped box".
2. **Vertical OSC descent into basket** → walled at z=0.589, can't reach cavity
   (attempt 1). *Fix:* `move_pose target_pitch=0.5` reliably reaches z≈0.58;
   pitch 0.6 clips rim, pitch ≥0.7 over-tilts and box pops out.
3. **`set_gripper +1` after pi0_pick** → thin box squeezed out of fingers during
   carry; lost cream cheese (attempt 4, qpos 0.0484→0.001) and butter (attempt 5).
   *Fix:* NEVER `set_gripper +1`; `gripper:1` in move_to is enough (Pi0 closes to
   ~0.018-0.025, held by friction).
4. **`pi0_pick` from home pose** → grabs milk / orange_juice / tomato_sauce / ketchup
   instead of named target (attempts 1, 2, 4). *Fix:* pre-position EEF over the
   wrist-refined xy at z≈0.55 before EVERY pi0_pick.
5. **Manual carry of butter** → butter slips on lift to 0.70 every time, even with
   gripper:1 only (attempt 5). *Fix:* don't carry butter at all — autonomous Pi0
   from a low pre-pose handles pick+place in one call.
6. **Carry transit below z=0.66 over basket** → bumps basket, can displace it out
   of FOV (attempt 2). *Fix:* keep z≥0.70 in lateral transit; do NOT lower z while
   traversing in y.
7. **Re-pi0_pick after a failed place from stale pose** → grabs WRONG object
   (attempt 2 grabbed milk). *Fix:* re-localize the bounced box via wrist,
   pre-position there, then re-pi0_pick.
8. **Cream cheese pi0_pick descends but doesn't close** (thin/rotated) — do NOT
   `set_gripper +1`; move up 5 cm and re-pi0_pick from slightly higher pre-pose
   (this was the saving move on attempt 5).
9. **Tried to placement-correct butter with manual move_to after pi0_pick** →
   interrupts Pi0's trained place trajectory, causes slip+bounce loop (attempt 5).
   *Fix:* once butter pi0_pick (step 16) starts, issue nothing after it.

## Scene re-localization (do this first)

- **Basket cavity `(bx, by, floor_z)`** — first `move_to (-0.30, 0, 0.85) gripper=-1`
  to park back. Then on `world_hi`, keep pixels with `z ∈ [0.43, 0.47]`;
  `bx, by = median(x), median(y)`, `floor_z ≈ 0.456`. Expect cavity span Δx≈0.14,
  Δy≈0.13; `bx ∈ [-0.03, +0.08]`, `by ∈ [+0.21, +0.25]`. `front_y = by - 0.03`,
  `back_y = by + 0.03`. If <30 hits, widen band to `[0.42, 0.47]`. **NEVER use SAM3
  basket prompts.**
- **Cream cheese `(Cx, Cy)`** at top z≈0.455 — almost always occluded by
  orange_juice. Pre-pose `move_to (-0.02, -0.20, 0.65) gripper=-1`, then on
  the `image_wrist_hi_path` returned by `view_driver_state`, find the **high-saturation blue 'Cream Cheese' label cluster** on
  silver/white box; project to world. Fallback: SAM3 wrist 'small box with blue
  Cream Cheese label'. Expect `Cx ∈ [-0.06, +0.02]`, `Cy ∈ [-0.27, -0.20]`.
- **Butter `(Bx, By)`** at top z≈0.454 — almost always occluded by tomato_sauce_can
  (SAM3 'red box' from agentview returns milk/ketchup/can, never butter). Pre-pose
  `move_to (-0.08, 0.06, 0.65) gripper=-1`, then inspect `image_wrist_hi_path` and find the
  **deep-red 'FARM FRESH BUTTER' label cluster** on flat-lying box; project to
  world. Expect `Bx ∈ [-0.10, -0.05]`, `By ∈ [+0.03, +0.09]`.
- **Declutter fallback (only if a wrist scan still fails after one sweep)**:
  pi0_pick the occluder by name (`orange_juice` or `tomato_sauce_can`), `move_to
  (-0.25, -0.20, 0.65) release`, refresh perception.

## Difficulty and reliability

**6 attempts** to win (5 failed → 1 success on attempt 6). Failures cycled through
SAM3-basket-rim-bias, OSC vertical wall, set_gripper-induced slip, wrong-object
pi0_pick from home pose, and butter-carry slip even without set_gripper. This is a
**high-difficulty** task in a new scene; the recipe converges first-try ONLY when:
(a) basket center is derived from world_hi z-filter, (b) every pi0_pick is
preceded by a wrist-refined pre-pose at z≈0.55, (c) no `set_gripper +1` is ever
issued after a pick, and (d) butter is left to autonomous Pi0 (no manual carry,
no follow-up move_to after its pi0_pick).

See: [[feedback_move_pose_covarying_reach]] (the z≈0.589 wall this whole recipe
works around), [[feedback_read_image_before_decide]] (wrist hi-res perception is
required because both targets are occluded in agentview).
