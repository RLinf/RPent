---
name: dual-object-anchor-and-relative-placement
description: Reusable plate-centered recipe combining a mug-rim grasp, a box grasp, and relative-side placement.
aliases:
  - put the red mug on the plate and put the chocolate pudding to the right of the plate
  - red mug plate chocolate pudding right side
metadata:
  node_type: memory
  type: skill
---

# Dual-object anchor and relative placement

## Applicable pattern

Use this pattern when one graspable object must be placed on an anchor surface and a second object must be placed at a directional offset from that same anchor. Re-localize both objects and the anchor in the current scene; treat the reference values below as sanity ranges, not absolute coordinates.

## Reference task alias

`put the red mug on the plate and put the chocolate pudding to the right of the plate`

## Winning technique

Two-stage place: red mug onto the white plate, then chocolate pudding box at plate_y + 0.100 (image-right). Single-camera agentview + SAM3 is sufficient — the porcelain-mug distractor is rejected by semantic prompt selectivity. The whole task is a sequence of three SAM3 segments + two `pi0_pick`s, each followed immediately by `set_gripper +1` to firm the grasp before carry. The winning technique is **mug-first, short pick prompts, hard-firm grip, carry-high, descend-to-OSC-contact-stall, release**.

## Magic numbers (ranges from observed runs)

- Mug pre-grasp xyz: `(mug.x + 0..0.010, mug.y + 0.005, 0.65)` — z ≈ mug_z + 0.116.
- Mug carry altitude: `z = 0.72`.
- Approach-above-plate xyz: `(plate.x + 0..0.017, plate.y + 0.029, 0.70)`.
- Mug commanded release z: `0.49` (OSC stalls at `z ≈ 0.552` — that is the mug-bottom-on-plate contact height; stable in observed runs).
- Mug descent `step_clip` ladder: `0.025 → 0.015 → 0.008`.
- Pudding pre-grasp xyz: `(pudding.x, pudding.y, 0.62)` — z ≈ pudding_z + 0.154.
- Pudding carry waypoint: `(0.0, 0.05, 0.66)` (stable in observed runs).
- Pudding place xyz: `(plate.x, plate.y + 0.100, 0.62)` then descend to `0.52`. **+y = image-right; magnitude 0.097–0.100.**
- `set_gripper +1 steps` after pick: **20 for mug** (rim, narrow tolerance), **12 for pudding** (box, wide).
- `pi0_pick max_chunks`: **12 for mug, 15 for pudding**. Both finish in 4–5 chunks; the headroom matters if the rim is slightly off.
- **Rim-hook fallback geometry** (if standard mug.y+0.005 pre-grasp doesn't seat): pre-position eef at `(mug.x, mug.y + 0.045, 0.68)` with `pi0_pick max_chunks=8` + `set_gripper +1 steps=20`. Held-mug eef↔mug delta becomes `(+0.011 x, +0.041 y)` — subtract from the plate-approach target: `(plate.x + 0.011, plate.y + 0.041, 0.70)`. This is an observed fallback measurement; re-localize before use.

## Failure modes to avoid

- **Mug tip during carry** — Pi0 reports grasp success but rim slips mid-carry → mug tips → z drops to ~0.455 → unrecoverable. Fix: `set_gripper +1 steps=20` *immediately* after `pi0_pick`. This was the observed first-run failure pattern.
- **Full task_language as Pi0 prompt** — causes place-mode bleed during pick (Pi0 starts translating sideways before lift). Fix: short verb-phrase prompts: `"pick up the red mug"`, `"pick up the chocolate pudding"`.
- **Brand-noun pudding prompt** — `"chocolate pudding box"` scores ~0.02 on SAM3. Use the geometric description `"small dark rectangular box on the table"` (scores 0.44–0.70).
- **Distractor graze** — porcelain mug sits image-left of red mug, along the natural carry path. Always stage through `(0.0, ~0.025, 0.72)` so you never drop below z=0.7 outside the plate column.
- **Pudding-first ordering (primary recipe)** — canonical order is **mug first, then pudding**. Pudding-first DOES work, but only as a fallback when mug-first keeps slipping despite firm-grip; mug-first has the stronger transfer evidence.
- **Narrow "right of plate" predicate box** — in an observed run, a release at `plate.y + 0.16` fell **outside** the predicate region and did NOT terminate, while `plate.y + 0.10` was inside. Tolerance ≈ ±0.02 m. If first place doesn't terminate, RE-PICK the pudding and place tighter at `plate.y + 0.10` exactly — do not assume wrong object.
- **Over-descent** — commanding z below 0.49 wastes steps; OSC contact-stall at 0.552 is the real release height for the mug. Trust the stall.

## How to re-localize per-scene (agentview SAM3 only — no wrist needed)

- **red_coffee_mug** ← prompt `"the red coffee mug with red and white pattern"` (score ≥ 0.85). The "red + white pattern" wording is what rejects the porcelain distractor.
- **plate** ← prompt `"the white plate with red rings"` (score ≈ 0.98).
- **chocolate_pudding** ← prompt `"small dark rectangular box on the table"` (score ≥ 0.4). If score ≈ 0.02 you used a brand noun — switch to geometric.

Observed ranges are useful only as sanity checks; treat the current scene localization as the input, never as a constant.

## Difficulty and reliability

Observed runs: **4 attempts total** — the first failed when the mug tipped, then the firm-grip version succeeded in the next 3 attempts. With that lever in place, this recipe is single-shot.

## Cross-task links

- Pi0 grasp-then-place-mode-bleed when prompts are too long is the same failure pattern noted in [[feedback_pi0_chunks_egl_crash]] — keep grasp prompts to verb + object.
- Distractor rejection via semantic SAM3 (porcelain mug ignored without two-camera disambiguation) is the same trick used in [[feedback_read_image_before_decide]].
