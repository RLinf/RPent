---
name: fragile-box-and-rigid-can-basket-placement
description: Reusable mixed-rigidity basket recipe for a fragile flat box and a rigid can in a cluttered scene.
aliases:
  - put both the cream cheese and the tomato sauce in the basket
  - cream cheese tomato sauce wicker basket
metadata:
  node_type: memory
  type: skill
---

# Fragile-box and rigid-can basket placement

## Applicable pattern

Use this pattern when a fragile flat package and a rigid cylindrical object share a basket target in a cluttered scene and need different grip handling.

## Reference task alias

`put both the cream cheese and the tomato sauce in the basket`

Reference setup: a wicker basket with orange juice, butter, alphabet soup, milk cartons, and a jam jar as distractors.

## Winning technique

**Two clean `pi0_pick` calls from upright home, each followed by manual `move_to`
carry at z=0.62 + `release`. NO `pi0_doubled` anywhere. NO `set_gripper +1` anywhere.**
Pick TOMATO first (easy tall bottle), then re-home with `gripper=-1` to restore
orientation, then pick CREAM CHEESE (fragile thin box). Carry uses only the
`gripper=1` flag in `move_to` — pi0's own ~0.04 closure on cream_cheese equals
box width and holds; `set_gripper +1` would clamp to 0.0002 and squeeze the box
out laterally.

## Magic numbers

| Parameter | Value | Range | Why |
|---|---|---|---|
| Tomato `pi0_pick` | `max_chunks=25, lift_thresh=0.05, gripper_closed_thresh=0.06` | hard | Expect `peak_lift_m` ~0.07, `min_gripper_opening` ~0.050 |
| Cream-cheese `pi0_pick` | `max_chunks=20, lift_thresh=0.05, gripper_closed_thresh=0.06` | hard | Expect `peak_lift_m` ~0.14, `min_gripper_opening` ~0.043 (= box width) |
| Carry / release z | **0.62** | 0.60-0.64 | Low enough that slip lands inside basket; z=0.70 dropped cream_cheese 6cm short (att. 2) |
| Tomato carry `step_clip` | **0.015** | 0.012-0.020 | Bottle tolerates faster motion |
| Cream-cheese carry `step_clip` | **0.012** | 0.010-0.015 | Thin box slips laterally above 0.015 |
| Re-home pose `H` | `[-0.058, 0.0, 0.68]` with `gripper=-1, step_clip=0.025` | DO NOT CHANGE | Canonical home reference; restores upright orientation between picks |
| Split waypoint | mid-y between cream and basket, z=0.62 | as-needed | Required when post-pick `dy > 0.30` (move_to xy limit) |
| Release | `gripper` opens to ~0.078 | n/a | Item drops from z=0.62 |
| Diagnostic: cream_cheese held opening | **0.020–0.060** | — | <0.020 = squeezed empty; >0.060 = never closed |

## Failure modes (mined from attempts 1–7) → fixes that finally worked

1. **`pi0_doubled` BEFORE/BETWEEN picks drifts wrist quat by ≥60°** → next `pi0_pick`
   descends crooked, either misses or squeezes empty (attempts 3, 6, 7).
   - **Fix:** ZERO `pi0_doubled` calls. Use two `pi0_pick` from clean home, with a
     re-home (`gripper=-1`) in between to restore upright orientation.
2. **`set_gripper +1` after pi0_pick clamps to 0.0002 and squeezes the thin
   cream_cheese box out laterally** (attempts 4, 5, 7).
   - **Fix:** never `set_gripper` on either item. Pi0 owns the closure; carry with
     `move_to gripper=1` (hold only, don't re-actuate).
3. **`pi0_doubled("put the cream cheese in the basket")` from on-table state does
   NOT descend** (min eef_z = 0.683, OSC vertical wall — attempts 3, 6).
   - **Fix:** never use partial-task `pi0_doubled` for cream_cheese.
4. **Release at z=0.70 → cream_cheese tumbles ~6 cm in front of basket front edge**
   (attempt 2).
   - **Fix:** carry and release at z=0.62.
5. **SAM3 `"orange and red small box"` grounds to BUTTER at score 0.879** (att. 4).
   - **Fix:** cream_cheese prompts: `cream cheese` / `small box of cream cheese` /
     `silver and blue cream cheese box`. Never include "orange" or "red".
6. **`pi0_pick "cream cheese"` from home with `max_chunks=25` grabs neighboring
   orange_juice or alphabet_soup** (attempts 1, 4).
   - **Fix:** keep `max_chunks=20` for cream cheese (short window forces commit to
     correct target).
7. **`move_to` pre-positioning before `pi0_pick` walls at z≈0.685** (OSC ceiling)
   and then pi0 grounds wrong neighbor from drifted approach (attempt 3).
   - **Fix:** call both `pi0_pick`s directly from clean home — never pre-position.
8. **Second `pi0_pick` after carrying first item descends tilted** because the
   carry leaves eef orientation drifted (attempt 7).
   - **Fix:** re-home with `gripper=-1` between the two picks (step 4 of recipe).

## Re-localize per-scene (SAM3 — pre-flight only)

| Entity | Prompt | Expected score | Backup |
|---|---|---|---|
| basket interior | `wicker basket interior` | ≥ 0.70 (observed run: ~0.9) | `inside of the basket` / `basket` |
| tomato_sauce (sanity) | `tomato sauce bottle` | ≥ 0.50 (observed run: 0.65) | `tomato sauce` / `red bottle of tomato sauce` |
| cream_cheese (sanity) | `cream cheese` | ≥ 0.50 (observed run: ~0.86) | `small box of cream cheese` / `silver and blue cream cheese box` |

**Forbidden cream_cheese prompts:** anything containing "orange" or "red" → BUTTER.

**Used by the recipe:**
- `basket.xy` → release point `R = [basket.x, basket.y, 0.62]` for both items.
- `cream_cheese.y` → split-waypoint y midpoint when `dy > 0.30`.
- Home pose is robot-frame constant; not localized per-scene.
- Tomato xy is NOT consumed (pi0 grounds it from home view).

**Sanity gates:** `cream_cheese.z ∈ [0.45, 0.47]` (on-table); reject if ≥ 0.50
(already moved). `basket` interior z ∈ [0.52, 0.58].

## Difficulty and reliability

**7 failed attempts before the winning eighth.** Among the hardest tasks in the
suite — cream_cheese is a ~3.8 cm thin box that punishes both over-tight closure
and high-altitude release. Single-shot transfer carries genuine risk in the
second `pi0_pick` (cream_cheese). If the verifier supports retries on the
cream-cheese step, allow at least 2 retries of step 4→5 (re-home → re-pick)
before declaring failure.

## Cross-refs

- [[feedback_read_image_before_decide]] — use 1024px world_hi for SAM3 sanity checks.
- [[feedback_pi0_chunks_egl_crash]] — Pi0 chunking conventions.
- [[feedback_read_image_before_decide]] — `silver+blue+box` defeats the butter trap.
- [[feedback_move_pose_covarying_reach]] — why `pi0_doubled("put cream cheese")` doesn't descend.
