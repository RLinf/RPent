---
name: relative-side-first-anchor-placement
description: Reusable two-object recipe that places the tight relative-side target before the anchor-surface object.
aliases:
  - put the white mug on the plate and put the chocolate pudding to the right of the plate
  - white mug plate chocolate pudding right side
metadata:
  node_type: memory
  type: skill
---

# Relative-side-first anchor placement

## Applicable pattern

Use this pattern when one object has a narrow relative-position target beside an anchor and should be placed before a second object occupies the anchor surface.

## Reference task alias

`put the white mug on the plate and put the chocolate pudding to the right of the plate`

## Winning technique

Two pick-and-place sub-goals on a four-entity scene (porcelain_mug,
red_coffee_mug, chocolate_pudding box, plate). **Order is pudding → mug**, not
the reverse: the right_target_zone is a TIGHT region (~10×5 cm centred at
`plate.y + 0.10`), and placing the lower-margin object while the workspace is
empty is what wins. SAM3-localize four entities → pre-pos above pudding at z≈0.60 → `pi0_pick "pick up the chocolate pudding"` (max_chunks=12) → firm
grip → 3-waypoint arc to `(PLATE.x, PLATE.y + 0.10, 0.49)` → release. Home
retreat → pre-pos above mug at `(MUG.x, MUG.y + 0.024, 0.60)` →
`pi0_pick "pick up the white mug"` (max_chunks=10; "white" disambiguates from
the red distractor) → firm grip → carry-high to z=0.70 → 4-step slow descent
to `(PLATE.x − 0.037, PLATE.y + 0.078, 0.56)` (the y bias is the rim-grasp
carry-shift compensator, NOT a place-off-center) → release. `libero_terminated`
fires on the mug release. "Right of plate" = world +y, NOT +x.

## Magic numbers

| Lever | Value (observed run win) | Range | Notes |
|---|---|---|---|
| **Order** | pudding → mug | fixed | mug-first lost on observed run attempts 1 & 2 |
| Pudding `pi0_pick max_chunks` | 12 | 10–15 | observed run used 5 chunks; `peak_lift_m` 0.152 m |
| Pudding `lift_thresh / gripper_closed_thresh` | 0.05 / 0.06 | standard | observed `min_gripper_opening` 0.047 |
| Mug `pi0_pick max_chunks` | 10 | 8–14 | rim-grasp is quick |
| Mug `lift_thresh / gripper_closed_thresh` | 0.05 / 0.06 | standard | rim signature: **`min_gripper_opening` ≈ 0.028 (range 0.022–0.038)** |
| `set_gripper +1, steps` after every pi0_pick | 15 (pudding), 12 (mug) | 10–18 | mandatory both stages |
| Pudding pre-pose z | 0.60 | 0.58–0.62 | hover ≈ PUD.z + 0.13 |
| Pudding **final xy** | `(PLATE.x, PLATE.y + 0.10)` | `Δy ∈ [+0.085, +0.12]`, `|Δx| ≤ 0.03` | the right_target_zone is TIGHT — never exceed `+0.15` |
| Pudding final z | 0.49 | 0.47–0.50 | ≈ PLATE.z + 0.04 |
| Mug pre-pick **y bias** | `MUG.y + 0.024` | `+0.015 … +0.035` | aligns wrist over rim plane |
| Mug carry-high z | 0.70 | 0.68–0.72 | clears placed pudding |
| Mug **final eef bias** | `(PLATE.x − 0.037, PLATE.y + 0.078)` | y `+0.055..+0.090`, x `−0.02..−0.05` | cancels held-mug centroid offset of `(~-0.05, ~-0.075, ~-0.04)` from eef |
| Mug **final z** | **0.56** | 0.54–0.58 (= PLATE.z + 0.11) | z=0.59 → edge-perched, predicate fails (attempt 1) |
| Mug **final step_clip** | 0.008 | ≤ 0.012 | mandatory slow descent so held mug stops swinging |
| Home retreat between stages | `(0, 0, 0.68)` | mandatory | avoids clipping placed pudding |
| `gripper = -1` on every pre-pick `move_to` | always | — | holding +1 between calls confuses next pi0_pick |

## Failure modes to avoid (mined from two failed attempts)

**Attempt 1 — mug-first, edge-perched then tipped on recovery.** Mug released
at z=0.59 → mug bottom landed at z=0.425 (table level), not on plate top
(0.447). On predicate did NOT fire even though it looked centred in the cam
image. Tried to re-pi0_pick to nudge — pi0's approach motion tipped the mug
(z dropped to 0.479 = side-lying), unrecoverable.
- **Fix:** mug release z = **0.56** (NOT 0.59); 4-step slow descent with final
  `step_clip = 0.008`; **never re-pi0_pick a placed standing mug** (it'll
  knock it over) — use scripted nudges only.

**Attempt 2 — mug-first, mug placed correctly but pudding overshot the zone.**
Mug ended at `(0.177, 0.024, 0.526)` over plate `(0.172, -0.002, 0.447)` —
looked perfect. Pudding then placed at `(0.178, 0.234, 0.466)` — `Δy = +0.236`
from plate y. In(right_target_zone) did NOT fire — the zone is `Δy ~+0.10`,
NOT a half-plane. Tried to re-pick pudding to recenter — pi0_pick failed (lift
1.2 cm), pudding likely tipped. Out of budget.
- **Fix:** pudding goes **FIRST** (not last) and lands at exactly
  `(PLATE.x, PLATE.y + 0.10, 0.49)`. Never `Δy > +0.15`.

**General lessons (failed attempts plus the winning trail):**
1. **Two-mug ambiguity.** Always say `"the white mug"` (or include "white"/"porcelain") in pi0_pick prompts. `"the mug"` alone hits the red mug ~half the time.
2. **Rim-grasp signature.** `min_gripper_opening` **must** land in `[0.022, 0.038]` for the porcelain mug. <0.015 = grabbed air; >0.05 = grabbed body (slip-prone). **The `pi0_pick` result field `success` may be False at the valid rim opening (`diagnostics.descent_done` was false on the observed run)** — trust the rim-opening value + carry behavior, NOT the heuristic. See [[feedback_pi0_false_positive_lift]].
3. **Held-mug carry shift.** With rim grasp, mug centroid sits ≈ `eef + (-0.05, -0.075, -0.04)`. To place mug centroid on plate center, displace eef by `(-0.037, +0.078)` in xy. Targeting `eef.xy = PLATE.xy` lands the mug at the plate's −y edge → edge-perch → fail.
4. **OSC wall at z ≈ 0.555 is a feature here:** mug at z=0.56 is just past it, pudding at z=0.49 is below it. Do NOT push past with `move_pose`/`rotate_pitch` — tilting ejects either object sideways.
5. **`max_chunks ≤ 25`** — EGL_NOT_INITIALIZED crash risk (see [[feedback_pi0_chunks_egl_crash]]).

## How to re-localize per-scene

Four SAM3 prompts on agentview (`world_hi` top-down as disambiguator if scores are borderline):

| Entity | Primary prompt | Fallback | Score floor |
|---|---|---|---|
| `PLATE` | `"the white plate with red rim"` | `"the round white plate near the front"` | 0.90 |
| `PUD` | `"the small brown chocolate pudding box"` | `"the small dark food package between the two mugs"` | 0.40 |
| `MUG` (white) | `"the white mug on the left"` | `"the small white ceramic mug, not the red one"` | 0.40 |
| `RED` (distractor; cache for sanity + recovery) | `"the red mug with handle"` | `"the larger red coffee mug"` | 0.35 |

**Sanity gates (re-segment if any fails):**
- `MUG.y < PUD.y < RED.y` (white-left → pudding-middle → red-right in image, world `-y → +y`).
- Two distinct mug masks; centroids > 0.05 m apart.
- `PLATE` unique.
- Compute `PLACE_PUD = (PLATE.x, PLATE.y + 0.10)` — NOT `(PLATE.x + 0.10, PLATE.y)`.

**Post-pick truth checks (mandatory):**
- After mug pi0_pick: check `min_gripper_opening ∈ [0.022, 0.038]`. If outside,
  the grasp failed (air or body) — abort, re-pose, retry. Optionally wrist-segment
  `"the white mug below the gripper"` to measure `held_offset_y = eef.y − mug_center.y`;
  use that as the bias instead of the default `+0.078`.
- After pudding pi0_pick: check `peak_lift_m > 0.08`. If pi0 grabbed the RED mug
  by mistake (rare, difficult layouts), re-segment `RED` — if it moved > 5 cm,
  apply parking recovery (`move_to (-0.05, +0.20, 0.54)`, release, retreat),
  then redo pudding.

## Difficulty and reliability

**3 attempts to win** — attempt 1 (mug-first edge-perch + tip-on-recovery),
attempt 2 (mug-first OK but pudding overshot zone at Δy=+0.236), attempt 3
(verbatim replay of a known-good pudding-first recipe = win). True difficulty:
**medium-hard** (compound predicate with TIGHT zone + rim-grasp placement
precision). This recipe encodes the order, the zone offset, and the
mug-release z to remove both failure modes a priori.

## Related

  sequence, lever ranges, layout-fragile fallbacks, and per-scene re-localization.
- [[ordered_dual_can_basket_placement]] / [[occluded_thin_box_basket_placement]] / [[upright_book_back_compartment_placement]] — sibling
  swap-suite tasks with the same `pi0_pick "pick up the X"` short-prompt
  pattern; [[occluded_thin_box_basket_placement]] has the matching distractor-aware pre-pose discipline.
  world −y/+y.
- [[feedback_pi0_pre_pos_can_hurt]] — pre-pose is usually a net negative;
  pudding here is an explicit exception (small/dim + salient distractor).
- [[feedback_pi0_false_positive_lift]] — `pi0_pick.success` is provisional;
  use `min_gripper_opening`, `peak_lift_m`, current images, and carry behavior.
- [[feedback_bowl_eef_y_offset]] — rim-offset principle (~+0.078 in y here for
  porcelain mug rim grasp).
- [[feedback_pi0_chunks_egl_crash]] — `max_chunks ≤ 25` hard rule.
- [[feedback_move_pose_covarying_reach]] — z ≈ 0.555 wall; in this recipe the
  mug lands at z=0.56 just past it and the pudding at z=0.49 below it.
- [[feedback_read_image_before_decide]] — porcelain mug / chocolate
  pudding box are gallery-listed names.
