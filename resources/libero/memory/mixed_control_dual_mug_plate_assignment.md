---
name: mixed-control-dual-mug-plate-assignment
description: Reusable identity-aware two-mug placement pattern using manual control for one plate and closed-loop delivery for the harder side.
aliases:
  - put the white mug on the left plate and put the yellow and white mug on the right plate
  - white mug left plate yellow white mug right plate
metadata:
  node_type: memory
  type: skill
---

# Mixed-control dual-mug plate assignment

## Applicable pattern

Use this pattern when two visually similar mugs have identity-specific plate targets and the two target sides have different OSC reachability.

## Reference task alias

`put the white mug on the left plate and put the yellow and white mug on the right plate`

## Winning technique

Two-mug swap into two flanking plates with a **red coffee mug DECOY** between
the mug cluster and the +y plate. Convention: image-left = −y world. Sequence:

1. `pi0_pick "pick up the white mug"` (short canonical, `max_chunks=8`) →
   manual carry to **−y / left plate `LP`** → release (clean OSC-friendly side).
2. Retreat with **`gripper=−1`** (mandatory), bypass RED at carry z ≥ 0.65,
   pre-pose above YELLOW at `YELLOW.top_z + 0.10`.
3. **`pi0_doubled "put the yellow and white mug on the right plate"
   `max_chunks=40` for the YELLOW→RP leg** — Pi0 self-completes pick+carry+place
   in 10–15 chunks, threading the OSC y-wall that manual aim-past cannot.

**Key shift from the prior recipe:** *Manual* aim-past + descend on the +y
plate is now a **fallback**, not the primary. Reason: the OSC y-wall position varies by scene;
one observed run stalled 7.5 cm higher than expected (eef z=0.602 vs expected
z=0.527). Releasing from z=0.602 tipped the mug and the closed-gripper push-fix
then toppled it (unrecoverable). `pi0_doubled` bypasses the wall entirely.

## Magic numbers

| Lever | Value | Range |
|---|---|---|
| Pre-pose z above mug (both legs) | `mug.top_z + 0.10` | +0.08 … +0.14 |
| `pi0_pick max_chunks` (WHITE leg) | 8 | 6–15 (bump to 15 before escalating prompt) |
| `pi0_doubled max_chunks` (YELLOW leg) | 40 | 35–50 |
| Pi0 prompt — WHITE leg | `"pick up the white mug"` | short canonical |
| Pi0 prompt — YELLOW leg | `"put the yellow and white mug on the right plate"` | **full task language, DEFAULT** |
| `lift_thresh / gripper_closed_thresh` | `0.05 / 0.06` | standard rung |
| Firm-grip after WHITE pick | `set_gripper +1, steps=10` | 8–14 |
| Carry altitude over LP (manual leg) | 0.58 | 0.55–0.62 |
| Bypass altitude over RED (step 9) | 0.65 | 0.65–0.70 (raise to 0.70 if `RED.top_z > 0.54`) |
| Bypass waypoint xy | example `(−0.10, −0.10)`; general: `midpoint(LP, YELLOW) + (+0.10, +0.05)` | scene-relative |
| Place descent on LP | `LP_top_z + 0.04` ≈ 0.49 | OSC stalls ~`LP_top_z + 0.07` — don't fight it |
| Step-8 retreat gripper | **`−1` (open)** | mandatory before YELLOW pre-pose |
| Push-fix (Fallback C only) | `(RP.x − 0.10, RP.y, 0.50)` → close → `(RP.x + 0.04, RP.y, 0.50)` | x-shift 0.08–0.14 m, `step_clip=0.012` |
| Legacy manual aim-past (Appendix A) | `RP.y + 0.07` (was +0.05 in old recipe) | only used if `pi0_doubled` fails |

Observed-run telemetry: the WHITE `pi0_pick` returned
`peak_lift_m=0.053, chunks_used=4`. The YELLOW `pi0_doubled` motion trace
showed 0.0995 m EEF ascent over 11 chunks, minimum gripper opening 0.0103,
and final eef `(0.001, 0.330, 0.529)`; these motion values are historical
trace measurements, not fields returned by the current `pi0_doubled` API.
Another WHITE pick returned `peak_lift_m=0.113, chunks_used=5`.

## Failure modes to avoid (from observed attempt logs)

1. **Manual aim-past on +y plate (PRIOR PRIMARY) → OSC y-wall stalls 7+ cm
   short, release tips mug.** The failed run stalled at z=0.602 vs recipe expected
   z=0.527. Releasing high dropped the mug onto the plate edge upright but
   not-on-centroid. **Fix:** make `pi0_doubled` the default for the YELLOW leg.
2. **Closed-gripper +x push to recover an upright off-center mug → TIPS THE
   MUG (unrecoverable).** The failed run used Fallback B; the push tool hit yellow
   above CoG and toppled it onto its side. **Fix:** only push after release has
   landed the mug nearly-centered; `pi0_doubled` avoids this state in the first
   place.
3. **OSC re-grasp of an upright mug to "recover" off-center placement.**
   Descending + closing on a mug rim makes jaws enter the cavity, close on air;
   the lift tips the mug. **TIPPED MUG = TASK OVER.** Only use the push
   primitive on a known-upright mug.
4. **Carrying with `gripper=+1` into the second pre-pose (skipping the
   open-jaw retreat step 8).** Pi0 expects to manage jaws itself on the second
   leg. **Fix:** always retreat with `gripper=−1` after the first release.
5. **Pi0 short prompt fails to ground the WHITE grasp** (observed failure).
   `peak_lift_m < 0.04` or `min_gripper_opening > 0.05` = miss. **Fix ladder:** re-pose at
   fresh `WHITE.top_z + 0.10` → `max_chunks=15` → `pi0_doubled "put the white
   mug on the left plate" max_chunks=40` → manual rim-grasp (`set_gripper +1,
   steps=20` at `top_z − 0.02`).
6. **Convention reversal** (observed failure). Some runs assumed
   image-left = +y. With swapped targets, both mugs landed on the WRONG plate
   even though Pi0 grasps + carries succeeded. **Fix:** verify `LP.y < 0 < RP.y`
   from SAM3 BEFORE issuing the first pick.
7. **Picking yellow first.** The yellow place is the predicate-firing event
   and the harder leg (Pi0 grasp offset + OSC y-wall). Failing it after white
   wastes the easy pick. **Fix:** WHITE first, always.

## How to re-localize per-scene

Run SAM3 on `agentview` hi-res `world_hi`:

| Entity | Primary prompt | Fallback | Score ≥ |
|---|---|---|---|
| `WHITE` | `"the plain white porcelain mug"` | `"the white mug"`, `"the matte white coffee mug"` | 0.40 |
| `YELLOW` | `"the yellow and white mug"` | `"the yellow mug with white band"`, `"the yellow coffee mug"` | 0.40 |
| `RED` (decoy) | `"the red coffee mug"` | `"the red mug with white text"` | 0.40 |
| `LP` (−y plate) | `"the plate on the left"` | `"the plate closer to the front-left"` | 0.40 |
| `RP` (+y plate) | `"the plate on the right"` | `"the plate closer to the front-right"` | 0.40 |

**Sanity gates (re-segment if any fails):** `WHITE.y < 0`, `RED.y > 0.05`,
`LP.y < −0.20`, `RP.y > +0.20`; mug `top_z` ∈ [0.51, 0.54]; plate `top_z` ∈
[0.44, 0.45]. Disambiguate mugs by **color** (see [[feedback_read_image_before_decide]]), never by xy. Confirm `LP.y < 0 < RP.y` before issuing
the first pick (convention check).

Derive all xyz in the recipe from these entities — **NEVER copy example literal
coordinates**. Reference-run coordinates:
`WHITE=(-0.061,-0.150,0.518)`, `YELLOW=(-0.192,0.026,0.516)`,
`RED=(-0.046,0.119,0.536)`, `LP=(-0.029,-0.298,0.448)`, `RP=(0.002,0.279,0.447)`.

## Difficulty and reliability

The all-manual variant is unreliable because the harder-side placement is
sensitive to the OSC y-wall, coordinate convention, and grasp-offset
compensation. Use `pi0_doubled` as the primary method for the YELLOW leg; keep
manual aim-past only as a fallback.

This remains a medium-hard pattern. Follow the mixed-control assignment from
the start instead of switching to it only after a manual placement fails.

## Related

- [[ordered_dual_can_basket_placement]] / [[occluded_thin_box_basket_placement]] — same "short canonical
  `pick up the X`" pattern.
- [[moka_placement_with_deferred_stove_activation]] — single-objective Pi0 calls vs. full task language.
- [[feedback_move_pose_covarying_reach]] — root cause of the +y plate y-wall.
- [[feedback_read_image_before_decide]] — color disambiguation of WHITE
  / YELLOW / RED mugs.
- [[feedback_no_pi0_end_to_end]] — Pi0 prompt-rung ladder (escalation tree).
