---
name: upright-book-back-compartment-placement
description: Reusable upright-book recipe that uses held-object overhang to place into a rear caddy compartment despite an OSC descent wall.
aliases:
  - pick up the book and place it in the back compartment of the caddy
  - upright black book back caddy compartment
metadata:
  node_type: memory
  type: skill
---

# Upright book back-compartment placement

## Applicable pattern

Use this pattern when a tall thin object hangs well below the gripper and can enter a rear compartment even when the end effector stalls above the cavity.

## Reference task alias

`pick up the book and place it in the back compartment of the caddy`

## Winning technique

The black book stands upright in front
of a wooden 4-compartment desk caddy. Pi0 picks it reliably with the short
prompt `"pick up the black book"` (4–7 chunks). Carry at `z ≈ WALL_TOP_Z +
0.20` (~1.23) over the **BACK-MIDDLE** compartment center, descend straight
down — the OSC gripper-down wall stalls the eef at `z ≈ 1.15`, but because
the held book hangs ~15 cm below the grip, its bottom is already **inside**
the compartment. Release at the walled pose (book drops the remaining ~10 cm
on gravity), then retreat upward with `gripper=−1`; predicate fires on
release or retreat.

## Magic numbers

| Knob | Value | Range | Notes |
|---|---|---|---|
| Pi0 pick prompt | `"pick up the black book"` (short) | short → full lang on miss | Worked every round (r1–r6) |
| `pi0_pick max_chunks` | 20 | 15–30 | Solved in 4–7 chunks across rounds |
| Pre-pose z | `BOOK.top_z + 0.10` | +0.08 … +0.13 | Lower → Pi0 thinks already-grasped |
| `set_gripper` after pick | `+1, steps=10` | 8–12 | **Mandatory** — firms spine grip |
| Carry-leg `gripper` field | `+1` on every carry `move_to` | `+1` only | **Mandatory** — `-1` or omitted opens jaws |
| Carry altitude | `WALL_TOP_Z + 0.18..0.22` (≈ 1.20–1.27) | `WALL_TOP_Z + 0.15 … +0.25` | Clears divider tops + hanging-book bottom |
| Descent target z | `FLOOR_Z + 0.13` (≈ 1.05) | `FLOOR_Z + 0.10 … +0.17` | OSC will wall ~7–10 cm short — by design |
| Descent `step_clip` | 0.015 | 0.012–0.020 | Avoid jamming the upright book |
| OSC wall stall z | ≈ 1.15 (observed) | not tunable | Recognize and release |
| `release` steps | 10 | 8–12 | |
| Retreat altitude | `WALL_TOP_Z + 0.20` (≈ 1.25), `gripper=−1` | `WALL_TOP_Z + 0.18 … +0.25` | Open jaws **before** retreat |
| Single-leg `|Δxy|` | `< 0.30` | hard OSC tracking limit | Use lift waypoint to clip |
| Tilt / pitch / `move_pose` | **none** | **none** | Would jam the upright book against a wall |

## Failure modes (mined from r1–r6 inter-round refinement) + the fix

1. **Carry-z too low** (early plans tried `WALL_TOP_Z + 0.14`). Held-book
   bottom barely above rim; brush risk on lateral motion. **Fix:** carry at
   `WALL_TOP_Z + 0.18..0.22` (≥ 0.18 m clearance over divider tops).
2. **Missing `set_gripper +1` after `pi0_pick`**, or `gripper` field omitted
   on a carry `move_to`. Jaws relax → book slips mid-carry (silent failure —
   telemetry healthy, book vanished). **Fix:** always `set_gripper +1
   steps=10` directly after the pick, and pass `gripper=+1` on every carry
   `move_to`.
3. **Targeting the wrong compartment.** Naïve SAM3 prompts can ground the
   wide LEFT column ("back-left"). The named "back compartment" is the back
   half of the **MIDDLE** column (split by an internal x-divider). **Fix:**
   derive `MB.x, MB.y` from a top-down topology sweep of `world_hi` — look
   for **three** x-ridges (back wall + internal divider + front edge) in the
   same y-band; that y-band IS the middle column.
4. **Fighting the OSC vertical wall** (r5 attempted re-descend at smaller
   `step_clip`). Zero progress, wasted chunks. **Fix:** release at the
   walled pose — held book hangs ~15 cm below grip, so its bottom is inside
   the compartment already. Trust the geometry.
5. **Predicate not firing on release alone** (r5). Book seated mid-air on
   release, took a moment to settle. **Fix:** add a retreat-upward step with
   `gripper=−1`; predicate fires as physics settles the book onto the floor.
6. **Tilt / `rotate_pitch` to "reach deeper".** Would tip the upright book
   against a compartment wall. **Fix:** straight vertical descent only. The
   [[feedback_move_pose_covarying_reach]] memo applies as a *physics limit*
   here, but tilt is NOT the workaround for this task — the hanging book is.

## per-scene re-localization

**SAM3 prompts:**

| Entity | Primary | Fallback | Score floor | Output |
|---|---|---|---|---|
| `BOOK` | `"the black book on the table"` | `"the upright black book"`, `"the dark book standing on the desk"` | 0.40 | `BOOK.xyz`, `BOOK.top_z` |
| `CADDY` | `"the brown wooden desk caddy organizer"` | `"the dark brown wooden box"`, `"the wooden organizer with compartments"` | 0.40 | `CADDY.xyz`, world xy bbox |

> **Brand-noun warning:** `"caddy"` alone scores ~0.03 (r2 saw this). The
> fallback `"the dark brown wooden box"` reliably scores ≥ 0.90.

**Topology sweep (after CADDY bbox is known):**

```
mask_top   = (world_hi[..., 2] >= 0.97) & inside CADDY_bbox
mask_floor = (world_hi[..., 2] >= 0.88) & (world_hi[..., 2] < 0.94) & inside CADDY_bbox

BACK_WALL.x        = min world_x in mask_top
FRONT_WALL.x       = max world_x in mask_top
INTERNAL_DIVIDER.x = middle x-ridge in mask_top (y-band with 3 x-ridges → middle column)
Y_WALL_LEFT, Y_WALL_RIGHT = min/max world_y of mask_top in middle column

WALL_TOP_Z = median z of mask_top    (≈ 1.05)
FLOOR_Z    = median z of mask_floor  (≈ 0.92)

MB.x = (BACK_WALL.x + INTERNAL_DIVIDER.x) / 2
MB.y = (Y_WALL_LEFT  + Y_WALL_RIGHT)      / 2
```

**Sanity gates:**

- `BACK_WALL.x < INTERNAL_DIVIDER.x < FRONT_WALL.x`, all in `[−0.50, −0.30]`.
- Middle-column width `Y_WALL_RIGHT − Y_WALL_LEFT ∈ [0.11, 0.16]`.
- BACK-MIDDLE depth `INTERNAL_DIVIDER.x − BACK_WALL.x ∈ [0.05, 0.08]`.
- `WALL_TOP_Z − FLOOR_Z ∈ [0.11, 0.16]`.
- `|MB.y − CADDY.y| < 0.06` (MB straddles caddy y-axis).
- `BOOK.y ∈ [−0.06, +0.04]`, `BOOK.x ∈ [−0.12, +0.00]`.

If the middle-column sweep returns < 3 x-ridges, drop the `mask_top`
threshold to `z >= 0.94` and retry; the smallest-x-range ridge is the
divider.

## Difficulty and reliability

6 rounds, all successful. Final replay = **1 uninterrupted episode, 8 commands**.
Difficulty: **easy-medium**, localization-bound. The manipulation technique
is robust; the only real risk is mis-identifying which compartment is "back".

## See also

  fallbacks
- [[feedback_move_pose_covarying_reach]] — same gripper-down OSC wall biting
  the descent; here it's WAI because the held book extends reach below the
  eef
- [[feedback_read_image_before_decide]] — `world_hi_00.npy` is what the
  topology sweep walks; Pi0 still runs on 256-px channel
- [[feedback_read_image_before_decide]] — disambiguate the caddy from
  other brown objects in perturbed layouts
