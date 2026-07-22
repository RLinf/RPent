---
name: mixed-can-box-basket-recovery
description: Reusable mixed-shape basket recipe with can-delivery recovery and tilt-descend placement for a flat box.
aliases:
  - put both the alphabet soup and the cream cheese box in the basket
  - alphabet soup cream cheese silver basket
metadata:
  node_type: memory
  type: skill
---

# Mixed can-box basket recovery

## Applicable pattern

Use this pattern when a cylindrical can and a flat box share a basket target but require different placement and rim-recovery strategies.

## Reference task alias

`put both the alphabet soup and the cream cheese box in the basket`

Reference setup: two pick-and-place into the silver woven basket on the +y / image-right side
of the table. Living-room scene with 5 movables (per `object_names`):
`alphabet_soup_1, basket_1, cream_cheese_1, ketchup_1, tomato_sauce_1`.
Distractor field: `tomato_sauce_1` (red/green can in back row), `ketchup_1`
(orange/yellow squeeze bottle, front-middle). Only the two named targets must
end up inside the basket; distractors that also fall inside are HARMLESS.

## Winning technique

**SOUP first** via `pi0_doubled` with the **FULL task language** (max_chunks=25)
— Pi0's training has the pick-and-place chain baked in, so it
grasps the alphabet-soup can. When (as on observed run) Pi0 burns its chunk budget
mid-flight and drops the can at z ≈ 0.50, recover with `set_gripper +1, steps=8`
(qpos → 0.022 body grip) + scripted lift to z=0.70 + transit to (BSK_X, BSK_Y,
0.70) + release. **THEN CHEESE**: low pre-pos at z=0.65 over `CHEESE_xy` →
`pi0_pick "pick up the cream cheese" max_chunks=20 gripper_closed_thresh=0.06`
→ carry-high at z=0.75 → **preventive tilt-descend with `move_pose
target_pitch=0.5` to z=0.60** (breaks OSC vertical wall at z≈0.589, lands box
inside cavity). **Fallback if the box still rim-perches:** `set_gripper +1,
steps=5` (closed = solid plunger) → top-down push to z=0.56 at
**step_clip=0.012** — this is what fired `libero_terminated` on observed run.

## Magic numbers (firm)

- `pi0_doubled` chunk cap: **`max_chunks ≤ 25`** per call (see `[[feedback_pi0_chunks_egl_crash]]`).
- Mid-air drop recovery: **`set_gripper +1, steps=8`** → qpos ≈ 0.022 (body grip on can).
- Soup transit lift: **`z = 0.70`** (above all table objects, above basket rim).
- Cheese low pre-pos: **`(CHEESE_xy, z = 0.65)`**, gripper open. Range 0.60-0.68.
- `pi0_pick` (cheese, thin box): `max_chunks=20, lift_thresh=0.05, gripper_closed_thresh=0.06`.
- Cheese carry-high transit: **`z = 0.75`**.
- Preventive tilt-descend: **`move_pose target_pitch=0.5, z=0.60, step_clip=0.012`** (range pitch 0.4-0.6, z 0.58-0.62).
- Closed-gripper push recovery: **`set_gripper +1, steps=5`** then `move_to (BSK_X, BSK_Y, 0.65) gripper=1` then `move_to (BSK_X, BSK_Y, 0.56) gripper=1 step_clip=0.012` (range z 0.55-0.58, step_clip 0.010-0.015).
- OSC vertical wall: **z ≈ 0.589** — never scripted-descend vertically; always tilt.
- Basket cavity floor: `z ∈ [0.43, 0.46]`. Basket rim: `BSK_TOP_Z` ≈ SAM3 centroid z (observed run: 0.528).

## Failure modes to avoid (and the fix)

1. **`pi0_doubled` chunk-budget exhaustion mid-air** → drops the held can at
   z ≈ 0.50 (seen on observed run). *Fix:* `set_gripper +1, steps=8` to re-grasp + scripted
   transit to (BSK_X, BSK_Y, 0.70) + release. Do NOT extend `max_chunks > 25` —
   the policy is just deciding "done".
2. **Cream-cheese flat box rim-perches** at z ≈ BSK_TOP_Z on release (observed run failure).
   *Fix:* **preventive tilt-descend** with `move_pose target_pitch=0.5` to z=0.60
   (thin-box placement lever). If still perched, recover with **closed-gripper top-down push**
   to z=0.56 at step_clip=0.012.
3. **Re-issuing `pi0_doubled "put the cream cheese in the basket"` as recovery**
   → on observed run this went rogue and grabbed the **ketchup** distractor. *Fix:* never
   use wide pi0 recovery while a distractor is reachable; use scripted closed-
   gripper push instead.
4. **Re-`pi0_pick`-ing a placed box to "nudge" it** (negative lesson from
   a related failed attempt) → tips the box, unrecoverable. *Fix:*
   only ever use the closed-gripper top-down push to re-seat a rim-perched box.
5. **`set_gripper +1` WHILE carrying the thin cheese box** (thin-box negative lesson)
   → squirts the box out of the fingers. *Fix:* leave Pi0's natural 0.02-0.04
   qpos during carry; only use `set_gripper +1` (a) to recover a dropped can
   or (b) as the closed-gripper push tool.
6. **SAM3 basket centroid used as drop xy** → biased to the silver rim (observed run
   had ~5 cm x-bias which is exactly what caused the cheese rim-perch). *Fix:*
   derive `(BSK_X, BSK_Y)` from `world_hi` z-filter on `z ∈ [0.43, 0.46]`
   inside the SAM3 basket bbox (thin-box lever, transfer-critical).
7. **SAM3 cannot ground the soup can by brand** — on observed run the prompt `"can with
   blue label in back"` grounded onto `tomato_sauce` (red/green-label can in
   center-back). *Fix:* world_hi pixel scan for "blue + yellow striped" among
   on-table cans; or trust Pi0_doubled with full task language to pick the
   right can (Pi0's libero_10 training disambiguates).
8. **Scripted vertical descent into basket** → walled by OSC at z ≈ 0.589.
   *Fix:* always use `move_pose target_pitch=0.5` to descend below z=0.60.
9. **Pi0_doubled leaves the wrist quat tilted** after Stage A → next
   `move_pose target_pitch=0.5` may refuse. *Fix:* restore quat with
   `move_pose target_pitch=0 target_yaw=0` to a high waypoint before B5
   (lesson from `[[dual_moka_obstacle_aware_stove_placement]]`).

## per-scene re-localization (do this first, EVERY scene layout)

- **`BASKET` cavity `(BSK_X, BSK_Y)`** — SAM3 agentview hi-res `"the silver
  woven basket on the right"` (fallbacks: `"the silver wicker basket"`,
  `"the metallic foil basket"`, min score 0.40). Then **world_hi z-filter on
  `z ∈ [0.43, 0.46]`** inside the loose basket bbox; midpoint = `(BSK_X, BSK_Y)`.
  Sanity: cavity y-span ≈ 0.13 m. Reference (observed run): `≈ (0.058, 0.229)`,
  `BSK_TOP_Z ≈ 0.528`. **DO NOT use SAM3 centroid as the drop xy** — it's
  biased to the rim and caused the observed run rim-perch.
- **`SOUP` (target #1, blue+yellow can)** — primary prompt `"the can with a
  blue and yellow sunflower label"`; fallbacks `"the can with a blue label
  in the back"`, `"the alphabet soup can"`. Min score 0.30. **MUST be the BLUE
  can** — if hi-res crop is red/green → grounded onto `tomato_sauce`; if
  orange → grounded onto `ketchup`; re-prompt or fall back to world_hi pixel
  scan for "blue+yellow striped" can. Reference (observed run): `≈ (-0.158, -0.141,
  0.491)`, back-left. SAM3 brand-grounding is unreliable here.
- **`CHEESE` (target #2, light-blue box)** — primary prompt `"the light blue
  box on the table"`; fallbacks `"the small light-blue carton"`, `"the
  rectangular box on the front-left of the table"`. Min score 0.25 (often
  low — verify against world_hi). Reference (observed run): `≈ (0.097, -0.213, 0.456)`.
  If occluded, wrist sweep from `(0.0, -0.20, 0.72)` and re-SAM3 wrist hi-res.
- **`TOMATO_SAUCE` distractor** — SAM3 `"the can with a red and green label
  in the back"` (min 0.35); cache as a GATE: if `SOUP_xy ≈ TOM_xy` within
  ~5 cm → SOUP grounded wrong, re-segment.
- **`KETCHUP` distractor** — SAM3 `"the orange ketchup bottle"` (min 0.35);
  cache to confirm post-pick that Pi0 didn't drift to ketchup.

## Difficulty and reliability

**1 attempt — explored solved on the first try** (no separate attempts
directory was created). However the winning episode required **25 commands**
including three mid-stream recoveries: (i) Pi0_doubled mid-air drop of soup →
`set_gripper +1` re-grasp + scripted transit; (ii) cheese rim-perch →
closed-gripper top-down push to z=0.56 at step_clip=0.012; (iii) implicit:
avoiding a wide pi0 recovery that grabbed the ketchup distractor (the
explorer did fall into this on observed run and worked around it by releasing the
ketchup and switching to scripted push). Treat as **medium difficulty in new
scenes**: the recipe converges first-try **only if** (a) `(BSK_X, BSK_Y)` is
the z-filter cavity midpoint, NOT the SAM3 rim centroid; (b) cheese descent
uses tilt `move_pose target_pitch=0.5` to z ≈ 0.60 (preventive); (c) any rim-perch
recovery uses the scripted closed-gripper push, NEVER a re-issued
`pi0_doubled`.

## Cross-references

- `[[occluded_thin_box_basket_placement]]` — sister "cream cheese + butter in basket"; origin
  of the `world_hi` cavity z-filter, the "no `set_gripper +1` while carrying
  thin box" rule, and the tilt `move_pose target_pitch=0.5` descent.
- `[[relative_side_first_anchor_placement]]` — related failed attempt: never re-pi0_pick
  a placed object to nudge it; tips & breaks unrecoverable.
- `[[dual_moka_obstacle_aware_stove_placement]]` — Pi0_doubled can leave wrist quat tilted; restore
  with `move_pose target_pitch=0 target_yaw=0`.
- `[[feedback_move_pose_covarying_reach]]` — z ≈ 0.589 vertical singularity; canonical
  bypass `move_pose target_pitch=0.5`.
- `[[feedback_pi0_chunks_egl_crash]]` — `max_chunks ≤ 25` per `pi0_*` call.
- `[[feedback_pi0_delivery_service]]` / `[[feedback_pi0_pick_full_prompt]]` —
  full-task-language rule that triggers Pi0's pick-and-place chain.
  basket sits on +y.
