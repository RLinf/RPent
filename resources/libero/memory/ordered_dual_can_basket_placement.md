---
name: ordered-dual-can-basket-placement
description: Reusable shared-basket recipe for two rigid cans with obstacle-aware carry and object-specific delivery behavior.
aliases:
  - put both the alphabet soup and the tomato sauce in the basket
  - alphabet soup tomato sauce woven basket
metadata:
  node_type: memory
  type: skill
---

# Ordered dual-can basket placement

## Applicable pattern

Use this pattern when two rigid cylindrical objects share one basket target but need different carry strategies because of visibility, obstacles, or delivery reliability.

## Reference task alias

`put both the alphabet soup and the tomato sauce in the basket`

## Winning technique
Both cans are wide, rigid, Pi0-graspable. **Tomato_sauce uses scripted carry+place** (`pi0_pick` to grasp → `set_gripper +1 steps=8` to firm the grip → intermediate waypoint at `(TOM_XY.x, TOM_XY.y+0.12, 0.70)` to dodge the ketchup bottle → `(CAV_XY, 0.70)` → descend to `(CAV_XY, 0.58)` → `release`). **Alphabet_soup uses Pi0 self-delivery**: from a pre-pose 14 cm above the soup can, `pi0_pick "pick up the alphabet soup" max_chunks=20` self-completes pick+carry+place and fires `libero_terminated`. Order is fixed: tomato first (visible/scripted), soup second (occluded, Pi0 self-delivers, predicate fires here). Critically — **do NOT** try Pi0 self-delivery on `tomato_sauce`: one observed failure lifted to peak 0.29 and dropped the can 14 cm short of the basket.

## Magic numbers (firm)

- Pre-pose z above either can: **`TOP_Z + 0.14`** (range +0.12 to +0.16), with `step_clip=0.025`.
- `pi0_pick` (both): `max_chunks=20, lift_thresh=0.05, gripper_closed_thresh=0.06`. Backup `max_chunks=28` for soup self-delivery on difficult layouts.
- `set_gripper gripper:+1, steps=8` after the tomato pi0_pick — **NEVER omit `steps=8`**; default-step closure crushes past the can width (observed qpos 0.035 → 0.0005, then the can dropped).
- Carry height: **z=0.70 ± 0.02** (must clear ketchup top ≈ 0.585 and milk carton).
- Drop z: **0.58 ± 0.01** (just past OSC vertical wall ≈ 0.589; can free-falls ~13 cm to cavity floor ≈ 0.45).
- Retreat z: **0.80** with `gripper=-1` so Pi0 controls the jaws on the next pick.
- Intermediate waypoint for tomato carry: **`(TOM_XY.x, TOM_XY.y + 0.12, 0.70)`** — stay at the start-x, lift first, then traverse in y. This is the ketchup-dodge.
- Transit toward soup pre-pose: lift to z=0.80 first, then descend to pre-pose z.
- `step_clip=0.025` on long traversals, `0.020` on the basket descent.
- Total commands when it works: ~13.

## Failure modes to avoid (from prior failure logs)

1. **Pi0 self-delivery for `tomato_sauce`** (the previous recipe's expectation) — **FAILED in an observed run**. `peak_lift_m` reached 0.29 and Pi0 released mid-carry at world y≈0.117, 14 cm short of the basket. *Fix:* always script the tomato_sauce carry; reserve Pi0 self-delivery for the alphabet_soup (second pick).
2. **`set_gripper gripper:+1` with default `steps`** after a pi0_pick on a can → crushes past the can width (qpos collapses, can drops). *Fix:* `steps=8` is the only safe value; or omit and rely on `gripper:+1` in the next move_to (which holds at qpos ≈ 0.06).
3. **Straight diagonal carry from tomato pre-pose to basket** → brushes the ketchup bottle (tall, central; top ≈ 0.585). *Fix:* lift in-x first (intermediate at `TOM_XY.x, TOM_XY.y+0.12, 0.70`), then traverse in y.
4. **Treating the `pi0_pick` result field `success` as placement truth** can misclassify the outcome. In the current API, that field reports the primitive's descend-close-ascend heuristic or official termination; it does not certify semantic placement by itself. *Fix:* confirm placement from current images and `libero_terminated`, not the primitive flag alone.
5. **Picking `alphabet_soup` first** → it's the harder grasp (occluded) AND the predicate-firing pick. Failing it before the easy tomato wastes the cleanest pi0_pick. *Fix:* tomato first, soup last.
6. **Shortened Pi0 prompts** (`"pick up tomato sauce"` no article, `"grasp the red can"`) → loses the place-in-basket training, may grasp ketchup/OJ. *Fix:* canonical noun phrase `"pick up the tomato sauce can"` / `"pick up the alphabet soup"` verbatim.
7. **Holding `gripper:+1` across the gap between picks** → breaks Pi0's grasp on the second can. *Fix:* `gripper:-1` on every approach move_to after the release.
8. **Wrist-rotated descent into the basket during recovery** hit an OSC singularity around step 71. *Fix:* never rotate the wrist for the canonical path; z=0.58 release is reachable from default upright EEF.
9. **Pi0_pick on `alphabet_soup` exhausts max_chunks** → bump to `max_chunks=28` and call `pi0_pick` again from the current mid-trajectory state (no new pre-pose). If still failing, fall through to scripted soup-carry mirroring the tomato branch (no `set_gripper +1` — soup is similarly tapered; hold with `gripper:+1` in move_to).

## Scene re-localization (do this first)

| Entity | Primary SAM3 prompt (score floor) | Fallbacks | What to extract |
|---|---|---|---|
| **`tomato_sauce`** | `"the short tomato sauce can with red and green label"` on agentview hi-res (≥ 0.40) | `"red and green striped tomato sauce can"`, `"short red soup can"` | `TOM_XY = centroid`; `TOM_TOP_Z` = height-map top (≈ 0.50 table-resting). Confirm against appearance gallery — SHORT striped can, NOT ketchup. |
| **`alphabet_soup`** | `"a soup can with blue label"` on agentview hi-res (≥ **0.30** — lower because it's often partially hidden) | `"blue label soup can"`, `"can with blue Campbell-style label"`, then wrist-cam top-down sweep from `(-0.10,-0.05,0.95)` and call `segment` with `camera="wrist"`; last resort `SOUP_XY ≈ MILK_XY + (-0.05, 0)`, `SOUP_TOP_Z ≈ MILK_TOP_Z + 0.05` | `SOUP_XY`, `SOUP_TOP_Z` |
| **`basket` cavity** | Prefer **`world_hi` z-filter `z∈[0.43,0.47]`** → `CAV_XY = bbox midpoint`. SAM3 `"the interior of the woven basket"` is acceptable fallback. | `"woven basket"` | `CAV_XY`; `CAV_FLOOR_Z ≈ 0.45` informational |
| **`ketchup`** (obstacle) | `"red ketchup bottle"` (≥ 0.30) | — | `KETCH_XY` — only used to confirm the intermediate waypoint clears it |
| **`milk_carton`** (occluder) | `"milk carton"` (≥ 0.30) | — | `MILK_XY`, `MILK_TOP_Z` — last-resort soup proxy |

**Color disambiguation:** three reddish silhouettes (tomato_sauce, ketchup, orange_juice). Always confirm `TOM_XY` matches the SHORT red+green-striped can — see [[feedback_read_image_before_decide]].

## Difficulty and reliability

**Difficulty: medium.** The first successful exploration completed in one attempt, while the prior "Pi0 self-completes both cans" recipe required 74 commands and still failed (both cans remained on the table). Perception is straightforward, but Pi0's self-delivery is unreliable for `tomato_sauce`. Treat reliability as medium and do not chase recovery on a displaced tomato; prevention is more reliable than trying to recover that state.

## Decisive insight

For this ordered dual-can pattern, Pi0 reliably carries-and-places ONLY the **second** can (`alphabet_soup`) — that's where `libero_terminated` fires. For the **first** can (`tomato_sauce`), Pi0 will grasp but its observed release behavior ranged from a successful 0.077 m lift to a 0.29 m lift followed by a drop 14 cm short. **Script the easy can; let Pi0 own the predicate-firing can.** This is a refined version of [[feedback_pi0_delivery_service]] — Pi0 is a delivery service for ONE object per task, not the whole multi-pick chain.

## Related

- [[feedback_pi0_delivery_service]] — Pi0 self-completes pick+carry+place; here only valid for the second can
- [[feedback_pi0_pick_full_prompt]] — canonical noun phrase beats shortened variants
- [[feedback_pi0_false_positive_lift]] — `pi0_pick.success` is provisional; combine `peak_lift_m`, gripper evidence, current images, and `libero_terminated`
- [[feedback_read_image_before_decide]] — 1024p side-channel; wrist-scan trick for occlusion recovery
- [[feedback_read_image_before_decide]] — red-silhouette disambiguation (tomato_sauce / ketchup / orange_juice)
- [[feedback_move_pose_covarying_reach]] — OSC wall ≈ z=0.589; release at z=0.58 just past it
