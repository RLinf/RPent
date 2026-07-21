---
name: identity-conditioned-dual-mug-tilted-placement
description: Reusable identity-aware two-mug recipe using tilted placement to reach opposing lateral plates around a distractor.
aliases:
  - put the yellow and white mug on the left plate and put the white mug on the right plate
  - yellow white mug left plate porcelain mug right plate
metadata:
  node_type: memory
  type: skill
---

# Identity-conditioned dual-mug tilted placement

## Applicable pattern

Use this pattern when two similar mugs have identity-specific targets on opposing sides and vertical placement stalls at lateral reach.

## Reference task alias

`put the yellow and white mug on the left plate and put the white mug on the right plate`

Reference setup: two target mugs, two plates, and a red-coffee-mug distractor.

## Winning technique

Pi0_pick each mug with a SHORT color-only grasp prompt (so Pi0 only lifts, doesn't complete its trained pick-and-place), lock with `set_gripper +1`, scripted multi-waypoint carry at z=0.72 routed clear of the red distractor and the placed first mug, then release via **`move_pose target_pitch=0.5`** (tilt the gripper forward) — NOT vertical `move_to`. The tilt clears the OSC vertical-reach singularity at the lateral plate distance (|y| ≈ 0.30); a vertical descent there walls at z≈0.52 AND drags the held mug 4–5 cm forward during the stall+release, putting the body off-plate (predicate fails). Tilt gives a clean stop with body on plate (predicate fires). After each tilted release, retreat via another `move_pose target_pitch=0.0` to un-tilt.

Assignment is by **mug identity** (NOT by starting y-half): `white_yellow_mug_1 → LEFT (−y) plate`, `porcelain_mug_1 → RIGHT (+y) plate`. The "white mug" in the language is the porcelain mug.

## Magic numbers

- Pi0 pick: `max_chunks=8` (typically finishes in 4-6 chunks), `lift_thresh=0.05`, `gripper_closed_thresh=0.06`.
- Pick prompts: **`"grasp the yellow mug"`** and **`"grasp the gray mug"`** — NEVER full task language ("white mug" / "porcelain mug" are empirically weaker for Pi0).
- `set_gripper`: `gripper=1`, `steps=10`. Key is `"gripper"`, **NOT `"value"`** (silently ignored — mug drops).
- Yellow pick: from home, no pre-pos required (the successful run skipped pre-pos).
- Porcelain pick: pre-pos `{move_to (GMx, GMy, 0.65)}` directly over body — porcelain prompt is weaker.
- Carry z: **0.72** (range 0.70–0.74), `step_clip=0.020`.
- Approach over plate: z=0.72 with xy near `(LPx, LPy+0.05)` (LEFT) or `(RPx, RPy−0.09)` (RIGHT).
- **TILTED PLACE (the lever): `move_pose target_pitch=0.5, step_clip=0.012, max_steps=120`**, xyz = `(LPx, LPy−0.020, 0.48)` for LEFT, `(RPx, RPy+0.010, 0.48)` for RIGHT. The z=0.48 is the target; eef actually stalls at z≈0.535–0.545 with the tilt — that is correct. The −0.020/+0.010 y-bias compensates for the tilt's forward swing so the mug body lands centered.
- Retreat after release: `move_pose target_pitch=0.0, gripper=-1` to un-tilt before next pick.

## Failure modes and fixes (mined from 11 failed attempts)

| Failure | Cause | Fix that finally worked |
|---|---|---|
| Pi0 dumped yellow past +y workspace edge (att 1) | Full task language + max_chunks=8 → Pi0 completed trained pick-and-place | Short grasp prompt; `set_gripper +1` immediately after Pi0 exits |
| Porcelain slips during carry (att 2) | Scripted gripper close on body at z=0.51 → laterally weak grip | Pi0_pick the porcelain too (`"grasp the gray mug"`) |
| Mug tips during pre-pos / Pi0 descent (att 3, 8) | Pre-pos at `mug_y ± 0.045–0.085` (opposite-handle side) | No pre-pos for yellow; pre-pos directly OVER mug center for gray |
| Asymmetric grip, body 6 cm off held point (att 4, 5) | Scripted handle grasp | Pi0_pick gives firm body-aligned grasp |
| **Visual pass, predicate fail (att 5, 6, 7, 9, 10, 11)** | Released with vertical `move_to` at z=0.50–0.55 — OSC walls at z≈0.52 AND drags mug forward 4–5 cm during stall+release | **Use `move_pose target_pitch=0.5` for release**; tilt clears OSC singularity, no slide, body lands on plate |
| Push to correct off-center mug → tips (att 6, 9) | Closed-gripper push levers mug over | NEVER push after release; rely on tilt + wrist refinement before descent |
| Re-grasp of placed/tipped mug fails (att 3, 6, 11) | Side-lying mug is unrecoverable (no side-grasp; Pi0 won't engage) | If a place fails, stop — don't re-grasp |
| Assignment inversion ALSO predicate-fails (att 7) | Both visual placements pass; predicate keyed on identity-not-position | Keep assignment `white_yellow→LEFT, porcelain→RIGHT` |
| `pi0_doubled` picks unused red mug (att 9B) | Full task language to Pi0 end-to-end is unsafe with 3-object scene | Keep the scripted carry; never use `pi0_doubled` or full task language here |
| `set_gripper` silently ignored (att 8) | Used `"value":1` instead of `"gripper":1` | Always `"gripper":1` |

## per-scene re-localization

Run all on the initial **agentview** (`step=0`); project SAM3 mask centroids via `world_hi`:

| Entity | SAM3 prompt (try in order) | Score floor | Use for |
|---|---|---|---|
| `white_yellow_mug` | `"the yellow mug"` → `"the white mug with yellow handle"` | 0.30 | Phase-1 source (no pre-pos needed) |
| `porcelain_mug` (gray) | `"the small gray mug on the left"` → `"the gray mug"` → `"the porcelain mug"` | 0.80 | Phase-2 pre-pos xy |
| `red_coffee_mug` (DISTRACTOR) | `"the red mug with white patterns"` | 0.60 | **record xy ONLY — to route carry around; NEVER pick** |
| LEFT plate | `"the left plate"` → `"the white plate with red stripe on the left side"` | 0.65 | Phase-1 placement xy |
| RIGHT plate | `"the right plate"` → `"the white plate with red stripe on the right side"` | 0.65 | Phase-2 placement xy |

**Wrist refinement (optional, only when agentview plate score < 0.85):** while carrying mug at z=0.72, segment the plate from wrist; refine plate xy from the wrist mask. Compute held-mug offset `Δ = wrist_mug_centroid_xy − wrist_eef_xy`; subtract from placement xy. Fallback if wrist plate score < 0.80: `LPy -= 0.02` / `RPy += 0.02`.

Wrist refinement helps the *visual* landing but is NOT the lever — the TILT is what makes the predicate fire.

## Carry routing

- **Yellow → LEFT:** `carry_x_L = clamp(YMx-0.01, -0.07, -0.03)`. If `RMx ∈ [-0.07, -0.03]` (red distractor in corridor), shift to `+0.02` and insert intermediate `(+0.02, YMy, 0.72)`.
- **Porcelain → RIGHT:** `carry_x_R = clamp(GMx+0.03, -0.05, +0.02)`. Must never pass back over the placed yellow at LPy. Stay z ≥ 0.70 throughout.
- Pick order is fixed: yellow first, porcelain second (porcelain-first tipped in attempt 3).

## Difficulty and reliability

**Win was attempt 12 of 12** (11 failed first). This is a **HIGH-difficulty** cell — the visual-pass / predicate-fail loop alone consumed 6 attempts (5/6/7/9/10/11) before the tilt lever was found. A single-attempt run must follow this recipe verbatim; improvising on the placement primitive (using `move_to` instead of `move_pose`+pitch) will produce visual-only success.

## Related memory

- [[feedback_move_pose_covarying_reach]] — the OSC stall at z≈0.52 kills vertical placement here; `move_pose` with `target_pitch` is the documented workaround for deep lateral reach.
- [[feedback_pi0_pick_full_prompt]] — when full task language is OK for Pi0 vs when (here) it triggers a destructive end-to-end pick-and-place.
- [[feedback_read_image_before_decide]] — SAM3 segmentation conventions (1024p world_hi).
- [[feedback_scripted_pick_limits]] — why scripted body close on porcelain failed.
