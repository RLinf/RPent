---
name: can-and-small-package-basket-placement
description: Reusable shared-basket recipe for a rigid can and a small package when a single closed-loop instruction outperforms manual placement.
aliases:
  - put both the alphabet soup and the butter in the basket
  - alphabet soup butter wicker basket
metadata:
  node_type: memory
  type: skill
---

# Can and small-package basket placement

## Applicable pattern

Use this pattern when two visually distinct small objects share a basket target and manual descent is blocked by the basket rim or contact geometry.

## Reference task alias

`put both the alphabet soup and the butter in the basket`

## Winning technique
Put BOTH alphabet_soup (blue/white "Dairy Fresh" can) and butter (small red "Farm Fresh Butter" package) into the wicker basket on a cluttered grocery table. **Technique: ONE primitive does the whole task.** After 7 failed hybrid attempts, manual perception plus scripted pick/carry/release repeatedly left items visibly perched on the back rim near z≈0.59. Attempt 8 was a SINGLE `pi0_doubled(FULL task_language verbatim, max_chunks=80)` and Pi0 finished both placements in 32 chunks with only 0.0475 m observed EEF ascent; `libero_terminated` flipped True on its own. **For LIBERO-10 multi-object-into-container tasks, default to `pi0_doubled` with the verbatim task prompt FIRST; manual primitives are a third-tier fallback, not the recipe.**

## Magic numbers (defaults + small ranges)
- `pi0_doubled` prompt: **EXACTLY** `"put both the alphabet soup and the butter in the basket"` (verbatim task_language). Truncated single-object prompts (attempts 2–7) bypass Pi0's trained two-object skill.
- `pi0_doubled max_chunks=80` (validated 32 used; 50–100 OK; <40 sometimes too tight).
- The primary recipe is one `pi0_doubled` call; it needs no manual localization or motion primitives.
- (Fallback only): pre-pos `z = obj_top_z + 0.16`; `set_gripper +1 steps=10`; tilted descent `move_pose target_pitch=0.5`; release `z ∈ [0.52, 0.55]` (never ≥ 0.58 — the observed OSC vertical wall leaves items on the rim); place items at OFFSET xy (e.g. basket front-left then basket front-right, never same xy or items will stack and remain visibly perched).

## Failure modes → fix that finally worked
| Failure (attempt) | Root cause | Fix |
|---|---|---|
| Manual carry + release at z=0.56–0.60 above basket; items looked close but the official task signal remained false (attempts 1, 3, 5, 7, 8) | The OSC vertical wall at z≈0.58–0.60 prevented deep descent; items remained perched on the basket back rim instead of settling on the interior floor. | Stop scripting the placement. `pi0_doubled(full_task_language)` produced a low-arc lift and deep tilted placement that settled both items inside the basket. |
| Object misidentification — milk_carton treated as butter for 5+ attempts (attempts 1–5) | Both at similar agentview xy; milk is at the position the agent kept pre-positioning to. Real butter is the small RED package, NOT the larger milk box. | Verify segmented crop's color & label: butter is small + RED + "Farm Fresh Butter"; reject milk (larger, blue/white). Best fix: skip manual localization entirely — Pi0's grounding is more reliable than ours here. |
| `pi0_pick "pick up the alphabet soup"` grabbed tomato_sauce instead (attempt 6, partial in 7) | SAM3 scores tomato_sauce HIGHER than alphabet_soup on the prompt `"alphabet soup can"` (0.711 vs 0.371) because both are cylindrical cans in close proximity. | Always inspect the crop COLOR: alphabet_soup = BLUE/WHITE "Dairy Fresh"; tomato_sauce = RED with GREEN stripes. Reject by color before trusting score. Or skip manual scripting — Pi0 distinguishes them via the natural-language prompt. |
| Items stacked at z≈0.597 and perched on each other near the basket back (attempts 1, 5, 8 early) | Released both at the same xy near the basket back wall (y ≈ basket_y); the second item landed on the first and neither settled on the usable interior floor. | If using fallback, place items at OFFSET xy: butter front-left (basket_x − 0.02, basket_y − 0.02), soup front-right (basket_x + 0.02, basket_y − 0.02). Aim for front interior (y < basket_y), not back rim. |
| Pi0 with `task_language` as `pi0_pick` prompt stopped after 4 chunks without placing (attempt 2) | `pi0_pick` is a single-pick primitive; it hits `lift_thresh` after grabbing first object and stops. Full-task prompt needs `pi0_doubled`, which is the trained multi-step skill. | Always use `pi0_doubled` (NOT `pi0_pick`) for the full task language. |
| Tilted release arc dropped butter outside basket (attempt 6) | `move_pose target_pitch=0.5` for descent then immediate release while arm still tilted → object arcs forward off basket. | If using tilted descent, hold pose stable for 1–2 steps before `release`. Better: skip the tilt and use `pi0_doubled`. |
| Butter pick from basket RIM (re-grasp after rim-perch) always failed (attempts 7, 8 cleanup) | Butter on thin rim has no graspable underside; Pi0 closes air, qpos→0.001. | Don't get into the rim-perch state to begin with — release deep, not at z=0.59. |

## per-scene re-localization (only needed for fallback)
The primary recipe needs ZERO explicit localization — Pi0 grounds visually. If the primary call does not complete the task, inspect the current images first. Localize only the missing targets when the basket remains upright and usable and those targets remain safely graspable.
- **BUT = butter** — `segment({"prompt":"red butter package","camera":"agentview","min_score":0.6})`. Fallbacks: `"small red box with cow"` → `"Farm Fresh Butter package"`. SANITY: smallest item in the front row, top_z ≈ 0.45 (lower than milk z≈0.52 and cans z≈0.50). Reject any mask centered on milk_carton (blue/white, ~2× the size) — this was the most expensive mis-segmentation in attempts 1–5.
- **AS = alphabet_soup** — `segment({"prompt":"alphabet soup can","camera":"agentview"})`. Score is unreliable — ALWAYS validate the crop is BLUE/WHITE "Dairy Fresh" label, NOT red+green tomato_sauce. Fallbacks: `"can with blue and white label"`, `"Dairy Fresh blue label can"`. If occluded behind milk on agentview, pre-position then call `segment` with `camera="wrist"` (score ~0.8 there). Sanity: top_z ≈ 0.50.
- **BK = basket** — `segment({"prompt":"basket","camera":"agentview","min_score":0.7})`; fallback prompt: `"wicker basket"`. Compute `(bx_center, by_center, b_interior_z)`; `by ∈ [0.15, 0.30]`, `b_interior_z ≈ 0.48–0.50`. AIM FRONT-INTERIOR: target_y = `by_center − 0.02` (the back rim and the observed OSC wall both obstruct a clean descent).
- **Do not cache absolute coordinates** - re-localize the objects and basket in the current scene.

## Difficulty and reliability
**8 attempts to a win.** Difficulty signal for MANUAL scripting: **VERY HIGH** (the basket rim and the observed OSC vertical wall can leave items looking close without settling inside). Difficulty for Pi0's trained skill: **LOW** (one call, 32 chunks, first try succeeded). If the first `pi0_doubled` call does not complete the task, inspect the current images before deciding. When the basket is still upright and usable and each missing target remains safely graspable, retry the same full-task call once from the current state. If a target is still missing afterward but the documented fallback geometry remains safe, re-localize only that target and use the fallback; do not remove extra objects unless they physically block the basket. If the basket has tipped, the usable interior is blocked, or a missing target is no longer safely graspable, no verified recovery is known from that state.

## Cross-refs
- See [[feedback_move_pose_covarying_reach]] — why the manual descent into the basket walls at z≈0.59 (gripper-down OSC wall), and why `move_pose target_pitch=0.5` is the workaround for the fallback.
- See [[feedback_pi0_delivery_service]] — `pi0_doubled(FULL prompt)` is Pi0's trained multi-object skill; prefer it over scripted pick+carry+release for LIBERO-10 placement tasks.
- See [[feedback_pi0_pick_full_prompt]] — `pi0_doubled` reliably acts on FULL task language; truncated single-object prompts (`"pick up the butter"`) don't trigger the placement phase.
- See [[feedback_read_image_before_decide]] — disambiguating butter from milk_carton and alphabet_soup from tomato_sauce by appearance, not SAM3 score.
- See [[fragile_box_and_rigid_can_basket_placement]] — the fragile-box + rigid-can pattern uses a TWO-stage hybrid (pi0_doubled then scripted) because cream-cheese grounding is fragile. **This pattern differs:** Pi0 grounds butter+soup well, so the recipe collapses to ONE pi0_doubled call. Do not carry the sibling pattern's two-stage scaffolding into this one.
