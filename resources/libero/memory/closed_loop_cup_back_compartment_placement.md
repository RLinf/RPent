---
name: closed-loop-cup-back-compartment-placement
description: Reusable closed-loop recipe for carrying a cup into a rear caddy cavity when manual placement is rim-limited.
aliases:
  - pick up the cup and place it in the back compartment of the caddy
  - white yellow cup back caddy compartment
metadata:
  node_type: memory
  type: skill
---

# Closed-loop cup back-compartment placement

## Applicable pattern

Use this pattern when a cup must enter a narrow rear compartment and an uninterrupted closed-loop contact trajectory is more reliable than manual rim descent.

## Reference task alias

`pick up the cup and place it in the back compartment of the caddy`

Reference setup: a white-yellow mug and a brown multi-cavity desk caddy.

## Winning technique
A **single `pi0_pick` call** with the **verbatim task_language** as prompt, `lift_thresh=0.5`, `gripper_closed_thresh=0.0`, `max_chunks=50`. This disables both pi0_pick early-exits (lift heuristic + gripper-closed heuristic), turning `pi0_pick` into an unbroken closed-loop contact driver that runs Pi0's full trained pick-carry-place trajectory until predicate fires. Pi0 owns localization, approach, grasp, carry path, cavity selection, and release — no manual primitives.

## Magic numbers
- `lift_thresh = 0.5` (only requirement: > realistic workspace lift ~0.35)
- `gripper_closed_thresh = 0.0` (strict — never raise)
- `max_chunks = 50` (wins seen at 32 and 36; fallback bump to 80 if rim-drop)
- No carry-z, no tilt, no step_clip, no set_gripper — those primitives are NOT used.

## Failure modes (mined from failed attempts) and the fix
1. **Short-prompt `pi0_pick "pick up the white yellow mug"` + manual `move_to` carry + `release`** → descent walls at z≈1.13 against cavity rim/divider; mug dropped sideways into wrong compartment. **Fix:** use full task_language so Pi0 keeps the placement skill; never hand-guide the descent.
2. **`pi0_doubled` with full prompt (60+50 chunks)** → mug parked on TOP of caddy rims (z≈1.04–1.11), never enters any cavity; second pass is a no-op. **Fix:** never chunk-segment; use one continuous `pi0_pick` with the high `lift_thresh` defang.
3. **Truncating the prompt to just "pick up the mug"** → Pi0 stops after grasp. **Fix:** verbatim full task_language including "back compartment of the caddy".

## per-scene re-localization
The recipe is coordinate-free, so re-localization is only a sanity check before firing — Pi0 does its own perception. Run these once on the world_hi (1024px) frame:
- **The cup:** SAM3 prompt `"white yellow mug"` (alt: `"ceramic mug with yellow interior"`). Expect exactly 1 hit, score > 0.4. Abort if 0 or > 1.
- **The caddy:** SAM3 prompt `"wooden desk organizer with compartments"` (alt: `"brown caddy"`). Expect 1 hit in the back half of the workspace.
- **Distractor (do not act):** `"flat black book"` — should match `black_book_1`, lying near the mug. Confirm it's flat (not a cup-shape mis-pick risk).
Cavity-level pixel geometry is NOT needed; Pi0 picks the back-row cavity itself from the prompt.

## Difficulty and reliability
Solved on **attempt 3** (after `pi0_pick`+manual-carry failed at the descent wall, and `pi0_doubled` placed on-top-of-rims). Difficulty signal: **moderate** — the technique is one line once you know it, but the two natural-looking approaches (manual carry, pi0_doubled) both fail, so a naive run will likely take 2–3 attempts to converge.

## Related memory

[[feedback_move_pose_covarying_reach]] [[feedback_pi0_delivery_service]]
