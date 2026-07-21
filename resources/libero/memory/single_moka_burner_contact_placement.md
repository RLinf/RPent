---
name: single-moka-burner-contact-placement
description: Reusable single-vessel stove recipe that requires base-to-burner contact rather than visual overlap alone.
aliases:
  - put the left moka pot on the stove
  - left silver moka pot red burner
metadata:
  node_type: memory
  type: skill
---

# Single-moka burner-contact placement

## Applicable pattern

Use this pattern when one handled vessel must make stable base contact with a burner and visual xy overlap alone is insufficient.

## Reference task alias

`put the left moka pot on the stove`

Reference setup: the left silver moka pot is camera-side (+y) relative to the other pot.

## Winning technique

As the first motion from the untouched initial task state, issue ONE `pi0_doubled` call with the
literal task string `"put the left moka pot on the stove"` and `max_chunks: 25`.
Pi0 was trained on this exact task; its closed-loop contact skill handles
pick → carry → place → base-contact in ~23 chunks. Do not script the placement —
OSC walls 4–5 cm short of the burner at that xy and the predicate needs base
contact, not xy overlap. **Difficulty evidence: 10 attempts to the win** (1
win after 9 scripted-path failures) — HIGH difficulty on scripted path, LOW on
trained-policy path.

## Magic numbers / levers

- `pi0_doubled` `max_chunks`: **25** (safe range 20–30; observed 23 used on observed run).
  `<20` cuts off mid-place; `>30` is just wallclock.
- Prompt: **literal LIBERO task string**. `pick up the left moka pot` covers
  only the grasp; dropping `"left"` → ~50 % wrong-pot pickup.
- No xyz, no `set_gripper`, no `rotate_pitch`, no `move_pose` in the recipe —
  all such scripted variants tested and lost (see failure modes).
- Issue `pi0_doubled` before any other motion primitive; prior `pi0_pick` calls
  leave cumulative wrist tilt that hurts the doubled skill.

## Failure modes mined from 9 failed attempts → fix

1. **OSC vertical wall at burner xy ≈ (−0.05, −0.20).** Straight-down descent
   walls at `eef_z ≈ 1.05–1.07`, pot base 2–5 cm above burner `z = 0.928`.
   Release → pot tips or floats. → **Fix:** `pi0_doubled` threads the wall with
   lateral corrections. See [[feedback_move_pose_covarying_reach]].
2. **Contact predicate ≠ xy overlap.** Attempt 9: pot upright, `xy` inside the
   burner red disc, predicate still false. Cook region is narrower than the
   visual disc AND requires base-mesh contact. → **Fix:** only `pi0_doubled`
   reliably achieves base contact.
3. **Pi0's left/right resolution is flaky on two-pot scenes** (attempts 2, 4, 7,
   9 each showed different behaviour). → **Fix:** preserve the closed-loop structure:
   run the doubled call before any other motion. After a failed call, inspect the
   current images. If the target pot remains upright and safely graspable and the
   burner remains clear, allow one same-episode retry with the identical full-task
   prompt and chunk budget. If the wrong pot occupies the burner, the target has
   tipped, or burner access is obstructed, do not blindly repeat the call; no
   verified recovery is known from that state.
4. **Manual re-grasp of a near-burner-but-wrong placement fails ~always**
   (attempts 7, 9) — Pi0 either grabs the other pot or air-grasps and tips the
   target. → **Fix:** do not repeat the same manual re-grasp. Continue only when
   current images satisfy the upright-target and clear-burner conditions above;
   otherwise no reliable physical recovery is known from that state.
5. **Manual push of pot toward burner tips it** off the platform (attempt 7). →
   **Fix:** do not push.
6. **`pi0_pick` accumulates wrist tilt** across chained calls (attempts 1, 8).
   → **Fix:** don't chain `pi0_pick`s; use one doubled call from the untouched
   initial task state.
7. **`pi0_pick` reports `success=true` on air-grasp** when descent misaligns
   (attempt 6) — verify via `final_gripper_opening` or by re-localizing the pot
   in agentview. Mooted by `pi0_doubled` which uses `libero_terminated` as
   success and isn't fooled by air-grasps.
8. **SAM3 `"stove"` returns the platform**, which is ~15 cm wider than the
   burner. Several attempts (1–6) targeted (−0.19, −0.20) — a platform corner,
   ~15 cm off the true burner at (−0.046, −0.243). → **Fix:** for diagnosis,
   use RGB red-pixel mask on agentview projected to world.

## Re-localization per-scene

`pi0_doubled` consumes the raw agentview and resolves "left moka pot" + "stove"
itself — no SAM3 prompt or world_hi pixel needed for the recipe. per-scene jitter
in pot/stove positions is handled inside Pi0.

For diagnosis only (not part of the recipe):
- `"silver moka pot"` SAM3 → 2 detections; LEFT = the one with **larger world-y**
  (camera-side / robot's left).
- Burner centroid: **RGB red-pixel mask on agentview projected to world** is
  trustworthy. SAM3 `"red circle"` / `"burner"` is an acceptable backup at score
  ≥ 0.30. SAM3 `"stove"` returns the platform (much larger than the burner) —
  do NOT use it as a target.
- `stove_z`: top-of-platform z (RGB stove-tile pixel + depth). observed run: 0.928.

## Difficulty signal

reference run: **10 attempts** (9 scripted-path failures, then `pi0_doubled` wins). The
recipe is short and high-confidence, but the underlying first-call success rate
of `pi0_doubled` is unknown. Choose `max_chunks: 25` (or 30 when extra contact
budget is justified) before execution. If the call does not solve the task,
inspect the current state: retry the identical full-task call once only when the
target remains upright and safely graspable and the burner remains clear. Do not
repeat from a tipped, obstructed, or wrong-pot-on-burner state.

the verbatim procedure.
