---
name: moka-placement-with-deferred-stove-activation
description: Reusable stove recipe that places a fragile moka pot before activating the knob to avoid orientation drift.
aliases:
  - turn on the stove and put the moka pot on it
  - moka pot stove burner knob
metadata:
  node_type: memory
  type: skill
---

# Moka placement with deferred stove activation

## Applicable pattern

Use this pattern when a fragile handled vessel must be placed on a burner and knob interaction would otherwise disturb the end-effector orientation needed for placement.

## Reference task alias

`turn on the stove and put the moka pot on it`

## Winning technique

Two-stage swap: (1) put the silver octagonal moka pot on the stove burner,
THEN (2) rotate the stove knob to ON. **The reverse of the natural order is
the win.** Doing `pi0_doubled` on the stove first tilts the eef quat
(`w ≈ 0.5`), which is not cleanly recoverable by `rotate_pitch`/`rotate_wrist`,
and every subsequent moka grasp fails (all 3 exploration attempts hit this).

The moka body tapers — no body-grip survives the lift (`qpos` slides past
the widest point, jaws collapse to ~0.001, pot drops). The working grasp is
a **top-hook**: descend ~3 cm below the moka top, `set_gripper +1 steps=20`
to close gently (`qpos ~0.037`, not 0), and lift — the jaws hook the
tapered cap. Then a scripted manual carry + place at constant `z=1.18`
clears the chefmate pan (the on-stove distractor). Pi0 owns only the final
knob turn, called with the **single-objective prompt `"turn on the stove"`**
(NOT the full task language), with the arm retreated to `(-0.21, 0.0, 1.18)`
so pi0 has a clean reach to the right-edge knob.

## Recipe shape (scene-relative, in order)

1. `segment "the silver octagonal moka pot"` → `(MOKA_x, MOKA_y, MOKA_top_z)`
2. `segment "the stove burner where pots are placed"` → `(BURNER_x, BURNER_y, BURNER_z)`
3. `move_to(MOKA_x, MOKA_y, 1.18, gripper=-1, step_clip=0.02)` — above moka
4. `move_to(MOKA_x, MOKA_y, MOKA_top_z - 0.03, gripper=-1, step_clip=0.01)` — descend to top-hook depth
5. `set_gripper(gripper=+1, steps=20)` — gentle close, qpos settles ~0.037
6. `move_to(MOKA_x, MOKA_y, 1.18, gripper=+1, step_clip=0.012)` — lift
7. `move_to(BURNER_x, BURNER_y, 1.18, gripper=+1, step_clip=0.015)` — carry
8. `move_to(BURNER_x, BURNER_y, BURNER_z + 0.11, gripper=+1, step_clip=0.01)` — descend over burner
9. `set_gripper(gripper=-1, steps=12)` — release
10. `move_to(-0.21, 0.0, 1.18, gripper=-1, step_clip=0.025)` — retreat LEFT
11. `pi0_doubled("turn on the stove", max_chunks=40)` — finishes in ~8 chunks

## Magic numbers (with safe ranges)

| Lever | reference run | Range / notes |
|---|---|---|
| Step ORDER | moka first, knob last | (fixed — knob first wrecks the grasp) |
| Grasp descent offset | `MOKA_top_z − 0.03` | `−0.025 … −0.04` |
| Close at grasp | `set_gripper +1, steps=20`, qpos ~0.037 | 16–24 steps; qpos > 0.06 = miss, qpos ~0 = body-grip (will slip) |
| Lift z (carry altitude) | 1.18 | 1.16–1.20 |
| Carry/lift `step_clip` | 0.012 / 0.015 | 0.010–0.018; faster shakes the loosely-hooked pot loose |
| Release height | `BURNER_z + 0.11` ≈ 1.04 | `+0.08 … +0.13`; higher tips, lower collides |
| Release | `set_gripper -1, steps=12` | 10–16 |
| Retreat target | `(-0.21, 0.0, 1.18)` | `x ∈ [-0.25, -0.18]`, off-knob side |
| Pi0 prompt | `"turn on the stove"` | DO NOT use full task language — re-targets moka |
| Pi0 max_chunks | 40 | 30–50; reference run finished in 8 |

reference run telemetry: moka landed at `(-0.0535, 0.184, 1.023)` on a burner at
`(-0.052, 0.199, 0.933)`; pi0 stove-on terminated `libero_terminated=true` in
8 chunks. Final qpos at close = 0.037 (top-hook signature).

## Failure modes to avoid (mined from the 3 failed attempts)

* **Body-grip + manual lift (attempts 1 & 2).** `set_gripper +1` at body
  `z ≈ 0.985–1.04` closed `qpos` to 0.007–0.024, but any `move_to gripper:+1`
  (tested `step_clip` 0.005–0.012) collapsed `qpos → 0.001` and dropped the
  pot. **Fix:** top-hook grasp at `MOKA_top_z − 0.03` instead — gentle close
  hooks the tapered cap, not the body.
* **Knob first (attempt 3).** `pi0_doubled "turn on the stove"` succeeds at
  the knob but leaves `quat w ≈ 0.5`. Subsequent `rotate_pitch 0`/
  `rotate_wrist 0` does NOT cleanly restore vertical; pi0_pick descends only
  to `z ≈ 1.084` and bails; scripted top-hook from the tilted pose misses.
  **Fix:** moka FIRST from clean home pose, pi0 stove-on LAST.
* **`pi0_doubled` with full task language (attempts 1 & 2).** At 50/50/80
  max_chunks it visibly moves things but doesn't fire the predicate or fails
  to grasp. **Fix:** never give pi0 the full multi-step task on this scene;
  give it exactly the *single* remaining objective ("turn on the stove")
  once everything else is done.
* **`pi0_pick "pick up the moka pot"` from home/far pose.** Lifts before
  descending → `descent_done=False`, never closes. Not used in the winning
  recipe — scripted top-hook is the right tool for this pot.

## per-scene re-localization

Always run on `agentview` (high-res `world_hi` if available):

| Entity | SAM3 prompt (primary) | Alternates | Used for |
|---|---|---|---|
| Moka pot | `"the silver octagonal moka pot"` | `"the silver coffee maker"`, `"the small octagonal silver pot"` | `MOKA_x, MOKA_y, MOKA_top_z` (top-Z of mask) |
| Burner | `"the stove burner where pots are placed"` | `"the black round burner on the stove"` | `BURNER_x, BURNER_y, BURNER_z` |

Score threshold ≥ 0.4. DO NOT use `"the pan"` / `"frying pan"` — that's the
left-side distractor (chefmate_8_frypan_1). Sanity-check geometry before
applying the recipe:

* `MOKA_y < 0`, `BURNER_y > 0` (moka in front half, stove in back half)
* `|MOKA_top_z − 1.00| < 0.05`, `|BURNER_z − 0.93| < 0.04` — outside these
  ranges the mask is probably wrong (e.g. caught the pan instead of burner)

The retreat target `(-0.21, 0.0, 1.18)` is workspace-fixed for the standard
LIBERO stove (knob on right edge). If on a new scene segmentation places the
knob on the left (very rare), mirror to `(+0.21, 0.0, 1.18)`.

## Difficulty and reliability

**4 attempts to win (3 failures + 1 success).** All 3 failures shared the
same root cause: trying to body-grip the tapered moka or trying to do the
knob first. Once the explorer flipped the order AND switched to a
top-hook grasp, it solved on the first try. Treat as **medium-hard**: on
a new scene, a single attempt should succeed if the recipe is followed and
SAM3 localization is verified. > 1 retry almost always means SAM3 mis-
detected the moka top-Z or the burner — re-localize before tweaking magic
numbers.

Links: [[feedback_move_pose_covarying_reach]] (the `z ≈ 1.038` OSC wall is
why the descent stops at `top_z − 0.03`); [[feedback_no_pi0_end_to_end]]
(why we split pi0 into single-objective calls instead of trusting full
task language).
