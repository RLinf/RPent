---
name: stove-activation-then-handle-grasped-pan-placement
description: Reusable stove recipe that activates the knob before carrying a stable handle-grasped pan to the burner.
aliases:
  - turn on the stove and put the pan on it
  - frying pan handle stove burner knob
metadata:
  node_type: memory
  type: skill
---

# Stove activation then handle-grasped pan placement

## Applicable pattern

Use this pattern when a stable wide vessel can tolerate post-knob orientation drift and should be carried by a handle with a known body offset.

## Reference task alias

`turn on the stove and put the pan on it`

## Winning technique

Two-stage: (1) rotate the stove knob to ON, (2) pick the chefmate frypan by its **handle**, carry-high over the moka-pot distractor, and place pan body on the burner. **Order: STOVE FIRST, PAN SECOND** (opposite of sibling [[moka_placement_with_deferred_stove_activation]]) — because the wide stable pan rim survives a slightly tilted post-knob eef whereas the swap_t2 moka would tip. The transfer-critical lever is `pi0_pick` repurposed as a CONTACT DRIVER for the knob: `lift_thresh=999, gripper_closed_thresh=0` forces Pi0 to keep rotating the knob for the full `max_chunks` ≈ 25 chunks. The pan is then grasped on its handle (gripper +0.08 m in x from pan body centroid) and placed at burner + (+0.12, +0.08) to compensate for the handle offset.

## Magic numbers (with safe ranges)

- **Knob contact driver:** `pi0_pick "turn on the stove" max_chunks=25 lift_thresh=999 gripper_closed_thresh=0.0` (range 20–35 chunks; add +10 if burner not glowing).
- **Pan Pi0 grasp:** `pi0_pick "pick up the frying pan" max_chunks=22 lift_thresh=0.05 gripper_closed_thresh=0.06` (11–15 chunks observed; `peak_lift_m` 0.10–0.15 m).
- **Firm-up after pan grasp:** `set_gripper +1 steps=8` (range 6–12; **DO NOT raise to 15+** — closes past the handle bar and pan slips out).
- **Carry altitude:** `z ≈ knob.z + 0.23` (≈ 1.15). Below 1.13 risks grazing the moka pot in the middle.
- **Handle grasp xy:** `(pan.x + 0.08, pan.y)` when handle not separately segmented; prefer segmenting `handle` directly and use `handle.xy`.
- **Place gripper xy:** `(burner.x + 0.12, burner.y + 0.08)` — mirrors the handle offset so pan BODY lands on burner. With segmented handle: `burner.xy + (handle.xy − pan.xy)`.
- **Final descent z:** command `burner.z + 0.08` (≈ 1.00); OSC stalls when pan rim contacts. step_clip ladder: 0.025 (transit) → 0.02 (carry) → 0.015 (final descent).
- **`tol`:** 0.015 (transit) → 0.012 (carry) → 0.01 (final).

## Failure modes to avoid

- **`pi0_doubled` with full task_language on the knob** — at max_chunks 50–80 it did not produce the required visible knob and burner state change in the observed runs. Fix: `pi0_pick` as contact driver with short prompt `"turn on the stove"`.
- **Long Pi0 grasp prompts (full task_language)** — cause place-mode bleed (Pi0 starts translating sideways before lifting). Fix: short verb-phrase `"pick up the frying pan"`.
- **`set_gripper +1 steps≥15` after pan grasp** — crushes past the handle bar; pan slips out mid-carry. Fix: `steps=8`.
- **Skipping `set_gripper +1`** — `step_clip=0.02` carry jolts the post-pi0 grip loose. Fix: 8-step firm-up.
- **Direct diagonal carry pan→burner at z<1.13** — clips the moka pot in the middle. Fix: explicit y=0 waypoint at z=carry-altitude on BOTH legs (empty approach AND loaded carry).
- **Placing gripper directly over `burner.xy`** — pan body ends up offset by −0.12 x, −0.08 y because grip was on the handle, not the center. Predicate fires on pan-body-on-burner. Fix: apply place offset `(+0.12, +0.08)` or symmetric handle-vs-body vector.
- **Final descent below `burner.z + 0.05`** — wastes step budget; OSC stall is at burner.z + 0.08–0.10. Fix: command +0.08 and release.
- **Reversed order (pan first, knob last)** — `pi0_pick "turn on the stove"` from a hand returning to home over the pan is still doable, but you traverse the moka column TWICE under load. Stove-first is cleaner.
- **Generic `"the pan"` SAM3 prompt** — on rare layouts segments the burner instead. Fix: `"the black frying pan"` and verify `pan_y < 0`.

## How to re-localize per-scene

agentview SAM3 only (wrist not needed — scene wide-spread):

| Entity | SAM3 prompt (primary) | Sanity bound |
|---|---|---|
| `pan` | `"the black frying pan"` | `pan_y < 0`, `pan_z ∈ [0.88, 0.94]` |
| `burner` | `"the round burner on the stove"` | `burner_y > 0`, `burner_z ∈ [0.91, 0.95]` |
| `knob` | `"the small black stove knob"` | `knob_y > 0`, `knob_x > burner_x` |
| `handle` (optional, preferred) | `"the black handle of the frying pan"` | score ≥ 0.4, `|handle.xy − pan.xy| ≤ 0.12` |

Score threshold ≥ 0.4. Geometry check: `pan_y < 0 < burner_y` and `knob_x > burner_x`. If any flips, re-prompt. Optional moka sanity check: `"the silver octagonal moka pot"` should land between pan and burner in y.

reference run telemetry: pan ≈ (−0.14..−0.21, −0.22..−0.10, 0.90–0.92), burner ≈ (−0.17..−0.13, 0.18–0.21, 0.92–0.93), grasp at (−0.064, −0.22, 1.05), final eef (0.00, 0.30, 1.00). Note the +0.08 x offset between gripper grasp xy and pan-body x — that is the handle bias to transfer.

## Difficulty and reliability

**1 attempt, 1 success** (verbatim replay of a previously-proven recipe — no additional exploration cost). The two transfer-critical levers (`pi0_pick` contact-driver for the knob; +x handle bias + symmetric place offset for the pan) were paid for during the sibling [[moka_placement_with_deferred_stove_activation]] 4-attempt exploration. Difficulty signal: **LOW–MEDIUM** — in a fresh scene a single attempt should succeed if SAM3 grounds pan/burner/knob cleanly and the handle offset is applied symmetrically.

## Cross-links

- [[moka_placement_with_deferred_stove_activation]] — same scene, opposite object (moka pot is the placed object there). Uses the OPPOSITE order (moka first, knob last) because the moka tips on a tilted-eef place. The pan in this pattern survives the tilt, so the knob can go first and save a moka traversal.
- [[feedback_stove_turnoff_strict]] — require a real visible knob state change and use the `pi0_pick`-as-contact-driver trick.
- [[dual_object_anchor_and_relative_placement]] — same pattern of short Pi0 prompts + immediate `set_gripper +1` firm-up; the mug-rim case uses steps=20, while this pan-handle case uses steps=8 (more forgiving geometry).
- [[feedback_pi0_pre_pos_can_hurt]] — the place-mode-bleed when Pi0 is given the full task_language.
- [[feedback_move_pose_covarying_reach]] — explains why the final descent stalls at burner.z + ~0.08–0.10.
