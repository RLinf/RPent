## LIBERO evaluation outcome

SINGLE-ATTEMPT MODE overrides reset/retry wording in the seed-0 lessons.
Target top-level `terminated == true`. In-place re-positioning, re-`pi0_pick`,
the prompt ladder, `rotate_pitch` and `move_pose` are allowed within this episode.
Report honest `terminated:false` when the sequence is exhausted.

## Available primitives and localization aid

FORBIDDEN: `exit`, `set_object_pose`, `articulate_to`,
`js_move_to`, `carry_object`.

⚠ INFRA NOTE: `pi0_doubled` IS implemented and callable in this runtime —
verified. It runs the Pi0 VLA on a CONTACT skill (drawer/door open-close, knob
turn) with success := `terminated` (no lift / no gripper-close
assumption — unlike `pi0_pick`). If ANY prior note or reference for this cell
concluded that `pi0_doubled` is "unknown action" / missing / that
drawer-or-door articulation is an unsolvable "structural dead-end" BECAUSE no
contact primitive existed — DISREGARD that specific conclusion and actually
USE `pi0_doubled` for the drawer/door step, alternating with short capped OSC
pushes/aligns as needed. Re-prove the cell from scratch; do not inherit the
dead-end verdict.

SAM3 localization aid — `segment` (no robot motion): instead of eyeballing
a pixel, call `segment({"prompt":"the black bowl on the cookies box",
"camera":"agentview"})`. It runs SAM3 on the current image, back-projects the
mask via the matching world map, and returns a robust median `world_xyz` plus
logical `segment_artifact` and `overlay_artifact` names. Inspect the embedded
overlay image to confirm the right object. Use `camera":"wrist"` (after parking the eef ~15–20 cm over the
target) for ±1–2 cm refinement, or `"point":[row,col]` for a point prompt.
Text `prompt` and `point` are mutually exclusive; provide exactly one.
⚠ PROMPT PHRASING (SAM3 is sensitive): use a plain colour+shape+RELATION phrase,
NEVER the internal/brand name from `object_names`/BDDL. `"the akita black bowl"`
scores ~0.03 (SAM3 can't ground "akita") whereas `"the black bowl on the stove"`
scores ~0.76. Strip proper nouns (akita, glazed_rim_porcelain_…) — say what it
LOOKS LIKE + where it is. Always inspect the returned overlay image to confirm
the mask landed on the right object before moving.
This is a CONVENIENCE alternative to manual back-projection — if it returns
`{"error":..., "fallback":...}` (server down / low score / no detection), walk
the prompt (drop the brand word, add the relation) or just pick a pixel in the
high-resolution image and call `back_project` yourself. It does NOT replace the
two-camera relation protocol for disambiguating identical objects.

## Recovery

RECOVERY (in-place ONLY — no reset): re-localize (objects may have moved),
re-pre-position + re-pi0_pick on the next prompt-ladder rung; split long
traversals into <0.30 xy waypoints; for a door/drawer/knob use a SHORT capped
OSC push or `pi0_doubled`, never one long push — it NaNs MuJoCo. If the task is
unrecoverable within this one episode, do NOT reset — write an honest
stuck-audit (`terminated:false`) and call `finish`. Never warp.

## Finish

WHEN top-level `terminated == true` in the latest tool result:
a. Write audit `{{output_dir}}/{{recipe_tag}}.json` with:
   suite, task_id, seed, regime:"strict_perception", strategy_notes (incl. how
   you localized), pick_result, final_state (latest state's `state`),
   terminated:true.
b. Call `finish`.
If your single attempt does not solve it, write `{{output_dir}}/{{recipe_tag}}.json` with
terminated:false + strategy_notes describing what you tried in this one
episode and where it stalled. Then call `finish`. (NO reset, NO second attempt.)
