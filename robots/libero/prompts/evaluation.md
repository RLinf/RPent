## LIBERO evaluation outcome

SINGLE-ATTEMPT MODE overrides reset/retry wording in the seed-0 lessons.
Target top-level `terminated == true`. In-place re-positioning, re-`pi0_pick`,
the prompt ladder, `rotate_pitch` and `move_pose` are allowed within this episode.
Report honest `terminated:false` when the sequence is exhausted.

## Recovery

Use the in-place recovery tools within this episode. If it is unrecoverable,
do not reset: write an honest stuck audit (terminated:false) and call finish.

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
