## LIBERO exploration protocol

Every task in this benchmark is physically solvable.
MULTI-ATTEMPT EXPLORE MODE: reset restores the scene. The task signal is `state.libero_terminated == true`.
The runtime refuses unsolved finish until the attempt budget is spent and refuses
reset after it is spent.

When ideas run out, try an untested class of physical approach:
     - scripted OSC pushes / servos;
     - the trained contact skill (`pi0_doubled`), from a clean pose;
     - changing how the servo advances (`step_clip`, `max_steps`, `tol`) rather
       than the target;
     - changing the contact GEOMETRY — where on the object you touch, and at
       what wrist pose. A wall along one axis often opens along another;
     - changing an EARLIER step so the blocking state never arises at all.
   If several classes are still untried, you are not out of ideas yet.

   When you do hand off, say in the audit which classes you exhausted and which
   you would try next. That sentence is the most valuable thing you leave the
   next agent.

## Recovery and reset

Also allowed: `reset`, after closing out a failed attempt within the budget.
Use in-place recovery first. If the episode is unrecoverable, close out the
attempt, reset, and try a changed plan. Never warp objects or robot poses.

## LIBERO archive fields

Close out every failed attempt, whether resetting or stopping. Number archives
from 1 continuously. Include suite, task, seed, libero_terminated:false,
final_state, changed_lever_vs_attempt<N-1> (omit on attempt 1), and strategy_notes.
Use `## Attempt <N>` headings in working notes.

## LIBERO completion and memory fields

On state.libero_terminated == True, save the audit, prepare eligible memory,
then finish. If unsolved, finish only when the runtime permits stopping.
For an unsolved audit record suite, task_id, seed, regime, libero_terminated:false,
total attempts, final_state, the stopping condition and where each attempt stalled.
For success record regime:"strict_perception", libero_terminated:true and attempts.
Put localization in one opening strategy_notes sentence; pick_result keys name
recipe steps, e.g. bowl_pi0_pick_step3, rather than bare object names. Use the
successful portion of states.json; the runner exports the recipe after finish.
Task-family id: task-family_<suite family>_<regime>_t<task_id>, e.g.
task-family_libero10_swap_t3. Regime is task|swap|lan|object, not strict_perception.

## Memory publication

A human reviews and merges the inbox proposals later. Leave them in the inbox;
do not edit the published corpus directly or claim they have been published.

## Memory artifact paths and metadata

Write each failed-attempt archive to
`{{output_dir}}/attempts/attempt_<N>_failed.json`.
Replace the angle-bracket fields with the actual attempt number; never overwrite a prior archive.

Write the task-family proposal to
`{{memory_inbox}}/task-family_{{recipe_tag}}_draft.md` and each global lesson to
`{{memory_inbox}}/new_global_<kind>_<slug>.md`.

Use a short kebab-case slug for each global `id`, without filename prefixes or suffixes.
Use the following task-family frontmatter, filling placeholders from the current
task and recorded evidence. `attempts` is the observed attempt count.

```yaml
---
id: task-family_<suite family>_<regime>_t<task_id>
scope: task-family
suite: <suite family, e.g. libero10>
regime: <task|swap|lan|object>
task_id: {{task}}
task_language: >-
  <verbatim current task instruction>
evidence:
  cells: [{{recipe_tag}}]
  attempts: <observed count>
  solved_seeds: [{{seed}}]
  failed_seeds: []
confidence: single-shot
related: []
---
```
