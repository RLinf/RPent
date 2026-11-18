## Working notes and failed attempts

Reserve enough of the remaining run budget to preserve evidence and prepare
eligible memory. Immediate stop and operator-abort rules take precedence.

During exploration, append concise evidence to `{{memory_inbox}}/wip/notes.md`
with session and attempt headings. Preserve the approach, actual commands and
parameters, observation references, progress, failed hypotheses and next named
change. Keep earlier notes and archives. Bound every failure claim by the method
and conditions actually tested; write "cause unknown" when the cause is unknown.
A successful retry does not establish which change helped. Never invent causes,
measurements or parameter ranges.
Update the working task-family write-up at each attempt close-out under `wip/`,
keeping the failure history and marking unvalidated approaches explicitly.

Before resetting a failed attempt or handing it off, close out that evidence.
Write the failed-attempt record at the path specified by the robot below, including
the task, attempt number, initial plan, actual calls, final state, bounded failure
evidence, next change and robot-specific verdict fields. Never overwrite an
earlier archive.

## Consolidation and eligibility

Re-read working notes and attempt evidence before consolidation. Create final
task-family/global drafts only after the
robot's authoritative success confirmation. If unsolved or aborted, leave working
notes, failure archives and an honest unsolved audit; do not claim a winning recipe.

Write memory only under the current inbox `{{memory_inbox}}/`. Working notes stay
in `wip/`; final Markdown proposals go directly in the inbox, using the robot-specific
task-family and global filenames. Never directly edit, rename or delete published
`task-family/`, `global/`, `task-specific/` or `MEMORY.md` files. The merge process
validates top-level drafts, not working subdirectories. Follow the publication instructions below. Do not claim publication before it completes.

## Proposal format

Each proposal begins with parseable YAML frontmatter delimited by `---`.
Use the robot-specific task-family template below. Replace example values with
recorded evidence. Quote values containing `: `, quoting the entire scalar.
Use IDs in `id` and `related`, not filenames or wikilinks; body links may use
`[[id]]`. Follow the robot's identifier rules. Where no identifier is prescribed,
use a short kebab-case slug naming the lesson rather than the task or its objects;
omit filename prefixes, extensions and the `_draft` suffix.

## Task-family write-up

Use these sections for working notes and eligible final task-family proposals.
Keep each section concise and grounded in recorded evidence:

- `## Applicable pattern`: what the task tests and the conditions under which
  the strategy applies.
- `## Winning technique`: phase order, actions and observable per-step success
  criteria. Until the approach has succeeded, use
  `## Best known approach (UNVALIDATED)` instead.
- `## Tested parameters`: actual tested settings and verified usable ranges.
  Mark an unknown range as unknown; one working value does not establish a range.
  Put each observed failure boundary or constraint on its own line.
- `## Failure modes`: a table with symptom, bounded cause (or "cause unknown"),
  attempt reference, and tested or proposed fix. Append evidence across attempts;
  mark earlier explanations contradicted by later evidence instead of erasing them.
- `## Re-localization per scene`: the observation and localization methods used,
  visual cues, confusable objects, rejection criteria and tested fallbacks.
  Record tool parameters only when supported and tested. Re-measure current
  geometry; keep episode coordinates in evidence rather than reusable targets.
- `## Fragility flags`: the most vulnerable step, its observable failure signal
  and supported recovery or fallback.
- `## Difficulty and reliability`: attempts to convergence, unresolved cases,
  measured reliability and transfer limits. State when cross-scene reliability
  has not been measured; do not invent a success rate.
- `## Cross-refs`: relevant `[[id]]` links to global memories.

Prepare the write-up from existing evidence. Do not add robot motion or restore
the scene merely to write memory. Final proposals still require confirmed success;
an unvalidated working write-up does not establish success.

## Global lessons and duplicate evidence

For each supported cross-task lesson, use `scope: global`, a `kind` from
`primitive|perception|strategy|failure`, non-empty `title` and `applies_when`, `confidence`, and
an `evidence` mapping with non-empty `cells`. Add search terms in `symptom` and
related IDs in `related`. Use one file per lesson with a short imperative title.
The lesson must apply beyond this task's objects or fixture; narrower observations
belong in the task-family write-up. Do not promote explanations contradicted by
later evidence. Qualify a still-supported claim with its actual `applies_when`.

Start the body with one sentence stating the claim, followed by:

- `**Why:**` the supported mechanism, or "observed, cause unknown".
- `**How to apply:**` tested settings, ordering, conditions and thresholds.
- `**Falsify:**` an observation that would disprove the claim.
- `**Related:**` relevant `[[id]]` links.

Before writing a new lesson, list and read plausible existing memory files.
Compare their bodies, not just index summaries. If the same claim is already
covered and your evidence confirms it, avoid a duplicate lesson. Append a review
record to `{{memory_inbox}}/corroborations.jsonl`:

```json
{"id":"<existing-id>","cell":"{{recipe_tag}}","effect":"confirmed","note":"<observed evidence>"}
```

This JSONL file is supporting evidence for review; the merger does not consume
it or raise confidence from it. Put conflicting observations in
`conflict_<id>.md` with valid proposal frontmatter and the existing ID, retaining
the old claim, new evidence, attempt references and conditions under which each
claim may hold. If no new global lesson is supported, say so.

One seed or layout warrants `single-shot` confidence even after several attempts.

## Task-specific trajectory evidence

A successful reference is the matched audit
`{{output_dir}}/{{recipe_tag}}.json` and exported
`{{output_dir}}/{{recipe_tag}}_recipe.jsonl`. Publish this pair only after confirmed
success. It must describe the successful attempt after the last valid scene
reset, not a splice of failed attempts. Preserve actual order and parameters of
every command eligible for this robot's export; explain recovery and controller
switches without inventing, omitting or reordering eligible calls.

Write the audit from recorded evidence and match its strategy explanation to the
runner's exported recipe. Keep failed attempts in their own records.
If unsolved, report the tried approaches and uncertainty without claiming a
successful trajectory. Save required artifacts before finishing.

In the completion summary, briefly report the task outcome, memory proposals,
corroborations and conflicts actually recorded. Do not claim artifacts were
published unless the publication result confirms it.

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

## Additional primitives

Also allowed: `reset` (🔁 allowed here —
close out the attempt first, see below).
FORBIDDEN: `exit`, `set_object_pose`, `articulate_to`, `js_move_to`,
`carry_object`.

## Recovery

RECOVERY (in-place FIRST, then reset): re-localize (objects may have moved);
re-pre-position and re-`pi0_pick` on the next prompt-ladder rung; split long
traversals into <0.30 xy waypoints; for a door/drawer/knob use a SHORT capped
OSC push or `pi0_doubled`, never one long push — it NaNs MuJoCo. If the episode
is unrecoverable, close out the attempt, `reset`, and try a changed plan.
Never warp.


## LIBERO archive fields

Close out every failed attempt, whether resetting or stopping. Number archives
from 1 continuously. Include suite, task, seed, libero_terminated:false,
final_state, changed_lever_vs_attempt<N-1> (omit on attempt 1), and strategy_notes.
Use `## Attempt <N>` headings in working notes.

## LIBERO completion and memory fields

WHEN state.libero_terminated == True — or one of the legacy Rule 4 (a)/(b)/(c)
conditions permits stopping — save the audit, run eligible distillation, then finish.
For an unsolved audit record suite, task_id, seed, regime, libero_terminated:false,
total attempts, final_state, the stopping condition and where each attempt stalled.
For success record regime:"strict_perception", libero_terminated:true and attempts.
Put localization in one opening strategy_notes sentence; pick_result keys name
recipe steps, e.g. bowl_pi0_pick_step3, rather than bare object names. Use the
successful portion of states.json; the runner exports the recipe after finish.
Task-family id: task-family_<suite family>_<regime>_t<task_id>, e.g.
task-family_libero10_swap_t3. Regime is task|swap|lan|object, not strict_perception.
