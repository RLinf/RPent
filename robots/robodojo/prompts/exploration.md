## RoboDojo initial observation

Call `view_env_state` first to read the instruction, observe the scene and check the native action budget before planning.

Before the first robot action, read robots/robodojo/guides/GUIDE_RPENT.md completely. Inspect view_env_state and head/wrist images, read the native instruction, and note both arms' initial end-effector positions and orientations. Bind arm roles from current observations.

## RoboDojo exploration protocol

Explore the fixed task, seed group and layout. reset restores those same settings. An attempt limit of 0 means no attempt-count limit; other run limits still apply. An unsolved finish is refused while a finite attempt budget remains; reset is refused after native success. Do not request hidden reward details.

Preserve arm roles required by the instruction; reconsider roles chosen for a prior layout.

The final audit {{output_dir}}/{{recipe_tag}}.json includes task, seed, layout, verbatim instruction, attempts, native_success, final_state, memory_files_read and uncertainties. Ordered strategy_notes account for every exported motion and SAM3 call with actual parameters, trace step, phase, observed result and controller-switch reason. The runner exports recorded motion commands and SAM3 calls that found a mask without an error, in invocation order. Finding a mask does not establish correct identity.

Record SAM3 descriptions and how identity was checked, required versus layout-selected arm roles, policy conditioning and observed takeover state, Primitive use and chunk choices. Record observed native action costs for productive phases, failed approaches, recovery and terminal motions using recorded counters; these are evidence, not fixed budgets. The shared finalizer merges valid inbox proposals and updates MEMORY.md after the run.

## Memory publication

At normal TaskRun completion, the finalizer merges eligible inbox proposals unless
`--no-auto-merge-memory` is set. Keep proposals in the inbox; do not edit the
published corpus directly or claim publication before the result confirms it.

## Memory artifact paths and metadata

Write each failed-attempt archive to
`{{output_dir}}/attempts/attempt_<N>_failed.json`.
Replace the angle-bracket fields with the actual attempt number; never overwrite a prior archive.

Write the task-family proposal to
`{{memory_inbox}}/task-family_{{recipe_tag}}_draft.md` and each global lesson to
`{{memory_inbox}}/new_global_<lesson-id>.md`.

Use a short kebab-case slug for each global `id`, without filename prefixes or suffixes.
Use the following task-family frontmatter, filling placeholders from the current
task and recorded evidence. `attempts` is the observed attempt count.

```yaml
---
scope: task-family
suite: robodojo
regime: {{env_cfg_type}}
task_id: {{task}}
task_language: >-
  <verbatim current task instruction>
evidence:
  cells: [{{recipe_tag}}]
  attempts: <observed count>
confidence: single-shot
---
```
