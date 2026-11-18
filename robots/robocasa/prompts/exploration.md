## RoboCasa exploration protocol

MULTI-ATTEMPT EXPLORE mode. Every task in the RoboCasa benchmark is solvable. Drive robocasa_terminated / state.success to true. reset starts a fresh attempt; use it only for an unrecoverable or deliberately completed experiment.

Read the initial success criteria, task_language and all camera views.
Inspect task_progress after every action; follow the navigation and RLDX tool
protocols throughout each attempt.

Write the final audit to {{output_dir}}/{{recipe_tag}}.json and consolidate inbox proposals before finish. The runner exports the clean winning commands after the final reset and merges validated inbox material. Task-family proposals must record task_progress milestones and a failure table. Use the verbatim initial task language.

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
`{{memory_inbox}}/new_global_<kind>_<slug>.md`.

Use a short kebab-case slug for each global `id`, without filename prefixes or suffixes.
Use the following task-family frontmatter, filling placeholders from the current
task and recorded evidence. `attempts` is the observed attempt count.

```yaml
---
scope: task-family
suite: robocasa
regime: {{split}}
task_id: {{task_name}}
task_language: >-
  <verbatim current task instruction>
evidence:
  cells: [{{recipe_tag}}]
  attempts: <observed count>
  solved_seeds: [{{seed}}]
  failed_seeds: []
confidence: single-shot
---
```
