## RoboTwin exploration protocol

You control a RoboTwin task in MULTI-ATTEMPT EXPLORE mode.
Reset starts an ordinary fresh episode and may resample layout.
Only fresh TASK_ENV.eval_success=true confirms success. Call finish exactly once.

## RoboTwin audit and proposal fields

Write {{output_dir}}/{{recipe_tag}}.json with task_name, task_config, seed,
eval_success, total attempts, final_state and successful_strategy.
successful_strategy lists every recipe-eligible command after the final reset
with actual parameters in order, excluding reset itself. A recorded success
step must exist after that reset. The runner exports successful commands and
merges validated memory.

## Memory publication

At normal TaskRun completion, the finalizer merges eligible inbox proposals unless
`--no-auto-merge-memory` is set. Keep proposals in the inbox; do not edit the
published corpus directly or claim publication before the result confirms it.

## Memory artifact paths and metadata

Write each failed-attempt archive to
`{{output_dir}}/attempts/attempt_<N>_failed.json`.
Replace the angle-bracket fields with the actual attempt number; never overwrite a prior archive.

Choose descriptive task-family and global Markdown filenames directly under
`{{memory_inbox}}/`.

Use `global_<kind>_<short-name>` for each global `id`.
Use the following task-family frontmatter, filling placeholders from the current
task and recorded evidence. `attempts` is the observed attempt count.

```yaml
---
id: task-family_robotwin_<task-name>
scope: task-family
suite: robotwin
regime: {{task_config}}
task_id: {{task_name}}
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
