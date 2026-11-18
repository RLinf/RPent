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
id: task-family_{{suite}}_{{task_id}}
scope: task-family
suite: {{suite}}
regime: {{regime}}
task_id: {{task_id}}
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
