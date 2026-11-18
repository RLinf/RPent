## Dual-Franka exploration protocol

Follow explicit operator corrections to the task preset and retain them in
handoff notes. Task solvability is not guaranteed.
Before the first motion in every session, call
request_scene_reset(reason, expected_scene_state). A human restores the objects
and confirms done; the tool then resets robot posture and records fresh
observations. Reset failure does not start a new attempt; keep motion stopped
until a reset completes.

Ask request_operator_verdict after apparent success or failure. The human answers
success, failure, continue or abort. Tool ok=True, gripper position, VLA
terminated/truncated, and your own finish status are not task-success evidence.
Any subsequent physical action invalidates the verdict. continue clears the
previous verdict. Obtain another judgment before finish.

Call describe_dual_franka_setup, inspect view_env_state, then follow the shared
memory-reading workflow and read prior attempt notes. Call request_scene_reset
before the first motion.
Do not finish successfully without a current human verdict.

## Dual-Franka evidence artifacts

Preserve {{output_dir}}/sessions/session_<NNN>/states.json and its RGB/depth,
camera metadata and per-arm state artifacts. operator_events.json records human
feedback against the attempt and observation step. Include that operator feedback
in each failed-attempt archive. Never replace these logs with a simulator trace.

After operator-confirmed success, write {{output_dir}}/{{recipe_tag}}.json with
observations, parameters and strategy_notes. The runner exports motion commands
after the last successful scene reset as {{recipe_tag}}_recipe.jsonl and adds
operator evidence to the audit. These are issued commands, not a promise that
replaying old coordinates is safe. Cite session/attempt/step evidence and identify
setup, camera, checkpoint and object assumptions that limit transfer.

## Memory publication

At normal TaskRun completion, the finalizer merges eligible inbox proposals unless
`--no-auto-merge-memory` is set. Keep proposals in the inbox; do not edit the
published corpus directly or claim publication before the result confirms it.

## Memory artifact paths and metadata

Write each failed-attempt archive to
`{{output_dir}}/attempts/attempt_s<session>_a<attempt>_failed.json`.
Replace the angle-bracket fields with the actual attempt and session numbers; never overwrite a prior archive.

Write the task-family proposal to
`{{memory_inbox}}/task-family_{{recipe_tag}}_draft.md` and each global lesson to
`{{memory_inbox}}/new_global_<kind>_<slug>.md`.

Use a short kebab-case slug for each global `id`, without filename prefixes or suffixes.
Use the following task-family frontmatter, filling placeholders from the current
task and recorded evidence. `attempts` is the observed attempt count.

```yaml
---
id: task-family_dual_franka_real_t{{task_id}}
scope: task-family
suite: dual_franka
regime: real
task_id: {{task_id}}
task_language: >-
  <verbatim current task instruction>
evidence:
  cells: [{{recipe_tag}}]
  attempts: <observed count>
confidence: single-shot
related: []
---
```
