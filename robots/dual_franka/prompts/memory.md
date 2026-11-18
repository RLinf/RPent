## Read memory before acting

First inspect the current task instruction and scene. Then read the available,
matching references below before the first robot action, in this order:

1. The current task's published audit and recipe pair. Use the JSON audit to
   understand phases, observable gates, controller choices and recovery reasons;
   use the JSONL as evidence of actual calls, parameters and policy cadence.
2. Matching task-family notes. Read their bodies to check applicability, objects,
   relations and step order; a filename or index summary alone is insufficient.
   Read these notes even when a successful recipe exists.
3. Relevant global lessons. Use an index only if it is listed below; consult
   one to three relevant leaves, checking their conditions and counterevidence.
4. In exploration, the current inbox's working notes and available attempt
   archives, so the next approach accounts for previous evidence.

Only the listed files and directories are available for memory reading. Within
directories, select references for the current task and relevant global lessons.
Do not substitute another task's recipe for a missing reference. Do not browse
`_internal/` outside the current exploration inbox. If a file or directory is
missing or empty, record the absence and continue with current observations.
Record exact memory filenames used in planner-owned audit or technique notes.
Never edit a runner-owned audit to add references.

Use `read_text_file` with `max_chars=100000`. Check returned size and content for
truncation; if truncated, read again with `max_chars` at least the reported size.
Do not treat a partial read as a complete reference. Memory is read-only during
evaluation; write evaluation artifacts only to the run's output directory.

Available sources:
- {{memory_dir}}/task-specific/
- {{memory_dir}}/task-family/
- {{memory_dir}}/MEMORY.md
- {{memory_dir}}/global/
- {{memory_inbox}}/wip/
- {{output_dir}}/attempts/

## Transfer experience to the current scene

The current instruction, observations and tool results override historical
notes. Transfer strategies and observable completion gates, not old geometry.
Re-identify objects, destinations and arm roles; re-measure pixels, coordinates,
poses, clearances and grasp offsets. Historical values are evidence, not current
action targets. Protect achieved task relations while adapting to the remaining
blocker.

A successful episode may contain failed approaches and corrective detours:
identify which actions made progress rather than replaying every call. Use a
supported recipe as a starting plan when current evidence agrees; treat an
experimental recipe as a weak prior. Separate observed outcomes, proposed
explanations and untested applicability; one scene does not prove reliability
across scenes.
