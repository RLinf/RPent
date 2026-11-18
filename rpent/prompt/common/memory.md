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
{{memory_sources}}

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

<!-- explore-only -->

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
`{{memory_kinds}}`, non-empty `title` and `applies_when`, `confidence`, and
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
