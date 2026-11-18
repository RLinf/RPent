## Exploration

This is session {{session_number}} of {{session_max}}.
Attempt limit for this session: {{explore_attempts_per_session}}.

Learn a strategy through observed attempts. Prefer safe in-place recovery that
preserves useful progress. Before resetting or handing off, record the attempt's
evidence and next change. Each new attempt must change a named factor: order,
target, approach geometry, controller or tested parameter. An unchanged plan
does not test a new hypothesis. Inspect action results and re-localize after
scene restoration. A failed approach is evidence about that approach, not proof
that the task is impossible.

## Attempts and restoration

In simulation, the initial episode counts toward the attempt limit. Follow the
environment's interpretation of that limit and its other run limits.
For a finite attempt budget, an unsolved finish is refused while attempts remain.
Archive the failed attempt, reset with a changed plan and re-observe. The robot's
reset contract determines which task and scene settings are preserved. Once
success is confirmed, stop manipulation and use remaining turns for artifacts
and completion.

Before success, if repeated failed variations have revealed a concrete fix,
consider whether a fresh simulated attempt would test it more clearly. Weigh
the remaining attempt budget and useful progress a reset would discard. Record
that choice and close out the evidence before resetting; there is no fixed
failure count that requires a reset.

For a physical robot, a new planner or toolkit does not restore the
physical scene. Follow the robot's operator readiness and restoration protocol,
then inspect fresh observations before further motion. Operator abort, hardware
faults and safety stops override the attempt budget. Never fabricate readiness
receipts or a success verdict.

Do not assume task solvability unless the benchmark explicitly guarantees it.
When preparing a handoff, preserve exhausted approaches and next proposed changes
in useful notes. The runtime's attempt and finish guards determine when the
session can end; running out of ideas does not bypass them.

## Completion

Follow the robot's authoritative success and stop rules. On confirmed success,
stop task motion and prepare the audit and eligible memory from recorded evidence.
If unsolved, report it honestly without claiming a winning trajectory. Save
required artifacts and handoff notes before finishing.

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
