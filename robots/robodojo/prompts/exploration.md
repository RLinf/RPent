## RoboDojo initial observation

Call `view_env_state` first to read the instruction, observe the scene and check the native action budget before planning.

Before the first robot action, read robots/robodojo/guides/GUIDE_RPENT.md completely. Inspect view_env_state and head/wrist images, read the native instruction, and note both arms' initial end-effector positions and orientations. Bind arm roles from current observations.

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
`primitive|perception|strategy|failure|infra`, non-empty `title` and `applies_when`, `confidence`, and
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

## RoboDojo exploration protocol

Explore the fixed task, seed group and layout. reset restores those same settings. An attempt limit of 0 means no attempt-count limit; other run limits still apply. An unsolved finish is refused while a finite attempt budget remains; reset is refused after native success. Do not request hidden reward details.

Preserve arm roles required by the instruction; reconsider roles chosen for a prior layout.

The final audit {{output_dir}}/{{recipe_tag}}.json includes task, seed, layout, verbatim instruction, attempts, native_success, final_state, memory_files_read and uncertainties. Ordered strategy_notes account for every exported motion and SAM3 call with actual parameters, trace step, phase, observed result and controller-switch reason. The runner exports recorded motion commands and SAM3 calls that found a mask without an error, in invocation order. Finding a mask does not establish correct identity.

Record SAM3 descriptions and how identity was checked, required versus layout-selected arm roles, policy conditioning and observed takeover state, Primitive use and chunk choices. Record observed native action costs for productive phases, failed approaches, recovery and terminal motions using recorded counters; these are evidence, not fixed budgets. The shared finalizer merges valid inbox proposals and updates MEMORY.md after the run.
