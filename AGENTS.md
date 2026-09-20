# Working on RPent

Follow [CONTRIBUTING.md](CONTRIBUTING.md) for development, testing, and pull
requests. Use the implementation and its callers to verify behavior; report
disagreements with documentation instead of assuming either is correct.

## Repository map

- `rpent/planner/`, `rpent/session/`, and `rpent/prompt/`: planner adapters,
  execution sessions, and shared prompt construction.
- `rpent/tools/` and `rpent/memory/`: tool execution, observations, and memory.
- `rpent/robots/`: robot discovery, descriptors, shared runtime, and Env/VLA
  components. `robots/<robot>/` owns each concrete integration.
- `rpent/cli/`, `rpent/dashboard/`, and `rpent/evaluation/`: entry points,
  interactive execution, and run results.
- `tests/`: offline tests and integration checks; see [tests/README.md](tests/README.md).
- `docs/source-en/` and `docs/source-zh/`: paired user and developer documentation.

## Development principles

- Confirm the working branch and relevant upstream state before editing. Read
  the closest current implementation and both sides of changed interfaces.
  Preserve unrelated working-tree changes.
- Reuse existing configuration, registration, RPC, runtime, and artifact
  helpers. Prefer RLinf's supported Env/VLA implementations with thin RPent
  adapters; keep robot-specific behavior in its owning package.
- Preserve existing contracts. Base compatibility paths and new abstractions
  on actual consumers or supported data, and explain intentional behavior
  changes and migrations in the PR.
- Validate external inputs at the boundary that owns the operation. Keep
  internal control flow direct, avoid repeated checks of established state,
  and preserve useful errors. Make resource ownership and cleanup explicit.
- Keep optional simulator/model imports at their use sites so basic imports,
  robot discovery, and CLI help work without every robot extra installed.
- Follow neighboring code for naming and structure, use the project logger,
  and follow CONTRIBUTING's type annotation and docstring conventions.
- Write prose about current behavior and non-obvious constraints. Keep review
  exchanges and implementation history in PR discussions or migration notes.
  Translate meaning into natural Chinese and preserve technical identifiers.
- Keep prompts and tool descriptions focused on the model's task. Remove
  repetition without dropping required behavior or operational constraints.
- Test observable behavior and concrete regressions, reusing existing cases
  and fixtures where possible. Avoid tests that merely restate implementation
  details. Never weaken failing tests to make a change pass.

## Task skills

Read the relevant skill when the task calls for it; ordinary edits do not
require loading every skill.

| Task | Skill |
| --- | --- |
| Explain and review a PR, or review an explicitly selected local diff | [review-pr](.agents/skills/review-pr/SKILL.md) |
| Check documentation, comments, translations, or user/model-facing prose | [docs-check](.agents/skills/docs-check/SKILL.md) |
| Select and run checks for a change, or prepare validation evidence | [verify-change](.agents/skills/verify-change/SKILL.md) |
| Add a robot integration or extend its Env/VLA/tool/runtime wiring | [add-robot](.agents/skills/add-robot/SKILL.md) |

Reuse repository tools and the commands documented in CONTRIBUTING and the
test guide. Report what ran, its result, and what remains unverified; keep
integration evidence distinct from benchmark task success.

Skill sources live in `.agents/skills/`. `.claude/skills` links to that whole
directory, and `CLAUDE.md` links to this file. Edit the canonical files so both
clients share the same instructions. Keep each skill self-contained in its
`SKILL.md` and link to existing repository resources.
