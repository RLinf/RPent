# Rules for every page

Every RPent docs page follows this file, whatever its type. Before writing, read this file, the page type's file below, and the language file for each language you write ([english.md](english.md), [chinese.md](chinese.md)). Before finishing, go through [mistakes.md](mistakes.md).

| Page type | Where it lives | Rules |
|---|---|---|
| Simulator & real-world robot guide | `simulators/`, `real_world_robots/` | [simulator.md](simulator.md) |
| Guide, Toolkit & Quick Start | `get_started/`, `guides/`, `toolkits/`, `real_world_demos/` | [guide.md](guide.md) |
| Concepts & Development | `development/` | [development.md](development.md) |
| Index & navigation | every `index.rst` | [index.md](index.md) |
| README highlight | `README.md`, `README.zh-CN.md` | [readme.md](readme.md) |

Leaderboard (`leaderboard/`) and Resources (`resources/`) pages follow this file and the language files; preserve published benchmark numbers, task/seed matrices, and coming-soon entries unless explicitly asked to change them.

## Write about what the reader does, not about the code

The reader of a user-facing page is an engineer running tasks with RPent who has not opened its source code.

- Describe what the reader runs, configures, and observes: CLI flags, install extras, environment variables, saved artifacts, and the environment's native success criterion.
- Name a piece of code only when the reader types, runs, opens, or edits it: commands, CLI flags, config keys, paths, and—on Development pages—the public classes and hooks a contributor implements or calls.
- Never explain user-facing behavior by touring `rpent/`, `robots/`, or `components/` modules.
- Rewrite code-tour sentences around observable behavior:
  - Before: "`rpent/cli/main.py` calls `RobotSpec.init_runtime`, which spawns `ProcessDaemon` servers and passes handles to `Toolkit`."
  - After: "Running `rpent --robot libero` starts the environment, VLA, and perception services, executes tool calls from the planner, and saves step images and run status under `--output-dir`."

## Lead with the purpose

- The first prose sentence (after any opening figure) states what the page lets the reader do, understand, choose, or look up.
- State prerequisites (OS, Python version, install extra, GPU/hardware, assets, checkpoints) before the first command.
- Describe current behavior and real operational constraints. Leave out review exchanges, history of replaced implementations ("previously", "the fixed installer no longer..."), and sentences that only say what is not needed. Keep negatives only when they state a real limit, failure condition, safety rule, or compatibility boundary.

## Prefer simplicity and readability over verbosity

When completeness and readability pull in opposite directions, trim the main path rather than piling prose onto the page.

- **Write in plain, readable language.** Explain ideas in concrete words a peer understands on the first read.
- **Apply the delete test.** Every section, paragraph, table row, and callout must earn its space. Push optional variants and full parameter tables to dedicated guides or Command-Line Reference (`guides/cli.rst`).
- **Say each thing once.** State a fact or flag where the reader acts on it; do not repeat it across the intro, command prose, and tables. Avoid negative-contrast framing ("this is not A, not B, but C") and defensive caveats.
- **Write for scanning.** Keep paragraphs short (1–3 sentences on simulator and user guides), use `.. list-table::` for specs/comparisons, numbered lists for procedures, and adjacent `.. note::` / `.. warning::` callouts in 1–3 sentences.
- **Keep one copyable main path.** Give one complete command sequence from setup to result; put alternative modes afterwards or link to their dedicated guide.

## Be accurate

- Verify every command, CLI flag, config key, install extra, default, path, and success signal against `rpent/`, `robots/`, and `pyproject.toml`.
- Keep bounded functional completion (`finish`) distinct from the environment's native success criterion (`terminated` or benchmark verifier reward).
- Update all affected EN/ZH pages, docstrings, and README entries when public behavior changes.
- Use one term per concept ([glossary.md](glossary.md)).

## Structure and formatting

- One visible H1 per page (`=` for page title, `-` for H2 sections, `~` for H3 subsections). Use Title Case for English headings. Never skip a level.
- Headings name tasks or ideas the reader understands; reserve raw identifiers for CLI/API reference sections.
- Frame each command with one sentence on what it does, followed by the few flags the reader actually changes and the expected output or artifact that confirms success. Write commands without a `$` prompt and explain `<placeholders>`.
- Link inside `docs/` with `:doc:` and `:ref:`, never hardcoded `rpent.readthedocs.io` URLs. Factor repeated blocks that share the exact same contract into underscore-prefixed `_*.rst` partials.
- Keep prose and cards left-aligned using the shared CSS; do not style individual paragraphs or add duplicate raw-HTML titles. Use direct HTTP 200 media URLs with alt text, captions, and video controls, avoid replacing ordinary text with screenshots, and check transparency/contrast in both light and dark themes (no forced white background on transparent logos).
- When moving or renaming a page, preserve useful anchors and add old-to-new URL redirects in both `docs/source-en/conf.py` and `docs/source-zh/conf.py`.

## Preserve information during restructuring

- Track where important commands, dependency options, parameter defaults, outputs, tables, figures, videos, and constraints move; brevity alone is not a reason to delete them.
- Preserve task/seed coverage, denominators, failure and timeout handling, aggregation, model settings, resource revisions, and timing boundaries alongside every reported result — moving a table must not detach its qualifications.
- Preserve existing coming-soon integrations and clearly state their documented status; explain deliberate removals in the change description.

## Two languages

- Every docs page lives at the same relative path under `docs/source-en/` and `docs/source-zh/`. Edit both in the same pass, preserving sections, commands, flags, numbers, and links.
- Write each language natively from the page plan; never machine-translate or translate clause by clause into Chinese.
- Never translate identifiers: CLI flags, config keys, robot/planner names, paths, or filenames.
