---
name: write-doc
description: Write or streamline RPent documentation (paired EN/ZH Sphinx RST under docs/) in a concise, task-first style — clear prerequisites, one copyable main-path command, compact tables instead of prose walls, and strict code/build verification. Use when drafting new docs or slimming down verbose pages.
---

# Write concise task-first documentation

Follow [AGENTS.md](../../../AGENTS.md), [CONTRIBUTING.md](../../../CONTRIBUTING.md),
and [docs/STYLE_GUIDE.md](../../../docs/STYLE_GUIDE.md). Pair with
[docs-check](../docs-check/SKILL.md) to verify code accuracy and bilingual parity.

## Core writing rules

1. **Lead with the task and prerequisites**
   - State in the first 1–2 sentences what task the reader will accomplish, then
     list concrete prerequisites (OS, Python/extras, hardware, weights/assets)
     before the first command.
   - Cut throat-clearing, promotional claims, review history, and narration of
     replaced implementations. Keep negative statements only for real operational
     limits, safety constraints, or compatibility boundaries.
2. **Keep one copyable main path; move variants out of the way**
   - Give one complete, copy-pasteable command sequence from setup to task run.
   - Frame each command with one short sentence on its purpose, then follow it
     with the 3–5 flags/config keys the user actually changes, expected output
     artifacts, and the environment's native success criterion (do not confuse
     the planner's `finish` call with the benchmark's evaluation outcome).
   - Do not restate YAML or CLI blocks in prose. Place optional modes after the
     main path, in compact tabs/tables, or link to their dedicated guide.
3. **Prefer tables, numbered steps, and adjacent callouts over long paragraphs**
   - Use `.. list-table::` for task suites, observation/action contracts, and
     option comparisons; use numbered lists for multi-step procedures.
   - Place `.. note::` (constraints) and `.. warning::` (real operational risks
     such as package conflicts, GPU/device selection, headless rendering, or
     service startup order) directly beside the affected step in 1–3 sentences
     stating the condition and required action.
   - Factor repeated instructions that share the exact same contract into
     underscore-prefixed `_*.rst` includes.
4. **Follow RPent page anatomy**
   - **Directory / index pages**: brief routing intro + cards or table + hidden
     `.. toctree::`; never bare link lists.
   - **Simulator guides**: optional credited figure (`:width: 90%`) + short task
     intro -> `Overview / 概览` (`.. grid:: 2 4 4 4` with `:gutter: 2` in fixed
     order: `Action Models / 动作模型`, `Planners / 规划器`, `Tasks / 任务`,
     `Hardware / 硬件`) -> H3 `Tasks / 任务` and `Observation and Action / 观测与动作`
     tables (`Observation`, `Action`, `Reward / success`, `Task prompt`) ->
     installation & downloads -> model/planner config & one runnable task ->
     result files & native success criterion -> env options/memory/exploration ->
     reproduction protocol -> common failures.
   - **Concepts & development pages**: concrete engineering problem -> minimal
     end-to-end usage example -> core component contract and extension points,
     rather than a file-by-file class inventory.
5. **Maintain strict EN/ZH parity and clean Chinese RST**
   - All user and developer documentation lives as paired Sphinx `.rst` pages
     under `docs/source-en/` and `docs/source-zh/` (do not add standalone `.md`
     guides outside `docs/`). Edit matching EN and ZH paths together, keeping
     headings, section order, commands, flags, config keys, and reported
     numbers/denominators aligned.
   - Write natural Chinese instead of clause-by-clause translation. Keep Chinese
     prose paragraphs on one line or end wrapped source lines with `\` so Sphinx
     does not insert spaces between characters. Leave spaces around inline code
     (e.g. `使用 ``base_camera`` 相机`) and avoid gluing `**...**` directly to
     Chinese characters.

## Workflow and verification

1. Inspect the owning implementation (`rpent/`, `robots/<robot>/`, configs,
   `pyproject.toml`) to confirm every flag, config key, default, and path.
2. Preserve existing leaderboard data, reproduction matrices, anchors, and
   coming-soon entries unless explicitly asked to change them; add redirects in
   both `conf.py` files when moving pages.
3. Build both documentation trees with warnings treated as errors:

```bash
sphinx-build -W --keep-going docs/source-en docs/build/html-en
sphinx-build -W --keep-going docs/source-zh docs/build/html-zh
```

4. Run [docs-check](../docs-check/SKILL.md) to verify rendered Chinese inline
   markup, links, media URLs, and code-to-doc consistency.
