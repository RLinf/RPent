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
5. **Plan the page as one continuous article**
   - Before drafting, outline the reader's starting point and intended outcome.
     For each section, note why it follows the previous one.
   - Use the first prose sentence, after any opening figure, to state what the
     page helps the reader do, understand, choose, or look up. Outline the scope
     and steps on longer pages. A short directory page may only need to name
     the choice it helps the reader make.
   - Open each section by connecting what the reader has learned or done to the
     question it answers. Introduce extensions when they become relevant. For
     example, explain which services RPent starts locally before showing how
     to reuse a service through an external endpoint.
   - For an interface workflow, explain why the example uses the API, then walk
     through each public RPent call in the order the caller makes it. Show the
     key inputs, what the call returns or changes, and how the caller uses the
     result. When lifecycle matters, say who handles setup and cleanup.
6. **Maintain strict EN/ZH parity and clean Chinese RST**
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

## Natural-language gate

After drafting, read each language on its own.

- In **English**, write as one engineer explaining the system to another.
  Name the component and say what it does. A short instruction can stand on
  its own. Give a design choice enough room to explain why it matters.
  Cut stock transitions and sentences that only announce the next section.
- In **Chinese**, reorganize the explanation around natural Chinese logic
  instead of translating sentence by sentence. Use direct, written technical
  language. Avoid conversational filler such as `看看长什么样`, `等需要时再看`,
  and `不用跟着改`, and bureaucratic phrasing such as `本文旨在` and
  `进行相关操作`. Follow the
  [project terminology](../../../docs/STYLE_GUIDE.md#english-and-chinese).
  For other terms, follow usage in guides on the same topic, including familiar
  English terms such as `endpoint` and `worker`.
- In **both languages**, make headings understandable on first reading. In
  concepts, guides, and extension pages, explain a component's role before
  using its code name in a heading. In an API, CLI, or configuration reference,
  the exact identifier can be the heading when readers look it up by name.

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
5. For each language version, first read only the introduction and the opening
   paragraph of each section. Together they should form a logical outline.
   Revise any opening that leaves it unclear why the section comes next.
6. Then read the full page to check that each section leads naturally into the next
   and that explanations follow the steps a reader would take to solve the
   problem. Make sure each example includes the steps and explanations needed
   to follow it and that the page explains new concepts when they first appear.
7. For indexes and reference pages, use a shorter review. Check that the
   introduction tells readers which pages they can reach or what information
   they can look up. The cards, tables, or entries should follow a deliberate
   order that helps readers find what they need.
