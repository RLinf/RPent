# RPent Documentation Style Guide

Use this guide for English and Chinese documentation, examples, comments, and
user-facing messages. Verify behavior against the code being documented. Build
instructions are in [README.md](README.md).

This guide adapts the writing and presentation principles in
[RLinf's documentation guide](https://github.com/RLinf/RLinf/blob/010990e5c81ae17d141d3eab584a3bd1791df950/docs/STYLE_GUIDE.md)
to RPent's task execution, memory, and robot workflows. RPent's navigation and
page boundaries are defined below.

## Write for someone running their first task

Open with what the reader will accomplish. State the prerequisites before the
first command. Explain each command's purpose, then show the expected result
and where to look if it fails. Keep one complete, copyable path through a
tutorial; place alternatives after that path or link to their own guide.

Use short, direct sentences and familiar technical terms. Explain a concept
before introducing its internal identifier. Keep implementation details in
developer pages unless they affect a user's choice or operation.

Keep explanations connected. Introduce each section with its purpose before
showing a command, table, or interface. Explain the expected output and how it
informs the next step. Concepts pages should explain how components cooperate,
not just enumerate classes and methods. Concision must preserve prerequisites,
conditions, exceptions, ownership, and the reasons a reader needs to act.

Describe current behavior and necessary constraints. Do not narrate a review
exchange or the process of replacing one implementation with another.

| Avoid | Prefer |
| --- | --- |
| The corrected installer no longer treats a nonempty directory as complete. | `--skip-existing` checks the downloaded file inventory. |
| This isn't just a few more motions; it unlocks endless evolution. | The planner checks observations after each tool call and selects the next action. |
| 修正后的安装器不再覆盖旧资源。 | 内容不同的已有资源会保留；使用 `--overwrite` 可替换安装范围内的资源。 |

Keep negative statements when they explain a real limit, failure condition,
safety requirement, or compatibility boundary. Put migration history and
version-specific changes in release or migration notes.

## Navigation and page ownership

Keep the same sections and ordering in both languages:

1. **Get Started / 开始使用**: Introduction to RPent and Quick Start.
2. **Leaderboard / 排行榜**: the existing performance and time/token pages.
3. **Real-World Demos / 真实世界演示**: direct robot demo pages, each linking to
   the corresponding installation and usage guide.
4. **Guides / 使用指南**: memory and exploration; action primitives and tools;
   planners and model services; CLI and configuration; Dashboard; Flash Mode;
   trajectory collection and data flywheel; remote services and parallel runs.
5. **Simulators / 仿真环境**: LIBERO, RoboCasa365, and RoboTwin. Each page owns
   its installation, first run, reproduction protocol, and troubleshooting.
6. **Real-World Robots / 真实机器人**: Single-Arm Franka, Dual-Arm Franka, YAM,
   and SO-101. Preserve coming-soon pages and describe their actual status.
7. **Concepts & Development / 原理与开发**: architecture, interfaces, memory
   design, and extension guides for environments, tools, and planners.
8. **Resources / 项目资源**: Harness VLA, contributing, and release notes.

The home page is a short introduction with task-oriented links. Installation
and the first LIBERO-PRO task form one Quick Start. Keep reproduction inside
each environment page; logs, success criteria, and common failures belong
next to the procedure they explain. Keep the leaderboard's published data and
two-page organization intact when reorganizing prose.

Preserve important commands, reported data, figures, and their experimental
conditions when moving content. Record their destination during a restructure;
brevity alone is not a reason to delete them. Verify intentional corrections
against the source and document the reason in the change description.

Use short sidebar labels and descriptive page headings. Keep existing page
URLs where practical. If content moves, preserve useful anchors and provide a
clear route from the old page. Do not delete an integration or placeholder
solely because it is incomplete.

## Page structure

### Home and directory pages

Use a centered logo and heading on the home page, followed by a short
introduction and task-oriented cards. Keep ordinary page text left-aligned.
Directory pages introduce the reader's choice and route to child pages with
cards or a table containing a short description for each entry. Do not use
bare link lists as the visible directory. Use hidden toctrees where a page
owns navigation; compatibility entry pages can link to pages already owned
by the main navigation without adding another toctree.

Use the shared sidebar, font stack, and spacing. Keep equivalent navigation
labels consistent in both languages. Check short pages as well as long ones:
removing a local table of contents must not leave an empty column before the
article. Site features such as an AI assistant or live repository star count
are separate from documentation content requirements.

### Quick Start

Use this sequence: requirements, installation, assets and checkpoints, planner
connection check, one task, expected artifacts and success criterion, common
failures. Keep exhaustive option tables in CLI and Configuration.

### Simulator guides

When an official overview figure helps explain the environment, place it below
the page title, followed by a short explanation of the task the reader can run
with RPent. Follow the RLinf LIBERO and RoboTwin pages: center the figure, use
`:width: 90%`, preserve its aspect ratio, and credit its source in the caption.
Use an image of the documented environment version. Figures are optional;
do not add a decorative image just to fill this position. Describe upstream
results within a figure as upstream results, not RPent experiment evidence.
Give the image descriptive alt text and keep section anchors outside the
figure directive. Check the rendered image, caption, and following heading in
both languages.

Start with **Overview / 概览**. Use four `sphinx-design` cards with the same
titles and order across simulator pages: **Action Models / 动作模型**,
**Planners / 规划器**, **Tasks / 任务**, **Hardware / 硬件**. Use
`.. grid:: 2 4 4 4` with `:gutter: 2` and brief, left-aligned descriptions.
List supported configurations, and link special modes to their prerequisites.
RPent coordinates task execution, so these cards describe planners rather
than RL training algorithms.

Under Overview, add **Tasks / 任务** and **Observation and Action / 观测与动作**
as H3 sections. Use tables for both. The task table identifies suites or task
categories and their scope; keep detailed task/seed matrices with reproduction.
The interface table uses the same rows: **Observation / 观测**, **Action / 动作**,
**Reward / success / 奖励与成功判定**, and **Task prompt / 任务指令**. Distinguish
what the planner sees from model inputs or native control commands where it
matters. State the native success signal without inventing a training reward
or treating the planner's `finish` response as the evaluation outcome.

After Overview, cover:

1. System and Python requirements, installation, and resource downloads.
2. Model and planner configuration, with one concrete runnable task.
3. Result files and the environment's native success criterion.
4. Environment-specific options, memory, and exploration.
5. Reproduction: code version, resource revisions, task/seed matrix, model
   settings, limits, retry policy, and aggregation method where available.
6. Common failures and relevant diagnostics.

Use the same headings for sections with the same purpose:

| English | Chinese |
| --- | --- |
| Overview | 概览 |
| Tasks | 任务 |
| Observation and Action | 观测与动作 |
| Installation and Resources | 安装与资源准备 |
| Run a Task | 运行一个任务 |
| View Results | 查看结果 |
| Task Memory | 任务记忆 |
| Exploration Mode | 探索模式 |
| Experiment Reproduction | 实验复现 |

Keep installation and downloads before Run a Task, followed immediately by
View Results. Put memory, exploration, reproduction, diagnostics, and developer
details after this first-run path. Keep the same order for shared sections;
preserve environment-specific steps and their prerequisites.

Keep headings accurate to their contents. A section covering only package
installation can remain **Installation / 安装** when downloads have separate
sections. A combined memory and exploration section can use **Task Memory and
Exploration Mode / 任务记忆与探索模式**. Add a benchmark qualifier when useful, such as **Experiment Reproduction
(Target50) / 实验复现（Target50）**. Keep published results and
environment-specific diagnostics where they help the reader; do not add
empty sections to match another page. Preserve existing named anchors when
renaming headings.

Use links to shared guides instead of repeating generic planner, memory, or
Dashboard instructions. Preserve environment-specific requirements. Never
present a single successful task as validation of a complete benchmark.

State software and accelerator requirements from the supported setup. Report
verified GPU models and memory usage only when there is evidence; distinguish
measurements from estimates or minimum requirements. Add a separate hardware
backend subsection only when its installation, device options, rendering, run
command, and limitations are documented and supported. Do not copy another
project's accelerator claims.

### Real-world robots and demos

An introductory photo should clearly show the robot's body, gripper, or
workbench layout. Choose a view where the robot occupies enough of the image
to be recognizable; avoid video posters dominated by subtitles or task objects.
Prefer photos from RPent demonstrations or the linked upstream deployment
guide, or clear product photos from the robot manufacturer. Place the figure
below the title with alt text and a short descriptive caption. Keep source URLs
in the RST source; visible source credits are not required unless the asset's
terms require attribution. Describe product photos as hardware illustrations,
without implying they show an RPent deployment.

Use `:figclass: rpent-robot-figure` for introductory robot figures. The shared
stylesheet provides a centered 600 px maximum width, a 4:3 image area, and
consistent caption spacing; figures shrink to fit narrow screens. Choose
images with comparable framing and a clearly visible robot, rather than
mixing full robot views with close-ups of individual parts. Use images at
least as wide as the display area, preserve their proportions, and keep the
base and gripper visible. Adjust framing only to remove empty background;
do not crop out hardware to fill the frame. Verify the result on desktop
and narrow screens. Keep pending documentation clearly marked even when
a photo is available.

Deployment pages cover hardware and controller requirements, calibration,
configuration, motion checks, task execution, result inspection, and stopping.
Keep operator responsibilities and physical safety constraints explicit.

Demo pages explain the visible task, embed the video with a useful caption,
and link to that robot's deployment page. If deployment instructions are
pending, say so at the link. A video is not an installation guide or evidence
that every published workflow is reproducible.

### Guides and developer pages

User guides explain a task with prerequisites and examples. Developer pages
explain ownership, interfaces, call flow, and extension points. Verify
signatures, argument types, defaults, lifecycle, and actual callers. Keep
illustrative code clearly distinct from executable examples.

Use CLI and Configuration as the shared reference for optional dependencies,
parameter defaults, environment variables, and output files. Link to it from
tutorials. Specify which environment or mode owns a default. Keep internal
storage formats separate from supported inspection tools and public APIs.

## English and Chinese

Keep matching page paths, navigation, commands, prerequisites, limits, and
results. Translate meaning into natural Chinese; do not copy English sentence
structure. Preserve code identifiers, model names, flags, and filenames.

| English | Chinese usage |
| --- | --- |
| planner | 规划器; keep `--planner` unchanged |
| memory | 记忆; use “任务经验” when describing its contents |
| exploration | 探索模式 |
| recipe | 动作序列 or 任务策略, depending on the file's contents; keep `_recipe.jsonl` unchanged |
| checkpoint | 模型权重 or checkpoint, depending on context |
| rollout / episode | 一次任务运行 / 一次尝试, when that is the intended meaning |
| success criterion | 成功判定 |
| Real-World Robots | 真实机器人 |
| Simulators | 仿真环境 |

Use Chinese punctuation in Chinese prose and spaces between Chinese text and
Latin identifiers. Keep each Chinese prose paragraph or list-item paragraph
on one source line so line wrapping does not insert spaces between Chinese
characters. Preserve indentation and line structure in directives, tables,
code, and raw HTML. English prose may wrap at normal source line lengths.

RST inline markup needs boundaries around Chinese text. Write
`使用 ``base_camera`` 相机` rather than gluing backticks to adjacent Chinese
words or punctuation. For emphasis, prefer plain Chinese or quotation marks
when `**...**` would touch Chinese characters. Check the generated HTML for
literal asterisks, backticks, and code spans that accidentally include prose.
Use Chinese punctuation in prose while preserving exact command syntax.

Use “RPent 简介” as the introduction title. Use “RoboCasa365” for the documented
simulator version, while keeping the CLI identifier `robocasa` unchanged.

## Formatting and media

- Give each page one visible H1. Use `=` for the page title, `-` for sections,
  and `~` for subsections. Use title case for English headings.
- Keep prose and cards left-aligned. Use the shared CSS for font stacks,
  spacing, content width, and media. Do not style individual paragraphs or
  add duplicate raw-HTML titles.
- Use `:doc:` and `:ref:` for internal links. Keep hidden toctrees on the home
  page for sidebar organization.
- Use tables for comparisons, numbered lists for procedures, and code blocks
  for commands. Introduce examples and explain what their output means.
- Use `.. note::` for a relevant constraint and `.. warning::` for an actual
  operational risk, such as incompatible packages, device selection, or robot
  startup order. Put the warning beside the affected step and explain the
  condition and the required action. Keep optional details out of the main
  installation path.
- Put reusable RST fragments in underscore-prefixed files excluded from page
  discovery. Share repeated instructions when they have the same contract;
  do not merge instructions with different environment prerequisites.
- Use direct media URLs, meaningful alt text, captions, and video controls.
  Check media links and both light and dark themes. Keep media responsive and
  avoid replacing ordinary text with screenshots.

Verify image, poster, and video URLs return HTTP 200 before handoff. Inspect
new figures and confirm video playback in the rendered page. Preserve useful
alt text, visible captions, and source credits. Report blocked checks instead
of treating an unchanged URL or a successful Sphinx build as availability
evidence. Check transparency and contrast in dark mode; avoid a forced white
background on transparent logos.

## Preserve information during restructuring

Compare against an explicit Git revision and the pre-edit working tree. Track
where important commands, dependency options, parameter defaults, outputs,
tables, figures, videos, and constraints move. Explain deliberate removals in
the change description and verify replacements against the current code.

Compare tables by their labels and values, not only their count. Preserve
task/seed coverage, denominators, failure and timeout handling, aggregation,
model settings, resource revisions, and timing boundaries with every reported
result. Moving a table must not detach its qualifications. A simplified quick
start should link to complete reference information elsewhere.

Retain useful page URLs and section anchors when renaming or moving content.
Check their destinations in generated HTML, including cross-language links.
Preserve coming-soon integrations and clearly state their documented status.

## Verify before handing off

Read every changed paragraph in both languages. Check commands, paths,
defaults, support claims, and result definitions against their code consumers.
Check that required content and existing placeholders remain reachable, that
reproduction data is unchanged, and that demo links reach the right robot.

Build both languages with warnings treated as errors:

```bash
sphinx-build -W --keep-going docs/source-en docs/build/html-en
sphinx-build -W --keep-going docs/source-zh docs/build/html-zh
```

Inspect rendered pages on desktop and narrow screens, including dark mode,
sidebar order, headings, tables, code, media, and the language switcher.
Check command blocks for shell syntax and bilingual consistency. Review all
changed prose manually: matching commands do not establish semantic parity.
Inspect index presentation, section introductions, and Chinese inline markup
in addition to build warnings. Compare preserved content with the recorded
baseline after the final edit, including leaderboard files and media URLs.
Report static checks, builds, and actual service or robot runs separately.
A clean documentation build does not establish that a command completed a
real task.
