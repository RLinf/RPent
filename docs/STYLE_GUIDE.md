# RPent Documentation Style Guide

RPent's documentation rules live in [`.agents/skills/docs/rules/`](../.agents/skills/docs/rules/) so contributors and coding agents read one source of truth split by page type and language.

## Rules

- **Every page**: [base.md](../.agents/skills/docs/rules/base.md) — reader-first framing, simplicity over verbosity, accuracy, formatting, preservation during restructuring, media/dark-mode checks, and bilingual parity.
- **By page type**:
  - Simulator and real-world robot guides (`simulators/`, `real_world_robots/`): [simulator.md](../.agents/skills/docs/rules/simulator.md) (templates: [simulator.rst](../.agents/skills/docs/templates/simulator.rst), [simulator.zh.rst](../.agents/skills/docs/templates/simulator.zh.rst))
  - Quick Start, user guides, toolkits, and demos (`get_started/`, `guides/`, `toolkits/`, `real_world_demos/`): [guide.md](../.agents/skills/docs/rules/guide.md)
  - Concepts and development (`development/`): [development.md](../.agents/skills/docs/rules/development.md)
  - Navigation, sidebar groups, index pages, and redirects (`index.rst`, `conf.py`): [index.md](../.agents/skills/docs/rules/index.md)
  - README highlights (`README.md`, `README.zh-CN.md`): [readme.md](../.agents/skills/docs/rules/readme.md)
- **By language**: [english.md](../.agents/skills/docs/rules/english.md), [chinese.md](../.agents/skills/docs/rules/chinese.md), and [glossary.md](../.agents/skills/docs/rules/glossary.md).
- **Common mistakes and automated checks**: [mistakes.md](../.agents/skills/docs/rules/mistakes.md), [reference.md](../.agents/skills/docs/reference.md), and the [docs skill](../.agents/skills/docs/SKILL.md).

## Build and Verify

Run the docs check before opening a PR; it validates EN/ZH parity, CJK inline markup, development code symbols, writing rules, and builds both languages with Read the Docs' strict warning gate:

```bash
python3 .agents/skills/docs/check.py
```

Or build both trees directly with warnings treated as errors (setup instructions are in [docs/README.md](README.md)):

```bash
sphinx-build -b html -E -N -W --keep-going docs/source-en docs/build/html-en
sphinx-build -b html -E -N -W --keep-going docs/source-zh docs/build/html-zh
```
