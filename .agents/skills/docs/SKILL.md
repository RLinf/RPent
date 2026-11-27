---
name: docs
description: "RPent documentation in docs/source-en, docs/source-zh, README.md, README.zh-CN.md, CONTRIBUTING.md, and CONTRIBUTING_zh.md: write, edit, or check a page in English and Chinese. Holds RPent's writing rules split by page type (simulator, guide, development, index, readme) and by language, common mistakes, templates, and the automated check for Sphinx EN/ZH builds, CJK inline markup, code-font symbols, EN/ZH parity, and writing style. Use before creating or editing any docs page, README, or user/model-facing prose, and to check docs."
paths:
  - docs/**
  - README.md
  - README.zh-CN.md
  - CONTRIBUTING.md
  - CONTRIBUTING_zh.md
---

# Docs

RPent's documentation rules live in [rules/](rules/) (indexed from [docs/STYLE_GUIDE.md](../../../docs/STYLE_GUIDE.md)), split so a writer reads only what the page in hand needs: the rules for every page, the rules for its type, and the rules for each language. This skill says what to read, how to write a page, and how to check it.

## Write

1. **Classify the page** by where it lives ([rules/base.md](rules/base.md) has the table), and open a current page of the same type in the same section, or its template in [templates/](templates/).
2. **Read** [rules/base.md](rules/base.md), the file for the page type, [rules/english.md](rules/english.md) and [rules/chinese.md](rules/chinese.md) for the languages you write, and [rules/glossary.md](rules/glossary.md) for terms.
3. **Check the facts** the page will state against the code (`rpent/`, `robots/`, `pyproject.toml`): every command, CLI flag, config key, install extra, default, path, and success criterion.
4. **Plan the page**: its type, its reader, the result the reader reaches, and one line per section saying what it answers.
5. **Write the English and the Chinese page from the plan**, each in its own language. Write the Chinese natively from the plan rather than translating English sentences one by one.
6. **Place the page** in the matching `:hidden:` toctree in `docs/source-en/index.rst` and `docs/source-zh/index.rst`, plus any directory routing table without adding a duplicate toctree ([rules/index.md](rules/index.md)). When moving or renaming a page, add the old-to-new redirect to both `conf.py` files. Add a README highlight for a major user-visible feature ([rules/readme.md](rules/readme.md)).
7. **Go through [rules/mistakes.md](rules/mistakes.md).**
8. **Run a simplicity and language pass.** Give a fresh agent, which has seen nothing else, only the changed pages, [rules/base.md](rules/base.md), and the language file for each. Ask it first to cut anything the reader does not need for the main path (long-winded explanations, redundant paragraphs, repeated facts, defensive caveats, internal code citations, complex shell glue, and repeated negative-contrast framing such as "this is not A, not B, but C"), and then to rewrite any remaining sentence that reads as translated, padded, or built on code names. Apply the cuts and rewrites that hold.
9. **Run the check** below and fix what it reports.

### A simulator guide

- Write `docs/source-en/rst_source/simulators/<name>.rst` and the matching path under `docs/source-zh/` from [templates/simulator.rst](templates/simulator.rst) and [templates/simulator.zh.rst](templates/simulator.zh.rst), following [rules/simulator.md](rules/simulator.md).
- Register the page in the matching `:hidden:` toctree in `docs/source-en/index.rst` and `docs/source-zh/index.rst`, add a row in `simulators/index.rst` in both languages (without adding another toctree), and add the feature to `README.md` and `README.zh-CN.md` ([rules/readme.md](rules/readme.md)).
- Keep existing leaderboard entries, reproduction conditions, and coming-soon placeholders intact unless explicitly asked to change them.

## Check

```bash
python3 .agents/skills/docs/check.py              # docs changed against main
python3 .agents/skills/docs/check.py FILE ...     # specific pages
python3 .agents/skills/docs/check.py --no-build   # skip the Sphinx build while iterating
```

Page findings count only on the lines the change adds, because older pages may predate current rules; `--all-lines` reports every line of the selected pages. The build always covers both trees (`docs/source-en` and `docs/source-zh`) using `docs/requirements.txt` (`$DOCS_VENV`, `<repo>/.docs-venv`, or the active Python when Sphinx and extensions are already installed). Findings use the shared format of [../README.md](../README.md#findings):

| Rule | Severity | Reports |
|---|---|---|
| `docs/build` | critical | a Sphinx warning in either language; both Read the Docs projects build with `fail_on_warning: true`, independently ([reference.md](reference.md#the-build-gate)) |
| `docs/markup` | critical or major | inline markup next to CJK text that breaks the build (critical) or silently mis-renders (major) ([reference.md](reference.md#inline-markup-next-to-cjk-punctuation)) |
| `docs/symbol` | major | a name in code font, on a Concepts & Development page, that nothing in the repository defines |
| `docs/code-name` | major | internal `rpent/`, `robots/`, or `components/` classes, functions, Sphinx API roles, or source paths on a simulator, toolkit, or user guide page |
| `docs/structure` | major | a new simulator page missing required sections (`Overview`, `Tasks`, `Observation and Action`, `Installation [and Resources]`, `Run a Task`, `View Results`) |
| `docs/rtd-link` | major | a hardcoded `rpent.readthedocs.io` link inside `docs/` instead of `:doc:` or `:ref:` |
| `docs/parity` | major or minor | a page without its counterpart, or toctrees that differ (major); a page, README, or CONTRIBUTING changed while its counterpart did not (minor) |
| `docs/heading` | minor | a code identifier in a heading outside Development and CLI reference |
| `docs/en-style` | minor | a filler or hype word/phrase that [rules/english.md](rules/english.md) cuts |
| `docs/zh-style` | minor | 您, 「」, 值得注意的是, 进行, 当……时, 不是/并非……而是……, half-width punctuation in a Chinese sentence, or missing spaces around English/digits ([rules/chinese.md](rules/chinese.md)) |
| `docs/zh-hard-wrap` | minor | unescaped hard-wrapped Chinese prose inside a paragraph or list item, which renders as a visible space |

Then read every changed page in full, in both languages, as a reader new to RPent who has not seen its code, and inspect every command block:

- [ ] The checklist in the page type's rules file holds.
- [ ] **Readability and conciseness**: apply [rules/base.md](rules/base.md#prefer-simplicity-and-readability-over-verbosity). Flag stiff prose, machine-translated or clause-by-clause Chinese, long-winded explanations, internal code citations or developer internals leaking into user pages / duplicating `development/`, repeated negative-contrast framing, and pages bloatier than a clean `main` sibling.
- [ ] **Code examples**: snippets and commands are complete, copyable, and runnable as written — no placeholder domains (`example.com`), fictional flags/paths, or unframed pseudo-code mixed into runnable steps (illustrative sketches on `development/` are labeled as such).
- [ ] **Commands and sibling alignment**: one clean copyable main path; commands match actual CLI flags, install extras, and defaults; simulator guides distinguish planner `finish` from native `terminated`.
- [ ] **Cross-references**: links inside `docs/` use `:doc:` / `:ref:` (no hardcoded `rpent.readthedocs.io` URLs), related guides link instead of duplicating shared setup or CLI tables, and moved/renamed pages have redirects in both `conf.py` files.
- [ ] **Docs ↔ code accuracy**: every command, flag, env var, install extra, path, and capability claim matches `main` plus the change; public additions, renames, or removals update all affected EN/ZH pages and prompts/docstrings.
- [ ] **Parity, assets, and README**: paired EN/ZH pages agree on sections, commands, keys, claims, and numbers; media URLs return HTTP 200 directly ([reference.md § Asset and media references](reference.md#asset-and-media-references)); noteworthy features get a README highlight ([rules/readme.md](rules/readme.md)).
- [ ] None of the mistakes in [rules/mistakes.md](rules/mistakes.md) remains.

Report each changed page with its type and result (pass or findings). Rate a page `major` when it fails any bullet above as a whole — for example, whole-page machine-translated or literal Chinese, developer internals leaking into a user guide or duplicating `development/`, non-runnable placeholder examples (such as `example.com`), missing prerequisites or broken main-path commands, or EN/ZH content drift. Rate isolated wording or phrasing tweaks `minor`. Quote the line, name the rule, and give the rewrite.
