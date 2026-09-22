---
name: docs-check
description: Check RPent documentation, comments, translations, prompts, and visible messages for accuracy, natural language, and consistency with the implementation. Use for prose review or documentation updates accompanying behavior changes.
---

# Check documentation and prose

Follow [AGENTS.md](../../../AGENTS.md) and the documentation requirements in
[CONTRIBUTING.md](../../../CONTRIBUTING.md). Establish the changed files and
intended audience, then read their owning implementation and nearby examples.

## Check meaning and coverage

- Verify commands, flags, configuration keys, defaults, paths, API names, and
  capability claims against the relevant revision and actual consumers. A
  successful build does not prove these claims. For public behavior changes,
  identify the affected guides, API docstrings, examples, and README entries.
- Read the paired pages under `docs/source-en/` and `docs/source-zh/` when both
  cover the feature, even if only one changed. Preserve equivalent behavior,
  commands, prerequisites, limitations, and reported results in both languages.
- Translate meaning into natural Chinese using nearby terminology. Preserve
  identifiers, commands, and proper names; choose context-appropriate technical
  terms instead of literal substitutions. Do not force identical sentence
  structure across languages.
- Describe current behavior and useful non-obvious constraints. Remove review
  exchanges, authoring notes, redundant implementation narration, and obsolete
  comparisons while retaining factual qualifications and necessary rationale.
  Historical explanations belong in migration or change records when needed.
- Match neighboring pages for headings, links, code blocks, and examples.
  Keep comments near their owning contract and avoid repeating the same rule
  across several files.
- Inspect prompts and tool descriptions as model inputs: check tool names,
  placeholders, actual rendered context, conflicting rules, and unnecessary
  repetition. Preserve behavioral instructions and distinguish prose cleanup
  from changes that require runtime evidence.

## Use existing validation

Use [docs/Makefile](../../../docs/Makefile) for local targets. See the
[English](../../../docs/source-en/.readthedocs.yaml) and
[Chinese](../../../docs/source-zh/.readthedocs.yaml) Read the Docs configurations
for CI build settings. From the repository root, with
[documentation dependencies](../../../docs/requirements.txt) installed:

```bash
sphinx-build -W --keep-going docs/source-en docs/build/html-en
sphinx-build -W --keep-going docs/source-zh docs/build/html-zh
```

Run these when the change affects Sphinx pages or their build inputs. Standalone
Markdown instructions and source comments need path/link and semantic checks;
the Sphinx build does not validate them. Reuse relevant prompt/rendering tests
when model-visible behavior changes, following
[verify-change](../verify-change/SKILL.md).

## Report

Give exact locations, the implementation or terminology evidence, and concrete
replacement wording in the affected language. Name any counterpart page that
needs updating. Report checks and limitations, and keep review-only requests
read-only unless the user also requests edits.
