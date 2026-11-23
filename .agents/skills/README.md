# Project skills

This directory is the single source for RPent's project skills:

- **Codex, Cursor, and Gemini CLI** discover `.agents/skills/` directly, and [`.gemini/settings.json`](../../.gemini/settings.json) instructs Gemini CLI to load [AGENTS.md](../../AGENTS.md) as project context.
- **Claude Code** discovers the same skills through the directory symlink `.claude/skills -> ../.agents/skills`, and `CLAUDE.md` symlinks to [AGENTS.md](../../AGENTS.md).

Edit skills only under `.agents/skills/<name>/` and give each skill a folder whose name matches the `name` in its `SKILL.md`.

## Available skills

| Skill | Purpose | Helper script |
| --- | --- | --- |
| [create-pr](create-pr/SKILL.md) | Write and lint an RPent PR title and description against `CONTRIBUTING.md` and `.github/PULL_REQUEST_TEMPLATE.md` | `python3 .agents/skills/create-pr/lint_pr.py` |
| [review-pr](review-pr/SKILL.md) | Review an RPent PR, re-review its revisions, or review a selected local diff, and lint/post a structured GitHub review | `python3 .agents/skills/review-pr/lint_review.py` |
| [write-doc](write-doc/SKILL.md) | Write or streamline paired EN/ZH Sphinx RST documentation in concise, task-first style | `python3 .agents/skills/docs-check/check_rst_markup.py` |
| [docs-check](docs-check/SKILL.md) | Check RPent documentation, RST inline markup, translations, prompts, and code-to-doc consistency | `python3 .agents/skills/docs-check/check_rst_markup.py` |
| [verify-change](verify-change/SKILL.md) | Select and run unit, contract, docs, and integration checks for a change | — |
| [add-robot](add-robot/SKILL.md) | Add a robot integration or extend its Env/VLA, tool, prompt, configuration, or runtime wiring | — |

## Adding or updating a skill

1. Create or edit `.agents/skills/<name>/SKILL.md` (`.claude/skills/<name>` resolves automatically through the directory symlink).
2. Register new skills in the table above and in [AGENTS.md](../../AGENTS.md).
3. Add offline CPU unit tests under `tests/unit_tests/skills/` for any helper script added under `.agents/skills/`.
