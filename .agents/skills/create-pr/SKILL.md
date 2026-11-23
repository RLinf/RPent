---
name: create-pr
description: Open a GitHub pull request for RPent, or fix an existing one — checks the PR title against Conventional Commits, writes a precise description that follows .github/PULL_REQUEST_TEMPLATE.md, and lints both before pushing. Use when asked to create/open/submit a PR, write or rewrite a PR description, or fix a PR title or body.
---

# Create a PR

Write the title and description, then **run the linter before you post**:

```bash
python3 .agents/skills/create-pr/lint_pr.py lint \
    --title "fix(toolkit): retain the original handler error" \
    --body-file /tmp/pr-body.md
```

Exit code 1 means at least one ERROR — fix it and re-run. Point the same checker at an open PR:

```bash
python3 .agents/skills/create-pr/lint_pr.py lint --pr 364
```

The rules come from [CONTRIBUTING.md](../../../CONTRIBUTING.md) ("Step 5. Commit and open a pull request") and [.github/PULL_REQUEST_TEMPLATE.md](../../../.github/PULL_REQUEST_TEMPLATE.md).

## 1. Gather the facts from the diff

Read the diff against `main` rather than relying on memory:

```bash
git fetch origin main
git log --oneline origin/main..HEAD
git diff --stat origin/main...HEAD
```

Record:

- **Which component the diff touches** (`planner`, `toolkit`, `cli`, `dashboard`, `flywheel`, `libero`, `robocasa`, `robotwin`, `robodojo`, `efe`, `docs`, `skills`, etc.) — that becomes the scope.
- **Every user-visible change** — new CLI flags, config keys, tool contracts, moved doc pages, or compatibility changes.
- **Exact automated checks run** — `pre-commit run --all-files`, `pytest tests/unit_tests/... -v`, `sphinx-build -W --keep-going ...`.
- **Manual verification** — GPU, simulator, model-service, or real-robot runs, or `"Not needed"` when offline tests fully cover the change.

## 2. Title

```text
<type>(<scope>): <description>
```

- **type**: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`, `perf`, `build`, `ci`, `revert` (lowercase).
- **scope**: lowercase `[a-z0-9_-]+` naming the touched component (optional; omit rather than using vague scopes like `update` or `new`).
- **description**: **≤ 50 characters** after `: `, starts with a lowercase ASCII letter `a-z`, imperative mood (`add`, `fix`, `support`, never `added` or `fixed`), no trailing period, plain ASCII only, and no trailing `(#123)`.

## 3. Body

Start from the template skeleton so all four headings from [.github/PULL_REQUEST_TEMPLATE.md](../../../.github/PULL_REQUEST_TEMPLATE.md) stay in order:

```bash
python3 .agents/skills/create-pr/lint_pr.py template > /tmp/pr-body.md
```

| Section | What goes in it |
| --- | --- |
| `## Description` | Background, motivation, and main changes (compatibility, migrations, new dependencies, runtime impact). ≤ 250 words, ≤ 12 bullets. Required. |
| `## Testing` | Exact automated commands run and their results. Required unless `--docs-only` is passed. |
| `## Manual verification` | GPU, simulator, model-service, or real-robot verification details, or `Not needed` when offline tests cover the change. |
| `## Checklist` | Tick every applicable box from `.github/PULL_REQUEST_TEMPLATE.md`. Required. |

Style rules enforced by `lint_pr.py`:

- **No uniform leading indentation** on headings, prose, or table rows.
- **Delete `<!-- ... -->` template comments** once filled.
- **Nothing above the first heading** — put introductory text inside `## Description`.
- **Balanced fenced code blocks**.

## 4. Lint, then create or fix

```bash
python3 .agents/skills/create-pr/lint_pr.py lint \
    --title "$TITLE" --body-file /tmp/pr-body.md
gh pr create --base main --title "$TITLE" --body-file /tmp/pr-body.md
```

To repair an existing PR body with paste-indentation or leftover HTML comments:

```bash
python3 .agents/skills/create-pr/lint_pr.py fix --pr 364 --unwrap > /tmp/fixed.md
python3 .agents/skills/create-pr/lint_pr.py lint --title "$TITLE" --body-file /tmp/fixed.md
gh pr edit 364 --body-file /tmp/fixed.md
```
