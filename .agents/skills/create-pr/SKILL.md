---
name: create-pr
description: Open a GitHub pull request for RPent, or fix an existing one — runs area checks, checks the PR title against Conventional Commits, writes a precise description that follows .github/PULL_REQUEST_TEMPLATE.md, and lints both before pushing. Use when asked to create/open/submit a PR, write or rewrite a PR description, or fix a PR title or body.
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

The rules come from [CONTRIBUTING.md](../../../CONTRIBUTING.md) ("Step 5. Commit and open a pull request") and [.github/PULL_REQUEST_TEMPLATE.md](../../../.github/PULL_REQUEST_TEMPLATE.md). All paths below are relative to the repository root.

## 1. Gather the facts before writing a word

Read the actual diff against `main` instead of relying on memory:

```bash
git fetch origin main
git log --oneline origin/main..HEAD
git diff --stat origin/main...HEAD
```

Record concretely:

- **Which component the diff touches** (`planner`, `toolkit`, `cli`, `dashboard`, `flywheel`, `libero`, `robocasa`, `robotwin`, `robodojo`, `robolab`, `efe`, `franka`, `yam`, `docs`, `skills`, etc.) — that becomes the scope.
- **Every user-visible change** — new CLI flags, config keys, install extras, tool contracts, moved doc pages, or compatibility changes. Leave internal churn out of the summary.
- **Exact automated checks run** — `pre-commit run --all-files`, `pytest tests/unit_tests/... -v`, `python3 .agents/skills/docs/check.py`. "Tested locally" is not an answer.
- **Manual verification** — GPU, simulator, model-service, or real-robot runs with command and result, or `Not needed` when offline tests cover the change.
- **The issue it closes**, if any (`Fixes #123` or `Refs #123`).

Then run the check of every area the branch touches, and fix every critical or major finding or explain in the description why it stays:

```bash
python3 .agents/skills/route.py check
```

Record its result in `## Testing` alongside the unit tests and pre-commit commands you ran.

## 2. Title

```text
<type>(<scope>): <description>
```

- **type**: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`, `perf`, `build`, `ci`, `revert` (lowercase).
- **scope**: lowercase `[a-z0-9_-]+` naming the touched component. Optional; omit rather than using vague scopes like `update` or `new`.
- **description**: **≤ 72 characters** after `: `, starts with a lowercase ASCII letter `a-z`, imperative mood (`add`, `fix`, `support`, never `added` or `fixed`), no trailing period, plain ASCII only, and no trailing `(#123)` (GitHub appends the PR number on squash-merge).

Check just the title on its own:

```bash
python3 .agents/skills/create-pr/lint_pr.py lint --title "feat(planner): add custom adapter"
```

## 3. Body

Start from the template skeleton so all four headings from [.github/PULL_REQUEST_TEMPLATE.md](../../../.github/PULL_REQUEST_TEMPLATE.md) stay in order:

```bash
python3 .agents/skills/create-pr/lint_pr.py template > /tmp/pr-body.md
```

Fill in every section, keeping all four headings in order:

| Section | What goes in it |
| --- | --- |
| `## Description` | Background, motivation, linked issue (`Fixes #NNN`), and main user-visible changes (compatibility, migrations, new dependencies, runtime impact). ≤ 250 words, ≤ 12 bullets. Note if an AI assistant was used. Required. |
| `## Testing` | Exact automated commands run (`route.py check`, `pre-commit`, `pytest`, docs check) and their outcomes. Warns if missing or empty without `--docs-only`. |
| `## Manual verification` | GPU, simulator, model-service, or real-robot runs (environment, command, result), or `Not needed` when offline tests cover the change. Keep bounded integration evidence distinct from benchmark task success. |
| `## Checklist` | Tick every applicable box from `.github/PULL_REQUEST_TEMPLATE.md`. Required. |

Style rules enforced by `lint_pr.py`:

- **No uniform leading indentation** on headings, prose, or table rows.
- **Delete `<!-- ... -->` template comments** once filled.
- **Nothing above the first heading** — put introductory text inside `## Description`.
- **One line per paragraph** so GitHub does not render artificial line breaks.
- **Balanced fenced code blocks** (supports nested CommonMark backtick/tilde fences).

Follow [docs/rules/english.md](../docs/rules/english.md) for prose quality: lead with what changed in behavior, avoid hype/hedging, and name files/symbols only where needed to locate the change.

## 4. Lint, then create

```bash
python3 .agents/skills/create-pr/lint_pr.py lint \
    --title "$TITLE" --body-file /tmp/pr-body.md
```

Pass `--docs-only` on documentation-only PRs to suppress the missing-testing warning, and `--quiet` to hide fix hints. Once it exits 0 and the user approves creating the PR:

```bash
gh pr create --base main --title "$TITLE" --body-file /tmp/pr-body.md
```

Commits must follow Conventional Commits.

## 5. Fixing an open PR

```bash
python3 .agents/skills/create-pr/lint_pr.py lint --pr 364
```

To mechanically strip paste-indentation, pull headings/tables flush left, drop leftover HTML template hints, and optionally unwrap hard-wrapped paragraphs:

```bash
python3 .agents/skills/create-pr/lint_pr.py fix --pr 364 --unwrap > /tmp/fixed.md \
    && python3 .agents/skills/create-pr/lint_pr.py lint --body-file /tmp/fixed.md
```

Read `/tmp/fixed.md` and confirm `lint` exits 0 before updating GitHub:

```bash
gh pr edit 364 --body-file /tmp/fixed.md
```

If the title also needs fixing, set `TITLE="..."`, run `lint` with `--title "$TITLE"`, and update it with `gh pr edit 364 --title "$TITLE"`.

## Gotchas

- **`lint --pr N` shells out to `gh pr view`** and is strictly read-only; nothing changes on GitHub until you run `gh pr create` or `gh pr edit`.
- **The `template` skeleton intentionally fails its own lint** until you tick at least one box under `## Checklist`.
- **`###` subheadings stay inside their parent `##` section** (for example `### Motivation` inside `## Description`), while renaming a top-level template heading triggers `section-missing`.
- **`fix` cannot repair an unclosed code fence.** If `body-fence` fires, close the stray fence by hand.
- **Warnings do not fail the run** (exit 0), but reviewers notice `body-hardwrap`, `body-comments`, `section-order`, and `title-scope-vague`.
- **Title description limit is 72 characters after `: `.** Keep PR titles concise so they stay readable in GitHub and `git log --oneline`.

## Troubleshooting

- `error: gh pr view N failed: ... could not resolve to a PullRequest` — run from an RPent checkout or pass a valid RPent PR number.
- `[E] body/body-indented` persists after `fix` — the body mixes multiple paste offsets; dedent manually.
- `[W] body/section-order` — all template sections exist, but their order differs from `.github/PULL_REQUEST_TEMPLATE.md`.
- Empty `--title ""` or empty `--body-file` — both fail with `title-empty` / `body-empty`; pass non-empty content for any flag you supply.
