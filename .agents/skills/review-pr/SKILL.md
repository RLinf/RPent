---
name: review-pr
description: Reviews an RPent pull request, re-reviews its revisions, or reviews an explicitly selected local diff for correctness, design, and code reuse against RPent's patterns, runs the check of every area the PR touches (code, docs, robots) through those skills and the reproduction when the host can, reports concise, deduplicated findings with evidence, and, when the user asks, posts them as one GitHub review. Use only when the user asks to review, re-review, check before merge, or post review comments; do not use when asked only to explain or summarize a PR, draft a PR description, or open or merge a PR.
---

# Review a PR

Produce one concise, deduplicated review in the step 8 format: a verdict, a two-bullet reproduction summary, the findings table, detail sections only for non-inline (`here`) findings, inline comments printed once, and a two-bullet check summary (not posted). Post to GitHub only when the user asks ([posting.md](posting.md)). For an explicitly selected local diff, compare against `RLinf/RPent:main` and stop after step 8 (writing in the user's language).

Focus on correctness, clean design, reuse of what RPent and RLinf already have, and unnecessary divergence from `main`. Area checks live in [code](../code/SKILL.md), [docs](../docs/SKILL.md), and [robots](../robots/SKILL.md). Read [code/patterns.md](../code/patterns.md) and [reference.md](reference.md) before step 5.

## 1. Set up

Cross-reference against `main` of `github.com/RLinf/RPent`, never the local working tree:

```bash
N=<PR number>
REMOTE=$(git remote -v | awk '/github\.com[:\/]RLinf\/RPent(\.git)? \(fetch\)/ {print $1; exit}')
git fetch "$REMOTE" main "pull/$N/head:pr-$N"
BASE="$REMOTE/main"; PR_HEAD="pr-$N"
SKILLS="$(git rev-parse --show-toplevel)/.agents/skills"

gh pr view "$N" --repo RLinf/RPent --json title,body,author,headRefOid,commits,files
gh pr checks "$N" --repo RLinf/RPent
git log --oneline "$BASE..$PR_HEAD"
git diff --stat "$BASE...$PR_HEAD"
git show "$BASE:<path>"
git grep "<pattern>" "$BASE"
gh api "repos/RLinf/RPent/pulls/$N/reviews" --paginate
gh api "repos/RLinf/RPent/issues/$N/comments" --paginate
gh api "repos/RLinf/RPent/pulls/$N/comments" --paginate
```

Check out `$PR_HEAD` in a separate git worktree (with an isolated `uv` `.venv` when installing dependencies) for steps 3, 6, and verification; never edit or push the PR branch. Without `gh`, read `<PR URL>.diff` and files on `github.com/RLinf/RPent/blob/main/<path>`; the review can still run, though posting needs `gh`.

On a re-review, inspect every earlier review summary, discussion comment, and inline thread first. Verify each fix in the current code (never trust commit messages alone) and check for regressions or shifted bugs. Carry open findings forward (reply on open inline threads and mark those rows `Posted = thread`; keep earlier IDs on open main-comment findings; number new findings after the highest earlier ID), name fixed earlier findings in the verdict, and do not reopen minor preferences the author already answered reasonably.

## 2. Understand the PR

Read the PR description, linked issue, commits, and diff, and note:

- the problem and core idea;
- touched areas and what could regress (planners, robots, CLI/Dashboard runs, RPC/daemon lifecycle, prompts and tool schemas, packaging, docs);
- closest current siblings on `main` for code (`git log "$BASE" -- <dir>`) and docs;
- every claim in the PR description ("behavior-preserving", test coverage, benchmark numbers, hardware runs) as a hypothesis to verify against the diff and tests.

## 3. Route the change

In the PR worktree:

```bash
python3 "$SKILLS/route.py" list --base "$BASE"
```

Check every listed area in step 6; review unrouted files yourself in step 5.

## 4. Run the reproduction

When a PR adds or changes a user-facing workflow (a new robot, action model, planner, tool, or feature with docs) or claims metrics, run a reproduction whenever this host has the hardware and assets:

1. **Check prerequisites** from the PR's docs against the host (`nvidia-smi` first; never stop another user's job; verify install extras, checkpoints, simulator assets, or external endpoints).
2. **Launch a fresh agent (never a fork)** in the background while you review, working in its own worktree at `$PR_HEAD`. Give it only the PR URL, head SHA, doc pages to follow, allowed hardware, and local install rules. It must follow only user docs, never open `rpent/`, `robots/`, or `tests/` to fill gaps, fix nothing, and report the minimal setup, first failure, every intervention, and bounded functional completion (`finish` + daemon cleanup) separately from native environment/benchmark success (`terminated`).
3. **If blocked**, state what hardware, assets, checkpoints, or credentials are missing (or report the user's run if provided).

Never reproduce the PR yourself after reading the implementation. An undocumented step or failure is a `docs` finding; a passing run still requires step 6 checks on command clarity and workflow alignment. Unit tests and area scripts are not reproductions—put them in Check summary.

## 5. Review the code

Work through [reference.md](reference.md) against [code/patterns.md](../code/patterns.md). File each finding under one category, ordered within each severity as:

| Category | Covers |
|---|---|
| `correctness` | bugs, edge cases, concurrency, cancellation, daemon/RPC lifecycle, refactor regressions |
| `design` | extension points, containment, lazy imports, abstractions, test-driven guards, prompt/tool contracts, scope creep |
| `reuse` | duplicated logic and the existing RPent or RLinf helper to reuse |
| `tests` | unit/e2e coverage and layout, mocks, install extras, `pyproject.toml` conflicts, CI checks |
| `docs` | docs-to-code accuracy, EN/ZH parity, README highlights, media/asset URLs, workflow usability |
| `docs-style` | readability, conciseness, build/markup, prose/structure rules for page type and language |
| `style` | Ruff lint/format, docstrings, type hints, logging, headers, commits, PR title/description |

- **Large PRs** (> ~15 files or ~1,000 changed lines): partition by concern (core logic, robots/packaging, tests/CI, docs) and review each group explicitly—using parallel focused agents when helpful—so late files (tests, prompts, Chinese docs) are not skimmed.
- **Side-by-side and caller sweep**: compare every changed function against `main` (label `main`-derived findings as `main-branch cross-check`). Whenever a signature, `RobotSpec` hook, RPC route, tool schema, CLI flag, or prompt variable changes, `git grep` all callers, subclasses, planners, tests, and docs at `$PR_HEAD` to catch broken untouched consumers.

## 6. Check each area

In the PR worktree, run the automated checks and follow each routed area's `SKILL.md § Check` judgment list:

```bash
python3 "$SKILLS/route.py" check --base "$BASE"
python3 "$SKILLS/create-pr/lint_pr.py" lint --pr "$N"
```

- **Docs readability pass (fresh agent)**: `docs/check.py` checks build, markup, symbols, and mechanical prose. Delegate the full EN/ZH reading to a **fresh agent (never a fork)** that sees only the changed doc pages, closest `main` sibling pages, and [docs/SKILL.md](../docs/SKILL.md) + `docs/rules/`—with no `rpent/`, `robots/`, or `components/` source context. Prioritize readability and conciseness: flag (1) stiff prose or machine-translated/literal Chinese, (2) long-winded explanations, (3) internal code citations on user pages, (4) developer internals leaking into user guides or duplicating `development/`, (5) complex commands, (6) non-runnable placeholder examples (such as `example.com`), and (7) repeated negative-contrast framing ("not A, not B, but C" / "不是……而是……").
- **Script-to-category mapping**:
  - `tests`: `code/test-layout`, `robots/packaging`, `robots/e2e-syntax`, `robots/e2e-targets`
  - `design`: `robots/exports`, `robots/lazy-import`
  - `docs`: `docs/symbol`, `docs/rtd-link`, `docs/parity`, docs-to-code accuracy, asset/media URLs, command usability
  - `docs-style`: `docs/build`, `docs/markup`, `docs/code-name`, `docs/structure`, `docs/heading`, `docs/en-style`, `docs/zh-style`, `docs/zh-hard-wrap`, readability/conciseness
  - `style`: `code/pre-commit`, `code/init`, `code/license`, `code/shell-syntax`, PR lint
- Report any area script that could not run (and what it needs) in Check summary.

## 7. Verify every finding

Try to falsify each candidate before reporting it. Run cheap offline CPU checks in the PR worktree whenever possible (imports, `enumerate_robots()`, CLI `--help`, focused `pytest` cases, `docs/check.py`) instead of relying solely on static traces.

- **confirmed**: proven by a run or a complete line-cited trace on `$PR_HEAD`.
- **unresolved**: plausible, but needs missing hardware, assets, or author intent; state what settles it.
- **dismissed**: cannot happen or matches `main`; drop silently (never list in the output).

Merge findings with the same root cause into one finding listing affected locations. Assign severity strictly per [README.md § Findings](../README.md#findings): `critical` requires a concrete failure path (wrong result, crash, data loss, broken backend/robot, failed docs build, or uncompletable workflow); whole-page docs readability failures, developer internals leaking into user guides or duplicating `development/`, and non-runnable placeholder examples (such as `example.com`) are `major`, not `minor`; downgrade or drop speculative edge cases and low-value nits. If nothing survives, write `No findings.` and one line on what was checked.

## 8. Report

Keep the output tight and free of repetition: print the main comment, then each inline comment once, then a two-bullet check summary (not posted). Use English for publishable GitHub text unless asked otherwise.

- **Verdict**: bold recommendation (`Ready for senior review`, `Changes requested`, or `Blocked`) + severity counts + head/base SHA, then one short sentence on what blocks merge or needs a maintainer decision.
- **Reproduction**: two short bullets (under ~300 chars each). `Result:` starts with `**Passed**`, `**Failed**`, `**Partial**`, or `**Not run**` + one-sentence functional/metric conclusion. `Setup:` names only hardware and command/config (omit host, OS, driver, venv paths, and SHA).
- **No duplicate findings**: the table indexes every finding (`F1`, `F2`, ...), sorted by severity then category. Write a `#### F<n>` section in the main comment **only for `Posted = here` findings**. `inline` findings live once in `### Inline comments` (each carrying problem, `Suggested fix:`, and one-line `Evidence:`), and carried-forward replies on existing threads use `Posted = thread` (with no `####` section or new `comments[]` entry).
- **No trailing noise**: no `Dismissed:` list, no restated Goal/Idea/Changes/Risk bullets, and no tables of passing doc pages.

````markdown
**<Recommendation>** · <n critical, n major, n minor, or no findings> · head `<sha7>` · base `main@<sha7>`

<One short sentence: what blocks merge, or why it is ready.>

### Reproduction

- Result: **<Passed | Failed | Partial | Not run>** — <one short sentence: bounded functional status and native task result vs. baseline, or first blocker / what is missing>
- Setup: <hardware and command/config run; or what a run needs if not run>

### Findings

| ID | Severity | Category | Location | Summary | Status | Posted |
|---|---|---|---|---|---|---|
| F1 | critical | correctness | `rpent/x.py:42` | <one line> | confirmed | inline |
| F2 | major | docs | `docs/source-zh/...` | <one line> | confirmed | here |

#### F2 · major · docs: <plain title>

`docs/source-zh/...`, compared with `main:...` (main-branch cross-check)

<Problem and impact in 1–3 plain sentences.>

Suggested fix: <one sentence, plus code block if code changes.>

Code walkthrough:

<Quoted snippets under `path:line` lines, then input-to-failure steps. Omit when not about code.>

Evidence: <one line: command and result, or trace; for unresolved, what settles it.>
````

````markdown
### Inline comments (posted on diff lines)

`rpent/x.py:42`

**F1 · critical · correctness: <plain title>**

<Problem (1–3 sentences), `Suggested fix:` + verified `suggestion`/code block, and one-line `Evidence:`.>
````

````markdown
## Check summary (not posted)

- Checks: `route.py check` (<result>) · `lint_pr` (<result>) · tests (<command, hardware, result; or skipped + why>)
- Context: compared with <closest sibling on main>; <any area check that could not run and why>
````

## 9. Post only when asked

Stop after step 8; never post by default. Skip this step for local diff reviews. When the user asks to post, update, or reply on GitHub, read [posting.md](posting.md) and run `lint_review.py lint` before posting. Approve only when explicitly asked.
