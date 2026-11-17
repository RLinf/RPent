---
name: review-pr
description: Reviews an RPent pull request, re-reviews its revisions, or reviews an explicitly selected local diff for correctness, design, code reuse, tests, and docs against RPent's conventions and docs/STYLE_GUIDE.md, reports the findings in a fixed format, and, when asked, posts them as one GitHub review with a main comment that answers the review questions and inline comments with suggested fixes. Use only when the user asks to review, re-review, check before merge, or post review comments; do not use when asked only to explain or summarize a PR, draft a PR description, or open or merge a PR.
---

# Review a PR

A review produces two things: a report in the terminal for the user, in the fixed format of step 6, and, only when the user asks or says yes, a GitHub review on the PR made of one main comment that answers the nine review questions and one inline comment per finding on a diff line, each with a suggested fix.

Most of the review is about correctness, clean design, reuse of what RPent and RLinf already have, and changes that deviate from the existing codebase without need. Area checks follow RPent's existing skills: robot and runtime wiring in [add-robot](../add-robot/SKILL.md), tests and validation in [verify-change](../verify-change/SKILL.md), docs accuracy and parity in [docs-check](../docs-check/SKILL.md), and docs structure and style against [docs/STYLE_GUIDE.md](../../../docs/STYLE_GUIDE.md) and [write-doc](../write-doc/SKILL.md) (question 8). PR formatting and commit conventions are checked but should not dominate. Read [AGENTS.md](../../../AGENTS.md), [CONTRIBUTING.md](../../../CONTRIBUTING.md), and [tests/README.md](../../../tests/README.md) before step 4.

## 1. Set up

Take a PR URL or number (or an explicitly selected local diff; for a local diff, compare against `RLinf/RPent:main` and stop after step 6). All cross-references are against the `main` branch of `github.com/RLinf/RPent`, never the local working tree, which may be on another branch or hold uncommitted work:

```bash
N=<PR number>
REMOTE=$(git remote -v | awk '/github\.com[:\/]RLinf\/RPent(\.git)? \(fetch\)/ {print $1; exit}')
git fetch "$REMOTE" main "pull/$N/head:pr-$N"
BASE="$REMOTE/main"; PR_HEAD="pr-$N"

gh pr view "$N" --repo RLinf/RPent --json title,body,author,headRefOid,commits,files
gh pr checks "$N" --repo RLinf/RPent                     # CI status
git log --oneline "$BASE..$PR_HEAD"                      # what the PR adds
git diff --stat "$BASE...$PR_HEAD"                       # the same diff GitHub shows
git show "$BASE:<path>"                                  # a file on main
git grep <pattern> "$BASE"                               # search main
gh api "repos/RLinf/RPent/pulls/$N/comments" --paginate  # earlier review threads
```

Check out `$PR_HEAD` in a separate git worktree (with an isolated `uv` `.venv` when installing dependencies) whenever running code or tests. Never edit the PR branch or push to it. Without `gh`, read the PR page, `<PR URL>.diff` (or `<PR URL>.patch`), and files on `https://github.com/RLinf/RPent/blob/main/<path>`; the review can still run, though posting needs `gh`.

On a second round of review, go through every earlier thread first. Verify in the new code which ones the new commits fix and which are still open, and carry the open ones forward rather than writing them again.

## 2. Understand the PR

Read the description, the linked issue, the commits, and the whole diff. Then write down, for step 6:

- the problem the PR solves and its core idea, in one or two sentences;
- the areas it touches (for example planner, toolkit, robot integration, packaging, docs, CI) and what could regress (robots, planners, CLI or Dashboard runs, prompts and tool schemas);
- the closest current sibling on `main` it should look like (`git log "$BASE" -- <dir>` shows which sibling is recent).

## 3. Run the reproduction

A PR that adds or changes a user-facing workflow (a new robot, action model, planner, tool, or feature with docs) or that claims metrics needs a reproduction, reported in questions 1 to 3:

1. Check what the workflow needs in the PR's docs (GPU or hardware, install extra, checkpoints, simulator assets) against the host (`nvidia-smi` first; never stop another user's job).
2. When the host has what it needs, run the documented workflow in the PR worktree following only the user docs without fixing undocumented gaps. Otherwise, questions 1 and 2 state what is missing and what a run would need—or report the user's own run record if they provide one.

Record the first failure and every intervention, and keep bounded functional completion (`finish` and cleanup) separate from native environment or benchmark task success; a run that needed undocumented changes does not show that the workflow is reproducible. The unit tests and checks run in step 5 belong in question 7, not here.

## 4. Review

Judge each change by category and map every finding to its main-comment question:

| Category | Covers | Question |
| --- | --- | --- |
| `correctness` | bugs, edge cases, concurrency, cancellation, daemon/RPC lifecycle, refactor regressions | 4 |
| `design` | extension points, module boundaries, lazy imports, packaging, prompt/tool contracts, dead code | 5 |
| `reuse` | duplicated logic and the existing RPent or RLinf helper to reuse | 6 |
| `tests` | unit and e2e coverage, mocks, install extras, CI checks | 7 |
| `docs` | docs against code, code against untouched docs, EN/ZH parity, README updates, asset links, usability | 7 (3 for usability) |
| `docs-style` | changed pages and README text against `docs/STYLE_GUIDE.md` | 8 |
| `style` | Ruff lint/format, docstrings, type hints, logging, license headers, commits, PR metadata | 9 |

For every changed function, read the `main` version and the PR version side by side. Label a finding that came from reading `main` rather than the diff alone as a `main-branch cross-check`. Check that robot-specific policy stays in `robots/<robot>/` while shared code uses RPent's extension points (`RobotSpec`, `Toolkit`, `Planner`, RPC facades) rather than robot- or planner-name branches, keeps optional imports lazy, preserves owned-daemon vs. borrowed-endpoint cleanup, and keeps rendered prompts and tool schemas accurate and concise.

Apply the matching skill for each touched area—[add-robot](../add-robot/SKILL.md) for robot, Env/VLA, or tool wiring, [verify-change](../verify-change/SKILL.md) for unit/e2e coverage, `pyproject.toml` extras, and CI, and [docs-check](../docs-check/SKILL.md) for docs against code (in both directions) and EN/ZH parity.

Check `docs-style` as a pass of its own after `docs-check`. Read every page the PR adds or changes in full and in both languages, as a reader who is new to RPent and has not seen its code, classify its page type, and check it against [docs/STYLE_GUIDE.md](../../../docs/STYLE_GUIDE.md) and [write-doc](../write-doc/SKILL.md).

## 5. Verify every finding

Before a finding goes in the report, run the relevant checks via [verify-change](../verify-change/SKILL.md) and settle each candidate as one of:

- **`confirmed`**: there is evidence—a focused unit test, a scratch check in the PR worktree, a strict docs build, or a complete trace citing every line on the PR head.
- **`unresolved`**: plausible, but it needs missing hardware, assets, or the author's intent to settle; say what would settle it.
- **`dismissed`**: the trace shows it cannot happen, or the code matches the convention on `main`; keep one line on why.

Then give each `confirmed` or `unresolved` finding a severity:

- **`critical`**: wrong results, a crash, data loss, a broken existing path or robot backend, or a documented workflow that cannot be completed.
- **`major`**: breaks RPent design, module-boundary, packaging, RPC, or prompt/tool contracts, or omits tests, docs, or CI required for a user-facing change; it must change before merge. A `docs-style` finding is `major` when a page fails as a whole (first-time reader cannot tell what the feature does or how to run it, explains code layout instead of behavior, has no lead or connected flow, breaks its page-type anatomy or docs-tree placement, translates Chinese clause-by-clause, or adds build warnings).
- **`minor`**: a local fix such as naming, a comment, a small style issue, or 1–2 line doc phrasing polish; it should change but does not block.

Report only problems. When correctness, design, and reuse yield nothing, say so and list what was checked; do not invent findings to fill a category.

## 6. Report

Print the review in the user's language (keeping step 8 comments in English unless asked otherwise) in exactly this shape. List findings by severity, and within one severity keep step 4's category order so tests, docs, docs-style, and style findings sit together after the code findings. Quote the code itself under `Code walkthrough:` so a reader never has to chase bare line numbers, and list every added or changed doc page under `## Docs style` so passing pages are visible too:

````markdown
# Review of #<N>: <PR title>

<PR URL> · head `<sha7>` · base `main@<sha7>` · <YYYY-MM-DD>

## What the PR does

- Goal: <the problem it solves>
- Idea: <the core idea in one or two sentences>
- Changes: <areas touched, with file counts>
- Risk: <robots, planners, paths, or services that could regress>
- Compared with: <the closest sibling on main>

## Findings

| ID | Severity | Category | Location | Summary | Status | Posted in |
|---|---|---|---|---|---|---|
| F1 | critical | correctness | `rpent/x.py:42` | <one line> | confirmed | inline |
| F2 | major | docs | `docs/source-zh/...` | <one line> | confirmed | question 7 |
| F3 | minor | docs-style | `docs/source-en/...:3` | <one line> | confirmed | inline |

### F1 · critical · correctness: <plain title>

`rpent/x.py:42`, compared with `main:rpent/x.py:40` (main-branch cross-check)

<The comment text from step 8, exactly as it will be posted.>

Code walkthrough:

<The key snippets from the PR, and from main when compared, each under a `path:line` header, then the steps that lead from the input to the bug.>

Evidence: <the command and its result, or the trace; for an unresolved finding, what would settle it>

## Docs style

Checked against `docs/STYLE_GUIDE.md` · build: <the EN and ZH sphinx-build results, or why it did not run>

| Page | Type | Result |
|---|---|---|
| `docs/source-en/rst_source/<path>.rst` | simulator guide | F3 |
| `docs/source-zh/rst_source/<path>.rst` | simulator guide | pass |

<Or, when the PR changes no docs: "The PR changes no docs.">

## Dismissed

- <candidate>: <why it does not apply>

## Main comment

<The nine sections from step 8, exactly as they will be posted.>
````

## 7. Ask before posting

Skip this step for local diff reviews. Otherwise, posting speaks for the user, so ask first: `"Post this review to PR #N?"` with **Post as a comment** *(recommended)*, **Post as request changes**, and **Don't post**. If the user answers with edits, apply them, show the changed text, and ask again. Approve only when the user explicitly asks for it.

## 8. Post

Write the review as the JSON payload of GitHub's create-review API with `json.dump` so escaping is right, then lint and post it:

```bash
python3 .agents/skills/review-pr/lint_review.py template
python3 .agents/skills/review-pr/lint_review.py lint --pr "$N" review.json
python3 .agents/skills/review-pr/lint_review.py post --pr "$N" review.json
```

```json
{
  "commit_id": "<PR head SHA>",
  "event": "COMMENT",
  "body": "<the main comment>",
  "comments": [
    {"path": "rpent/x.py", "line": 42, "body": "<inline comment>"},
    {"path": "robots/libero/y.py", "start_line": 10, "line": 12, "body": "<inline comment>"}
  ]
}
```

`event` is `COMMENT` or `REQUEST_CHANGES`, as the user chose. `lint` checks the nine headings, the recommendation, the `Suggested fix:` paragraph of every inline comment, hard-wrapped lines, and—with `--pr`—that every comment sits on a diff line and not on a line that already has a thread. GitHub rejects the whole review when one comment is outside the diff. `post` runs the same lint, posts everything as one review, and prints its URL. Report that URL and the number of inline comments to the user.

### Where a finding goes

A finding goes inline when a line of the diff carries it: the line with the bug, the line where a special case enters shared code, or, for a block, `start_line` to `line` (`side: "LEFT"` only for a deleted line that should stay). A `docs-style` finding goes on the line it rewrites; when one rule fails on many lines of a page, comment on the first and list the others under question 8. A finding with no such line goes in the main comment under its question: a missing doc page, a missing test, PR metadata. Reply in an existing thread instead of opening a new one on the same line: `gh api "repos/RLinf/RPent/pulls/$N/comments/<id>/replies" -f body=...`.

### Inline comments

Write for a reader with little background in the module: the PR author knows their own code but may not know RPent's internals, and the human reviewer may know RPent but not this module. Each comment covers one finding:

1. Say in 1–3 sentences what the code tries to do, what happens instead, and why it matters—starting from the situation the reader recognizes before naming identifiers.
2. When the cause or pattern to follow lives elsewhere, point to `main:<path>:<line>` or quote the line.
3. End with `Suggested fix:` stating what to change in one sentence, followed by a `suggestion` block (when replacing the commented lines) or a fenced code block with a language tag.

Keep the language plain: short sentences, active voice, no hedging, no filler, no praise, and no severity labels except `Minor:` at the start of a minor finding. Keep each paragraph on one line, because GitHub turns every newline into a line break. Write in English unless the user asks otherwise.

A correctness finding:

````markdown
When the action model server fails `try_wait_server`, `init_runtime` returns without tearing down the environment daemon spawned just before it, leaving an orphaned simulator process holding GPU memory after CLI startup aborts. Passing `daemons` into `try_spawn_server` lets `stop_owned_daemons` clean up earlier owned servers (`main:rpent/robots/runtime.py:53`).

Suggested fix: pass `daemons` into `try_spawn_server` and `try_wait_server` instead of registering the process only after readiness succeeds.

```python
_, env_rpc = try_spawn_server(daemons, dashboard_events, "env", spawn_env)
```
````

A design finding:

````markdown
The CLI runner now branches on `args.robot == "robocasa"` to skip global memory sync before starting runtime services, so the next robot with local-only memory will have to add another branch here. Per-robot memory preparation belongs on `RobotSpec.prepare_memory` (`main:rpent/robots/robot_spec.py:88`).

Suggested fix: implement `prepare_memory` on the RoboCasa `RobotSpec` and drop the robot-name check from `rpent/cli/main.py`.

```python
def _prepare_robocasa_memory(args: argparse.Namespace, config: RunConfig) -> None:
    config.prompt_vars["memory_dir"] = str(args.memory_dir or get_memory_dir("robocasa"))
```
````

A docs-style finding:

````markdown
This paragraph explains task execution by listing `rpent/session/base.py` and `rpent/tools/toolkit.py`, which a reader new to RPent never opens or edits to run a task. The style guide asks pages to explain behavior through what the user runs and observes (`docs/STYLE_GUIDE.md`, "Write for someone running their first task").

Suggested fix: describe the CLI options and saved run artifacts instead of internal modules.

```suggestion
Each run records step images, tool calls, and the final environment status under `--output-dir`, letting you inspect why an episode stopped without opening source files.
```
````

### Main comment

The main comment holds only the nine sections, each under its question as a `###` heading, word for word (`lint_review.py template` prints them). Answer in short bullets, keep each one to one line, point to inline comments by `path:line`, and put long logs in a `<details>` block.

1. **What settings did you use to reproduce this PR?** The PR revision (head SHA), hardware and GPU, software environment, robot, model and dataset, config, and commands. Link a saved summary rather than listing everything. Without a run, say so and what a run needs.
2. **What were the reproduction results?** Claimed vs. observed behavior or metrics with links to logs and results; state bounded functional completion (`finish`, state artifacts, and cleanup) and native task success separately, plus the first failure, any interventions, and what was not reproduced.
3. **Is the documentation clear and usable?** Whether every step from setup to expected output could be followed; unclear lines, missing prerequisites, failed commands, and workarounds. Without a run, say that the docs were read rather than executed.
4. **Are there bugs or unresolved concerns?** Confirmed issues and unresolved concerns with triggers, evidence, and inline pointers, then dismissed findings with one line each on why. `"No issue found"` states what was checked.
5. **Does the integration follow RPent's design?** Closest sibling compared against; whether `RobotSpec`, `Toolkit`, `Planner`, RPC, lazy imports, containment, and prompt/tool contracts match it; mismatches; and points needing a maintainer's judgment.
6. **Does it reuse existing modules/functions properly?** Avoidable duplication and the `main:<path>:<line>` helper (or RLinf component) to reuse, or else the implementation compared against.
7. **Are tests and documentation adequate?** Whether offline CPU tests cover the changed behavior; docs, EN/ZH updates, README highlights where warranted, and `pyproject.toml` extras/CI coverage; tests and checks run (command, hardware/scope, result) and skipped checks with reasons.
8. **Does the documentation follow the docs style guide?** Every page the PR adds or changes, in both languages, with its type and whether it follows `docs/STYLE_GUIDE.md` (first-time reader clarity, lead and flow, voice, page anatomy, and natural phrasing in each language), failing rules with inline pointers, and both EN and ZH build results. When the PR changes no docs, say so; a missing page belongs in question 7.
9. **What is your recommendation?** Exactly one of `Ready for senior review` (no open critical or major finding, and reproduction passed or was not needed), `Changes requested` (open critical or major findings the author can fix), or `Blocked` (needs a maintainer decision, unavailable hardware/assets, or an unresolved failure outside the author's fix), followed by remaining actions and needed expert review.
