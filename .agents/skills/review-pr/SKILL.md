---
name: review-pr
description: Reviews an RPent pull request, re-reviews its revisions, or reviews an explicitly selected local diff for correctness, design, code reuse, tests, and docs against RPent's conventions and docs/STYLE_GUIDE.md, previews a publishable review, and, when asked, posts that body and its inline suggestions to GitHub. Use only when the user asks to review, re-review, check before merge, or post review comments; do not use when asked only to explain or summarize a PR, draft a PR description, or open or merge a PR.
---

# Review a PR

Write one review body with a recommendation, reproduction results, and findings. Preview it with the proposed inline comments in step 6. When the user approves publication in step 7, post that same text.

Most of the review is about correctness, clean design, reuse of what RPent and RLinf already have, and changes that deviate from the existing codebase without need. Area checks follow RPent's existing skills: robot and runtime wiring in [add-robot](../add-robot/SKILL.md), tests and validation in [verify-change](../verify-change/SKILL.md), docs accuracy and parity in [docs-check](../docs-check/SKILL.md), and docs structure and style against [docs/STYLE_GUIDE.md](../../../docs/STYLE_GUIDE.md) and [write-doc](../write-doc/SKILL.md). PR formatting and commit conventions are checked but should not dominate. Read [AGENTS.md](../../../AGENTS.md), [CONTRIBUTING.md](../../../CONTRIBUTING.md), and [tests/README.md](../../../tests/README.md) before step 4.

## 1. Set up

For a PR URL or number, use GitHub's base and head commits to identify what the PR changes. Use upstream `main` for design and reuse comparisons. For an explicitly selected local diff, compare against `RLinf/RPent:main` and stop after step 6.

```bash
N=<PR number>
REMOTE=$(git remote -v | awk '/github\.com[:\/]RLinf\/RPent(\.git)? \(fetch\)/ {print $1; exit}')
read -r BASE_NAME BASE PR_HEAD < <(
  gh pr view "$N" --repo RLinf/RPent --json baseRefName,baseRefOid,headRefOid \
    --jq '[.baseRefName, .baseRefOid, .headRefOid] | @tsv'
)
git fetch "$REMOTE" main "$BASE" "$PR_HEAD"
printf 'head %.7s · base %s@%.7s\n' "$PR_HEAD" "$BASE_NAME" "$BASE"

gh pr view "$N" --repo RLinf/RPent --json title,body,author,commits,files
gh pr checks "$N" --repo RLinf/RPent                     # CI status
git log --oneline "$BASE..$PR_HEAD"                      # what the PR adds
git diff --stat "$BASE...$PR_HEAD"                       # the same diff GitHub shows
git show "$REMOTE/main:<path>"                           # a file on upstream main
git grep "<pattern>" "$REMOTE/main"                      # search upstream main
```

Check out `$PR_HEAD` in a separate git worktree (with an isolated `uv` `.venv` when installing dependencies) whenever running code or tests. Never edit the PR branch or push to it. Without `gh`, read the PR page, `<PR URL>.diff` (or `<PR URL>.patch`), and files on `https://github.com/RLinf/RPent/blob/main/<path>`; the review can still run, though posting needs `gh`.

Read earlier feedback from all three places. On a re-review, check each request against the current code and link unresolved findings to the original discussion. Use the original finding IDs when reporting which issues are fixed and which remain open. Number new findings after the highest ID used in earlier rounds. Refer to earlier feedback without an ID by its original discussion link.

```bash
gh api "repos/RLinf/RPent/pulls/$N/reviews" --paginate    # review summaries
gh api "repos/RLinf/RPent/issues/$N/comments" --paginate  # PR discussion
gh api "repos/RLinf/RPent/pulls/$N/comments" --paginate   # inline comments and replies
```

## 2. Understand the PR

Read the description, the linked issue, the commits, and the whole diff. Then write down, for step 6:

- the problem the PR solves and its core idea, in one or two sentences;
- the areas it touches (for example planner, toolkit, robot integration, packaging, docs, CI) and what could regress (robots, planners, CLI or Dashboard runs, prompts and tool schemas);
- the closest current sibling on `main` it should look like (`git log "$REMOTE/main" -- <dir>` shows which sibling is recent).

## 3. Run the reproduction

A PR that adds or changes a user-facing workflow (a new robot, action model, planner, tool, or feature with docs) or that claims metrics needs a reproduction. Report the run in the review body's `Reproduction` section.

1. Check what the workflow needs in the PR's docs (GPU or hardware, install extra, checkpoints, simulator assets) against the host (`nvidia-smi` first; never stop another user's job).
2. When the host has what the workflow needs, start a fresh agent that does not inherit the review conversation. Give it only the PR worktree path, the user docs to follow, the hardware it may use, and any local installation rules. Have it follow the documented steps and report its commands, results, and the first step it could not complete. It should not consult source code or tests to fill gaps in the docs. If a fresh agent is unavailable, report any commands you ran and state that the docs were not independently reproduced. If the host lacks prerequisites, state in `Reproduction` what is missing and what a run would need, or report the user's own run record if they provide one.

Record the first failure and every intervention, and keep bounded functional completion (`finish` and cleanup) separate from native environment or benchmark task success; a run that needed undocumented changes does not show that the workflow is reproducible. Record the checks from step 5 in the review notes and cite any that provide evidence for a finding.

## 4. Review

Review each change against the relevant categories.

| Category | Covers |
| --- | --- |
| `correctness` | bugs, edge cases, concurrency, cancellation, daemon/RPC lifecycle, refactor regressions |
| `design` | extension points, module boundaries, lazy imports, packaging, prompt/tool contracts, dead code |
| `reuse` | duplicated logic and the existing RPent or RLinf helper to reuse |
| `tests` | unit and e2e coverage, mocks, install extras, CI checks |
| `docs` | docs against code, code against untouched docs, EN/ZH parity, README updates, asset links, usability |
| `docs-style` | changed pages and README text against `docs/STYLE_GUIDE.md` |
| `style` | Ruff lint/format, docstrings, type hints, logging, license headers, commits, PR metadata |

For each changed function, compare the full versions at the merge base and the PR head. Label a finding that relies on a separate comparison with upstream `main` as a `main-branch cross-check`. Check that robot-specific policy stays in `robots/<robot>/` while shared code uses RPent's extension points (`RobotSpec`, `Toolkit`, `Planner`, RPC facades) rather than robot- or planner-name branches, keeps optional imports lazy, preserves owned-daemon vs. borrowed-endpoint cleanup, and keeps rendered prompts and tool schemas accurate and concise.

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

Preview the exact GitHub review body, followed by the proposed inline comments with their locations. Use English for the publishable text unless the user requests another language. For a local diff review, use the user's language.

### Review body

Run `python3 .agents/skills/review-pr/lint_review.py template` for the body skeleton. Start with one recommendation on its own line. Use **Ready for senior review** when there are no open critical or major findings and reproduction passed or was unnecessary, **Changes requested** when the author can fix an open critical or major finding, or **Blocked** when progress needs a maintainer decision, missing resources, or resolution of a failure outside the author's control. Briefly explain the change and the reason for the verdict. Include limitations that affect the decision. On a re-review, identify earlier findings the revision fixed.

```markdown
**Changes requested**

<One or two sentences explaining the verdict.>

### Reproduction

<Who ran the workflow, the revision and settings, the commands, and the observed result. If it was not run, explain why and what remains unverified.>

### Findings

#### F2 · major · correctness · <Title>

<Location, trigger, impact, and evidence.>
```

For a reproduction, include the relevant hardware, software, robot, model, dataset, and configuration. Compare claimed and observed results. For metrics, include the workload, seeds, and baseline. Follow step 3 when reporting failures, interventions, and task success.

Give each confirmed or unresolved finding one heading as shown above, in severity order and then step 4's category order. Start at F1 on the first review and follow step 1's numbering on re-review. Cite the relevant `path:line` and the check result or reasoning that supports the finding. For code findings, quote the lines needed to follow the behavior. For an unresolved issue, explain what would settle it. If there are no findings, write `No findings.` in this section.

### Where a finding goes

When a finding belongs on a diff line, put the suggested fix in an inline comment on that line. Keep the evidence in the body and identify the inline location there. For a deleted line that should stay, use `side: "LEFT"`. For a range from `start_line` to `line`, set both `start_side` and `side` to `RIGHT` for the new code or `LEFT` for the old code. For a repeated `docs-style` issue on one page, comment on the first affected line and list the other locations under the same finding. For a finding without a suitable diff line, such as a missing page or test, put its suggested fix in the body under its finding ID. Reply in an existing thread instead of opening a new one on the same line: `gh api "repos/RLinf/RPent/pulls/$N/comments/<id>/replies" -f body=...`.

### Inline comments

Write for a reader with little background in the module: the PR author knows their own code but may not know RPent's internals, and the human reviewer may know RPent but not this module. Each comment covers one finding:

1. Say in 1–3 sentences what the code tries to do, what happens instead, and why it matters—starting from the situation the reader recognizes before naming identifiers.
2. When the cause or pattern to follow lives elsewhere, point to `main:<path>:<line>` or quote the line.
3. End with `Suggested fix:` stating what to change in one sentence, followed by a `suggestion` block (when replacing the commented lines) or a fenced code block with a language tag.

Keep the language plain: short sentences, active voice, no hedging, no filler, no praise, and no severity labels except `Minor:` at the start of a minor finding. Keep each paragraph on one line, because GitHub turns every newline into a line break.

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

### Terminal review notes

After the publishable text, add brief notes in the user's language. Include the PR URL and revisions (or the local diff baseline), the goal, approach, risks, and the closest implementation compared on `main`. For each affected area from step 2, give the file count, checks and results, and reasons for skipped checks. Refer to a finding by ID when it already gives the evidence.

For each changed documentation page, record its type and style result, including pages that passed, and the EN/ZH build results or why the builds were skipped. Note any dismissed candidate and the reason for dismissing it.

## 7. Ask before posting

Skip this step for local diff reviews. Otherwise, posting speaks for the user, so ask first: `"Post this review to PR #N?"` with **Post as a comment** *(recommended)*, **Post as request changes**, and **Don't post**. If the user answers with edits, apply them, show the changed text, and ask again. Approve only when the user explicitly asks for it.

## 8. Post

Return to the root of the RPent checkout where you loaded this skill. Use `json.dump` to write the approved review body to `body` and the approved inline comments to `comments` in `review.json`. Keep the terminal review notes out of the payload. Run the following commands after preparing the file.

```bash
python3 .agents/skills/review-pr/lint_review.py lint --pr "$N" review.json
python3 .agents/skills/review-pr/lint_review.py post --pr "$N" review.json
```

```json
{
  "commit_id": "<PR head SHA>",
  "event": "COMMENT",
  "body": "<the approved review body>",
  "comments": [
    {"path": "rpent/x.py", "line": 42, "body": "<inline comment>"},
    {"path": "robots/libero/y.py", "start_line": 10, "start_side": "RIGHT", "line": 12, "side": "RIGHT", "body": "<inline comment>"}
  ]
}
```

`event` is `COMMENT` or `REQUEST_CHANGES`, as the user chose. `lint` checks the recommendation and body sections, the `Suggested fix:` paragraph of every inline comment, hard-wrapped lines, and—with `--pr`—that every comment sits on a diff line. It warns about existing threads. GitHub rejects the whole review when one comment is outside the diff. `post` runs the same lint, posts everything as one review, and prints its URL. Report that URL and the number of inline comments to the user.
