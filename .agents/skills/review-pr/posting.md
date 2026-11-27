# Posting a GitHub review

Read this file only when the user asks to post, update, or reply to a review on GitHub.

## 1. Build, lint, and post

Return to the root of the RPent checkout where you loaded this skill. Write the GitHub create-review API payload as JSON (`json.dump` for clean escaping), lint it, then post it. `body` is the main comment from `review-pr/SKILL.md` step 8 (without `### Inline comments` or `## Check summary`):

```bash
python3 .agents/skills/review-pr/lint_review.py template      # main-comment skeleton
python3 .agents/skills/review-pr/lint_review.py lint --pr "$N" review.json
python3 .agents/skills/review-pr/lint_review.py post --pr "$N" review.json
python3 .agents/skills/review-pr/lint_review.py update --pr "$N" --review <id> review.json
```

```json
{
  "commit_id": "<PR head SHA>",
  "event": "COMMENT",
  "body": "<the main comment>",
  "comments": [
    {"path": "rpent/x.py", "line": 42, "side": "RIGHT", "body": "<inline comment>"},
    {"path": "robots/libero/y.py", "start_line": 10, "start_side": "RIGHT", "line": 12, "side": "RIGHT", "body": "<inline comment>"}
  ]
}
```

- `event` is `COMMENT`, or `REQUEST_CHANGES` when asked (`APPROVE` only when explicitly asked).
- `lint_review.py lint` enforces the step 8 structure (verdict, `Result:`/`Setup:` reproduction bullets, findings table, `####` sections only for `here` findings, paired inline comments with matching `**F<n> · <severity> · <category>: <title>**` headers plus `Suggested fix:` and `Evidence:`, plain Markdown without HTML/placeholders/hard-wrapped paragraphs/unclosed fences, 65,536-char GitHub body limit, and with `--pr`, diff-hunk line validity and existing-thread checks) plus warnings for long reproduction bullets, unquoted `Code walkthrough:` blocks, hedging phrases, and >5 minor findings.
- `post` runs the same lint, posts everything as one review, and prints its URL. Report the URL and inline comment count to the user.
- Update a posted review in place (`lint_review.py update` for the body; edit or reply on inline threads). Only a new review round on a new head posts a new review. If `post` fails without printing a URL, check `gh api "repos/RLinf/RPent/pulls/$N/reviews"` before retrying so you do not double-post.

## 2. Where each finding goes

- **Inline (`comments[]`)**: any finding anchored to a diff line (or range `start_line`..`line`, setting both `start_side` and `side`). Use `"side": "LEFT"` only for a deleted line that should stay. For a docs-style rule repeated on many lines of one page, comment on the first line and note the remaining lines rather than opening a thread per line.
- **Main comment (`here`)**: findings with no diff line (missing doc page, missing test, untouched file that should have changed, PR metadata).
- **Existing threads (`thread`)**: reply instead of opening a duplicate thread on the same line, and mark the carried-forward row `Posted = thread` in the Findings table (omitting it from `comments[]` and from `####` sections):

```bash
gh api "repos/RLinf/RPent/pulls/$N/comments/<id>/replies" -f body="..."
```

## 3. Writing inline comments

Write for a reader who may not know both RPent internals and this module. Each inline comment is self-contained (not repeated in the main comment) and has four parts:

1. **Header**: first line `**F<n> · <severity> · <category>: <plain title>**` matching its row in the Findings table.
2. **Problem (1–3 sentences)**: start from what the code or doc is trying to do, state what happens instead and why it matters, then cite identifiers or `main:<path>:<line>` as needed.
3. **`Suggested fix:`**: one sentence stating what to change, followed by a `suggestion` block when replacing the exact commented lines (verify indentation, target lines, and syntax), or a fenced code block otherwise.
4. **`Evidence:`**: one short line with the check/command result or exact trace (for unresolved findings, what settles it).

Keep prose plain and direct: short sentences, active voice, one line per paragraph (GitHub turns newlines into line breaks), no HTML, no hedging, no filler, and no praise. Write in English unless asked otherwise.

### Examples

Correctness:

````markdown
**F1 · critical · correctness: failed action-model startup leaves env daemon running**

When the action-model server fails `try_wait_server`, `init_runtime` returns without tearing down the environment daemon spawned right before it, leaving an orphaned simulator process holding GPU memory after CLI startup aborts. Passing `daemons` into `try_spawn_server` lets `stop_owned_daemons` clean up earlier owned servers (`main:rpent/robots/runtime.py:53`).

Suggested fix: pass `daemons` into `try_spawn_server` and `try_wait_server` instead of recording the process only after readiness succeeds.

```python
env_daemon, env_rpc = try_spawn_server(daemons, dashboard_events, "env", spawn_env)
```

Evidence: `rpent/robots/runtime.py:66` calls `stop_owned_daemons(daemons, ...)` only on daemons already registered in `daemons`.
````

Design:

````markdown
**F2 · major · design: robot-name branch in CLI runner for local memory**

The CLI runner now branches on `args.robot == "robocasa"` to skip global memory sync before starting services, forcing the next robot with local-only memory to add another branch here. Per-robot memory setup belongs on `RobotSpec.prepare_memory` (`main:rpent/robots/robot_spec.py:98`).

Suggested fix: implement `prepare_memory` on the RoboCasa `RobotSpec` and remove the robot-name check from `rpent/cli/main.py`.

```python
def _prepare_robocasa_memory(args: argparse.Namespace, config: RunConfig) -> None:
    config.prompt_vars["memory_dir"] = str(args.memory_dir or get_memory_dir("robocasa"))
```

Evidence: `rpent/robots/robot_spec.py:98` defines `prepare_memory` for robot-owned sync (`code/patterns.md` § Generalize; don't special-case).
````

Docs style:

````markdown
**F3 · minor · docs-style: user guide tours internal session and toolkit modules**

This paragraph explains task execution by listing `rpent/session/base.py` and `rpent/tools/toolkit.py`, which a first-time user never opens or edits to run a task (`docs/rules/base.md`, "Write about what the reader does, not about the code").

Suggested fix: describe the CLI options and saved run artifacts under `--output-dir` instead of internal module paths.

```suggestion
Each run records step images, tool calls, and the final environment status under ``--output-dir``, so you can inspect why an episode stopped without opening source files.
```

Evidence: `docs/code-name` flagged 2 internal paths on this user guide page.
````
