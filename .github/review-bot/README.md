# Review Bot

PR slash commands backed by Claude Code on self-hosted runners.

| Command | What runs | What it posts |
| --- | --- | --- |
| `/review-design` | Claude Code with the design prompt | Design and contract findings |
| `/review-pr` | Claude Code with the review prompt | Code review findings |
| `/review-suggest` | Claude Code with the suggestion prompt and a JSON schema | Inline `suggestion` comments the author may accept or reject |
| `/review-reproduce` | Claude Code with the reproduction prompt and shell access on a GPU runner | A reproduction report and uploaded artifacts |

The command must be the first line of the comment. Only owners, members, and
collaborators may trigger it.

The prompts and the procedure the agent follows live in a private review kit
that the runner provides at `$REVIEW_KIT`, not in this repository. The workflow
copies the kit to `.review-kit/` in the workspace, then runs the prompt for the
requested command. That keeps internal review rules and deployment details out
of the public tree while still letting the code review use this repository's
own guidance.

The review agent checks out the PR head, so it can read callers and full files
rather than only the diff. It also receives `.review/diff.patch`, the base
version of each changed file, and the PR description. `CLAUDE.md`, `AGENTS.md`,
`CONTRIBUTING.md`, and `.agents/skills/` are restored from the base branch
before the agent starts: a pull request cannot rewrite the guidance that
reviews it.

## Runner contract

Commands run on a self-hosted runner labelled `review-bot`; GPU reproduction
runs on one labelled `gpu`. The runners must provide:

- `claude-review`, a wrapper that launches Claude Code with the review model and
  credentials already configured, either on `PATH` or as `CLAUDE_REVIEW`;
- `claude-reproduce`, the same launcher with shell and write tools enabled,
  either on `PATH` or as `CLAUDE_REPRODUCE`;
- `REVIEW_KIT`, the checkout of the private review kit (`prompts/`, `skills/`,
  `schemas/`);
- outbound access to the model endpoint and to the reproduction mirrors.

The wrappers own deployment details: API credentials, base URL, model name,
tool allowlists, and token budgets. The workflow only passes the prompt on
stdin and expects the answer on stdout. It sets `MAX_BUDGET_USD` as a per-run
cost ceiling and leaves all other harness flags to the wrapper.

### Review isolation

The review job checks out PR code, but the agent runs with:

- `Read`, `Grep`, and `Glob` only — no shell, no writes, no network tools;
- restricted mode and no MCP servers, so repository settings and hooks do not
  apply;
- no GitHub token, so it cannot comment, push, or read other resources.

The diff and the base file versions are computed by the workflow, not by the
agent.

The result is uploaded as a workflow artifact and published by a separate job
on `ubuntu-latest` with a repository-scoped token. `/review-suggest` validates
each suggestion against the diff and folds anything outside a commentable line
into the review body.

### GPU reproduction

`/review-reproduce` checks out the PR head and runs the kit's reproduction
prompt with `claude-reproduce`, which adds shell, write, and edit tools, under:

- `CUDA_VISIBLE_DEVICES=0`, `MUJOCO_GL=egl`, `PYOPENGL_PLATFORM=egl`;
- `PR_NUMBER`, `PR_URL`, `BASE_REF`, `HEAD_SHA`, `PR_REPOSITORY`;
- `REPRODUCE_OUTPUT_DIR`, an empty directory outside the checkout for logs and
  artifacts.

The agent builds an isolated environment and runs the smallest experiment that
distinguishes the PR from its base, guided by the kit's reproduction skill. It
writes only to the workspace and `REPRODUCE_OUTPUT_DIR`, and the prompt forbids
committing, pushing, commenting, or touching the network beyond package
mirrors. The report becomes a PR comment; the artifact directory is uploaded
and linked from that comment. Because this executes PR code on a GPU runner,
only trusted collaborators should trigger it.

## Scheduling

Commands are grouped by PR. Commands on the same PR run one at a time; commands
on different PRs run in parallel when enough runners are online.
