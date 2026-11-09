# Review Bot

PR slash commands backed by [PR-Agent](https://github.com/The-PR-Agent/pr-agent)
on self-hosted runners.

| Command | What runs | What it posts |
| --- | --- | --- |
| `/review-design` | PR-Agent `review` with design-only instructions | Design and contract findings |
| `/review-pr` | PR-Agent `review` | Code review findings |
| `/review-autofix` | PR-Agent `improve` with committable suggestions | Suggested changes the author may accept or reject in GitHub |

The command must be the first line of the comment. Only owners, members, and
collaborators may trigger it. Each run posts its own review comment; PR-Agent's
persistent-comment mode is disabled so the design and code reviews cannot
overwrite each other.

## Runner contract

The review job runs on a self-hosted runner labelled `review-bot`. The runner
must provide:

- the `pr-agent` executable, either as `PR_AGENT` or on `PATH`;
- model access through PR-Agent's environment variables, such as
  `ANTHROPIC__KEY`, `ANTHROPIC_API_BASE`, `CONFIG__MODEL`, and any token-budget
  settings the model requires;
- outbound access to the GitHub API and the model endpoint.

The workflow supplies `GITHUB__USER_TOKEN` from `github.token`; no long-lived
token is stored in the repository. It never checks out or executes PR code, and
PR-Agent reads the diff through the GitHub API.

## Scheduling

Review jobs are grouped by PR. Commands on the same PR run one at a time;
commands on different PRs run in parallel when more than one `review-bot`
runner is online.
