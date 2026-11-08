# Review Bot

PR slash commands for this repository, backed by
[PR-Agent](https://github.com/The-PR-Agent/pr-agent) on self-hosted runners.

| Command | What runs | What it posts |
| --- | --- | --- |
| `/review-design` | PR-Agent `review` with design-only instructions | Design and contract findings |
| `/review-pr` | PR-Agent `review` | Code review findings |
| `/review-autofix` | PR-Agent `improve` with committable suggestions | Suggested changes the author may accept or reject in GitHub |

The command must be the first line of the comment. Only owners, members, and
collaborators may trigger it.

## Runner

The review job runs on a self-hosted runner labelled `review-bot`. On the
review host, install PR-Agent and Python 3.12 under
`/mnt/public/rpent_review_agent/`:

```bash
R=/mnt/public/rpent_review_agent
export UV_CACHE_DIR="$R/.uv_cache" UV_PYTHON_INSTALL_DIR="$R/python" TMPDIR="$R/tmp"
uv python install 3.12
uv venv --python 3.12 "$R/venv"
uv pip install --python "$R/venv/bin/python" "pr-agent[github]"
```

Register the runner with the `review-bot` label. The workflow never checks out
or executes PR code; PR-Agent reads the diff through the GitHub API.

## Repository secrets

| Secret | Value |
| --- | --- |
| `ANTHROPIC_API_KEY` | Key for the Anthropic-compatible gateway |
| `ANTHROPIC_API_BASE` | Gateway base URL, without `/v1` |

`CONFIG__MODEL` in the workflow is `anthropic/deepseek-v4.1-flash-1017`; change
it if the gateway serves a different model id.
