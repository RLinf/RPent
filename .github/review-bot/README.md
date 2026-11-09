# Review Bot

PR slash commands for this repository, backed by
[PR-Agent](https://github.com/The-PR-Agent/pr-agent) on self-hosted runners.

| Command | What runs | What it posts |
| --- | --- | --- |
| `/review-design` | PR-Agent `review` with design-only instructions | Design and contract findings |
| `/review-pr` | PR-Agent `review` | Code review findings |
| `/review-autofix` | PR-Agent `improve` with committable suggestions | Suggested changes the author may accept or reject in GitHub |

The command must be the first line of the comment. Only owners, members, and
collaborators may trigger it. `/review-pr` keeps one rolling review comment per
PR; `/review-design` posts a separate comment for each run so the two do not
overwrite each other.

## Runner

The review job runs on a self-hosted runner labelled `review-bot`. On the
review host, install PR-Agent and Python 3.12 under
`/mnt/public/rpent_review_agent/`:

```bash
R=/mnt/public/rpent_review_agent
export UV_CACHE_DIR="$R/.uv_cache" UV_PYTHON_INSTALL_DIR="$R/python" TMPDIR="$R/tmp"
# On hosts that cannot reach GitHub releases, point at a reachable mirror.
export UV_PYTHON_INSTALL_MIRROR="https://ghproxy.net/https://github.com/astral-sh/python-build-standalone/releases/download"
uv python install --no-bin 3.12
uv venv --python "$R"/python/cpython-3.12*/bin/python3.12 "$R/venv"
UV_DEFAULT_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple \
  uv pip install --python "$R/venv/bin/python" "pr-agent[github]"
```

Model credentials are read at run time from `$R/etc/model.env` (mode 600):

```bash
ANTHROPIC__KEY=...
ANTHROPIC_API_BASE=https://cloud.infini-ai.com/maas
```

Install the GitHub Actions runner under `$R/runner`, register it with the
`review-bot` label, and start it with `RUNNER_ALLOW_RUNASROOT=1 ./run.sh` when
the service runs as root. It is not installed as a system service because the
review host keeps all state under `$R`.

Register the runner with the `review-bot` label. The workflow never checks out
or executes PR code; PR-Agent reads the diff through the GitHub API.

## Model credentials

Credentials are read from `/mnt/public/rpent_review_agent/etc/model.env`
(mode 600) at run time and are not stored in the repository:

| Variable | Value |
| --- | --- |
| `ANTHROPIC__KEY` | Key for the Anthropic-compatible gateway |
| `ANTHROPIC_API_BASE` | Gateway base URL, without `/v1` |

`CONFIG__MODEL` in the workflow is `anthropic/deepseek-v4.1-flash-1017`. The
model is a reasoning model, so the workflow also raises
`CONFIG__MAX_MODEL_TOKENS`/`CONFIG__CUSTOM_MODEL_MAX_TOKENS` to 128000 and keeps
`CONFIG__MAX_OUTPUT_TOKENS` at 48000; the defaults truncate prompts or leave no
room for the answer after thinking.
