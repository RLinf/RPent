# EFE-compatible coding evaluation

[English](README.md) | [简体中文](README.zh-CN.md)

Status: experimental, **EFE-compatible / non-official**.
Harbor `0.23.0` is the adapter's supported host version. It requires Python 3.12.
The installed RPent runner also uses Python 3.12. It installs no robot extras.

## Scope

RPent runs the existing `CodexPlanner` inside Harbor's Agent Container with
RPent's prompt bundle, an optional loaded memory corpus, RPent's shared file
and memory tools with their authorization rules, and optional multi-session
handoff. EFE publishes no RobotSpec, so the session loads no robot primitive
tools, VLA, or robot runtime. The agent still operates the simulator only
through the task's public Robot Client.

Harbor owns isolation and execution budgets. EFE's verifier scores independently.

`finish(status="completed")` means the agent finished its own work.
It does not mean task success and produces no Reward. The CLI exit code reports
only whether the Runner failed.
This phase supports no online RL, Pi0.5, VLA, or task and scoring changes.

## Trial isolation

Each run creates independent state:

- `memory/` is the empty `MemoryManager` root for that Trial. The Runner does
  not download or merge a corpus. An uploaded `memory_source` directory
  replaces this root.
- `codex-home/` is the `CODEX_HOME` for that Trial. The Runner does not copy
  host history, sessions, or authentication files.
- Model and credentials come only from explicit environment variables. The
  adapter forwards `OPENAI_API_KEY`, `CODEX_API_KEY`, `OPENAI_BASE_URL`,
  `CODEX_BASE_URL`, and the Codex budget variables.

Without a corpus the run measures RPent without prior experience. The Runner
sets `CODEX_HOME` in its own process. Run one Trial per agent process.

## Install and entry points

The host needs the Docker environment required by EFE, cloud model credentials,
and an EFE checkout. The CPU-only simulator does not need a GPU. The first run
still downloads the container and task assets.
Prepare the host and `tasks/autobio/autobio-pickup-i00-standard/` as described
by EFE. That task declares CPU physics and OSMesa rendering.
Its agent container uses Python 3.12.14, workdir `/app`, and a client under
`/opt/tools`. These settings come from source inspection. See Validation
status for real runs. Keep the original EFE task, simulator, verifier, and
resource limits.

The adapter packages Python sources under `rpent/` and `robots/`, package data
declared in `pyproject.toml`, and installation metadata from the current
checkout. It uploads the source to `/opt/rpent-efe/source` and creates a `venv`
with the task Python. The selected task provides Python 3.12; the container
does not need `uv`.
The source archive excludes the host `.git`, tests, credential files, and run
logs. `robots/` is part of the existing RPent distribution and starts no Robot
Runtime.
The host `PYTHONPATH` must import the adapter from this repository.
`source_root` defaults to the repository root of the adapter's source checkout.
Pass `source_root` explicitly when the adapter is imported from an installed
wheel.

```bash
export RPENT_ROOT=/absolute/path/to/RPent
export EFE_ROOT=/absolute/path/to/embodied-first-exam
export EFE_TASK="$EFE_ROOT/tasks/autobio/autobio-pickup-i00-standard"
export EFE_MODEL=openai/your-model
export OPENAI_API_KEY=your-key
```

Do not write credentials into the repository. The adapter forwards only model
environment variables and explicit Harbor `--agent-env` settings.
It maps `OPENAI_API_KEY` and `OPENAI_BASE_URL` to RPent `CODEX_API_KEY` and
`CODEX_BASE_URL`. Existing `CODEX_*` settings take precedence.
The adapter does not forward the host `CODEX_HOME`. The Runner creates that
directory inside the agent container for each Trial.

## P0: Native baseline

First follow EFE's official run instructions for the same original task with
native Codex. This command uses Harbor's native agent and does not load RPent:

```bash
uv run --project "$EFE_ROOT" harbor run \
  --path "$EFE_TASK" --agent codex --model "$EFE_MODEL" \
  --n-concurrent 1 --n-attempts 1 \
  --jobs-dir "$RPENT_ROOT/logs/efe-native"
```

Inspect the Trial `result.json` and the verifier output.
Container startup, simulator interaction, and verifier execution need real
evidence. A clean native-agent exit does not prove task success.

## P1: RPent Installed Agent [recommended]

```bash
PYTHONPATH="$RPENT_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
uv run --project "$EFE_ROOT" harbor run \
  --path "$EFE_TASK" \
  --agent rpent.benchmark.efe.harbor_agent:RPentEfeAgent \
  --model "$EFE_MODEL" --n-concurrent 1 --n-attempts 1 \
  --agent-kwarg source_root="$RPENT_ROOT" \
  --agent-kwarg timeout_s=3500 --agent-kwarg max_turns=200 \
  --agent-kwarg reasoning_effort=none \
  --jobs-dir "$RPENT_ROOT/logs/efe-rpent"
```

The default workdir is `/app`. Pass `--agent-kwarg workdir=/path` for a task
with a different workdir.
Harbor's task budget is always the outer hard limit. `timeout_s` is the RPent
inner limit. The selected task has a 3600-second original agent limit, and the
adapter defaults to 3500 seconds so RPent writes `session.json` before Harbor
cancels the agent. Set `timeout_s` below the task's agent timeout; an equal
value lets Harbor time out first and lose the report.
Use the same effective time budget in comparison runs.
`max_turns` counts model responses in the existing planner, not shell commands.

The alternative entry point diagnoses an existing Agent Container:

```bash
rpent-efe --instruction-file /logs/agent/rpent-instruction.txt \
  --workdir /app --output-dir /logs/agent/manual-rpent \
  --model your-model --timeout-s 600 --max-turns 200
```

The CLI creates no Harbor container and starts no simulator or verifier.
Full evaluation uses the Installed Agent entry point.

## P2: Prior experience and exploration

Add `memory_source` to load a prior RPent corpus and `explore_sessions` to run
several independent sessions.

```bash
PYTHONPATH="$RPENT_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
uv run --project "$EFE_ROOT" harbor run \
  --path "$EFE_TASK" \
  --agent rpent.benchmark.efe.harbor_agent:RPentEfeAgent \
  --model "$EFE_MODEL" --n-concurrent 1 --n-attempts 1 \
  --agent-kwarg source_root="$RPENT_ROOT" \
  --agent-kwarg timeout_s=3500 --agent-kwarg max_turns=200 \
  --agent-kwarg reasoning_effort=none \
  --agent-kwarg memory_source="$RPENT_ROOT/memory/efe" \
  --agent-kwarg explore_sessions=3 \
  --jobs-dir "$RPENT_ROOT/logs/efe-rpent-exp"
```

`memory_source` is a host directory. The adapter uploads it to
`/opt/rpent-efe/memory` before the session. Without it, the session starts with
an empty memory root. RPent's file tools keep the published corpus read-only
unless `explore_sessions` is greater than 1; exploration writes working notes
only under the current inbox. Native Codex shell tools are not covered by that
policy, but each Trial uploads a fresh copy of `memory_source`, so container
writes do not persist to the host.

`explore_sessions` runs independent planner sessions in sequence. Each session
gets a fresh Codex context and reads the previous sessions' notes. `timeout_s`
is the total budget across sessions; each session receives the remaining time.
`max_turns` applies to each session.

No session is promoted into the corpus, because EFE's verifier runs outside the
agent container. Treat `memory_source` as the only prior experience.

## P3: Records and comparison

Harbor synchronizes each Trial's agent logs. RPent artifacts are under
`agent/rpent/`:

| File | Content |
| --- | --- |
| `session.json` | Original task, configuration, memory mode, versions, per-session stats, agent report, usage, elapsed time, and errors |
| `codex.txt` | Readable planner transcript |
| `codex.txt.stream.jsonl` | Raw Codex SDK events, including native tool calls |
| `codex.txt.last` | Final reply; absent when startup fails |
| `run.log` | RPent run log |

`rpent-source.json` in the agent log root stores the SHA-256 of the uploaded
source archive.
If Harbor stops the agent before it writes the report, the adapter sums the
last usage event of every session trace. An all-zero event is treated as
unobserved, not as measured usage.
`usage-recovery.json` then marks `complete=false`. These counters cover only
responses written to the traces.
These logs contain no final Reward from RPent. Harbor's Trial `result.json`
retains the EFE verifier score and error classification.

Fix the task and asset revision, model, endpoint, Codex version, reasoning
effort, time limit, and retry policy for comparison.

| Setting | Native `--agent codex` | RPent |
| --- | --- | --- |
| Codex binary | `@openai/codex@latest` from npm unless `--agent-kwarg version=` pins it | bundled `openai-codex-cli-bin`; pass `CODEX_BIN` with `--agent-env CODEX_BIN=...` to override |
| Reasoning effort | unset, so the Codex or model default applies | `none` |
| System prompt | none; the instruction is the user prompt | RPent system prompt |
| Project docs and skills | Codex defaults | `project_doc_max_bytes=0`; repository skills disabled |
| Context window and compaction | Codex defaults | Codex defaults unless `CODEX_MODEL_CONTEXT_WINDOW` or `CODEX_AUTO_COMPACT_TOKEN_LIMIT` is set |
| Turn cap | none | `max_turns` per session, default 200 |
| Time limit | task agent timeout | `timeout_s`, default 3500 in the adapter |
| Sandbox and approvals | `--dangerously-bypass-approvals-and-sandbox` | SDK `full_access` and `deny_all` approvals |

Align the native agent with `--agent-kwarg reasoning_effort=none` and
`--agent-kwarg version=<codex-version>`. Read the RPent CLI version from
`session.json` `versions`; do not assume the two installs match.
`session.json` records the `openai-codex` and `openai-codex-cli-bin` versions.
The container install log records resolved dependency versions.
Keep both complete Trial results and logs. Do not compare only the `finish` text.

| Metric | Read and interpretation |
| --- | --- |
| Task result | Use the verifier Reward; list infrastructure or agent errors separately |
| Token | Compare input, cache, and output in Harbor Context; input includes cache |
| Time | Compare agent execution time separately from total Trial time with installation |
| Tool behavior | Read the raw trace; separate shell, file operations, and Robot Client requests |
| Cost | The adapter does not estimate `cost_usd`; keep missing measurements empty |

The adapter writes measured token totals to Harbor `AgentContext` and
`model_usage`.
The current delivery is the raw SDK trace, not an ATIF converter.
Do not claim RPent is better than native Codex before a real experiment completes.

## Offline validation

```bash
pytest tests/unit_tests/rpent/benchmark/efe -q
pytest tests/integration/efe -q
```

The basic offline tests need no Docker or model credentials.
Harbor contract tests need `harbor==0.23.0` in a Python 3.12 environment.
The real Harbor Smoke test needs an explicit task and model; see its opt-in
condition in the test file.
That Smoke test waits up to 4 hours for container build, the 3500-second agent
budget, and task teardown.
Offline success proves only interfaces, tool transport, and artifact contracts.
It does not prove EFE task success.

## Validation status

One Oracle, one native Codex, and one RPent Trial ran on WSL with Docker
`29.9.2` on 2026-11-02. All three verifiers returned `success=1`.
P0 and P2 used a local model proxy with a dummy key, not a public API key.
Each path ran once; these results are not a statistical comparison.
The RPent Trial used an empty `memory/` and a `codex-home/` with no
`auth.json`.
These Trials cover the Harbor adapter, the source upload, and the usage
recovery. The RPent Trial ran before the adapter loaded RPent's prompt and
shared file tools, so it does not cover the current agent configuration.

The `sim` image was built locally from the pinned AutoBio commit because
BuildKit could not clone it through this host's proxy. The task and scoring
files were not changed.

## Sources

- [EFE setup and execution](https://github.com/mll-lab-nu/embodied-first-exam/blob/149a3836999cd0f4596b579e5e1a77bd28f3c478/docs/protocol/running.md)
- [Selected CPU task and limits](https://github.com/mll-lab-nu/embodied-first-exam/blob/149a3836999cd0f4596b579e5e1a77bd28f3c478/tasks/autobio/autobio-pickup-i00-standard/task.toml)
- [Harbor custom agent lifecycle](https://github.com/harbor-framework/harbor/blob/1e5c5c6db929a10a140d05e606882c671ae20729/docs-mintlify/agents/custom-agents.mdx)
- [Harbor usage fields](https://github.com/harbor-framework/harbor/blob/1e5c5c6db929a10a140d05e606882c671ae20729/src/harbor/models/agent/context.py)
