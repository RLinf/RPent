# EFE-compatible tool-driven evaluation

[English](README.md) | [简体中文](README.zh-CN.md)

Status: experimental, **EFE-compatible / non-official**.
Harbor `0.23.0` is the adapter's supported host version. It requires Python 3.12.
The installed RPent runner also uses Python 3.12. It installs no robot extras.

## Scope

RPent runs the existing `CodexPlanner` inside Harbor's Agent Container as the
trial's only agent process. `robots/efe` treats the task's public
`rcb_client.Robot` service as the robot and registers it with RPent's Robot
Registry. Both arms call `efe_spec`, `efe_observe`, `efe_status`, and
`efe_recover`. The raw arm also gets `efe_submit`, which sends one action chunk
in a layout from the spec. The skills arm instead gets `move_to_xyz` and
`set_gripper`: a closed IK loop over public proprioception and a gripper command
that holds the arm pose. A task with a reset budget also gets `efe_reset`; a
task with agent-ended episodes also gets `efe_end_episode`.

The agent operates the simulator only through that client. RPent starts no
simulator, sends no reset unless the task declares a budget and the planner
asks for one, reads no privileged state, and does not change the task or
scoring.

Harbor owns isolation and execution budgets. EFE's verifier scores independently.

`finish(status="completed")` means the agent finished its own work. It does not
mean task success and produces no Reward. `solved()` reports the service's own
end reason, and the Verifier still grades the episode. The CLI exit code
reports only whether the Runner failed. This phase supports no online RL,
Pi0.5, VLA, or task and scoring changes.

## Task and mode coverage

EFE ships 32 benchmark families and 2109 task directories in three modes:
1066 `standard` (`eai-standard/2.1`, `2.2`, `2.4`), 84 `standard-realtime`
(`eai-standard/2.3`), and 959 `privileged`. The adapter takes every task's
capabilities from the service spec, so it carries no task-specific code and
covers all standard and standard-realtime tasks. Standard-realtime runs on a
clock that does not stop, but the service holds the last targets while the
planner thinks, so the same tools apply. Privileged tasks are not supported:
they share the simulator container and accept a trajectory or policy
deliverable instead of a service client.

## Raw tools and skills

The raw arm exposes the service protocol directly: the planner reads the
capability spec, chooses joint targets, and sends action chunks with
`efe_submit`.

The skills arm replaces `efe_submit` with one high-level layer over the same
client. `move_to_xyz` turns a world-frame target into an IK solution, submits
short chunks, and re-reads proprioception until the tool frame reaches the
target, so the planner can reason in Cartesian space instead of joint angles.
`set_gripper` sets the finger opening while holding the current arm pose. The
arm exposes no joint-level action tool.

A task whose action layout does not match what the skills need (kinematics
joint order plus one trailing gripper column) gets no skills: the toolkit keeps
the raw `efe_submit` path, logs the reason, and records it as `skills_error` in
`session.json`. The prompt lists the tools the session actually registers. The
two arms therefore differ in available tools, not in model, budget, or task.

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
and an EFE checkout. Follow EFE's instructions to prepare the host and the
task. Simulator requirements come from the task. The first run still downloads
the container and task assets. Keep the original EFE task, simulator, verifier,
and resource limits.

The adapter packages Python sources under `rpent/` and `robots/`, package data
declared in `pyproject.toml`, and installation metadata from the current
checkout. It uploads the source to `/opt/rpent-efe/source` and creates a `venv`
with the task Python. The task image provides the Python the runner uses; the
container does not need `uv`.
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
export EFE_TASK="$EFE_ROOT/tasks/<family>/<task>"
export EFE_MODEL=openai/your-model
export OPENAI_API_KEY=your-key
```

Do not write credentials into the repository. The adapter forwards only model
environment variables and explicit Harbor `--agent-env` settings.
It maps `OPENAI_API_KEY` and `OPENAI_BASE_URL` to RPent `CODEX_API_KEY` and
`CODEX_BASE_URL`. Existing `CODEX_*` settings take precedence.
The adapter does not forward the host `CODEX_HOME`. The Runner creates that
directory inside the agent container for each Trial.

## Installed Agent entry point

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
inner limit. Set it below the task's agent timeout: an equal value lets Harbor
time out first and lose the report. RPent writes `session.json` at the end of a
Trial, so leave enough headroom.
`max_turns` counts model responses in the existing planner, not shell commands.

This custom agent declares no model endpoint, so Harbor cannot infer the
inference host. Pass `--allow-agent-host <model-host>`. An explicit allowlist
replaces the defaults, so also allow the package hosts the install needs:
`--allow-agent-host pypi.org --allow-agent-host files.pythonhosted.org`.
The first container install runs `pip`, and Harbor's default setup timeout is
360 seconds; add `--agent-setup-timeout-multiplier 4` on hosts where a cold
install is slower.

The alternative entry point diagnoses an existing Agent Container:

```bash
rpent-efe --instruction-file /logs/agent/rpent-instruction.txt \
  --workdir /app --output-dir /logs/agent/manual-rpent \
  --model your-model --timeout-s 600 --max-turns 200
```

The CLI creates no Harbor container and starts no simulator or verifier.
Full evaluation uses the Installed Agent entry point.

## Raw and skills arms

Both arms run the same Runner. `skills=false` registers the raw tools; the
default skills arm replaces the joint-level submit tool with the skill layer.
The arm comparison changes the tool set, not the model or the budget. The
Runner defaults to the Codex `read-only` sandbox: shell commands cannot reach
the simulator, so only RPent tools act on the episode. Pass
`--agent-kwarg sandbox=full-access` to let the planner script the service
directly. That mode measures the agent's free-form behavior; its shell actions
do not appear in the RPent tool counters. The generic `rpent --robot efe`
entry point builds the shared planner without a sandbox setting, so it always
runs with `full-access`; use `rpent-efe` or the Harbor adapter when the shell
must stay away from the simulator.

```bash
# RPent-Raw
PYTHONPATH="$RPENT_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
uv run --project "$EFE_ROOT" harbor run \
  --path "$EFE_TASK" \
  --agent rpent.benchmark.efe.harbor_agent:RPentEfeAgent \
  --model "$EFE_MODEL" --n-concurrent 1 --n-attempts 1 \
  --agent-kwarg source_root="$RPENT_ROOT" \
  --agent-kwarg timeout_s=3500 --agent-kwarg max_turns=200 \
  --agent-kwarg skills=false \
  --jobs-dir "$RPENT_ROOT/logs/efe-raw"

# RPent-Skills
PYTHONPATH="$RPENT_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
uv run --project "$EFE_ROOT" harbor run \
  --path "$EFE_TASK" \
  --agent rpent.benchmark.efe.harbor_agent:RPentEfeAgent \
  --model "$EFE_MODEL" --n-concurrent 1 --n-attempts 1 \
  --agent-kwarg source_root="$RPENT_ROOT" \
  --agent-kwarg timeout_s=3500 --agent-kwarg max_turns=200 \
  --jobs-dir "$RPENT_ROOT/logs/efe-skills"
```

Summarize both runs with the Verifier reward and the RPent counters:

```bash
rpent-efe-report "$RPENT_ROOT/logs/efe-raw" "$RPENT_ROOT/logs/efe-skills"
```

The report reads `result.json` (Verifier reward, tokens, wall time) and
`agent/rpent/session.json` (tool calls, failed calls, skill calls, submit calls,
control steps). It labels each trial from `skills` in `session.json`, so both
arms can run under one root. Separate `--jobs-dir` paths still keep the logs
readable. `--json` prints the same data for scripts.

Keep the model, endpoint, reasoning effort, task revision, time limit, and
attempt count fixed across arms. A single trial is not a comparison.

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
