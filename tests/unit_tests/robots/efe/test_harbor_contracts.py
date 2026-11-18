# Copyright 2026 The RPent Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Contracts against the optional Harbor dependency used by EFE."""

from __future__ import annotations

import asyncio
import io
import json
import os
import shlex
import tarfile
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

pytest.importorskip(
    "harbor", reason="Install EFE's harbor==0.23.0 in a Python 3.12 environment."
)

AgentContext = import_module("harbor.models.agent.context").AgentContext
RPentEfeAgent = import_module("robots.efe.harbor_agent").RPentEfeAgent


class UploadEnvironment:
    def __init__(self, default_user=None):
        self.uploads = {}
        self.archive_members = []
        self.default_user = default_user

    async def upload_file(self, *, source_path, target_path):
        self.uploads[target_path] = Path(source_path).read_bytes()
        if str(source_path).endswith(".tar.gz"):
            with tarfile.open(source_path) as archive:
                self.archive_members = archive.getnames()


def test_run_preserves_instruction_and_quotes_paths(tmp_path, monkeypatch):
    agent = RPentEfeAgent(
        logs_dir=tmp_path,
        model_name="openai/test-model",
        workdir="/app/task with spaces",
        timeout_s=42,
        max_turns=7,
    )
    environment = UploadEnvironment()
    execute = AsyncMock()
    monkeypatch.setattr(agent, "exec_as_agent", execute)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.delenv("CODEX_API_KEY", raising=False)
    instruction = "Task with 'quotes', $(commands), and\nnewlines."

    asyncio.run(agent.run(instruction, environment, AgentContext()))

    assert (
        environment.uploads[
            str(agent.environment_logs_dir / "rpent-instruction.txt")
        ].decode()
        == instruction
    )
    assert execute.call_args.kwargs["cwd"] == "/app/task with spaces"
    assert execute.call_args.kwargs["env"]["CODEX_API_KEY"] == "test-key"
    assert execute.call_args.kwargs["timeout_sec"] == 72
    command = shlex.split(execute.call_args.kwargs["command"])
    assert command[:4] == [
        "/opt/rpent-efe/venv/bin/rpent",
        "--robot",
        "efe",
        "--planner",
    ]
    assert command[4] == "codex"
    assert command[command.index("--model") + 1] == "test-model"
    assert command[command.index("--planner-timeout-s") + 1] == "31"
    assert command[command.index("--efe-connect-timeout-s") + 1] == "10.5"
    assert command[command.index("--efe-trial-timeout-s") + 1] == "42"
    assert command[command.index("--max-turns") + 1] == "7"
    assert command[command.index("--reasoning-effort") + 1] == "none"
    assert command[command.index("--codex-sandbox") + 1] == "read-only"
    assert command[command.index("--memory-profile") + 1] == "local"
    assert command[command.index("--memory-dir") + 1] == "/logs/agent/rpent/memory"
    assert command[command.index("--instruction-file") + 1].endswith(
        "rpent-instruction.txt"
    )
    assert command[command.index("--workdir") + 1] == "/app/task with spaces"
    assert command[command.index("--output-dir") + 1] == "/logs/agent/rpent"
    assert execute.call_args.kwargs["env"]["CODEX_HOME"] == (
        "/logs/agent/rpent/codex-home"
    )
    assert "--timeout-s" not in command
    assert "--no-skills" not in command


def test_run_names_the_task_from_the_trial_configuration(tmp_path, monkeypatch):
    """The uploaded instruction names the log directory, not the task."""
    trial_config = {"task": {"path": "/tasks/pick_the_tube"}}
    (tmp_path / "config.json").write_text(json.dumps(trial_config), encoding="utf-8")
    agent = RPentEfeAgent(logs_dir=tmp_path / "logs", model_name="openai/test-model")
    environment = UploadEnvironment()
    execute = AsyncMock()
    monkeypatch.setattr(agent, "exec_as_agent", execute)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    asyncio.run(agent.run("Task", environment, AgentContext()))

    command = shlex.split(execute.call_args.kwargs["command"])
    assert command[command.index("--task-name") + 1] == "pick_the_tube"


def test_run_names_a_dataset_task_from_its_registry_name(tmp_path, monkeypatch):
    trial_config = {"task": {"name": "embodied-first-exam/pick-the-tube"}}
    (tmp_path / "config.json").write_text(json.dumps(trial_config), encoding="utf-8")
    agent = RPentEfeAgent(logs_dir=tmp_path / "logs", model_name="openai/test-model")
    environment = UploadEnvironment()
    execute = AsyncMock()
    monkeypatch.setattr(agent, "exec_as_agent", execute)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    asyncio.run(agent.run("Task", environment, AgentContext()))

    command = shlex.split(execute.call_args.kwargs["command"])
    assert command[command.index("--task-name") + 1] == "pick-the-tube"


def test_run_without_a_trial_configuration_leaves_the_task_name_implicit(
    tmp_path, monkeypatch
):
    agent = RPentEfeAgent(logs_dir=tmp_path, model_name="openai/test-model")
    environment = UploadEnvironment()
    execute = AsyncMock()
    monkeypatch.setattr(agent, "exec_as_agent", execute)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    asyncio.run(agent.run("Task", environment, AgentContext()))

    command = shlex.split(execute.call_args.kwargs["command"])
    assert "--task-name" not in command


def test_run_forwards_the_configured_codex_sandbox(tmp_path, monkeypatch):
    agent = RPentEfeAgent(
        logs_dir=tmp_path,
        model_name="openai/test-model",
        sandbox="full-access",
    )
    execute = AsyncMock()
    monkeypatch.setattr(agent, "exec_as_agent", execute)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.delenv("CODEX_API_KEY", raising=False)

    asyncio.run(agent.run("Task", UploadEnvironment(), AgentContext()))

    command = shlex.split(execute.call_args.kwargs["command"])
    assert command[command.index("--codex-sandbox") + 1] == "full-access"


def test_known_planner_budget_stop_does_not_fail_the_harbor_trial(
    tmp_path, monkeypatch
):
    agent = RPentEfeAgent(logs_dir=tmp_path, model_name="openai/test-model")
    execute = AsyncMock(side_effect=RuntimeError("CLI exited with status 1"))
    monkeypatch.setattr(agent, "exec_as_agent", execute)
    monkeypatch.setattr(agent, "_transcript_reports_budget_stop", lambda _: True)

    asyncio.run(agent.run("Task", UploadEnvironment(), AgentContext()))

    execute.assert_awaited_once()


def test_transcript_budget_stop_detection_is_narrow(tmp_path):
    output_dir = tmp_path / "rpent"
    output_dir.mkdir()
    agent = RPentEfeAgent(logs_dir=tmp_path, model_name="openai/test-model")
    transcript = output_dir / "transcript_task.json"
    transcript.write_text(
        json.dumps({"error": "Codex SDK timed out after 12s"}), encoding="utf-8"
    )
    assert agent._transcript_reports_budget_stop(output_dir)
    transcript.write_text(
        json.dumps({"error": "ConnectionError: Request timed out after 12s"}),
        encoding="utf-8",
    )
    assert not agent._transcript_reports_budget_stop(output_dir)


def test_unknown_codex_sandbox_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="sandbox"):
        RPentEfeAgent(logs_dir=tmp_path, model_name="openai/test", sandbox="none")


def test_run_forwards_trial_local_codex_home_not_host_home(tmp_path, monkeypatch):
    agent = RPentEfeAgent(logs_dir=tmp_path, model_name="openai/test")
    execute = AsyncMock()
    monkeypatch.setattr(agent, "exec_as_agent", execute)
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "host-codex"))

    asyncio.run(agent.run("Task", UploadEnvironment(), AgentContext()))

    assert execute.call_args.kwargs["env"]["CODEX_HOME"] == (
        "/logs/agent/rpent/codex-home"
    )
    assert execute.call_args.kwargs["env"]["CODEX_HOME"] != os.environ["CODEX_HOME"]


def test_install_uploads_one_wheel_and_records_its_hash(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    for filename in (
        "pyproject.toml",
        "README.md",
        "LICENSE",
        "robots/efe/robot_spec.py",
        "robots/__init__.py",
        "rpent/cli/main.py",
    ):
        target = source / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("placeholder")
    wheel = tmp_path / "rpent-0.1.0-py3-none-any.whl"
    wheel.write_bytes(b"wheel")
    agent = RPentEfeAgent(
        logs_dir=tmp_path / "logs", model_name="openai/test", source_root=str(source)
    )
    environment = UploadEnvironment()
    execute = AsyncMock()
    execute.return_value = SimpleNamespace(stdout="rpent==0.1.0\n")
    monkeypatch.setattr(agent, "exec_as_root", execute)
    monkeypatch.setattr(agent, "_build_wheel", lambda output_dir: wheel)

    asyncio.run(agent.install(environment))

    commands = [call.kwargs["command"] for call in execute.await_args_list]
    install = next(command for command in commands if "pip install" in command)
    uploaded = environment.uploads["/opt/rpent-efe/rpent-0.1.0-py3-none-any.whl"]
    assert uploaded == b"wheel"
    assert "python3 -m venv" in install
    assert "--no-cache-dir" in install
    assert "--disable-pip-version-check" in install
    assert "/opt/rpent-efe/rpent-0.1.0-py3-none-any.whl" in install
    assert "pip freeze --all" in commands[-1]
    assert (tmp_path / "logs/setup/pip-freeze.txt").read_text() == "rpent==0.1.0\n"
    wheel_record = json.loads((tmp_path / "logs/rpent-wheel.json").read_text())
    assert wheel_record["filename"] == wheel.name
    assert len(wheel_record["sha256"]) == 64


def test_default_source_root_is_the_repository_root(tmp_path):
    agent = RPentEfeAgent(logs_dir=tmp_path, model_name="openai/test")

    assert (agent.source_root / "pyproject.toml").is_file()
    assert (agent.source_root / "robots/efe/harbor_agent.py").is_file()


def test_post_run_usage_keeps_cached_tokens_inside_input(tmp_path):
    agent = RPentEfeAgent(logs_dir=tmp_path, model_name="openai/test")
    (tmp_path / "rpent").mkdir()
    (tmp_path / "rpent/transcript_task.json").write_text(
        json.dumps(
            {
                "stats": {
                    "usage_observed": True,
                    "total_input_tokens": 100,
                    "total_cached_input_tokens": 40,
                    "total_output_tokens": 12,
                }
            }
        )
    )
    context = AgentContext()

    agent.populate_context_post_run(context)

    assert context.n_input_tokens == 100
    assert context.n_cache_tokens == 40
    assert context.n_output_tokens == 12
    assert context.cost_usd is None
    assert context.model_usage["openai/test"].n_input_tokens == 100


def test_agent_uploads_memory(tmp_path, monkeypatch):
    memory = tmp_path / "corpus"
    (memory / "global").mkdir(parents=True)
    (memory / "global" / "lesson.md").write_text("# Lesson\n")
    agent = RPentEfeAgent(
        logs_dir=tmp_path / "logs",
        model_name="openai/test",
        memory_source=str(memory),
    )
    environment = UploadEnvironment()
    execute = AsyncMock()
    execute.return_value = SimpleNamespace(stdout="")
    agent_execute = AsyncMock(return_value=SimpleNamespace(stdout="1000\n1000\n"))
    monkeypatch.setattr(agent, "exec_as_root", execute)
    monkeypatch.setattr(agent, "exec_as_agent", agent_execute)
    wheel = tmp_path / "rpent-0.1.0-py3-none-any.whl"
    wheel.write_bytes(b"wheel")
    monkeypatch.setattr(agent, "_build_wheel", lambda output_dir: wheel)

    asyncio.run(agent.install(environment))
    asyncio.run(agent.run("Task", environment, AgentContext()))

    # The corpus is staged under the agent log directory, so the trial can read
    # it after the container stops.
    archive_path = "/logs/agent/rpent/memory.tar.gz"
    assert archive_path in environment.uploads
    with tarfile.open(fileobj=io.BytesIO(environment.uploads[archive_path])) as archive:
        assert "./global/lesson.md" in archive.getnames()
    # ``docker cp`` needs the destination directory, and the agent must own the
    # tree so it can extract the archive and remove it even when the task runs
    # the agent as a non-root user.
    root_commands = [call.kwargs["command"] for call in execute.call_args_list]
    staging = [
        command for command in root_commands if "mkdir -p /logs/agent/rpent" in command
    ]
    assert staging == [
        "mkdir -p /logs/agent/rpent && chown -R 1000:1000 /logs/agent/rpent"
    ]
    command = shlex.split(agent_execute.call_args.kwargs["command"])
    assert command[command.index("--memory-dir") + 1] == "/logs/agent/rpent/memory"


def test_malformed_usage_events_are_skipped(tmp_path):
    agent = RPentEfeAgent(logs_dir=tmp_path, model_name="openai/test")
    output_dir = tmp_path / "rpent"
    output_dir.mkdir()
    (output_dir / "transcript_task.json").write_text(
        json.dumps({"stats": {"usage_observed": False}})
    )
    events = [
        {"method": "thread/tokenUsage/updated"},
        {"method": "thread/tokenUsage/updated", "payload": None},
        {"method": "thread/tokenUsage/updated", "payload": {"token_usage": None}},
        {
            "method": "thread/tokenUsage/updated",
            "payload": {"token_usage": {"total": None}},
        },
        {
            "method": "thread/tokenUsage/updated",
            "payload": {
                "token_usage": {
                    "total": {"input_tokens": 4, "output_tokens": 2},
                }
            },
        },
    ]
    (output_dir / "codex.txt.stream.jsonl").write_text(
        "".join(json.dumps(event) + "\n" for event in events)
    )
    context = AgentContext()

    agent.populate_context_post_run(context)

    assert context.n_input_tokens == 4
    assert context.n_output_tokens == 2


def test_zero_usage_trace_is_not_recovered_as_measured(tmp_path):
    agent = RPentEfeAgent(logs_dir=tmp_path, model_name="openai/test")
    output_dir = tmp_path / "rpent"
    output_dir.mkdir()
    (output_dir / "transcript_task.json").write_text(
        json.dumps({"stats": {"usage_observed": False}})
    )
    event = {
        "method": "thread/tokenUsage/updated",
        "payload": {
            "token_usage": {
                "total": {
                    "input_tokens": 0,
                    "cached_input_tokens": 0,
                    "output_tokens": 0,
                }
            }
        },
    }
    (output_dir / "codex.txt.stream.jsonl").write_text(json.dumps(event) + "\n")
    context = AgentContext()

    agent.populate_context_post_run(context)

    assert context.n_input_tokens is None
    assert context.n_cache_tokens is None
    assert context.n_output_tokens is None
    assert not (output_dir / "usage-recovery.json").exists()


def test_codex_key_overrides_openai_key_for_the_default_provider(tmp_path, monkeypatch):
    agent = RPentEfeAgent(logs_dir=tmp_path, model_name="openai/test")
    environment = UploadEnvironment()
    execute = AsyncMock()
    monkeypatch.setattr(agent, "exec_as_agent", execute)
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    monkeypatch.setenv("CODEX_API_KEY", "codex-key")

    asyncio.run(agent.run("Task", environment, AgentContext()))

    assert execute.call_args.kwargs["env"]["OPENAI_API_KEY"] == "codex-key"
    assert execute.call_args.kwargs["env"]["CODEX_API_KEY"] == "codex-key"


def test_trial_codex_home_isolated_from_host_codex_home(tmp_path, monkeypatch):
    agent = RPentEfeAgent(logs_dir=tmp_path, model_name="openai/test")
    environment = UploadEnvironment()
    execute = AsyncMock()
    monkeypatch.setattr(agent, "exec_as_agent", execute)
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "host-codex"))

    asyncio.run(agent.run("Task", environment, AgentContext()))

    assert execute.call_args.kwargs["env"]["CODEX_HOME"] == (
        "/logs/agent/rpent/codex-home"
    )
    assert execute.call_args.kwargs["env"]["CODEX_HOME"] != str(tmp_path / "host-codex")


def test_failed_start_does_not_invent_usage(tmp_path):
    agent = RPentEfeAgent(logs_dir=tmp_path, model_name="openai/test")
    (tmp_path / "rpent").mkdir()
    (tmp_path / "rpent/transcript_task.json").write_text(
        '{"stats": {}, "error": "startup failed"}'
    )
    context = AgentContext()

    agent.populate_context_post_run(context)

    assert context.n_input_tokens is None
    assert context.n_cache_tokens is None
    assert context.n_output_tokens is None
    assert context.model_usage is None


def test_zeroed_stats_without_observed_usage_stay_unknown(tmp_path):
    agent = RPentEfeAgent(logs_dir=tmp_path, model_name="openai/test")
    (tmp_path / "rpent").mkdir()
    (tmp_path / "rpent/transcript_task.json").write_text(
        json.dumps(
            {
                "stats": {
                    "usage_observed": False,
                    "total_input_tokens": 0,
                    "total_cached_input_tokens": 0,
                    "total_output_tokens": 0,
                },
                "error": "401 Unauthorized",
            }
        )
    )
    context = AgentContext()

    agent.populate_context_post_run(context)

    assert context.n_input_tokens is None
    assert context.n_cache_tokens is None
    assert context.n_output_tokens is None
    assert context.model_usage is None


@pytest.mark.parametrize("partial_report", [None, '{"stats":'])
def test_hard_timeout_recovers_observed_usage_from_partial_stream(
    tmp_path, partial_report
):
    agent = RPentEfeAgent(logs_dir=tmp_path, model_name="openai/test")
    output_dir = tmp_path / "rpent"
    output_dir.mkdir()
    if partial_report is not None:
        (output_dir / "transcript_task.json").write_text(partial_report)
    event = {
        "method": "thread/tokenUsage/updated",
        "payload": {
            "token_usage": {
                "total": {
                    "input_tokens": 31,
                    "cached_input_tokens": 12,
                    "output_tokens": 7,
                }
            }
        },
    }
    (output_dir / "codex.txt.stream.jsonl").write_text(
        json.dumps(event) + '\n{"method":'
    )
    context = AgentContext()

    agent.populate_context_post_run(context)

    assert context.n_input_tokens == 31
    assert context.n_cache_tokens == 12
    assert context.n_output_tokens == 7
    assert (
        json.loads((output_dir / "usage-recovery.json").read_text())["complete"]
        is False
    )
    assert context.cost_usd is None


def test_the_wrapper_timeout_sits_inside_harbors_agent_timeout(tmp_path, monkeypatch):
    """Harbor kills the container at ``agent_timeout_sec``; the wrapper must not beat it."""
    agent = RPentEfeAgent(
        logs_dir=tmp_path,
        model_name="openai/test-model",
        agent_timeout_s=600,
    )
    execute = AsyncMock()
    monkeypatch.setattr(agent, "exec_as_agent", execute)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    asyncio.run(agent.run("Task", UploadEnvironment(), AgentContext()))

    command = shlex.split(execute.call_args.kwargs["command"])
    planner_budget = int(command[command.index("--planner-timeout-s") + 1])
    connect_wait = float(command[command.index("--efe-connect-timeout-s") + 1])
    assert planner_budget + connect_wait <= agent.timeout_s
    assert planner_budget > 0
    assert execute.call_args.kwargs["timeout_sec"] < 600


def test_an_explicit_planner_budget_is_capped_by_the_agent_timeout(tmp_path):
    agent = RPentEfeAgent(
        logs_dir=tmp_path,
        model_name="openai/test-model",
        timeout_s=5000,
        agent_timeout_s=600,
    )

    assert agent.timeout_s < 600


def test_a_planner_budget_shorter_than_the_agent_timeout_is_kept(tmp_path):
    agent = RPentEfeAgent(
        logs_dir=tmp_path,
        model_name="openai/test-model",
        timeout_s=120,
        agent_timeout_s=3600,
    )

    assert agent.timeout_s == 120
    assert agent.exec_timeout_s < 3600


def test_detected_harbor_timeout_replaces_the_default_planner_budget(tmp_path):
    agent = RPentEfeAgent(logs_dir=tmp_path, model_name="openai/test-model")

    agent._apply_agent_timeout(5000)

    assert agent.timeout_s == 5000 - 2 * 30
    assert agent.exec_timeout_s == 5000 - 30


def test_an_agent_timeout_too_small_for_one_session_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="too small"):
        RPentEfeAgent(logs_dir=tmp_path, model_name="openai/test", agent_timeout_s=30)


def test_the_trial_config_resolves_the_agent_timeout(tmp_path):
    """Harbor passes its resolved budget only through the trial's config."""
    trial = tmp_path / "trial"
    logs_dir = trial / "agent"
    logs_dir.mkdir(parents=True)
    agent = RPentEfeAgent(logs_dir=logs_dir, model_name="openai/test-model")

    assert agent._resolve_harbor_agent_timeout() is None

    (trial / "config.json").write_text(
        json.dumps(
            {
                "agent": {"override_timeout_sec": 900, "max_timeout_sec": 600},
                "agent_timeout_multiplier": 1.5,
            }
        )
    )

    assert agent._resolve_harbor_agent_timeout() == 900


def test_the_task_config_supplies_the_timeout_without_an_override(tmp_path):
    task = tmp_path / "task"
    task.mkdir()
    (task / "task.toml").write_text("[agent]\ntimeout_sec = 1200\n")
    trial = tmp_path / "trial"
    (trial / "agent").mkdir(parents=True)
    (trial / "config.json").write_text(
        json.dumps({"task": {"path": str(task)}, "timeout_multiplier": 0.5})
    )
    agent = RPentEfeAgent(logs_dir=trial / "agent", model_name="openai/test-model")

    assert agent._resolve_harbor_agent_timeout() == 600


def test_a_transcript_without_a_stats_object_falls_back_to_recovery(tmp_path):
    agent = RPentEfeAgent(logs_dir=tmp_path, model_name="openai/test-model")
    report = agent.logs_dir / "rpent" / "transcript_task.json"
    report.parent.mkdir(parents=True)
    report.write_text(json.dumps({"error": None, "stats": []}), encoding="utf-8")
    stream = report.parent / "codex.txt.stream.jsonl"
    stream.write_text(
        json.dumps(
            {
                "method": "thread/tokenUsage/updated",
                "payload": {
                    "token_usage": {
                        "total": {
                            "input_tokens": 7,
                            "cached_input_tokens": 1,
                            "output_tokens": 3,
                        }
                    }
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    context = AgentContext()

    agent.populate_context_post_run(context)

    assert context.n_input_tokens == 7
    assert context.n_output_tokens == 3
