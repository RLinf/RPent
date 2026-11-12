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
    assert shlex.split(execute.call_args.kwargs["command"])[-14:] == [
        "--model",
        "test-model",
        "--timeout-s",
        "42",
        "--efe-connect-timeout-s",
        "42",
        "--max-turns",
        "7",
        "--reasoning-effort",
        "none",
        "--explore-sessions",
        "1",
        "--codex-sandbox",
        "read-only",
    ]


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

    assert shlex.split(execute.call_args.kwargs["command"])[-2:] == [
        "--codex-sandbox",
        "full-access",
    ]


def test_unknown_codex_sandbox_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="sandbox"):
        RPentEfeAgent(logs_dir=tmp_path, model_name="openai/test", sandbox="none")


def test_install_uploads_local_source_without_tests_or_caches(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    for filename in (
        "pyproject.toml",
        "README.md",
        "LICENSE",
        "robots/efe/cli.py",
        "robots/__init__.py",
        "rpent/__pycache__/cached.pyc",
        "rpent/.env",
        "rpent/.codex/auth.json",
        "rpent/logs/run.log",
        "rpent/dashboard/static/style.css",
        "tests/private.py",
    ):
        target = source / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("placeholder")
    (source / "pyproject.toml").write_text(
        '[tool.setuptools.package-data]\n"rpent.dashboard" = ["static/*.css"]\n'
    )
    agent = RPentEfeAgent(
        logs_dir=tmp_path / "logs", model_name="openai/test", source_root=str(source)
    )
    environment = UploadEnvironment()
    execute = AsyncMock()
    execute.return_value = SimpleNamespace(stdout="rpent==0.1.0\n")
    monkeypatch.setattr(agent, "exec_as_root", execute)

    asyncio.run(agent.install(environment))

    commands = [call.kwargs["command"] for call in execute.await_args_list]
    install = next(command for command in commands if "pip install" in command)
    assert "robots/efe/cli.py" in environment.archive_members
    assert "rpent/__pycache__/cached.pyc" not in environment.archive_members
    assert "tests/private.py" not in environment.archive_members
    assert "rpent/.env" not in environment.archive_members
    assert "rpent/.codex/auth.json" not in environment.archive_members
    assert "rpent/logs/run.log" not in environment.archive_members
    assert "rpent/dashboard/static/style.css" in environment.archive_members
    assert "python3 -m venv" in install
    assert "--no-cache-dir" in install
    assert "--disable-pip-version-check" in install
    assert "pip freeze --all" in commands[-1]
    assert (tmp_path / "logs/setup/pip-freeze.txt").read_text() == "rpent==0.1.0\n"
    assert (
        len(json.loads((tmp_path / "logs/rpent-source.json").read_text())["sha256"])
        == 64
    )


def test_default_source_root_is_the_repository_root(tmp_path):
    agent = RPentEfeAgent(logs_dir=tmp_path, model_name="openai/test")

    assert (agent.source_root / "pyproject.toml").is_file()
    assert (agent.source_root / "robots/efe/harbor_agent.py").is_file()


def test_post_run_usage_keeps_cached_tokens_inside_input(tmp_path):
    agent = RPentEfeAgent(logs_dir=tmp_path, model_name="openai/test")
    (tmp_path / "rpent").mkdir()
    (tmp_path / "rpent/session.json").write_text(
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


def test_agent_uploads_memory_and_forwards_exploration(tmp_path, monkeypatch):
    memory = tmp_path / "corpus"
    (memory / "global").mkdir(parents=True)
    (memory / "global" / "lesson.md").write_text("# Lesson\n")
    agent = RPentEfeAgent(
        logs_dir=tmp_path / "logs",
        model_name="openai/test",
        memory_source=str(memory),
        explore_sessions=2,
    )
    environment = UploadEnvironment()
    execute = AsyncMock()
    execute.return_value = SimpleNamespace(stdout="")
    monkeypatch.setattr(agent, "exec_as_root", execute)
    monkeypatch.setattr(agent, "exec_as_agent", AsyncMock())

    asyncio.run(agent.install(environment))
    asyncio.run(agent.run("Task", environment, AgentContext()))

    # The corpus is staged under the agent log directory, so exploration notes
    # survive the container and the agent user can append to them.
    archive_path = "/logs/agent/rpent/memory.tar.gz"
    assert archive_path in environment.uploads
    with tarfile.open(fileobj=io.BytesIO(environment.uploads[archive_path])) as archive:
        assert "./global/lesson.md" in archive.getnames()
    command = shlex.split(agent.exec_as_agent.call_args.kwargs["command"])
    assert command[command.index("--explore-sessions") + 1] == "2"
    assert command[command.index("--memory-dir") + 1] == "/logs/agent/rpent/memory"


def test_multi_session_usage_recovery_sums_each_session_trace(tmp_path):
    agent = RPentEfeAgent(logs_dir=tmp_path, model_name="openai/test")
    output_dir = tmp_path / "rpent"
    for number, totals in ((1, (5, 1, 1)), (1, (10, 2, 1)), (2, (20, 3, 2))):
        session = output_dir / "sessions" / f"session_{number:03d}"
        session.mkdir(parents=True, exist_ok=True)
        event = {
            "method": "thread/tokenUsage/updated",
            "payload": {
                "token_usage": {
                    "total": {
                        "input_tokens": totals[0],
                        "cached_input_tokens": totals[1],
                        "output_tokens": totals[2],
                    }
                }
            },
        }
        with (session / "codex.txt.stream.jsonl").open("a", encoding="utf-8") as file:
            file.write(json.dumps(event) + "\n")
    (output_dir / "session.json").write_text(
        json.dumps({"stats": {"usage_observed": False}})
    )
    context = AgentContext()

    agent.populate_context_post_run(context)

    assert context.n_input_tokens == 30
    assert context.n_cache_tokens == 5
    assert context.n_output_tokens == 3
    assert (
        json.loads((output_dir / "usage-recovery.json").read_text())["complete"]
        is False
    )


def test_zero_usage_trace_is_not_recovered_as_measured(tmp_path):
    agent = RPentEfeAgent(logs_dir=tmp_path, model_name="openai/test")
    output_dir = tmp_path / "rpent"
    output_dir.mkdir()
    (output_dir / "session.json").write_text(
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


def test_agent_rejects_invalid_exploration_sessions(tmp_path):
    with pytest.raises(ValueError):
        RPentEfeAgent(
            logs_dir=tmp_path,
            model_name="openai/test",
            explore_sessions=0,
        )


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


def test_host_codex_home_is_not_forwarded_to_the_trial(tmp_path, monkeypatch):
    agent = RPentEfeAgent(logs_dir=tmp_path, model_name="openai/test")
    environment = UploadEnvironment()
    execute = AsyncMock()
    monkeypatch.setattr(agent, "exec_as_agent", execute)
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "host-codex"))

    asyncio.run(agent.run("Task", environment, AgentContext()))

    assert "CODEX_HOME" not in execute.call_args.kwargs["env"]


def test_failed_start_does_not_invent_usage(tmp_path):
    agent = RPentEfeAgent(logs_dir=tmp_path, model_name="openai/test")
    (tmp_path / "rpent").mkdir()
    (tmp_path / "rpent/session.json").write_text(
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
    (tmp_path / "rpent/session.json").write_text(
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
        (output_dir / "session.json").write_text(partial_report)
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
    planner_budget = int(command[command.index("--timeout-s") + 1])
    connect_wait = float(command[command.index("--efe-connect-timeout-s") + 1])
    assert planner_budget < 600
    assert connect_wait == planner_budget
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


def test_an_agent_timeout_too_small_for_one_session_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="too small"):
        RPentEfeAgent(logs_dir=tmp_path, model_name="openai/test", agent_timeout_s=30)


def test_a_session_report_without_a_stats_object_falls_back_to_recovery(tmp_path):
    agent = RPentEfeAgent(logs_dir=tmp_path, model_name="openai/test-model")
    report = agent.logs_dir / "rpent" / "session.json"
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
