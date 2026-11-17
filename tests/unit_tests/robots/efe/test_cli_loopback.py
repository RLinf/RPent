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

"""Run the generic EFE CLI with local processes and HTTP MCP transport."""

from __future__ import annotations

import json
import socket
import subprocess
import sys
from collections.abc import AsyncIterator
from pathlib import Path

import httpx
import openai_codex
import pytest
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

from robots.efe import env_client
from tests.unit_tests.robots.efe.efe_support import FakeEfeClient


class LocalTurn:
    def __init__(self, workdir: Path, mcp_url: str) -> None:
        self.workdir = workdir
        self.mcp_url = mcp_url
        self.tools: list[str] = []
        self.finish_response: dict[str, object] = {}
        self.finish_arguments = {
            "status": "completed",
            "summary": "The Python child wrote control-result.txt.",
        }

    async def interrupt(self) -> None:
        """The local turn always completes; there is nothing to interrupt."""

    async def stream(self) -> AsyncIterator[dict[str, object]]:
        command = [
            sys.executable,
            "-c",
            "from pathlib import Path\n"
            "Path('control-result.txt').write_text('control program executed\\n')\n"
            "print('control child completed')\n",
        ]
        child = subprocess.run(
            command,
            cwd=self.workdir,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        yield {
            "method": "item/completed",
            "payload": {
                "item": {
                    "type": "commandExecution",
                    "command": command,
                    "cwd": str(self.workdir),
                    "status": "completed",
                    "exit_code": child.returncode,
                    "aggregated_output": child.stdout,
                }
            },
        }

        async with httpx.AsyncClient(trust_env=False, timeout=10) as client:
            async with streamable_http_client(self.mcp_url, http_client=client) as (
                read,
                write,
                _get_session_id,
            ):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    declarations = await session.list_tools()
                    self.tools = [tool.name for tool in declarations.tools]
                    response = await session.call_tool("finish", self.finish_arguments)
                    self.finish_response = response.model_dump(
                        mode="json", by_alias=True
                    )
        yield {
            "method": "item/completed",
            "payload": {
                "item": {
                    "type": "mcpToolCall",
                    "tool": "mcp__rpent__finish",
                    "status": "completed",
                    "arguments": self.finish_arguments,
                    "result": self.finish_response,
                }
            },
        }
        yield {
            "method": "item/completed",
            "payload": {
                "item": {"type": "agentMessage", "text": "Local control program ended."}
            },
        }
        yield {
            "method": "thread/tokenUsage/updated",
            "payload": {
                "token_usage": {
                    "total": {
                        "input_tokens": 17,
                        "cached_input_tokens": 4,
                        "output_tokens": 9,
                        "reasoning_output_tokens": 2,
                    }
                }
            },
        }
        yield {
            "method": "turn/completed",
            "payload": {"turn": {"status": "completed"}},
        }


class LocalThread:
    def __init__(self, turn: LocalTurn) -> None:
        self.local_turn = turn
        self.prompt = ""
        self.turn_options: dict[str, object] = {}

    async def turn(self, prompt: str, **options: object) -> LocalTurn:
        self.prompt = prompt
        self.turn_options = options
        return self.local_turn


class LocalCodex:
    def __init__(self, config: openai_codex.CodexConfig) -> None:
        self.config = config
        self.closed = False
        self.thread_options: dict[str, object] = {}
        mcp_override = next(
            value
            for value in config.config_overrides
            if value.startswith("mcp_servers.rpent.url=")
        )
        mcp_url = json.loads(mcp_override.split("=", 1)[1])
        self.thread = LocalThread(LocalTurn(Path(config.cwd), mcp_url))

    async def thread_start(self, **options: object) -> LocalThread:
        self.thread_options = options
        return self.thread

    async def close(self) -> None:
        self.closed = True


@pytest.fixture
def local_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> list[LocalCodex]:
    instances: list[LocalCodex] = []
    socket_connect = socket.socket.connect

    def connect_loopback(connection: socket.socket, address: tuple[str, int]) -> None:
        if address[0] not in {"127.0.0.1", "::1"}:
            raise AssertionError(f"The offline test rejected a connection: {address}")
        socket_connect(connection, address)

    def create_codex(config: openai_codex.CodexConfig) -> LocalCodex:
        instance = LocalCodex(config)
        instances.append(instance)
        return instance

    monkeypatch.setattr(socket.socket, "connect", connect_loopback)
    monkeypatch.setattr(openai_codex, "AsyncCodex", create_codex)
    return instances


def _run_generic_cli(
    monkeypatch: pytest.MonkeyPatch,
    *,
    workdir: Path,
    output_dir: Path,
    instruction: str,
    memory_dir: Path | None = None,
) -> int:
    from rpent.cli import main as cli

    clients: list[FakeEfeClient] = []

    def connect(*, base_url, connect_timeout_s, deadline):
        del base_url, connect_timeout_s
        client = FakeEfeClient()
        client.set_deadline(deadline)
        clients.append(client)
        return client

    monkeypatch.setattr(env_client, "EfeEnvClient", connect)
    monkeypatch.setenv("CODEX_HOME", str(output_dir.parent / "host-codex"))
    arguments = [
        "rpent",
        "--robot",
        "efe",
        "--planner",
        "codex",
        "--model",
        "offline-codex",
        "--instruction-file",
        str(workdir / "instruction.txt"),
        "--workdir",
        str(workdir),
        "--output-dir",
        str(output_dir),
        "--memory-profile",
        "local",
        "--memory-dir",
        str(memory_dir or output_dir / "memory"),
        "--max-turns",
        "3",
        "--reasoning-effort",
        "low",
        "--efe-trial-timeout-s",
        "30",
    ]
    (workdir / "instruction.txt").write_text(instruction, encoding="utf-8")
    monkeypatch.setattr(sys, "argv", arguments)
    exit_code = cli.main()
    assert clients and clients[0].deadline is not None
    return exit_code


def test_generic_cli_loads_efe_memory_and_tools_through_the_real_mcp_transport(
    tmp_path: Path, local_backend: list[LocalCodex], monkeypatch: pytest.MonkeyPatch
) -> None:
    workdir = tmp_path / "public workspace"
    workdir.mkdir()
    output_dir = tmp_path / "agent logs"
    memory_dir = tmp_path / "corpus"
    (memory_dir / "global").mkdir(parents=True)
    (memory_dir / "global" / "pickup.md").write_text("# Pickup lesson\n")

    exit_code = _run_generic_cli(
        monkeypatch,
        workdir=workdir,
        output_dir=output_dir,
        instruction="Execute a Python control program in the public workspace.",
        memory_dir=memory_dir,
    )

    assert exit_code == 0
    backend = local_backend[0]
    assert backend.thread.local_turn.tools == [
        "read_text_file",
        "write_text_file",
        "list_dir",
        "finish",
        "efe_spec",
        "efe_observe",
        "efe_status",
        "efe_recover",
        "view_env_state",
        "move_to_pose",
        "set_gripper",
    ]
    assert str(memory_dir.resolve()) in backend.thread.prompt
    assert (workdir / "control-result.txt").is_file()


def test_generic_cli_preserves_child_output_and_accounts_mcp_finish(
    tmp_path: Path,
    local_backend: list[LocalCodex],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workdir = tmp_path / "public workspace"
    workdir.mkdir()
    output_dir = tmp_path / "agent logs"
    instruction = "Execute a Python control program in the public workspace."

    exit_code = _run_generic_cli(
        monkeypatch,
        workdir=workdir,
        output_dir=output_dir,
        instruction=instruction,
    )

    assert exit_code == 0
    transcript_path = next(output_dir.glob("transcript_*.json"))
    transcript = json.loads(transcript_path.read_text(encoding="utf-8"))
    result = json.loads((output_dir / "efe-result.json").read_text(encoding="utf-8"))
    assert transcript["error"] is None
    assert transcript["finish"] == {
        "_finish": True,
        "status": "completed",
        "summary": "The Python child wrote control-result.txt.",
    }
    assert "reward" not in result
    assert (
        transcript["stats"].items()
        >= {
            "backend": "codex_sdk",
            "turns_used": 1,
            "tool_calls": 2,
            "total_input_tokens": 17,
            "total_cached_input_tokens": 4,
            "total_output_tokens": 9,
            "total_reasoning_output_tokens": 2,
        }.items()
    )
    assert (workdir / "control-result.txt").read_text() == "control program executed\n"
    assert "control child completed" in (output_dir / "codex.txt").read_text()
    assert (output_dir / "codex.txt.last").read_text() == "Local control program ended."
    assert json.loads(
        (output_dir / "codex.txt.stream.jsonl").read_text().splitlines()[-1]
    ) == {"method": "turn/completed", "payload": {"turn": {"status": "completed"}}}
    assert len(local_backend) == 1
    assert local_backend[0].closed
    assert local_backend[0].config.env["CODEX_HOME"] == str(
        output_dir.resolve() / "codex-home"
    )
    assert local_backend[0].thread_options["cwd"] == str(workdir.resolve())
    assert local_backend[0].thread_options["model"] == "offline-codex"
    assert instruction in local_backend[0].thread.prompt
    assert local_backend[0].thread.local_turn.mcp_url.startswith("http://127.0.0.1:")
    assert local_backend[0].thread.local_turn.tools == [
        "read_text_file",
        "write_text_file",
        "list_dir",
        "finish",
        "efe_spec",
        "efe_observe",
        "efe_status",
        "efe_recover",
        "view_env_state",
        "move_to_pose",
        "set_gripper",
    ]
    assert local_backend[0].thread.local_turn.finish_response["isError"] is False
    assert json.loads(
        local_backend[0].thread.local_turn.finish_response["content"][0]["text"]
    ) == transcript["finish"]
