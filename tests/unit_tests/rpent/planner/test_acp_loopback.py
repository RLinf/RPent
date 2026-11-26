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

"""Offline ACP subprocess driving the real RPent HTTP MCP bridge."""

import asyncio
import base64
import json
import os
import sys
from pathlib import Path
from unittest.mock import Mock

import acp
import httpx
import pytest
from acp import schema
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

from rpent.dashboard.events import NullDashboardEventSink
from rpent.memory import MemoryManager
from rpent.planner.utils.acp import AcpPlanner, AcpSession
from rpent.planner.utils.agent_runtime import agent_environment
from rpent.tools import Toolkit, ToolResult, tool
from rpent.utils.logging import init_output_dir

PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAIAAAD8GO2jAAAAKElEQVR4nO3NMQEAAAjDMMC/ZzDBvlRA01vZJvwHAAAAAAAAAAAAbx2jxAE/i2AjOgAAAABJRU5ErkJggg=="
)


class LocalAgent:
    def on_connect(self, connection):
        self.client = connection

    async def initialize(self, protocol_version, **kwargs):
        self.record = {"pid": os.getpid(), "url": None, "cwd": os.getcwd()}
        Path(sys.argv[1]).write_text(json.dumps(self.record))
        if sys.argv[2] == "no-capabilities":
            # Preserve explicit null, which the SDK model serializer omits.
            return {"protocolVersion": protocol_version, "agentCapabilities": None}
        return schema.InitializeResponse(
            protocol_version=protocol_version,
            agent_capabilities=schema.AgentCapabilities(
                mcp_capabilities=schema.McpCapabilities(http=sys.argv[2] != "no-http")
            ),
        )

    async def new_session(self, cwd, mcp_servers, **kwargs):
        self.url = mcp_servers[0].url if mcp_servers else None
        self.record.update(url=self.url)
        Path(sys.argv[1]).write_text(json.dumps(self.record))
        return schema.NewSessionResponse(
            session_id="local",
            config_options=[
                schema.SessionConfigOptionSelect(
                    id="agent-model",
                    name="Model",
                    category="model",
                    type="select",
                    current_value="default",
                    options=[
                        schema.SessionConfigSelectOption(
                            value="test-model", name="Test"
                        )
                    ],
                )
            ]
            if sys.argv[2] != "no-model"
            else [],
        )

    async def set_config_option(self, session_id, config_id, value, **kwargs):
        assert session_id == "local"
        if config_id == "agent-model":
            assert value == "test-model"
            self.selected_model = value
        else:
            assert (config_id, value) == ("agent-effort", "high")
            self.record["effort"] = value
            Path(sys.argv[1]).write_text(json.dumps(self.record))
        options = [schema.SessionConfigSelectOption(value="high", name="High")]
        if sys.argv[2] == "effort-grouped":
            options = [
                schema.SessionConfigSelectGroup(
                    group="levels", name="Levels", options=options
                )
            ]
        return schema.SetSessionConfigOptionResponse(
            config_options=[
                schema.SessionConfigOptionSelect(
                    id="agent-effort",
                    name="Effort",
                    category="thought_level",
                    type="select",
                    current_value="default",
                    options=options,
                )
            ]
            if sys.argv[2] != "no-effort"
            else []
        )

    async def prompt(self, session_id, prompt, **kwargs):
        assert self.selected_model == "test-model"
        permission = await self.client.request_permission(
            session_id=session_id,
            tool_call=schema.ToolCallUpdate(tool_call_id="shell", title="run shell"),
            options=[
                schema.PermissionOption(
                    option_id="allow", name="Allow", kind="allow_once"
                )
            ],
        )
        assert permission.outcome.outcome == "cancelled"
        if self.url:
            async with httpx.AsyncClient(trust_env=False) as http:
                async with streamable_http_client(self.url, http_client=http) as (
                    read,
                    write,
                    _,
                ):
                    async with ClientSession(read, write) as client:
                        await client.initialize()
                        assert {
                            tool.name for tool in (await client.list_tools()).tools
                        } == {"observe", "finish"}
                        result = await client.call_tool("observe", {})
                        assert base64.b64decode(result.content[1].data) == PNG
                        result = await client.call_tool("finish", {"status": "success"})
                        assert not result.isError
        secret = os.environ["TEST_HARNESS_API_KEY"]
        print(secret, file=sys.stderr)
        await self.client.session_update(
            session_id=session_id,
            update=schema.AgentMessageChunk(
                session_update="agent_message_chunk", content=acp.text_block(secret)
            ),
        )
        return schema.PromptResponse(stop_reason="end_turn")


@pytest.mark.parametrize(
    ("mode", "error"),
    [
        ("ok", None),
        ("no-http", "must support HTTP MCP"),
        ("no-capabilities", "must support HTTP MCP"),
        ("no-model", "no model selector"),
        ("effort", None),
        ("effort-grouped", None),
        ("no-effort", "no thought_level selector"),
        ("unsupported-effort", "available values: high"),
    ],
)
def test_harness_reuses_acp_tools_probe_and_cleanup(tmp_path, monkeypatch, mode, error):
    init_output_dir(tmp_path / "log")
    record = tmp_path / "agent.json"
    secret = "test-harness-secret-value"
    monkeypatch.setenv("TEST_HARNESS_API_KEY", secret)

    class LocalHarnessPlanner(AcpPlanner):
        backend = "local_harness"
        label = "Local harness"

        def _create_session(
            self,
            root,
            control,
            emit,
            stderr_path,
            *,
            mcp_url,
            tool_names,
            redact,
        ):
            return AcpSession(
                command=[
                    sys.executable,
                    str(Path(__file__).resolve()),
                    str(record),
                    mode,
                ],
                cwd=root,
                env=agent_environment(),
                mcp_url=mcp_url,
                model=self._model,
                reasoning_effort=self._reasoning_effort,
                control=control,
                emit_event=emit,
                stderr_path=stderr_path,
                redact=redact,
            )

    class TestToolkit(Toolkit):
        def _register_common_tools(self):
            pass

    toolkit = TestToolkit(
        dashboard_events=NullDashboardEventSink(),
        memory=MemoryManager(tmp_path / "memory"),
    )
    toolkit.add_tool(
        tool(name="observe", readonly=True)(
            lambda: ToolResult(data={"observed": True}, images=[PNG])
        )
    )

    @tool(name="finish", readonly=True)
    def finish(status: str) -> ToolResult:
        return ToolResult(data={"_finish": True, "status": status})

    toolkit.add_tool(finish, replace=True)
    toolkit.execute_tool = Mock(wraps=toolkit.execute_tool)
    toolkit.cancel_active_and_wait = Mock(wraps=toolkit.cancel_active_and_wait)
    planner = LocalHarnessPlanner(
        output_dir=tmp_path,
        timeout_s=15,
        dashboard_events=NullDashboardEventSink(),
        model="test-model",
        reasoning_effort="low"
        if mode == "unsupported-effort"
        else "high"
        if "effort" in mode
        else "none",
    )
    if mode not in {"no-model", "no-effort", "unsupported-effort"}:
        assert planner.probe("ping") == "<TEST_HARNESS_API_KEY>"
        assert json.loads(record.read_text())["url"] is None
    result = planner.solve(
        system_prompt="rules", user_message="task", toolkit=toolkit, max_turns=2
    )
    toolkit.cancel_active_and_wait.assert_called_once()
    child = json.loads(record.read_text())
    assert not Path(child["cwd"]).exists()
    with pytest.raises(ProcessLookupError):
        os.kill(child["pid"], 0)
    if error:
        assert error in result.error
        toolkit.execute_tool.assert_not_called()
        return
    assert result.error is None
    if mode in {"effort", "effort-grouped"}:
        assert child["effort"] == "high"
    assert result.stats["backend"] == "local_harness"
    assert result.finish_result == {"_finish": True, "status": "success"}
    assert result.stats["tool_calls"] == 2
    assert [call.args[0] for call in toolkit.execute_tool.call_args_list] == [
        "observe",
        "finish",
    ]
    assert child["url"].startswith("http://127.0.0.1:")
    for name in ("output_path", "raw_stream_path", "stderr_path"):
        assert secret not in Path(result.stats[name]).read_text()


if __name__ == "__main__":
    asyncio.run(acp.run_agent(LocalAgent()))
