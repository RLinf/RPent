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

"""Exercise the user-facing facade across a real local MCP transport."""

from __future__ import annotations

import base64
import sys
from pathlib import Path
from typing import Any

import pytest
from pydantic_ai import BinaryContent

from rpent.context import ContextDocument
from rpent.embodied_agent import EmbodiedAgent, McpServer
from rpent.llm import LLMConfig
from rpent.planner.base import PlannerResult
from rpent.planner.utils.http_mcp_server import HttpMcpServer
from rpent.tools.toolkit import ToolResult


class _RobotTools:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def get_tools_spec(self) -> list[dict[str, Any]]:
        return [
            {
                "name": "move_eef",
                "description": "Move to a target pose and return a camera frame.",
                "input_schema": {
                    "type": "object",
                    "properties": {"pose": {"type": "string"}},
                    "required": ["pose"],
                },
            },
            {
                "name": "snapshot",
                "description": "Capture the current camera frame.",
                "input_schema": {"type": "object", "properties": {}},
            },
        ]

    def execute_tool(self, name: str, input_dict: dict[str, Any]) -> ToolResult:
        self.calls.append((name, input_dict))
        if name == "move_eef":
            if input_dict["pose"] == "invalid":
                return ToolResult(name=name, result={"error": "pose unreachable"})
            return ToolResult(name=name, result={"pose": input_dict["pose"]})
        return ToolResult(name=name, result={"camera": "front", "_image_bytes": b"png"})


@pytest.mark.parametrize("with_context", [False, True])
def test_embodied_agent_discovers_calls_and_preserves_images(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, with_context: bool
) -> None:
    robot = _RobotTools()
    server = HttpMcpServer(robot)
    server.start()
    skill = tmp_path / "SKILL.md"
    skill.write_text("Use the front camera after each motion.", encoding="utf-8")
    seen: dict[str, Any] = {}

    class _Planner:
        def solve(self, **kwargs: Any) -> PlannerResult:
            seen.update(kwargs)
            toolkit = kwargs["toolkit"]
            specs = {item["name"]: item for item in toolkit.get_tools_spec()}
            assert specs["robot__move_eef"]["input_schema"]["required"] == ["pose"]
            assert toolkit.execute_tool(
                "robot__move_eef", {"pose": "target"}
            ).result == {"pose": "target"}
            rejected = toolkit.execute_tool("robot__move_eef", {"pose": "invalid"})
            assert rejected.result["error"] == "pose unreachable"
            snapshot = toolkit.execute_tool("robot__snapshot", {})
            assert snapshot.content_blocks[0]["type"] == "text"
            assert snapshot.content_blocks[1]["type"] == "image"
            assert (
                base64.b64decode(snapshot.content_blocks[1]["source"]["data"]) == b"png"
            )
            finish = toolkit.execute_tool(
                "finish", {"status": "success", "summary": "placed"}
            )
            assert finish.is_finish
            return PlannerResult(
                finish_result=finish.result,
                stats={
                    "total_input_tokens": 10,
                    "total_output_tokens": 5,
                    "cache_read_tokens": 3,
                    "cache_write_tokens": 2,
                    "requests": 1,
                },
            )

    planner_kwargs: dict[str, Any] = {}

    def build_planner(*args: Any, **kwargs: Any) -> _Planner:
        planner_kwargs.update(kwargs)
        return _Planner()

    monkeypatch.setattr("rpent.embodied_agent.build_planner", build_planner)
    llm = LLMConfig("openai", "gpt-4o", api_key="test")
    agent = EmbodiedAgent(
        mcp_servers=[McpServer(name="robot", url=server.url)],
        output_dir=tmp_path / "episode",
        llm=llm,
    )
    image = BinaryContent(data=b"initial-png", media_type="image/png")
    context_args = (
        {
            "memory": [
                ContextDocument("grasp", "Use the top grasp.", "memory/grasp.md")
            ],
            "initial_context": ["Initial camera", image],
        }
        if with_context
        else {}
    )
    try:
        result = agent.run(
            "Place the block.",
            system_prompt="Use safe poses.",
            skills=[skill],
            **context_args,
        )
    finally:
        server.stop()

    assert result.finish_result["status"] == "success"
    assert planner_kwargs["llm_config"] is llm
    assert result.stats["llm_usage"]["total_tokens"] == 15
    assert result.stats["llm_usage"]["cache_read_tokens"] == 3
    if with_context:
        assert seen["user_message"] == [
            "Place the block.\n\n## Memory: grasp\nSource: memory/grasp.md\n\nUse the top grasp.",
            "Initial camera",
            image,
        ]
        assert seen["user_message"][2] is image
    else:
        assert seen["user_message"] == "Place the block."
    assert seen["system_prompt"] == (
        f"Use safe poses.\n\n## Skill: {tmp_path.name}\n\n"
        "Use the front camera after each motion."
    )
    assert robot.calls == [
        ("move_eef", {"pose": "target"}),
        ("move_eef", {"pose": "invalid"}),
        ("snapshot", {}),
    ]


def test_missing_skill_fails_before_mcp_startup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def reject_start(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("MCP must not start before context is loaded")

    monkeypatch.setattr("rpent.embodied_agent._McpToolkit.start", reject_start)
    agent = EmbodiedAgent(
        mcp_servers=[McpServer(name="robot", url="http://localhost/mcp")],
        output_dir=tmp_path / "episode",
    )
    with pytest.raises(FileNotFoundError):
        agent.run("task", system_prompt="rules", skills=[tmp_path / "missing.md"])
    assert not (tmp_path / "episode").exists()


def test_embodied_agent_rejects_invalid_server_config(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="exactly one"):
        McpServer(name="robot")
    with pytest.raises(ValueError, match="exactly one"):
        McpServer(name="robot", url="http://localhost/mcp", command="python")
    with pytest.raises(ValueError, match="unique"):
        EmbodiedAgent(
            mcp_servers=[
                McpServer(name="robot", url="http://localhost/a"),
                McpServer(name="robot", url="http://localhost/b"),
            ],
            output_dir=tmp_path,
        ).run("task", system_prompt="rules")


def test_embodied_agent_starts_stdio_server(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = tmp_path / "robot_server.py"
    script.write_text(
        "from mcp.server.fastmcp import FastMCP\n"
        "server = FastMCP('robot')\n"
        "@server.tool()\n"
        "def snapshot() -> str:\n"
        "    return 'camera ready'\n"
        "server.run(transport='stdio')\n",
        encoding="utf-8",
    )

    class _Planner:
        def solve(self, **kwargs: Any) -> PlannerResult:
            toolkit = kwargs["toolkit"]
            result = toolkit.execute_tool("robot__snapshot", {})
            assert result.content_blocks == [{"type": "text", "text": "camera ready"}]
            return PlannerResult()

    monkeypatch.setattr(
        "rpent.embodied_agent.build_planner", lambda *a, **kw: _Planner()
    )
    agent = EmbodiedAgent(
        mcp_servers=[
            McpServer(name="robot", command=sys.executable, args=(str(script),))
        ],
        output_dir=tmp_path / "episode",
    )
    agent.run("Observe the scene.", system_prompt="Use snapshot.")
