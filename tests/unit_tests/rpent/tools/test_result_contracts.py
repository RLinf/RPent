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

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

import pytest

from rpent.dashboard.events import NullDashboardEventSink
from rpent.memory import MemoryManager
from rpent.planner.claude_code import _tool_result_to_mcp
from rpent.planner.utils.http_mcp_server import mcp_result
from rpent.session import EnvState
from rpent.tools import Toolkit, ToolResult, tool


@pytest.mark.parametrize("render", [mcp_result, _tool_result_to_mcp])
@pytest.mark.parametrize("error", [None, "motion failed", {"code": "blocked"}, "", {}])
def test_payload_error_and_images_reach_planner_without_mutation(render, error):
    payload = {"step": 3, "error": error}
    result = ToolResult(data=payload, images=[b"image"])

    response = render(result)

    assert response.get("isError", response.get("is_error", False)) is (
        error is not None
    )
    assert json.loads(response["content"][0]["text"]) == payload
    assert base64.b64decode(response["content"][1]["data"]) == b"image"
    assert payload == {"step": 3, "error": error}
    copied = result.to_dict()
    copied["error"] = "changed"
    assert payload["error"] == error


@pytest.fixture
def toolkit(tmp_path: Path) -> Toolkit:
    return Toolkit(
        dashboard_events=NullDashboardEventSink(),
        state=EnvState(tmp_path / "state"),
        memory=MemoryManager(tmp_path / "memory"),
    )


@pytest.mark.parametrize(
    ("action_error", "observation_error", "expected_error"),
    [
        ("motion failed", "camera failed", "motion failed"),
        ("motion failed", None, "motion failed"),
        (None, "camera failed", "camera failed"),
        (None, None, None),
    ],
)
def test_state_capture_preserves_action_error_and_images(
    toolkit, monkeypatch, action_error, observation_error, expected_error
):
    @tool
    def move() -> ToolResult:
        """Move the robot."""
        return ToolResult(
            data={"error": action_error, "interrupted": True}, images=[b"action"]
        )

    captured = []

    def observe(**kwargs: Any) -> ToolResult:
        captured.append(kwargs["result"])
        return ToolResult(
            data={"step": 1, "error": observation_error}, images=[b"observation"]
        )

    toolkit.add_tool(move)
    monkeypatch.setattr(toolkit, "get_env_state", observe)

    result = toolkit.execute_tool("move", {})

    assert captured == [{"error": action_error, "interrupted": True}]
    assert result.data["error"] == expected_error
    assert result.is_error is (expected_error is not None)
    assert result.data["step"] == 1
    assert result.images == [b"action", b"observation"]
    if action_error is not None:
        assert result.data["interrupted"] is True


@pytest.mark.parametrize("action_error", [None, "motion failed"])
def test_capture_exception_preserves_original_error(toolkit, monkeypatch, action_error):
    @tool
    def move() -> ToolResult:
        """Move the robot."""
        return ToolResult(data={"error": action_error})

    def observe(**kwargs: Any) -> ToolResult:
        raise RuntimeError("camera offline")

    toolkit.add_tool(move)
    monkeypatch.setattr(toolkit, "get_env_state", observe)

    result = toolkit.execute_tool("move", {})

    assert result.is_error
    assert result.data["error"] == (
        action_error or "failed to capture state after move: camera offline"
    )
    assert result.data["state_capture_error"] == "camera offline"


def test_unknown_tool_and_bad_arguments_are_error_results(toolkit):
    unknown = toolkit.execute_tool("missing", {})
    invalid = toolkit.execute_tool("read_text_file", {"path": 3})

    assert unknown.is_error
    assert unknown.data["error"] == "unknown tool: missing"
    assert invalid.is_error
    assert invalid.data["error"] == "bad arguments for read_text_file"
    assert invalid.data["errors"][0]["loc"] == ("path",)
