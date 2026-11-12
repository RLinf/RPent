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
from typing import Any

import pytest

from rpent.planner.claude_code import _tool_result_to_mcp
from rpent.planner.utils.http_mcp_server import mcp_result
from rpent.tools import ToolResult
from rpent.tools import base as tool_base


@pytest.mark.parametrize(
    "render,error_key", [(mcp_result, "isError"), (_tool_result_to_mcp, "is_error")]
)
@pytest.mark.parametrize("error", [None, "motion failed", {"code": "blocked"}, "", {}])
def test_payload_error_and_images_reach_planner_without_mutation(
    render, error_key, error
):
    payload = {"step": 3, "error": error}
    images = [b"main", b"camera", b"navigation", b"wrist"]
    result = ToolResult(data=payload, images=images)

    response = render(result)

    if error_key == "isError" or error is not None:
        assert response[error_key] is (error is not None)
    else:
        assert error_key not in response
    assert json.loads(response["content"][0]["text"]) == payload
    blocks = response["content"]
    assert [block["type"] for block in blocks] == ["text"] + ["image"] * 4
    assert [base64.b64decode(block["data"]) for block in blocks[1:]] == images
    assert all(block["mimeType"] == "image/png" for block in blocks[1:])
    assert images == [b"main", b"camera", b"navigation", b"wrist"]
    assert payload == {"step": 3, "error": error}
    copied = result.to_dict()
    copied["error"] = "changed"
    assert payload["error"] == error


@pytest.mark.parametrize("value", ["plain text", 17, ["one", "two"]])
def test_tool_result_encodes_payload_values(value: Any) -> None:
    result = ToolResult(data={"value": value})
    assert json.loads(result.to_text()) == {"value": value}


@pytest.mark.parametrize("finished", [True, False])
def test_tool_result_preserves_finish_signal(finished: bool) -> None:
    result = ToolResult(data={"_finish": finished})
    assert result.to_dict()["_finish"] is finished
    assert json.loads(result.to_text())["_finish"] is finished


def test_tool_result_text_limit_preserves_internal_payload(monkeypatch) -> None:
    monkeypatch.setattr(tool_base, "MAX_TOOL_TEXT_BYTES", 30)
    payload = {"value": "界" * 10}
    result = ToolResult(data=payload)
    text = result.to_text()
    assert len(text.encode("utf-8")) <= 30
    assert text.endswith("\n[truncated]")
    assert result.data == payload


@pytest.mark.parametrize("render", [mcp_result, _tool_result_to_mcp])
def test_plain_result_contains_only_text(render):
    response = render(ToolResult(data={"value": "plain"}))
    assert response["content"] == [{"type": "text", "text": '{\n  "value": "plain"\n}'}]
