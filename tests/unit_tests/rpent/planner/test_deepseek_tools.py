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

"""Native execution provenance and opt-in MCP result contracts."""

from __future__ import annotations

import hashlib
import json
import math

import pytest

from rpent.planner.deepseek_tools import _ToolBoundary
from rpent.planner.utils.http_mcp_server import mcp_result
from rpent.tools import ToolResult
from rpent.tools.common import CommonTools


class Toolkit:
    def __init__(self):
        self.calls = []
        self.cancelled = False
        self.result = ToolResult(data={"_finish": True, "status": "success"})

    def list_tools(self):
        return (CommonTools.finish,)

    def execute_tool(self, name, arguments):
        self.calls.append(name)
        return self.result

    def cancel_active_and_wait(self):
        self.cancelled = True


def test_native_finish_latches_before_later_call():
    toolkit = Toolkit()
    boundary = _ToolBoundary(toolkit)
    assert boundary.execute_tool("finish", {}).data["_finish"]
    assert boundary.execute_tool("move", {}).is_error
    assert toolkit.calls == ["finish"]
    assert boundary.finish_result == toolkit.result.data


@pytest.mark.parametrize("payload", [{"error": "refused"}, {"status": "success"}])
def test_refused_or_unmarked_finish_does_not_conclude(payload):
    toolkit = Toolkit()
    toolkit.result = ToolResult(data=payload)
    boundary = _ToolBoundary(toolkit)
    boundary.execute_tool("finish", {"status": "success"})
    boundary.execute_tool("move", {})
    assert boundary.finish_result is None
    assert toolkit.calls == ["finish", "move"]


def test_structured_payload_retains_marker_beyond_text_limit():
    result = ToolResult(data={"summary": "x" * 70000, "_finish": True})
    assert mcp_result(result)["content"][0]["text"].endswith("[truncated]")
    assert "structuredContent" not in mcp_result(result)
    assert mcp_result(result, include_structured=True)["structuredContent"]["_finish"]


def test_structured_coordinates_normalize_negative_zero_without_changing_native():
    result = ToolResult(
        data={"result": {"pose": [0.0705, -0.0, 1.0194]}, "note": "literal -0.0"}
    )
    payload = mcp_result(result, include_structured=True)
    structured = payload["structuredContent"]
    text = json.loads(payload["content"][0]["text"])
    assert structured == text == result.data
    assert math.copysign(1, structured["result"]["pose"][1]) == 1
    assert math.copysign(1, text["result"]["pose"][1]) == 1
    assert math.copysign(1, result.data["result"]["pose"][1]) == -1
    assert structured["note"] == "literal -0.0"


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_structured_coordinates_reject_nonfinite_values(value):
    with pytest.raises(ValueError, match="Out of range float"):
        mcp_result(ToolResult(data={"pose": [value]}), include_structured=True)


@pytest.mark.parametrize(
    "failure", [None, "nonfinite", "image", "uncertain", "truncated"]
)
def test_business_refusal_requires_a_complete_canonical_payload(tmp_path, failure):
    toolkit = Toolkit()
    toolkit.result = ToolResult(
        data={"error": "business refusal", "position": [-0.0]},
        images=[b"image"] if failure == "image" else [],
        execution_uncertain=failure == "uncertain",
    )
    if failure == "nonfinite":
        toolkit.result.data["position"] = [float("nan")]
    elif failure == "truncated":
        toolkit.result.data["details"] = "x" * 70000
    path = tmp_path / "ledger.json"
    boundary = _ToolBoundary(toolkit, path)
    result = boundary.execute_tool("act", {})
    ledger = json.loads(path.read_text())
    assert result is toolkit.result
    assert ledger["isError"]
    if failure is None:
        text = mcp_result(result, include_structured=True)["content"][0]["text"]
        assert (
            ledger["canonicalBusinessErrorSha256"]
            == hashlib.sha256(text.encode()).hexdigest()
        )
    else:
        assert ledger["canonicalBusinessErrorSha256"] is None


def test_cancel_prevents_queued_calls():
    toolkit = Toolkit()
    boundary = _ToolBoundary(toolkit)
    boundary.cancel_active_and_wait()
    assert boundary.execute_tool("move", {}).is_error
    assert toolkit.calls == []
    assert toolkit.cancelled


def test_uncertain_boundary_never_reopens():
    toolkit = Toolkit()
    toolkit.result = ToolResult(data={"acted": True}, execution_uncertain=True)
    boundary = _ToolBoundary(toolkit)
    boundary.execute_tool("move", {})
    boundary.cancel_active_and_wait()
    with pytest.raises(RuntimeError, match="uncertain"):
        boundary.resume_calls()
    assert boundary.execute_tool("move", {}).is_error
    assert toolkit.calls == ["move"]
