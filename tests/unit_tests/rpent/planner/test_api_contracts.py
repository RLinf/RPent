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

import asyncio
import base64
import queue
from typing import Any

import pytest
from pydantic_ai import BinaryContent, ToolReturn
from pydantic_ai.messages import (
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.usage import RequestUsage

from rpent.dashboard.events import TranscriptEvent, UsageEvent
from rpent.planner.api_loop import (
    ApiAgentLoop,
    _build_tools,
    _content_blocks_to_pydantic,
    _make_tool_function,
)
from rpent.tools.toolkit import ToolResult


class RecordingSink:
    def __init__(self) -> None:
        self.events: list[Any] = []

    @property
    def enabled(self) -> bool:
        return True

    def emit(self, event: Any) -> None:
        self.events.append(event)


class FakeToolkit:
    state = None

    def __init__(self, result: dict[str, Any] | None = None) -> None:
        self.result = result or {"ok": True}
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.cancel_calls = 0

    def get_tools_spec(self) -> list[dict[str, Any]]:
        return [
            {
                "name": "finish",
                "description": "Finish after the environment accepts the result.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "status": {"type": "string"},
                        "summary": {"type": "string"},
                    },
                    "required": ["status", "summary"],
                },
            }
        ]

    def execute_tool(self, name: str, args: dict[str, Any]) -> ToolResult:
        self.calls.append((name, args))
        return ToolResult(name, dict(self.result))

    def cancel_active_and_wait(self) -> None:
        self.cancel_calls += 1


def solve_with_model(
    function: Any,
    toolkit: FakeToolkit,
    sink: RecordingSink,
    *,
    timeout_s: float = 5,
):
    planner = ApiAgentLoop(
        FunctionModel(function),
        max_tokens=321,
        dashboard_events=sink,
        timeout_s=timeout_s,
    )
    return planner.solve(
        system_prompt="Use tools carefully.",
        user_message="complete the task",
        toolkit=toolkit,
        max_turns=3,
    )


def test_successful_finish_waits_for_its_tool_result() -> None:
    seen_instructions: list[str | None] = []

    def model(messages: list[Any], info: Any) -> ModelResponse:
        seen_instructions.append(info.instructions)
        assert info.model_settings["max_tokens"] == 321
        assert not any(
            isinstance(part, ToolReturnPart)
            for message in messages
            for part in message.parts
        )
        return ModelResponse(
            parts=[
                ToolCallPart(
                    "finish",
                    {"status": "success", "summary": "done"},
                    "finish-call",
                )
            ],
            usage=RequestUsage(input_tokens=7, output_tokens=3),
        )

    toolkit = FakeToolkit()
    sink = RecordingSink()
    result = solve_with_model(model, toolkit, sink)

    assert seen_instructions == ["Use tools carefully."]
    assert toolkit.calls == [("finish", {"status": "success", "summary": "done"})]
    assert result.finish_result == {
        "_finish": True,
        "status": "success",
        "summary": "done",
    }
    assert result.error is None
    assert result.stats == {
        "turns_used": 1,
        "tool_calls": 1,
        "total_input_tokens": 7,
        "total_output_tokens": 3,
        "cache_read_tokens": 0,
        "cache_write_tokens": 0,
        "requests": 1,
    }
    assert any(isinstance(event, TranscriptEvent) for event in sink.events)
    assert any(isinstance(event, UsageEvent) for event in sink.events)


def test_rejected_finish_does_not_end_the_run() -> None:
    def model(messages: list[Any], info: Any) -> ModelResponse:
        del info
        if any(
            isinstance(part, ToolReturnPart)
            for message in messages
            for part in message.parts
        ):
            return ModelResponse(parts=[TextPart("I could not finish.")])
        return ModelResponse(
            parts=[
                ToolCallPart(
                    "finish",
                    {"status": "success", "summary": "too early"},
                    "rejected-finish",
                )
            ]
        )

    toolkit = FakeToolkit({"error": "finish refused by environment"})
    result = solve_with_model(model, toolkit, RecordingSink())

    assert result.finish_result is None
    assert result.error is None
    assert result.stats["tool_calls"] == 1
    assert any(
        message.get("role") == "tool"
        and message.get("content") == '{\n  "error": "finish refused by environment"\n}'
        for message in result.messages
    )


def test_backend_failure_is_returned_without_escaping() -> None:
    def model(messages: list[Any], info: Any) -> ModelResponse:
        del messages, info
        raise RuntimeError("provider failed")

    result = solve_with_model(model, FakeToolkit(), RecordingSink())

    assert result.finish_result is None
    assert result.error == "RuntimeError: provider failed"
    assert result.messages == [{"role": "user", "content": "complete the task"}]


def test_timeout_cancels_active_toolkit_work() -> None:
    async def model(messages: list[Any], info: Any) -> ModelResponse:
        del messages, info
        await asyncio.sleep(10)
        return ModelResponse(parts=[TextPart("unreachable")])

    toolkit = FakeToolkit()
    result = solve_with_model(
        model,
        toolkit,
        RecordingSink(),
        timeout_s=0.01,
    )

    assert result.error == "API planner timed out after 0.01s"
    assert toolkit.cancel_calls == 1
    assert result.messages == [{"role": "user", "content": "complete the task"}]


def test_queue_and_dashboard_inputs_are_rejected_before_model_use() -> None:
    calls = 0

    def model(messages: list[Any], info: Any) -> ModelResponse:
        nonlocal calls
        del messages, info
        calls += 1
        return ModelResponse(parts=[TextPart("unused")])

    planner = ApiAgentLoop(
        FunctionModel(model),
        dashboard_events=RecordingSink(),
    )

    with pytest.raises(ValueError, match="cannot be used together"):
        planner.solve(
            system_prompt="",
            user_message="task",
            toolkit=FakeToolkit(),
            max_turns=1,
            input_queue=queue.Queue(),
            dashboard_interaction=object(),
        )

    assert calls == 0


def test_tool_schema_and_dispatch_are_mapped_to_pydantic_ai() -> None:
    toolkit = FakeToolkit()

    tools = _build_tools(toolkit)

    assert [tool.name for tool in tools] == ["read_image", "finish"]
    finish = tools[1]
    assert finish.description == "Finish after the environment accepts the result."
    assert (
        finish.function_schema.json_schema
        == toolkit.get_tools_spec()[0]["input_schema"]
    )


def test_tool_result_conversion_keeps_text_and_images_separate() -> None:
    raw_image = b"\x89PNG\r\ncontract-image"
    encoded = base64.b64encode(raw_image).decode()
    blocks = [
        {"type": "text", "text": "observation"},
        {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": encoded,
            },
        },
    ]

    text, images = _content_blocks_to_pydantic(blocks)

    assert text == "observation"
    assert images == [BinaryContent(data=raw_image, media_type="image/png")]
    assert blocks[1]["source"]["data"] == encoded


def test_no_images_mode_suppresses_binary_tool_content() -> None:
    toolkit = FakeToolkit({"value": "visible", "_image_bytes": b"secret pixels"})

    multimodal = _make_tool_function(toolkit, "finish")(
        status="success",
        summary="done",
    )
    text_only = _make_tool_function(toolkit, "finish", no_images=True)(
        status="success",
        summary="done",
    )

    assert isinstance(multimodal, ToolReturn)
    assert multimodal.return_value == '{\n  "value": "visible"\n}'
    assert len(multimodal.content or []) == 1
    assert isinstance(multimodal.content[0], BinaryContent)
    assert text_only == '{\n  "value": "visible"\n}'
    assert "secret" not in text_only
