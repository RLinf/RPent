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
import json
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
    _make_tool_function,
)

from ._native_helpers import PNG, FakeToolkit, RecordingSink


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


def test_successful_finish_waits_for_its_tool_result(
    make_toolkit,
) -> None:
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

    toolkit = make_toolkit()
    sink = RecordingSink()
    result = solve_with_model(model, toolkit, sink)

    assert seen_instructions == ["Use tools carefully."]
    assert toolkit.calls == [("finish", {"status": "success", "summary": "done"})]
    assert result.finish_result == {
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


def test_rejected_finish_does_not_end_the_run(
    make_toolkit,
) -> None:
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

    toolkit = make_toolkit({"error": "finish refused by environment"})
    result = solve_with_model(model, toolkit, RecordingSink())

    assert result.finish_result is None
    assert result.error is None
    assert result.stats["tool_calls"] == 1
    assert any(
        message.get("role") == "tool"
        and "finish refused by environment" in message.get("content", "")
        for message in result.messages
    )


def test_backend_failure_is_returned_without_escaping(
    make_toolkit,
) -> None:
    def model(messages: list[Any], info: Any) -> ModelResponse:
        del messages, info
        raise RuntimeError("provider failed")

    result = solve_with_model(model, make_toolkit(), RecordingSink())

    assert result.finish_result is None
    assert result.error == "RuntimeError: provider failed"
    assert result.messages[0] == {"role": "user", "content": "complete the task"}


def test_timeout_cancels_active_toolkit_work(
    make_toolkit,
) -> None:
    async def model(messages: list[Any], info: Any) -> ModelResponse:
        del messages, info
        await asyncio.sleep(10)
        return ModelResponse(parts=[TextPart("unreachable")])

    toolkit = make_toolkit()
    result = solve_with_model(
        model,
        toolkit,
        RecordingSink(),
        timeout_s=0.01,
    )

    assert result.error == "API planner timed out after 0.01s"
    assert toolkit.cancel_calls >= 1
    assert result.messages[0] == {"role": "user", "content": "complete the task"}


def test_queue_and_dashboard_inputs_are_rejected_before_model_use(
    make_toolkit,
) -> None:
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
            toolkit=make_toolkit(),
            max_turns=1,
            input_queue=queue.Queue(),
            dashboard_interaction=object(),
        )

    assert calls == 0


def test_tool_schema_and_dispatch_are_mapped_to_pydantic_ai(
    make_toolkit,
) -> None:
    toolkit = make_toolkit()

    tools = _build_tools(toolkit)

    assert [tool.name for tool in tools] == [
        "read_image",
        *[tool.name for tool in toolkit.list_tools() if tool.name != "read_image"],
    ]
    assert "read_image" in [tool.name for tool in tools]
    assert all(tool.sequential for tool in tools)
    finish = next(tool for tool in tools if tool.name == "finish")
    assert finish.function_schema.json_schema == next(
        tool.input_schema for tool in toolkit.list_tools() if tool.name == "finish"
    )


def test_tool_result_conversion_keeps_text_and_images_separate(
    make_toolkit,
) -> None:
    toolkit = make_toolkit({"value": "visible"}, images=[PNG])
    result = asyncio.run(_make_tool_function(toolkit, "inspect_scene")())
    assert isinstance(result, ToolReturn)
    assert json.loads(result.return_value) == {"value": "visible"}
    assert result.content == [BinaryContent(data=PNG, media_type="image/png")]


def test_no_images_mode_suppresses_binary_tool_content(
    make_toolkit,
) -> None:
    toolkit = make_toolkit({"value": "visible"}, images=[PNG])
    result = asyncio.run(
        _make_tool_function(toolkit, "inspect_scene", no_images=True)()
    )
    assert json.loads(result) == {"value": "visible"}
    assert "contract-image" not in result
