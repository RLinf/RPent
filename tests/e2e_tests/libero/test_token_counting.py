# Copyright 2026 The RPent Authors.
# Licensed under the Apache License, Version 2.0.
"""Opt-in real-SGLang accounting checks; max one output token, no simulator actions.

RPENT_RUN_TOKEN_COUNT_INTEGRATION=1 RPENT_TOKEN_COUNT_ENDPOINTS=<url[,url]>
pytest tests/e2e_tests/libero/test_token_counting.py -v
"""

from __future__ import annotations

import asyncio
import io
import os
from typing import Literal

import pytest
from PIL import Image
from pydantic_ai import RunContext, RunUsage
from pydantic_ai.messages import (
    BinaryContent,
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models import ModelRequestContext, ModelRequestParameters
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.usage import RequestUsage

from robots.rl.history import retained_history
from robots.rl.toolkit import episode_tools
from rpent.planner.compaction.window import RequestAwareSlidingWindow
from rpent.planner.qwen import QwenChatModel, qwen_model_settings
from rpent.tools import ToolResult

pytestmark = pytest.mark.skipif(
    os.environ.get("RPENT_RUN_TOKEN_COUNT_INTEGRATION") != "1",
    reason="opt-in checkpoint/model-server integration",
)
ENDPOINTS = os.environ.get(
    "RPENT_TOKEN_COUNT_ENDPOINTS", "http://127.0.0.1:30000/v1"
).split(",")


def image(size=1024, *, level=6, color="white"):
    buffer = io.BytesIO()
    Image.new("RGB", (size, size), color).save(
        buffer, format="PNG", compress_level=level
    )
    return BinaryContent(data=buffer.getvalue(), media_type="image/png")


def transcript(pairs=8):
    result = [
        ModelRequest(
            parts=[
                SystemPromptPart("Use the measured images and tools."),
                UserPromptPart("Inspect this task."),
            ]
        )
    ]
    for i in range(pairs):
        result += [
            ModelResponse(
                parts=[
                    ThinkingPart(f"Measured history reasoning {i}. " * 10),
                    ToolCallPart("view_env", {"camera": "all"}, tool_call_id=f"c{i}"),
                ],
                usage=RequestUsage(input_tokens=250000, output_tokens=20000),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        "view_env",
                        [
                            f"Observation {i}",
                            image(color=(i, 80, 170)),
                            image(color=(i, 90, 160)),
                        ],
                        tool_call_id=f"c{i}",
                    )
                ]
            ),
        ]
    return result


def parameters():
    return ModelRequestParameters(
        function_tools=[
            ToolDefinition(
                name=tool.name,
                description=tool.description,
                parameters_json_schema=tool.input_schema,
            )
            for tool in episode_tools(None, "libero", 3)
        ]
    )


def make_model(base_url, client):
    from pydantic_ai.providers.openai import OpenAIProvider

    return QwenChatModel(
        os.environ.get("RPENT_TOKEN_COUNT_MODEL", "qwen3.8-27b"),
        provider=OpenAIProvider(openai_client=client),
        profile={"openai_supports_strict_tool_definition": False},
    )


@pytest.fixture(scope="module")
def counter():
    """Exercise normal loopback discovery once and reuse small image metadata."""
    from openai import AsyncOpenAI

    async def load():
        async with AsyncOpenAI(base_url=ENDPOINTS[0], api_key="unused") as client:
            model = make_model(ENDPOINTS[0], client)
            return await model._get_prompt_counter()

    result = asyncio.run(load())
    assert result is not None and result.sglang
    return result


async def assert_wire(model, messages, settings, params):
    estimate = await model.count_prompt_tokens(messages, settings, params)
    prepared, params = model.prepare_request(settings, params)
    wire = await model._map_messages(messages, params, model_settings=prepared)
    tools, _ = model._get_tool_choice(prepared, params)
    response = await model.client.chat.completions.create(
        model=model.model_name,
        messages=wire,
        tools=tools,
        max_tokens=1,
        reasoning_effort=prepared.get("openai_reasoning_effort"),
        extra_body=prepared["extra_body"],
        temperature=1,
        top_p=0.95,
    )
    assert response.usage.prompt_tokens == estimate.total_tokens, (
        estimate.to_dict(),
        response.usage.model_dump(),
    )
    if estimate.image_tokens:
        assert (
            response.usage.prompt_tokens_details.image_tokens == estimate.image_tokens
        )
    return estimate


@pytest.mark.parametrize("base_url", ENDPOINTS)
@pytest.mark.parametrize(
    "case",
    [
        "initial1024",
        "initial512",
        "initial256",
        "uncompressed",
        "history_all",
        "history_r0_i1",
        "history_r3_i3",
        "history_r0_i0",
        "thinking_low",
        "thinking_medium",
        "thinking_off",
    ],
)
def test_current_request_matches_real_usage(base_url, case, counter):
    from openai import AsyncOpenAI

    async def run():
        async with AsyncOpenAI(
            base_url=base_url, api_key="unused", timeout=120
        ) as client:
            model = make_model(base_url, client)
            model._prompt_counter, model._prompt_counter_checked = counter, True
            settings = qwen_model_settings(max_tokens=1, reasoning_effort="xhigh")
            if case.startswith("history_"):
                messages = transcript()
                if case != "history_all":
                    reasoning, images = {
                        "history_r0_i1": (0, 1),
                        "history_r3_i3": (3, 3),
                        "history_r0_i0": (0, 0),
                    }[case]
                    messages, _ = retained_history(messages, reasoning, images)
            else:
                size = (
                    int(case.removeprefix("initial"))
                    if case.startswith("initial")
                    else 1024
                )
                messages = [
                    ModelRequest(
                        parts=[
                            SystemPromptPart("Inspect the image."),
                            UserPromptPart(
                                [
                                    "Current two cameras.",
                                    image(
                                        size, level=0 if case == "uncompressed" else 6
                                    ),
                                    image(size),
                                ]
                            ),
                        ]
                    )
                ]
                if case.startswith("thinking_"):
                    tier = case.removeprefix("thinking_")
                    settings = qwen_model_settings(
                        max_tokens=1,
                        reasoning_effort="none" if tier == "off" else tier,
                        thinking=tier != "off",
                    )
            await assert_wire(model, messages, settings, parameters())

    asyncio.run(run())


@pytest.mark.parametrize("base_url", ENDPOINTS)
@pytest.mark.parametrize("trigger", ["tokens", "messages", "pruned"])
def test_real_payload_after_compaction(base_url, trigger, counter):
    from openai import AsyncOpenAI

    async def run():
        async with AsyncOpenAI(
            base_url=base_url, api_key="unused", timeout=120
        ) as client:
            model = make_model(base_url, client)
            model._prompt_counter, model._prompt_counter_checked = counter, True
            settings = qwen_model_settings(max_tokens=1, reasoning_effort="xhigh")
            messages = transcript(41 if trigger == "messages" else 8)
            if trigger == "pruned":
                messages, _ = retained_history(messages, reasoning=0, images=1)
            before = await model.count_prompt_tokens(messages, settings, parameters())
            ctx = RunContext(deps=None, model=model, usage=RunUsage(), prompt="task")
            request = ModelRequestContext(
                model=model,
                messages=messages,
                model_settings=settings,
                model_request_parameters=parameters(),
            )
            window = RequestAwareSlidingWindow(
                max_messages=80,
                keep_messages=40 if trigger == "messages" else 2,
                max_tokens=10000 if trigger == "tokens" else 183500,
            )
            await window.before_model_request(ctx, request)
            after = await assert_wire(model, request.messages, settings, parameters())
            if trigger == "pruned":
                assert len(request.messages) == len(messages)
                assert after.total_tokens == before.total_tokens
                assert len(after.images) == 2  # no old images restored
            else:
                assert after.total_tokens < before.total_tokens
                assert len(request.messages) == (41 if trigger == "messages" else 3)
                assert len(after.images) == (40 if trigger == "messages" else 2)

    asyncio.run(run())


@pytest.mark.timeout(180)
def test_native_agent_loop_retains_then_compacts_and_records_real_usage(
    tmp_path, counter
):
    """Use the real SDK loop, tools, trace and compactor, with no simulator."""
    import json
    from dataclasses import replace

    from pydantic_ai.capabilities import AbstractCapability

    from robots.rl.history import HistoryRetention
    from robots.rl.qwen_grammar import bounded_tool_format
    from rpent.dashboard.events import NullDashboardEventSink
    from rpent.memory import MemoryManager
    from rpent.planner.api_loop import ApiAgentLoop
    from rpent.planner.base import build_api_model
    from rpent.tools import iter_tools, tool
    from rpent.tools.toolkit import Toolkit

    class Views(Toolkit):
        planner_builtin_tools = False

        def __init__(self):
            self.output_dir = tmp_path
            self.reasoning_history, self.image_history = 0, 1
            self.workflow_stats = {}
            self.views = 0
            super().__init__(
                dashboard_events=NullDashboardEventSink(),
                memory=MemoryManager(tmp_path / "memory"),
            )
            self.add_tools(iter_tools(self))

        def _register_common_tools(self):
            pass

        def initial_observation(self):
            return ToolResult(
                data={"test": "token accounting"},
                images=[image(color=(1, 2, 3)).data, image(color=(4, 5, 6)).data],
            )

        @tool(readonly=True)
        def view_env(self, camera: Literal["all"] = "all") -> ToolResult:
            """Return both saved test images without any physical action."""
            self.views += 1
            return ToolResult(
                data={"observation": self.views},
                images=[
                    image(color=(self.views, 20, 30)).data,
                    image(color=(self.views, 40, 50)).data,
                ],
            )

        @tool(readonly=True)
        def finish(self) -> ToolResult:
            """End this accounting check. No arguments."""
            return ToolResult(data={"_finish": True, "success": False})

    class Stages(AbstractCapability):
        async def before_model_request(self, ctx, request_context):
            parameters = request_context.model_request_parameters
            first = ctx.usage.requests < 2
            prepared = replace(
                parameters,
                function_tools=parameters.function_tools if first else [],
                output_tools=[] if first else parameters.output_tools,
                allow_text_output=False,
            )
            request_context.model_request_parameters = prepared
            definitions = [*prepared.function_tools, *prepared.output_tools]
            settings = request_context.model_settings
            settings["extra_body"] = {
                **settings["extra_body"],
                "tool_choice": "auto",
                "response_format": bounded_tool_format(definitions),
            }
            return request_context

    from rpent.utils.logging import init_output_dir

    init_output_dir(tmp_path)
    toolkit = Views()
    model = build_api_model(
        "openai-chat:" + os.environ.get("RPENT_TOKEN_COUNT_MODEL", "qwen3.8-27b"),
        ENDPOINTS[0],
        strict_tool_definitions=False,
        request_trace_dir=tmp_path / "api_trace",
    )
    model._prompt_counter, model._prompt_counter_checked = counter, True
    planner = ApiAgentLoop(
        model=model,
        dashboard_events=NullDashboardEventSink(),
        max_tokens=2048,
        reasoning_effort="xhigh",
        sequential_tools=True,
        timeout_s=150,
        compaction=lambda: RequestAwareSlidingWindow(
            max_messages=4, max_fraction=0.7, context_window=262144, keep_messages=2
        ),
        capabilities_factory=lambda tk, system, user, turns: [
            Stages(),
            HistoryRetention(tk),
        ],
    )
    result = planner.solve(
        system_prompt="This is an accounting check. Immediately call the only available tool.",
        user_message="Use the available tool; do not describe imaginary actions.",
        toolkit=toolkit,
        max_turns=3,
    )
    assert result.error is None and toolkit.views == 2
    assert result.stats["turns_used"] == 3 and result.finish_result["_finish"]
    audits = sorted((tmp_path / "api_trace/token_counts").glob("*-usage.json"))
    assert len(audits) == 3
    assert all(json.loads(path.read_text())["difference"] == 0 for path in audits)
    counts = [
        json.loads(path.read_text())
        for path in sorted((tmp_path / "api_trace/token_counts").glob("*-input.json"))
    ]
    assert all(len(row["after"]["images"]) == 2 for row in counts)
    assert counts[-1]["triggered"]
    assert counts[-1]["messages_before"] > counts[-1]["messages_after"]
    assert counts[-1]["after"]["total_tokens"] < counts[-1]["threshold_tokens"]
