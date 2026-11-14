# Copyright 2026 The RPent Authors.
# Licensed under the Apache License, Version 2.0.
"""Regressions for image bytes, rewritten histories and the actual API compactor."""

from __future__ import annotations

import asyncio
import base64
import io
import json
from types import SimpleNamespace

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
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.usage import RequestUsage

from robots.rl.history import retained_history
from rpent.planner.api_loop import sliding_window_compaction
from rpent.planner.compaction import count_image_tokens_in, estimate_request_tokens
from rpent.planner.compaction.window import RequestAwareSlidingWindow
from rpent.planner.token_counting import QwenPromptCounter


def png(size=1024, level=6):
    buffer = io.BytesIO()
    Image.new("RGB", (size, size), "white").save(
        buffer, format="PNG", compress_level=level
    )
    return buffer.getvalue()


def picture(size=1024, level=6):
    return BinaryContent(data=png(size, level), media_type="image/png")


def uri(size=1024, level=6):
    return "data:image/png;base64," + base64.b64encode(png(size, level)).decode()


class Processor:
    """Deliberately use a resize policy unlike the old ceil(width/32) rule."""

    def __init__(self):
        self.calls = 0
        self.seen = None
        self.tokenizer = SimpleNamespace(apply_chat_template=self.render)

    def _get_num_multimodal_tokens(self, *, image_sizes):
        self.calls += 1
        return {"num_image_tokens": [777 for _ in image_sizes]}

    def render(self, messages, *, tools, tokenize, add_generation_prompt, **kwargs):
        self.seen = messages, tools, kwargs
        assert tokenize and add_generation_prompt
        return [0] * 30  # two image placeholders and 28 other template tokens


@pytest.mark.parametrize("sglang", [False, True])
def test_template_counter_decodes_openai_argument_strings_without_mutating_wire(
    sglang,
):
    from jinja2 import Environment

    template = Environment().from_string(
        "{% for name, value in messages[0].tool_calls[0].function.arguments|items %}"
        "{{ name }}={{ value|tojson }};{% endfor %}"
    )

    def render(messages, **kwargs):
        return template.render(messages=messages).encode()

    arguments = {"camera": "all", "step": 0}
    call = {"function": {"name": "view_env", "arguments": json.dumps(arguments)}}
    wire = [{"role": "assistant", "content": "", "tool_calls": [call]}]
    counter = QwenPromptCounter(
        SimpleNamespace(tokenizer=SimpleNamespace(apply_chat_template=render)),
        fingerprint="template-with-mapping-arguments",
        sglang=sglang,
    )
    encoded = counter.count(wire, tools=[], template_kwargs={})
    mapped = [
        {
            **wire[0],
            "tool_calls": [
                {"function": {**call["function"], "arguments": arguments}},
            ],
        }
    ]
    assert encoded.total_tokens == len(render(mapped))
    assert isinstance(wire[0]["tool_calls"][0]["function"]["arguments"], str)


def test_checkpoint_processor_controls_visual_cost_and_counts_occurrences():
    processor = Processor()
    counter = QwenPromptCounter(processor, fingerprint="config-hash")
    wire = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "look"},
                {"type": "image_url", "image_url": {"url": uri()}},
                {"type": "image_url", "image_url": {"url": uri()}},
            ],
        }
    ]
    tools = [{"type": "function", "function": {"name": "finish"}}]
    count = counter.count(wire, tools=tools, template_kwargs={"enable_thinking": True})
    assert count.image_tokens == 1554
    assert count.total_tokens == 30 - 2 + 1554
    assert processor.calls == 1  # repeated asset, two charged occurrences
    assert processor.seen[0][0]["content"][1:] == [{"type": "image"}] * 2
    assert processor.seen[1] == tools and processor.seen[2]["enable_thinking"]
    assert count.images[0]["processor_sha256"] == "config-hash"
    assert wire[0]["content"][1]["type"] == "image_url"  # no mutation


def test_tokenizer_mapping_default_does_not_count_dictionary_keys():
    def render(messages, *, return_dict=True, **kwargs):
        ids = list(range(19))
        return {"input_ids": ids, "attention_mask": [1] * 19} if return_dict else ids

    counter = QwenPromptCounter(
        SimpleNamespace(tokenizer=SimpleNamespace(apply_chat_template=render)),
        fingerprint="mapping-default-tokenizer",
    )
    result = counter.count(
        [{"role": "user", "content": "task"}], tools=[], template_kwargs={}
    )
    assert result.total_tokens == 19


def test_equal_pixels_different_png_compression_do_not_change_tokens():
    counts = []
    for level in [0, 9]:
        processor = Processor()
        counter = QwenPromptCounter(processor, fingerprint="same")
        counts.append(
            counter.count(
                [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": uri(level=level)},
                            }
                        ],
                    }
                ],
                tools=[],
                template_kwargs={},
            ).total_tokens
        )
    assert counts[0] == counts[1]
    # Also protect the fallback used by non-Qwen planners.
    a = [ModelRequest(parts=[ToolReturnPart("view", [picture(level=0)])])]
    b = [ModelRequest(parts=[ToolReturnPart("view", [picture(level=9)])])]
    assert estimate_request_tokens(a) == estimate_request_tokens(b) < 1200


@pytest.mark.parametrize("size,expected", [(1024, 1024), (512, 256), (256, 64)])
def test_fallback_counts_user_and_tool_images_without_binary_text(size, expected):
    messages = [
        ModelRequest(
            parts=[
                UserPromptPart(["image", picture(size)]),
                ToolReturnPart("view", ["two", picture(size)]),
            ]
        )
    ]
    assert count_image_tokens_in(messages) == expected * 2
    assert expected * 2 <= estimate_request_tokens(messages) < expected * 2 + 100


def history(pairs=3):
    messages = [
        ModelRequest(parts=[SystemPromptPart("system"), UserPromptPart("task")])
    ]
    for i in range(pairs):
        messages += [
            ModelResponse(
                parts=[
                    ThinkingPart("old thought " * 50),
                    ToolCallPart("view", {}, tool_call_id=str(i)),
                ],
                usage=RequestUsage(input_tokens=250_000, output_tokens=20_000),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        "view", ["view", picture(), picture()], tool_call_id=str(i)
                    )
                ]
            ),
        ]
    return messages


@pytest.mark.parametrize("keep,images", [(0, 0), (1, 2), (3, 6), (-1, 6)])
def test_retention_recounts_current_content_not_old_provider_usage(keep, images):
    original = history()
    retained, _ = retained_history(original, reasoning=0, images=keep)
    assert count_image_tokens_in(retained) == images * 1024
    assert estimate_request_tokens(retained) < images * 1024 + 100
    assert estimate_request_tokens(original) > estimate_request_tokens(retained)
    assert original[1].usage.input_tokens == 250_000  # raw usage left intact
    assert count_image_tokens_in(original) == 6144


def context(messages):
    model = FunctionModel(lambda messages, info: ModelResponse(parts=[]))
    ctx = RunContext(deps=None, model=model, usage=RunUsage(), prompt="task")
    request = ModelRequestContext(
        model=model,
        messages=messages,
        model_settings={},
        model_request_parameters=ModelRequestParameters(),
    )
    return ctx, request


def test_rewritten_small_history_does_not_trigger_from_stale_usage():
    retained, _ = retained_history(history(), reasoning=0, images=0)
    ctx, request = context(retained)
    window = RequestAwareSlidingWindow(
        max_messages=80, max_fraction=0.7, context_window=262144, keep_messages=40
    )
    result = asyncio.run(window.before_model_request(ctx, request))
    assert result.messages == retained


def test_image_cost_triggers_window_and_preserves_complete_tool_pairs():
    original, _ = retained_history(history(6), reasoning=0, images=-1)
    ctx, request = context(original)
    window = RequestAwareSlidingWindow(
        max_messages=80, max_tokens=8000, keep_messages=2
    )
    result = asyncio.run(window.before_model_request(ctx, request))
    assert len(result.messages) == 3
    assert result.messages[0].parts[0].content == "system"
    assert result.messages[0].parts[1].content == "task"
    calls = {
        p.tool_call_id
        for m in result.messages
        for p in m.parts
        if isinstance(p, ToolCallPart)
    }
    returns = {
        p.tool_call_id
        for m in result.messages
        for p in m.parts
        if isinstance(p, ToolReturnPart)
    }
    assert calls == returns == {"5"}
    assert estimate_request_tokens(result.messages) < 8000
    assert window._last_count.image_tokens == 2048
    # It must remeasure the compacted view on the following request.
    again = asyncio.run(window.before_model_request(ctx, request))
    assert again.messages == result.messages


def test_message_count_trigger_still_uses_eighty_to_forty():
    original = history(41)
    ctx, request = context(original)
    window = sliding_window_compaction(context_window=262144, max_fraction=0.7)()
    assert isinstance(window, RequestAwareSlidingWindow)  # actual CLI factory
    result = asyncio.run(window.before_model_request(ctx, request))
    assert len(result.messages) == 41  # first prompt plus 20 complete pairs
    assert result.messages[-1] is original[-1]


def test_explicit_remote_sglang_counter_configuration_is_local_only(monkeypatch):
    from pydantic_ai.providers.openai import OpenAIProvider

    from rpent.planner.qwen import QwenChatModel

    monkeypatch.setenv("RPENT_TOKENIZER_PATH", "/local/checkpoint")
    monkeypatch.setenv("RPENT_TOKEN_COUNT_BACKEND", "sglang")
    seen = []
    expected = object()

    def load(path, *, sglang):
        seen.append((path, sglang))
        return expected

    monkeypatch.setattr(QwenPromptCounter, "from_checkpoint", load)
    model = QwenChatModel(
        "qwen3.8-27b",
        provider=OpenAIProvider(base_url="http://model.invalid/v1", api_key="unused"),
    )

    async def run():
        try:
            assert await model._get_prompt_counter() is expected
            assert await model._get_prompt_counter() is expected
            assert seen == [("/local/checkpoint", True)]
        finally:
            await model.client.close()

    asyncio.run(run())


def test_sglang_history_normalization_does_not_mutate_raw_messages():
    processor = Processor()
    counter = QwenPromptCounter(processor, fingerprint="config", sglang=True)
    messages = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "segment", "arguments": '{"prompt":"cup"}'},
                }
            ],
        },
        {
            "role": "tool",
            "content": [
                {"type": "text", "text": "one"},
                {"type": "text", "text": "two"},
            ],
        },
    ]
    counter.count(
        messages,
        tools=[{"type": "function", "function": {"name": "segment"}}],
        template_kwargs={"reasoning_effort": "low"},
    )
    mapped, tools, kwargs = processor.seen
    assert mapped[0]["content"] == ""
    assert mapped[0]["tool_calls"][0]["function"]["arguments"] == {"prompt": "cup"}
    assert mapped[1]["content"] == "one two"
    assert tools[0]["defer_loading"] is None
    assert kwargs["reasoning_effort"] == "low"
    assert messages[0]["content"] is None
    assert isinstance(messages[0]["tool_calls"][0]["function"]["arguments"], str)


def test_chat_template_effort_override_matches_server_precedence():
    from pydantic_ai.providers.openai import OpenAIProvider

    from rpent.planner.qwen import QwenChatModel, qwen_model_settings

    processor = Processor()
    model = QwenChatModel(
        "qwen3.8-27b",
        provider=OpenAIProvider(base_url="http://model.invalid/v1", api_key="unused"),
    )
    model._prompt_counter = QwenPromptCounter(processor, fingerprint="config")
    model._prompt_counter_checked = True
    settings = qwen_model_settings(max_tokens=10, reasoning_effort="xhigh")
    settings["extra_body"]["chat_template_kwargs"]["reasoning_effort"] = "low"

    async def run():
        try:
            await model.count_prompt_tokens(
                [ModelRequest(parts=[UserPromptPart("task")])],
                settings,
                ModelRequestParameters(),
            )
            assert processor.seen[2]["reasoning_effort"] == "low"
        finally:
            await model.client.close()

    asyncio.run(run())
