# Copyright 2026 The RPent Authors.
# Licensed under the Apache License, Version 2.0.
from __future__ import annotations

import pytest
from pydantic_ai import BinaryContent, ToolReturn
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)

from robots.rl.history import retained_history
from rpent.planner.compaction import images_in


def history():
    image = BinaryContent(data=b"image", media_type="image/png")
    messages = [ModelRequest(parts=[UserPromptPart(["task", image, image])])]
    for i in range(4):
        messages.extend(
            [
                ModelResponse(
                    parts=[
                        ThinkingPart(f"reasoning-{i}"),
                        ThinkingPart(f"second-{i}"),
                        TextPart("ordinary content"),
                        ToolCallPart(
                            "move_to", {"xyz": [i, 0, 1]}, tool_call_id=str(i)
                        ),
                    ]
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            "move_to", ["measured", image, image], tool_call_id=str(i)
                        )
                    ]
                ),
            ]
        )
    return messages


@pytest.mark.parametrize("n", [-1, 0, 1, 3])
def test_retention_removes_only_selected_modalities_and_keeps_source(n):
    original = history()
    filtered, stats = retained_history(original, n, n)
    expected_replies = 4 if n == -1 else min(n, 4)
    assert (
        sum(
            isinstance(p, ThinkingPart)
            for m in filtered
            if isinstance(m, ModelResponse)
            for p in m.parts
        )
        == expected_replies * 2
    )
    expected_groups = 5 if n == -1 else min(n, 5)
    assert (
        sum(
            len(images_in(p))
            for m in filtered
            if isinstance(m, ModelRequest)
            for p in m.parts
        )
        == expected_groups * 2
    )
    assert (
        sum(
            isinstance(p, ToolCallPart)
            for m in filtered
            if isinstance(m, ModelResponse)
            for p in m.parts
        )
        == 4
    )
    assert (
        sum(
            isinstance(p, TextPart)
            for m in filtered
            if isinstance(m, ModelResponse)
            for p in m.parts
        )
        == 4
    )
    assert (
        sum(
            isinstance(p, ThinkingPart)
            for m in original
            if isinstance(m, ModelResponse)
            for p in m.parts
        )
        == 8
    )
    assert (
        sum(
            len(images_in(p))
            for m in original
            if isinstance(m, ModelRequest)
            for p in m.parts
        )
        == 10
    )
    assert stats["image_groups_after"] == expected_groups


def test_image_history_zero_has_no_forced_latest_image():
    filtered, _ = retained_history(history(), 0, 0)
    assert not any(
        images_in(p) for m in filtered if isinstance(m, ModelRequest) for p in m.parts
    )
    text = str(filtered)
    assert "reasoning-0" not in text and "ignore</think>" not in text
    assert "ordinary content" in text and "measured" in text


def test_retention_before_real_compaction_does_not_restore_removed_content(tmp_path):
    from types import SimpleNamespace

    from pydantic_ai import Agent
    from pydantic_ai.messages import ModelRequest, ModelResponse, ThinkingPart
    from pydantic_ai.models.function import FunctionModel
    from pydantic_ai_harness.compaction import SlidingWindowCompaction

    from robots.rl.history import HistoryRetention

    tk = SimpleNamespace(
        output_dir=tmp_path, reasoning_history=0, image_history=0, workflow_stats={}
    )
    count = 0

    def model(messages, info):
        nonlocal count
        assert not any(
            isinstance(part, ThinkingPart)
            for m in messages
            if isinstance(m, ModelResponse)
            for part in m.parts
        )
        assert not any(
            images_in(part)
            for m in messages
            if isinstance(m, ModelRequest)
            for part in m.parts
        )
        count += 1
        if count == 7:
            return ModelResponse(parts=[TextPart("done")])
        return ModelResponse(
            parts=[
                ThinkingPart("private historical reasoning"),
                ToolCallPart("look", {}, tool_call_id=str(count)),
            ]
        )

    def look() -> ToolReturn:
        image = BinaryContent(data=b"image", media_type="image/png")
        return ToolReturn(return_value=["measured", image, image])

    agent = Agent(
        FunctionModel(model),
        tools=[look],
        capabilities=[
            HistoryRetention(tk),
            SlidingWindowCompaction(max_messages=5, keep_messages=4),
        ],
    )
    import asyncio

    result = asyncio.run(agent.run("task"))
    assert result.output == "done" and count == 7
    assert len(list((tmp_path / "history_retention").glob("*.json"))) == 7
