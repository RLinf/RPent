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

"""Image accounting, retention, and prompt preservation for planner compaction."""

from __future__ import annotations

import io

import pytest
from pydantic_ai import (
    BinaryContent,
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)

from rpent.planner.compaction import (
    DEFAULT_IMAGE_TOKENS,
    ImageRetentionFallback,
    count_image_tokens,
    count_image_tokens_in,
    estimate_request_tokens,
    images_in,
    preserve_prompts,
    retain_images,
    retained_image_keys,
)


def png(width: int, height: int) -> bytes:
    """Return a solid PNG of the given size."""
    PIL = pytest.importorskip("PIL.Image")
    buffer = io.BytesIO()
    PIL.new("RGB", (width, height)).save(buffer, format="PNG")
    return buffer.getvalue()


def image(width: int = 1024, height: int = 1024) -> BinaryContent:
    return BinaryContent(data=png(width, height), media_type="image/png")


def observation(call_id: str, images: int = 2) -> list:
    """A tool call with its image-bearing result, as a rollout planner sees it."""
    return [
        ModelResponse(
            parts=[ToolCallPart("view_env", {}, tool_call_id=call_id)],
        ),
        ModelRequest(
            parts=[
                ToolReturnPart(
                    "view_env",
                    [image() for _ in range(images)],
                    tool_call_id=call_id,
                )
            ]
        ),
    ]


def test_image_tokens_follow_the_served_geometry():
    # 16-pixel patches merged 2x2: a 1024x1024 camera image is 1024 tokens.
    assert count_image_tokens(image(1024, 1024)) == 1024
    assert count_image_tokens(image(512, 512)) == 256
    # Non-square images keep the merged grid.
    assert count_image_tokens(image(1024, 512)) == 512


def test_unreadable_image_bytes_fall_back_to_a_flat_charge():
    broken = BinaryContent(data=b"not a png", media_type="image/png")
    assert count_image_tokens(broken) == DEFAULT_IMAGE_TOKENS


def test_images_inside_a_tool_result_are_counted():
    history = [ModelRequest(parts=[UserPromptPart("task")]), *observation("c1")]
    images = [
        i
        for m in history
        if isinstance(m, ModelRequest)
        for p in m.parts
        for i in images_in(p)
    ]
    assert len(images) == 2
    assert count_image_tokens_in(history) == 2048


def test_images_inside_a_user_prompt_are_counted():
    history = [
        ModelRequest(parts=[UserPromptPart(content=["look", image()])]),
    ]
    assert count_image_tokens_in(history) == 1024


def test_image_cost_is_added_to_the_text_estimate():
    raw = bytes(image(1024, 1024).data)
    history = [ModelRequest(parts=[UserPromptPart(content=["go", image()])])]
    text_only = estimate_request_tokens(history) - count_image_tokens_in(history)
    assert count_image_tokens_in(history) == 1024
    # The text estimate must not be inflated by the image payload's characters.
    assert text_only < 100
    assert len(raw) > 1000


def test_retention_keeps_every_image_inside_the_recent_pairs():
    # Two recent pairs carry images, so both survive and the older one does not.
    history = [
        ModelRequest(parts=[UserPromptPart("task")]),
        *observation("old"),
        *observation("new1"),
        *observation("new2"),
    ]
    keep = retained_image_keys(history, keep_pairs=2)
    assert {key for _, key in keep} == {"new1", "new2"}

    kept = retain_images(history, keep)
    assert count_image_tokens_in(kept) == 1024 * 4  # two pairs x two images
    assert "[image omitted during compaction]" in str(kept[2].parts[0].content)


def test_retention_falls_back_to_the_latest_image_when_recent_pairs_have_none():
    # The recent pair is a text-only tool result, so only one older image
    # survives instead of the whole stale sequence.
    history = [
        ModelRequest(parts=[UserPromptPart("task")]),
        *observation("img1"),
        *observation("img2"),
        ModelResponse(parts=[ToolCallPart("move_to", {}, tool_call_id="text")]),
        ModelRequest(
            parts=[
                ToolReturnPart("move_to", {"status": "reached"}, tool_call_id="text")
            ]
        ),
    ]
    keep = retained_image_keys(history, keep_pairs=1)
    assert len(keep) == 1

    kept = retain_images(history, keep)
    assert count_image_tokens_in(kept) == 1024 * 2  # the latest pair's two images


def test_retention_keeps_nothing_when_the_history_has_no_images():
    history = [
        ModelRequest(parts=[UserPromptPart("task")]),
        ModelResponse(parts=[TextPart("no tools")]),
    ]
    assert retained_image_keys(history, keep_pairs=4) == set()
    assert retain_images(history, set()) == history


def test_preserve_prompts_restores_each_exactly_once():
    system = SystemPromptPart("you control one arm")
    user = UserPromptPart("the task follows")
    original = [
        ModelRequest(parts=[system, user]),
        ModelResponse(parts=[TextPart("ok")]),
    ]

    # A summarizing pass dropped both prompts.
    compacted = [ModelRequest(parts=[SystemPromptPart("<summary>did things")])]
    restored = preserve_prompts(original, compacted)

    systems = [
        part
        for message in restored
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, SystemPromptPart) and part.content == "you control one arm"
    ]
    users = [
        part
        for message in restored
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, UserPromptPart) and part.content == "the task follows"
    ]
    assert len(systems) == 1
    assert len(users) == 1


def test_preserve_prompts_drops_a_duplicate():
    system = SystemPromptPart("you control one arm")
    user = UserPromptPart("the task follows")
    original = [ModelRequest(parts=[system, user])]
    duplicated = [
        ModelRequest(parts=[SystemPromptPart("you control one arm")]),
        ModelRequest(parts=[UserPromptPart("the task follows")]),
    ]
    restored = preserve_prompts(original, duplicated)
    assert (
        sum(
            1
            for message in restored
            if isinstance(message, ModelRequest)
            for part in message.parts
            if isinstance(part, SystemPromptPart)
            and part.content == "you control one arm"
        )
        == 1
    )


@pytest.mark.anyio
async def test_retention_fallback_is_a_compaction_strategy():
    strategy = ImageRetentionFallback(keep_pairs=1)
    history = [
        ModelRequest(parts=[UserPromptPart("task")]),
        *observation("old"),
        *observation("new"),
    ]
    kept = await strategy.compact(history, None)
    assert count_image_tokens_in(kept) == 1024 * 2
