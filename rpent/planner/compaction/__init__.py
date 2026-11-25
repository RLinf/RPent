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

"""Image retention and current-request accounting for rollout compaction.

Qwen's API loop counts its mapped request with the checkpoint processor and
tokenizer. Other models use a labelled text/geometry estimate which never
stringifies image bytes or reuses usage from an edited history.
"""

from __future__ import annotations

import math
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, replace
from io import BytesIO
from typing import Any

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import (
    BinaryContent,
    FilePart,
    ImageUrl,
    ModelMessage,
    ModelRequest,
    ModelRequestPart,
    SystemPromptPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.tools import AgentDepsT, RunContext
from pydantic_ai_harness.compaction import (
    ClearToolResults,
    CompactionStrategy,
    SummarizingCompaction,
)
from pydantic_ai_harness.compaction._shared import (
    context_for_request,
    estimate_token_count,
    iter_tool_pairs,
    record_compaction_reclaim,
)

# Qwen3.5-4B vision geometry: 16-pixel patches merged 2x2, so one merged token
# covers a 32x32 pixel block and a 1024x1024 camera image is 1024 tokens.
PATCH_SIZE = 16
MERGE_SIZE = 2
#: Tokens charged for one image whose dimensions this process cannot read.
DEFAULT_IMAGE_TOKENS = 1024
#: Text substituted for a removed image, so the model still sees a view existed.
OMITTED = "[image omitted during compaction]"

Image = BinaryContent | ImageUrl


def count_image_tokens(image: Image) -> int:
    """Return the prompt tokens one image occupies for the served model."""
    data = getattr(image, "data", None)
    if data is None:
        return DEFAULT_IMAGE_TOKENS
    try:
        from PIL import Image as PILImage

        with PILImage.open(BytesIO(data)) as decoded:
            width, height = decoded.size
    except Exception:
        return DEFAULT_IMAGE_TOKENS
    block = PATCH_SIZE * MERGE_SIZE
    return math.ceil(width / block) * math.ceil(height / block)


def _is_image(item: Any) -> bool:
    return isinstance(item, (BinaryContent, ImageUrl))


def images_in(part: ModelRequestPart) -> list[Image]:
    """Return the images one request part sends to the model."""
    if isinstance(part, FilePart):
        return [part.content]
    if isinstance(part, UserPromptPart):
        if isinstance(part.content, str):
            return []
        return [item for item in part.content if _is_image(item)]
    if type(part) is ToolReturnPart:
        return list(part.files)
    return []


def count_image_tokens_in(messages: Sequence[ModelMessage]) -> int:
    """Return the image tokens a request built from *messages* would send."""
    return sum(
        count_image_tokens(image)
        for message in messages
        if isinstance(message, ModelRequest)
        for part in message.parts
        for image in images_in(part)
    )


def estimate_request_tokens(
    messages: Sequence[ModelMessage],
    tokenizer: Callable[[str], int] | None = None,
) -> int:
    """Fallback count of current text and image geometry without stale usage.

    Exact Qwen request counting lives on QwenChatModel and includes its mapped
    template and tools. This fallback never stringifies binary image content,
    and never reuses usage from a request before history was edited.
    """
    stripped = retain_images(list(messages), set())
    return estimate_token_count(stripped, tokenizer) + count_image_tokens_in(messages)


def _key(index: int, position: int, part: ModelRequestPart) -> tuple[int, str]:
    """Stable address of one image-bearing part within a history."""
    if isinstance(part, ToolReturnPart) and part.tool_call_id:
        return index, part.tool_call_id
    return index, str(position)


def _without_images(part: ModelRequestPart) -> ModelRequestPart:
    """Return *part* with its images replaced by a text marker."""
    if isinstance(part, FilePart):
        return UserPromptPart(content=OMITTED)
    if isinstance(part, UserPromptPart) and not isinstance(part.content, str):
        return replace(
            part,
            content=[OMITTED if _is_image(item) else item for item in part.content],
        )
    if type(part) is ToolReturnPart:
        items = [
            OMITTED if _is_image(item) else item
            for item in part.content_items(mode="raw")
        ]
        if all(isinstance(item, str) for item in items) and len(items) == 1:
            return replace(part, content=items[0])
        return replace(part, content=items)
    return part


def retain_images(
    messages: list[ModelMessage], keep: set[tuple[int, str]]
) -> list[ModelMessage]:
    """Return *messages* with every image outside *keep* replaced by a marker."""
    out: list[ModelMessage] = []
    for index, message in enumerate(messages):
        if not isinstance(message, ModelRequest):
            out.append(message)
            continue
        parts: list[ModelRequestPart] = []
        changed = False
        for position, part in enumerate(message.parts):
            if not images_in(part) or _key(index, position, part) in keep:
                parts.append(part)
                continue
            parts.append(_without_images(part))
            changed = True
        out.append(replace(message, parts=parts) if changed else message)
    return out


def retained_image_keys(
    messages: Sequence[ModelMessage], keep_pairs: int
) -> set[tuple[int, str]]:
    """Return the image addresses the retention policy keeps.

    Images inside the most recent ``keep_pairs`` tool pairs all survive when
    those pairs carry any image; otherwise only the single latest image in the
    history survives.
    """
    pairs = iter_tool_pairs(messages)
    recent = (
        {pair.tool_call_id for pair in pairs[-keep_pairs:]} if keep_pairs else set()
    )
    keep: set[tuple[int, str]] = set()
    for index, message in enumerate(messages):
        if not isinstance(message, ModelRequest):
            continue
        for position, part in enumerate(message.parts):
            if (
                type(part) is ToolReturnPart
                and part.tool_call_id in recent
                and images_in(part)
            ):
                keep.add(_key(index, position, part))
    if keep:
        return keep

    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        if not isinstance(message, ModelRequest):
            continue
        for position in range(len(message.parts) - 1, -1, -1):
            part = message.parts[position]
            if images_in(part):
                return {_key(index, position, part)}
    return set()


@dataclass
class ImageRetentionFallback(CompactionStrategy[AgentDepsT]):
    """Keep the images a rollout planner can still act on, and drop the rest."""

    keep_pairs: int = 8
    """Number of most-recent tool-call / tool-return pairs whose images survive."""

    async def compact(
        self,
        messages: list[ModelMessage],
        ctx: RunContext[AgentDepsT],
    ) -> list[ModelMessage]:
        """Apply the retention policy to *messages* unconditionally."""
        return retain_images(messages, retained_image_keys(messages, self.keep_pairs))


def _first_system_prompt(messages: Sequence[ModelMessage]) -> SystemPromptPart | None:
    """Return the run's own system prompt, ignoring compaction summaries."""
    for message in messages:
        if not isinstance(message, ModelRequest):
            continue
        for part in message.parts:
            if isinstance(part, SystemPromptPart):
                return part
    return None


def _first_user_prompt(messages: Sequence[ModelMessage]) -> UserPromptPart | None:
    for message in messages:
        if not isinstance(message, ModelRequest):
            continue
        for part in message.parts:
            if isinstance(part, UserPromptPart):
                return part
    return None


def _same_prompt(part: ModelRequestPart, prompt: ModelRequestPart) -> bool:
    """Match a prompt by role and content; equal content is not object equality."""
    if isinstance(prompt, SystemPromptPart):
        return isinstance(part, SystemPromptPart) and part.content == prompt.content
    if isinstance(prompt, UserPromptPart):
        return isinstance(part, UserPromptPart) and part.content == prompt.content
    return part is prompt


def _restore_prompt(
    messages: list[ModelMessage], prompt: ModelRequestPart
) -> list[ModelMessage]:
    """Keep the first copy of *prompt*, dropping later ones; insert it when missing."""
    out: list[ModelMessage] = []
    seen = False
    for message in messages:
        if not isinstance(message, ModelRequest):
            out.append(message)
            continue
        if not any(_same_prompt(part, prompt) for part in message.parts):
            out.append(message)
            continue
        if seen:
            parts = [part for part in message.parts if not _same_prompt(part, prompt)]
            if parts:
                out.append(replace(message, parts=parts))
            continue
        seen = True
        out.append(message)
    if seen:
        return out
    insert_at = 1 if out and _first_system_prompt(out[:1]) is not None else 0
    return [*out[:insert_at], ModelRequest(parts=[prompt]), *out[insert_at:]]


def preserve_prompts(
    original: Sequence[ModelMessage], compacted: list[ModelMessage]
) -> list[ModelMessage]:
    """Ensure the original system and opening user prompt each appear once."""
    result = compacted
    system_prompt = _first_system_prompt(original)
    if system_prompt is not None:
        result = _restore_prompt(result, system_prompt)
    user_prompt = _first_user_prompt(original)
    if user_prompt is not None:
        result = _restore_prompt(result, user_prompt)
    return result


@dataclass
class ImageAwareCompaction(AbstractCapability[AgentDepsT]):
    """Escalate through cheap-to-expensive tiers until the real prompt fits.

    Tiers are driven directly, so their own triggers are bypassed; the loop
    re-measures text plus images after each tier and stops at ``target_tokens``.
    Afterwards the original system prompt and opening user prompt are restored
    exactly once, and images outside the retention policy are dropped.
    """

    tiers: Sequence[CompactionStrategy[AgentDepsT]]
    """Strategies applied in order, cheap to expensive."""

    target_tokens: int
    """Prompt budget in tokens, images included."""

    tokenizer: Callable[[str], int] | None = None
    """Optional exact tokenizer for message text."""

    keep_pairs: int = 8
    """Retained tool pairs whose images always survive."""

    def __post_init__(self) -> None:
        if not self.tiers:
            raise ValueError("tiers must not be empty.")
        if self.target_tokens < 1:
            raise ValueError("target_tokens must be positive.")

    async def compact(
        self,
        messages: list[ModelMessage],
        ctx: RunContext[AgentDepsT],
    ) -> list[ModelMessage]:
        """Apply the tier list unconditionally."""
        return await self._escalate(list(messages), ctx)

    async def _escalate(
        self,
        messages: list[ModelMessage],
        ctx: RunContext[AgentDepsT],
        measure: Callable[[list[ModelMessage]], Awaitable[int]] | None = None,
    ) -> list[ModelMessage]:
        original = messages
        for tier in self.tiers:
            tokens = (
                await measure(messages)
                if measure is not None
                else estimate_request_tokens(messages, self.tokenizer)
            )
            if tokens <= self.target_tokens:
                break
            messages = await tier.compact(messages, ctx)
        messages = retain_images(
            messages, retained_image_keys(messages, self.keep_pairs)
        )
        return preserve_prompts(original, messages)

    async def before_model_request(self, ctx, request_context):
        """Use the mapped checkpoint count when the model provides one."""

        async def measure(messages):
            counter = getattr(request_context.model, "count_prompt_tokens", None)
            if counter is None:
                return estimate_request_tokens(messages, self.tokenizer)
            count = await counter(
                messages,
                request_context.model_settings,
                request_context.model_request_parameters,
            )
            return count.total_tokens

        messages = list(request_context.messages)
        before = await measure(messages)
        if before <= self.target_tokens:
            return request_context
        request_ctx = context_for_request(ctx, request_context)
        compacted = await self._escalate(messages, request_ctx, measure)
        request_context.messages = compacted
        record_compaction_reclaim(request_context, before, await measure(compacted))
        return request_context


def build_history_compaction(
    options: dict[str, Any],
    *,
    tokenizer: Callable[[str], int] | None = None,
    model: Any = None,
) -> ImageAwareCompaction:
    """Build a rollout planner's image-aware compaction from config options.

    Args:
        options: ``target_tokens`` plus ``clear_tool_results`` and
            ``summarizing`` tier options, or ``SlidingWindowCompaction``
            options when the recipe keeps the plain message window.
        tokenizer: Optional exact text tokenizer of the served policy.
        model: Model for the summarizing tier; ``None`` inherits the run's.

    Returns:
        The compaction capability for the planner run.
    """
    if "target_tokens" not in options:
        from rpent.planner.compaction.window import RequestAwareSlidingWindow

        return RequestAwareSlidingWindow(**options, tokenizer=tokenizer)
    clear = dict(options.get("clear_tool_results") or {})
    summarizing = dict(options.get("summarizing") or {})
    keep_pairs = int(clear.get("keep_pairs", 8))
    tiers: list[CompactionStrategy[Any]] = [
        ClearToolResults(**clear, tokenizer=tokenizer),
        ImageRetentionFallback(keep_pairs=keep_pairs),
        SummarizingCompaction(**summarizing, model=model, tokenizer=tokenizer),
    ]
    return ImageAwareCompaction(
        tiers=tiers,
        target_tokens=int(options["target_tokens"]),
        tokenizer=tokenizer,
        keep_pairs=keep_pairs,
    )
