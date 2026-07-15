"""Provider-independent tool-use agent loop built on pydantic-ai.

The loop wraps the agent's :class:`~rpent.tools.toolkit.Toolkit` as
pydantic-ai function tools and drives a single :class:`pydantic_ai.Agent` run,
streaming each turn so progress is logged in real time. Task completion is
signalled by the env-provided ``finish`` tool, whose result carries ``_finish``.
"""

from __future__ import annotations

import asyncio
import base64
import dataclasses
import json
from typing import Any

from pydantic_ai import Agent, BinaryContent, ModelSettings, Tool, ToolReturn
from pydantic_ai.capabilities import ProcessHistory, Thinking
from pydantic_ai.exceptions import UsageLimitExceeded
from pydantic_ai.messages import (
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    ModelMessage,
    ModelResponse,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    UserPromptPart,
)
from pydantic_ai.models import Model
from pydantic_ai.usage import RunUsage, UsageLimits

from rpent.cerebrum.base import CerebrumResult
from rpent.tools.toolkit import Toolkit
from rpent.utils.logging import get_logger

logger = get_logger("api_loop")

#: Console-log truncation limits (characters).
_TEXT_LOG_LIMIT = 500
_ARGS_LOG_LIMIT = 250
_TOOL_LOG_LIMIT = 350

#: Cap on cumulative decoded image bytes kept in the resent request history.
_MAX_HISTORY_IMAGE_BYTES = 4 * 1024 * 1024

#: Always retain at least this many of the most recent images, even if a single
#: frame exceeds the byte budget, so the model never loses its current view.
_MIN_RECENT_IMAGES = 2


class ApiAgentLoop:
    """Cerebrum that runs the tool-calling loop via a pydantic-ai ``Agent``."""

    def __init__(self, model: Model, max_tokens: int = 8192, dashboard: Any = None):
        """Store the pydantic-ai model, output-token cap, and dashboard hook."""
        self._model = model
        self._max_tokens = max_tokens
        self._dashboard = dashboard

    def solve(
        self,
        *,
        system_prompt: str,
        user_message: str,
        toolkit: Toolkit,
        max_turns: int,
    ) -> CerebrumResult:
        """Run the tool-calling loop until finish, normal stop, or budget."""
        return asyncio.run(
            self._solve(
                system_prompt=system_prompt,
                user_message=user_message,
                toolkit=toolkit,
                max_turns=max_turns,
            )
        )

    async def _solve(
        self,
        *,
        system_prompt: str,
        user_message: str,
        toolkit: Toolkit,
        max_turns: int,
    ) -> CerebrumResult:
        agent = Agent(
            self._model,
            instructions=system_prompt or None,
            tools=_build_tools(toolkit),
            model_settings=_build_model_settings(self._model, self._max_tokens),
            capabilities=[
                Thinking(effort="high"),
                ProcessHistory(processor=_prune_history_images),
            ],
        )

        dashboard = self._dashboard
        messages: list[dict[str, Any]] = [{"role": "user", "content": user_message}]
        finish_result: dict[str, Any] | None = None
        n_tool_calls = 0
        turns = 0
        last_error: str | None = None
        # Shared usage accumulator so token stats span every injected sub-run.
        usage = RunUsage()

        # Self-maintained pydantic-ai message history. It is rebuilt from the
        # nodes we iterate (request/response alternating) so that when a user
        # interjection breaks the run mid-flight, the history handed to the
        # next ``agent.iter`` always ends on a balanced tool-return request —
        # never on a dangling tool-call with no result.
        history: list[ModelMessage] = []
        next_prompt = user_message

        try:
            while True:
                stop = False
                pending_inject: list[str] = []
                if dashboard is not None:
                    dashboard.set_busy(True)
                async with agent.iter(
                    next_prompt,
                    message_history=history or None,
                    usage=usage,
                    # request_limit overrides pydantic-ai's default (50) so the
                    # manual max_turns break below is what bounds the loop.
                    usage_limits=UsageLimits(request_limit=max_turns + 1),
                ) as run:
                    async for node in run:
                        if Agent.is_model_request_node(node):
                            # A fresh model request means any prior tool returns
                            # have landed — a safe boundary to inject a user turn
                            # without splitting a tool call from its result.
                            request = getattr(node, "request", None)
                            if request is not None:
                                history.append(request)
                            if 0 < turns < max_turns:
                                pending_inject = _take_injection(dashboard)
                                if pending_inject:
                                    break
                        elif Agent.is_call_tools_node(node):
                            turns += 1
                            response = node.model_response
                            history.append(response)
                            messages.append(_serialize_response(response))
                            _log_response(response, run.usage, turns, max_turns)

                            async with node.stream(run.ctx) as stream:
                                async for event in stream:
                                    if isinstance(event, FunctionToolCallEvent):
                                        n_tool_calls += 1
                                        if event.part.tool_name == "finish":
                                            finish_result = {
                                                "_finish": True,
                                                **event.part.args_as_dict(),
                                            }
                                    elif isinstance(event, FunctionToolResultEvent):
                                        message = _serialize_tool_result(event)
                                        messages.append(message)
                                        _log_tool_result(message)

                            if finish_result is not None:
                                logger.info("FINISH called: %s", finish_result)
                                stop = True
                                break
                            if turns >= max_turns:
                                logger.info(
                                    "reached max_turns=%d. Stopping.", max_turns
                                )
                                stop = True
                                break
                        elif Agent.is_end_node(node):
                            logger.info("model ended turn without a tool call.")
                            stop = True
                            break

                if dashboard is not None:
                    dashboard.set_busy(False)
                if stop:
                    break
                # Nothing consumed mid-run? Pick up any message queued after the
                # run settled so a late submission still starts a new turn.
                if not pending_inject:
                    pending_inject = _take_injection(dashboard)
                if not pending_inject:
                    break

                next_prompt = "\n\n".join(pending_inject)
                messages.append({"role": "user", "content": next_prompt})
                logger.info(
                    "injecting %d user message(s) as a new turn", len(pending_inject)
                )
        except UsageLimitExceeded as e:
            logger.info("usage limit reached: %s", e)
        except Exception as e:  # noqa: BLE001 - surfaced via CerebrumResult.error
            last_error = f"{type(e).__name__}: {e}"
            logger.error("agent run failed: %s", last_error)

        return CerebrumResult(
            finish_result=finish_result,
            messages=messages,
            stats=_build_stats(usage, turns, n_tool_calls),
            error=last_error,
        )


def _take_injection(dashboard: Any) -> list[str]:
    """Drain queued dashboard messages, clearing any interrupt. ``[]`` if none."""
    if dashboard is None:
        return []
    msgs = dashboard.take_pending_messages()
    if msgs:
        dashboard.clear_interrupt()
    return msgs


def _build_model_settings(model: Model, max_tokens: int) -> ModelSettings:
    """Build model settings, enabling prompt caching for Anthropic models."""
    from pydantic_ai.models.anthropic import AnthropicModel, AnthropicModelSettings

    if isinstance(model, AnthropicModel):
        return AnthropicModelSettings(
            max_tokens=max_tokens,
            anthropic_cache_instructions=True,
            anthropic_cache_tool_definitions=True,
            anthropic_cache_messages=True,
        )
    return ModelSettings(max_tokens=max_tokens)


def _prune_history_images(messages: list[ModelMessage]) -> list[ModelMessage]:
    """Drop old camera images so the resent request body stays bounded."""
    # Every image in history, oldest -> newest: (msg_idx, part_idx, item_idx, nbytes).
    located: list[tuple[int, int, int, int]] = []
    for mi, message in enumerate(messages):
        for pi, part in enumerate(getattr(message, "parts", ()) or ()):
            if not isinstance(part, UserPromptPart) or not isinstance(
                part.content, list
            ):
                continue
            for ii, item in enumerate(part.content):
                if isinstance(item, BinaryContent) and item.media_type.startswith(
                    "image/"
                ):
                    located.append((mi, pi, ii, len(item.data)))

    if not located:
        return messages

    # Walk newest -> oldest, keeping images while under the byte budget.
    keep: set[tuple[int, int, int]] = set()
    total = 0
    for rank, (mi, pi, ii, nbytes) in enumerate(reversed(located)):
        if rank < _MIN_RECENT_IMAGES or total + nbytes <= _MAX_HISTORY_IMAGE_BYTES:
            keep.add((mi, pi, ii))
            total += nbytes

    if len(keep) == len(located):
        return messages

    drop_items_by_part: dict[tuple[int, int], set[int]] = {}
    for mi, pi, ii, _ in located:
        if (mi, pi, ii) not in keep:
            drop_items_by_part.setdefault((mi, pi), set()).add(ii)

    new_messages = list(messages)
    for (mi, pi), drop_items in drop_items_by_part.items():
        message = new_messages[mi]
        part = message.parts[pi]
        new_content = [
            "[earlier camera image omitted to bound request size]" if ci in drop_items else item
            for ci, item in enumerate(part.content)
        ]
        new_parts = list(message.parts)
        new_parts[pi] = dataclasses.replace(part, content=new_content)
        new_messages[mi] = dataclasses.replace(message, parts=new_parts)

    return new_messages


def _build_tools(toolkit: Toolkit) -> list[Tool]:
    """Wrap each toolkit tool as a pydantic-ai function tool from its schema."""
    tools: list[Tool] = []
    for spec in toolkit.get_tools_spec():
        name = spec["name"]
        tools.append(
            Tool.from_schema(
                function=_make_tool_function(toolkit, name),
                name=name,
                description=spec.get("description", ""),
                json_schema=spec.get("input_schema")
                or {"type": "object", "properties": {}},
                takes_ctx=False,
            )
        )
    return tools


def _make_tool_function(toolkit: Toolkit, name: str):
    """Return a callable that dispatches one tool call to the toolkit."""

    def _call(**kwargs: Any) -> Any:
        result = toolkit.execute_tool(name, kwargs)
        text, images = _content_blocks_to_pydantic(result.content_blocks)
        if images:
            return ToolReturn(return_value=text, content=images)
        return text

    _call.__name__ = name
    return _call


def _content_blocks_to_pydantic(
    blocks: list[dict[str, Any]],
) -> tuple[str, list[BinaryContent]]:
    """Split Anthropic-shaped content blocks into text and image content."""
    text_parts: list[str] = []
    images: list[BinaryContent] = []
    for block in blocks:
        block_type = block.get("type")
        if block_type == "text":
            text_parts.append(block.get("text", ""))
        elif block_type == "image":
            source = block.get("source") or {}
            data = source.get("data")
            if source.get("type") == "base64" and data:
                images.append(
                    BinaryContent(
                        data=base64.b64decode(data),
                        media_type=source.get("media_type", "image/png"),
                    )
                )
    text = "\n\n".join(part for part in text_parts if part) or "{}"
    return text, images


def _serialize_response(response: ModelResponse) -> dict[str, Any]:
    """Render one assistant turn as a serialisable transcript message."""
    content: list[dict[str, Any]] = []
    for part in response.parts:
        if isinstance(part, TextPart):
            if part.content:
                content.append({"type": "text", "text": part.content})
        elif isinstance(part, ThinkingPart):
            if part.content:
                content.append({"type": "thinking", "thinking": part.content})
        elif isinstance(part, ToolCallPart):
            content.append(
                {
                    "type": "tool_use",
                    "id": part.tool_call_id,
                    "name": part.tool_name,
                    "input": part.args_as_dict(),
                }
            )
    return {"role": "assistant", "content": content}


def _serialize_tool_result(event: FunctionToolResultEvent) -> dict[str, Any]:
    """Render one tool result as a serialisable transcript message (no images)."""
    part = event.part
    content = getattr(part, "content", None)
    if not isinstance(content, str):
        content = json.dumps(content, default=str)
    return {
        "role": "tool",
        "name": getattr(part, "tool_name", None),
        "tool_call_id": getattr(part, "tool_call_id", None),
        "content": content,
    }


def _build_stats(
    usage: RunUsage | None, turns: int, n_tool_calls: int
) -> dict[str, Any]:
    """Assemble the run stats dict from accumulated usage and counters."""
    stats: dict[str, Any] = {"turns_used": turns, "tool_calls": n_tool_calls}
    if usage is not None:
        stats.update(
            {
                "total_input_tokens": int(usage.input_tokens or 0),
                "total_output_tokens": int(usage.output_tokens or 0),
                "cache_read_tokens": int(usage.cache_read_tokens or 0),
                "cache_write_tokens": int(usage.cache_write_tokens or 0),
                "requests": int(usage.requests or 0),
            }
        )
    return stats


def _log_response(
    response: ModelResponse, usage: RunUsage, turn: int, max_turns: int
) -> None:
    """Log model text, thinking, tool calls, and cumulative usage for a turn."""
    logger.info("=== turn %d/%d ===", turn, max_turns)
    for part in response.parts:
        if isinstance(part, TextPart):
            text = (part.content or "").strip()
            if text:
                logger.info("[model] %s", text)
        elif isinstance(part, ThinkingPart):
            text = (part.content or "").strip()
            if text:
                logger.info("[think] %s", _clip(text, _TEXT_LOG_LIMIT))
        elif isinstance(part, ToolCallPart):
            args = json.dumps(part.args_as_dict(), default=str)
            logger.info("[tool>] %s(%s)", part.tool_name, _clip(args, _ARGS_LOG_LIMIT))
    logger.info(
        "[usage] in=%s out=%s cache_read=%s cache_write=%s requests=%s",
        usage.input_tokens,
        usage.output_tokens,
        usage.cache_read_tokens,
        usage.cache_write_tokens,
        usage.requests,
    )


def _log_tool_result(message: dict[str, Any]) -> None:
    """Log a one-line summary of a tool result."""
    content = " ".join((message.get("content") or "").split())
    logger.info("[tool<] %s: %s", message.get("name"), _clip(content, _TOOL_LOG_LIMIT))


def _clip(text: str, limit: int) -> str:
    """Truncate ``text`` to ``limit`` characters with an overflow marker."""
    if len(text) <= limit:
        return text
    return text[:limit] + "...(+%d)" % (len(text) - limit)
