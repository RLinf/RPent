"""Anthropic Messages API adapter for the shared API loop."""
from __future__ import annotations

import copy
import json
import time
from dataclasses import dataclass
from typing import Any, Callable

import anthropic

from physical_agent.cerebrum.adapters.base import ModelTurn, ToolCall, ToolResult
from physical_agent.utils.logging import get_logger

logger = get_logger("anthropic")


@dataclass
class AnthropicState:
    system: str | list[dict[str, Any]]
    tools: list[dict[str, Any]]
    messages: list[dict[str, Any]]


class AnthropicAdapter:
    """Adapter for Anthropic Messages API tool-use."""

    name = "Anthropic"

    def __init__(
        self,
        client: anthropic.Anthropic,
        model: str,
        max_tokens: int = 4096,
        *,
        thinking: bool = False,
        thinking_budget_tokens: int = 4096,
    ):
        self._client = client
        self._model = model
        self._thinking = bool(thinking)
        self._thinking_budget = int(thinking_budget_tokens)
        if self._thinking and max_tokens <= self._thinking_budget:
            new_max = self._thinking_budget + 1024
            logger.warning(
                "max_tokens=%d <= thinking budget_tokens=%d; bumping max_tokens to %d",
                max_tokens, self._thinking_budget, new_max,
            )
            max_tokens = new_max
        self._max_tokens = max_tokens

    def start(
        self,
        *,
        system_prompt: str,
        user_message: str,
        tools_spec: list[dict[str, Any]],
    ) -> AnthropicState:
        return AnthropicState(
            system=_cacheable_system(system_prompt),
            tools=_cacheable_tools(tools_spec),
            messages=[{"role": "user", "content": user_message}],
        )

    def call(self, state: AnthropicState) -> ModelTurn | None:
        response = self._call_with_retries(
            system=state.system,
            tools=state.tools,
            messages=state.messages,
        )
        if response is None:
            return None

        usage = response.usage
        return ModelTurn(
            raw_response=response,
            assistant_payload=response.content,
            stop_reason=response.stop_reason,
            tool_calls=_extract_tool_calls(response.content),
            usage={
                "input_tokens": int(getattr(usage, "input_tokens", 0) or 0),
                "output_tokens": int(getattr(usage, "output_tokens", 0) or 0),
                "cache_creation_input_tokens": int(
                    getattr(usage, "cache_creation_input_tokens", 0) or 0
                ),
                "cache_read_input_tokens": int(
                    getattr(usage, "cache_read_input_tokens", 0) or 0
                ),
            },
        )

    def append_assistant(self, state: AnthropicState, turn: ModelTurn) -> None:
        state.messages.append({"role": "assistant", "content": turn.assistant_payload})

    def append_tool_results(
        self,
        state: AnthropicState,
        tool_results: list[ToolResult],
        tool_result_formatter: Callable[[dict[str, Any]], list[dict[str, Any]]],
    ) -> None:
        content = []
        for tool_result in tool_results:
            content.append({
                "type": "tool_result",
                "tool_use_id": tool_result.call_id,
                "content": tool_result_formatter(tool_result.result),
            })
        state.messages.append({"role": "user", "content": content})

    def messages(self, state: AnthropicState) -> list[dict[str, Any]]:
        return state.messages

    def is_normal_stop(self, turn: ModelTurn) -> bool:
        return turn.stop_reason == "end_turn"

    def log_model_turn(
        self,
        turn: ModelTurn,
        *,
        usage_totals: dict[str, int],
    ) -> None:
        for block in turn.assistant_payload:
            block_type = getattr(block, "type", None)
            if block_type == "text" and getattr(block, "text", "").strip():
                logger.info("[claude] %s", block.text.strip())
            elif block_type == "thinking":
                text = (getattr(block, "thinking", "") or "").strip()
                if text:
                    if len(text) > 500:
                        text = text[:500] + "...(+%d)" % (len(text) - 500)
                    logger.info("[thinking] %s", text)
            elif block_type == "redacted_thinking":
                logger.info("[thinking] <redacted>")
            elif block_type == "tool_use":
                s = json.dumps(getattr(block, "input", {}), default=str)
                if len(s) > 250:
                    s = s[:250] + "...(+%d)" % (len(s) - 250)
                logger.info("[tool->] %s(%s)", getattr(block, "name", "?"), s)

        logger.info(
            "[usage] in=%s cache_create=%s cache_read=%s out=%s stop=%s "
            "total_in=%s total_cache_create=%s total_cache_read=%s total_out=%s",
            turn.usage.get("input_tokens", 0),
            turn.usage.get("cache_creation_input_tokens", 0),
            turn.usage.get("cache_read_input_tokens", 0),
            turn.usage.get("output_tokens", 0),
            turn.stop_reason,
            usage_totals.get("total_input_tokens", 0),
            usage_totals.get("total_cache_creation_input_tokens", 0),
            usage_totals.get("total_cache_read_input_tokens", 0),
            usage_totals.get("total_output_tokens", 0),
        )

    def api_failure_error(self) -> str:
        return "Anthropic API call failed after retries"

    def _call_with_retries(
        self,
        *,
        system: str | list[dict[str, Any]],
        tools: list[dict[str, Any]],
        messages: list[dict[str, Any]],
    ):
        last_err = None
        extra_kwargs: dict[str, Any] = {}
        if self._thinking:
            extra_kwargs["thinking"] = {
                "type": "enabled",
                "budget_tokens": self._thinking_budget,
            }
        for outer in range(3):
            try:
                return self._client.messages.create(
                    model=self._model,
                    max_tokens=self._max_tokens,
                    system=system,
                    tools=tools,
                    messages=messages,
                    **extra_kwargs,
                )
            except (
                anthropic.APIConnectionError,
                anthropic.APITimeoutError,
                anthropic.InternalServerError,
                anthropic.RateLimitError,
            ) as e:
                last_err = e
                wait = 10 * (outer + 1)
                logger.warning(
                    "API error '%s: %s' - sleeping %ds (retry %d/3)",
                    type(e).__name__, e, wait, outer + 1,
                )
                time.sleep(wait)
        logger.error("giving up after 3 retries; last error: %s", last_err)
        return None


def _extract_tool_calls(content: list[Any]) -> list[ToolCall]:
    tool_calls = []
    for block in content:
        if getattr(block, "type", None) != "tool_use":
            continue
        arguments = getattr(block, "input", {}) or {}
        if not isinstance(arguments, dict):
            arguments = {"value": arguments}
        tool_calls.append(ToolCall(
            id=getattr(block, "id", ""),
            name=getattr(block, "name", ""),
            arguments=arguments,
            raw_arguments=getattr(block, "input", None),
        ))
    return tool_calls


def _cacheable_system(system_prompt: str) -> list[dict[str, Any]] | str:
    """Return Anthropic system blocks with prompt caching enabled."""
    if not system_prompt:
        return system_prompt
    return [
        {
            "type": "text",
            "text": system_prompt,
            "cache_control": {"type": "ephemeral"},
        }
    ]


def _cacheable_tools(tools_spec: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Mark the stable tool surface as cacheable without mutating callers."""
    tools = copy.deepcopy(tools_spec)
    if tools:
        tools[-1]["cache_control"] = {"type": "ephemeral"}
    return tools
