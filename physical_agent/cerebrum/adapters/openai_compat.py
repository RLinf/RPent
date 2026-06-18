"""OpenAI-compatible Chat Completions adapter for the shared API loop."""
from __future__ import annotations

import copy
import json
import time
from dataclasses import dataclass
from typing import Any, Callable

from physical_agent.cerebrum.adapters.base import ModelTurn, ToolCall, ToolResult
from physical_agent.utils.logging import get_logger

logger = get_logger("openai")


@dataclass
class OpenAICompatState:
    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]]


class OpenAICompatibleAdapter:
    """Adapter for OpenAI-compatible Chat Completions tool calling."""

    name = "OpenAI-compatible"

    def __init__(
        self,
        client: Any,
        model: str,
        max_tokens: int = 4096,
        *,
        thinking: bool = False,
        reasoning_effort: str = "xhigh",
    ):
        self._client = client
        self._model = model
        self._max_tokens = max_tokens
        self._thinking = bool(thinking)
        self._reasoning_effort = reasoning_effort

    def start(
        self,
        *,
        system_prompt: str,
        user_message: str,
        tools_spec: list[dict[str, Any]],
    ) -> OpenAICompatState:
        messages: list[dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_message})
        return OpenAICompatState(
            messages=messages,
            tools=anthropic_tools_to_openai_tools(tools_spec),
        )

    def call(self, state: OpenAICompatState) -> ModelTurn | None:
        response = self._call_with_retries(
            tools=state.tools,
            messages=state.messages,
        )
        if response is None:
            return None

        usage = _get(response, "usage")
        choice = _first_choice(response)
        message = _get(choice, "message")
        assistant_message = _assistant_message_to_dict(message)
        return ModelTurn(
            raw_response=response,
            assistant_payload=assistant_message,
            stop_reason=_get(choice, "finish_reason"),
            tool_calls=_extract_tool_calls(assistant_message),
            usage={
                "input_tokens": int(_get(usage, "prompt_tokens", 0) or 0),
                "output_tokens": int(_get(usage, "completion_tokens", 0) or 0),
            },
        )

    def append_assistant(self, state: OpenAICompatState, turn: ModelTurn) -> None:
        state.messages.append(turn.assistant_payload)

    def append_tool_results(
        self,
        state: OpenAICompatState,
        tool_results: list[ToolResult],
        tool_result_formatter: Callable[[dict[str, Any]], list[dict[str, Any]]],
    ) -> None:
        tool_messages = []
        image_blocks = []
        for tool_result in tool_results:
            tool_text, tool_image_blocks = format_tool_result_for_openai(
                tool_result.result,
                tool_result_formatter,
                tool_name=tool_result.name,
            )
            tool_messages.append({
                "role": "tool",
                "tool_call_id": tool_result.call_id,
                "content": tool_text,
            })
            image_blocks.extend(tool_image_blocks)

        state.messages.extend(tool_messages)
        if image_blocks:
            state.messages.append({
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Images returned by the preceding tool calls.",
                    },
                    *image_blocks,
                ],
            })

    def messages(self, state: OpenAICompatState) -> list[dict[str, Any]]:
        return state.messages

    def is_normal_stop(self, turn: ModelTurn) -> bool:
        return turn.stop_reason in ("stop", "end_turn", None)

    def log_model_turn(
        self,
        turn: ModelTurn,
        *,
        usage_totals: dict[str, int],
    ) -> None:
        assistant_message = turn.assistant_payload
        content = assistant_message.get("content")
        if isinstance(content, str) and content.strip():
            logger.info("[openai] %s", content.strip())

        message = _get(_first_choice(turn.raw_response), "message")
        reasoning_text = _get(message, "reasoning_content")
        if isinstance(reasoning_text, str) and reasoning_text.strip():
            text = reasoning_text.strip()
            if len(text) > 500:
                text = text[:500] + "...(+%d)" % (len(text) - 500)
            logger.info("[thinking] %s", text)

        for tool_call in assistant_message.get("tool_calls") or []:
            function = tool_call.get("function") or {}
            name = function.get("name", "?")
            arguments = function.get("arguments") or "{}"
            if len(arguments) > 250:
                arguments = arguments[:250] + "...(+%d)" % (len(arguments) - 250)
            logger.info("[tool->] %s(%s)", name, arguments)

        logger.info(
            "[usage] in=%s  out=%s  stop=%s  total_in=%s  total_out=%s",
            turn.usage.get("input_tokens", 0),
            turn.usage.get("output_tokens", 0),
            turn.stop_reason,
            usage_totals.get("total_input_tokens", 0),
            usage_totals.get("total_output_tokens", 0),
        )

    def api_failure_error(self) -> str:
        return "OpenAI-compatible API call failed after retries"

    def _call_with_retries(
        self,
        *,
        tools: list[dict[str, Any]],
        messages: list[dict[str, Any]],
    ):
        last_err = None
        extra_kwargs: dict[str, Any] = {}
        if self._thinking:
            extra_kwargs["reasoning_effort"] = self._reasoning_effort
        for outer in range(3):
            try:
                response = self._client.chat.completions.create(
                    model=self._model,
                    messages=messages,
                    tools=tools,
                    tool_choice="auto",
                    max_tokens=self._max_tokens,
                    **extra_kwargs,
                )
                # Some OpenAI-compatible gateways answer 200 with an empty
                # body (no `choices`) when the upstream request fails. Treat that as
                # a transient, retryable error instead of crashing the agent loop.
                _first_choice(response)
                return response
            except Exception as e:  # noqa: BLE001 - SDK-compatible errors vary by provider.
                last_err = e
                if not _is_retryable_error(e):
                    raise
                wait = 10 * (outer + 1)
                logger.warning(
                    "API error '%s: %s' - sleeping %ds (retry %d/3)",
                    type(e).__name__, e, wait, outer + 1,
                )
                time.sleep(wait)

        logger.error("giving up after 3 retries; last error: %s", last_err)
        return None


def anthropic_tools_to_openai_tools(
    tools_spec: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Convert Anthropic-style tool definitions to OpenAI function tools."""
    tools = []
    for tool in tools_spec:
        tools.append({
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool.get("description", ""),
                "parameters": copy.deepcopy(tool.get("input_schema") or {}),
            },
        })
    return tools


def format_tool_result_for_openai(
    result: dict[str, Any],
    tool_result_formatter: Callable[[dict[str, Any]], list[dict[str, Any]]],
    *,
    tool_name: str,
) -> tuple[str, list[dict[str, Any]]]:
    """Convert Anthropic-style content blocks into OpenAI messages."""
    formatter_input = copy.deepcopy(result) if isinstance(result, dict) else result
    blocks = tool_result_formatter(formatter_input)
    text_parts: list[str] = []
    image_blocks: list[dict[str, Any]] = []

    for block in blocks:
        block_type = _get(block, "type")
        if block_type == "text":
            text = _get(block, "text", "")
            if text:
                text_parts.append(str(text))
        elif block_type == "image":
            image_url = _image_block_to_url(block)
            if image_url:
                image_blocks.extend([
                    {
                        "type": "text",
                        "text": f"Image returned by tool {tool_name}.",
                    },
                    {"type": "image_url", "image_url": {"url": image_url}},
                ])
        else:
            text_parts.append(json.dumps(_to_plain_data(block), default=str))

    text = "\n\n".join(part for part in text_parts if part)
    if not text:
        text = "{}"
    return text, image_blocks


def _extract_tool_calls(assistant_message: dict[str, Any]) -> list[ToolCall]:
    tool_calls = []
    for index, tool_call in enumerate(assistant_message.get("tool_calls") or [], start=1):
        call_id = tool_call.get("id") or f"tool_call_{index}"
        function = tool_call.get("function") or {}
        raw_arguments = function.get("arguments") or "{}"
        arguments, parse_error = _parse_tool_arguments(raw_arguments)
        tool_calls.append(ToolCall(
            id=call_id,
            name=function.get("name", ""),
            arguments=arguments,
            raw_arguments=raw_arguments,
            parse_error=parse_error,
        ))
    return tool_calls


def _image_block_to_url(block: Any) -> str | None:
    source = _get(block, "source") or {}
    source_type = _get(source, "type")
    if source_type == "base64":
        media_type = _get(source, "media_type", "image/png")
        data = _get(source, "data")
        if data:
            return f"data:{media_type};base64,{data}"
    url = _get(source, "url")
    return str(url) if url else None


def _assistant_message_to_dict(message: Any) -> dict[str, Any]:
    content = _get(message, "content")
    assistant_message: dict[str, Any] = {
        "role": "assistant",
        "content": content if content is not None else "",
    }
    tool_calls = _get(message, "tool_calls") or []
    if tool_calls:
        assistant_message["tool_calls"] = [
            _tool_call_to_dict(tool_call) for tool_call in tool_calls
        ]
    return assistant_message


def _tool_call_to_dict(tool_call: Any) -> dict[str, Any]:
    function = _get(tool_call, "function") or {}
    return {
        "id": _get(tool_call, "id"),
        "type": _get(tool_call, "type", "function"),
        "function": {
            "name": _get(function, "name"),
            "arguments": _get(function, "arguments", "{}"),
        },
    }


def _parse_tool_arguments(raw_arguments: Any) -> tuple[dict[str, Any], str | None]:
    if isinstance(raw_arguments, dict):
        return raw_arguments, None
    if raw_arguments in (None, ""):
        return {}, None
    try:
        parsed = json.loads(raw_arguments)
    except json.JSONDecodeError as e:
        return {}, f"invalid JSON tool arguments: {e}"
    if not isinstance(parsed, dict):
        error = f"tool arguments must decode to an object, got {type(parsed).__name__}"
        return {}, error
    return parsed, None


def _first_choice(response: Any) -> Any:
    choices = _get(response, "choices") or []
    if not choices:
        raise KeyError("OpenAI-compatible response did not include choices")
    return choices[0]


def _is_retryable_error(error: Exception) -> bool:
    name = type(error).__name__
    if name in {
        "APIConnectionError",
        "APITimeoutError",
        "InternalServerError",
        "RateLimitError",
        "Timeout",
        "TimeoutException",
        "KeyError",
    }:
        return True
    status_code = getattr(error, "status_code", None)
    return status_code in {408, 409, 429} or (
        isinstance(status_code, int) and status_code >= 500
    )


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _to_plain_data(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _to_plain_data(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_plain_data(v) for v in obj]
    if isinstance(obj, tuple):
        return [_to_plain_data(v) for v in obj]
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    return obj
