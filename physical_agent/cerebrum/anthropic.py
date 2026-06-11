"""Anthropic API cerebrum — multi-turn tool-use agent loop."""

from __future__ import annotations

import copy
import json
import time
from typing import Any, Callable

import anthropic

from physical_agent.cerebrum.base import Cerebrum, CerebrumResult
from physical_agent.utils.logging import get_logger

logger = get_logger("cerebrum.anthropic")


class AnthropicCerebrum:
    """Cerebrum backed by the Anthropic Messages API with tool-use.

    Constructor takes a pre-configured ``anthropic.Anthropic`` client,
    model id, and per-turn token budget.  Call ``solve()`` to run the
    agent loop against a set of tools.
    """

    def __init__(
        self,
        client: anthropic.Anthropic,
        model: str,
        max_tokens: int = 4096,
    ):
        self._client = client
        self._model = model
        self._max_tokens = max_tokens

    # ------------------------------------------------------------------
    # Cerebrum protocol
    # ------------------------------------------------------------------

    def solve(
        self,
        *,
        system_prompt: str,
        user_message: str,
        tools_spec: list[dict[str, Any]],
        tool_handler: Callable[[str, dict[str, Any]], dict[str, Any]],
        tool_result_formatter: Callable[[dict[str, Any]], list[dict[str, Any]]],
        max_turns: int = 80,
    ) -> CerebrumResult:
        messages: list[dict] = [{"role": "user", "content": user_message}]
        finish_result = None
        total_in = total_out = 0
        total_cache_create = total_cache_read = 0
        n_tool_calls = 0
        last_error = None
        system = _cacheable_system(system_prompt)
        tools = _cacheable_tools(tools_spec)

        for turn in range(1, max_turns + 1):
            logger.info("=== turn %d/%d ===", turn, max_turns)

            response = self._call_with_retries(
                system=system,
                tools=tools,
                messages=messages,
            )
            if response is None:
                break

            u = response.usage
            total_in += u.input_tokens
            total_out += u.output_tokens
            cache_create = int(getattr(u, "cache_creation_input_tokens", 0) or 0)
            cache_read = int(getattr(u, "cache_read_input_tokens", 0) or 0)
            total_cache_create += cache_create
            total_cache_read += cache_read

            self._log_response(
                response,
                u,
                total_in,
                total_out,
                total_cache_create,
                total_cache_read,
            )

            messages.append({"role": "assistant", "content": response.content})

            if response.stop_reason == "tool_use":
                tool_results, finish_result, n = self._execute_tools(
                    response, tool_handler, tool_result_formatter,
                )
                n_tool_calls += n
                messages.append({"role": "user", "content": tool_results})
                if finish_result is not None:
                    logger.info("FINISH called: %s", finish_result)
                    break
            elif response.stop_reason == "end_turn":
                logger.info("model ended turn without a tool call. Stopping.")
                break
            else:
                logger.warning("unexpected stop_reason: %s", response.stop_reason)
                break

        return CerebrumResult(
            finish_result=finish_result,
            messages=messages,
            stats={
                "total_input_tokens": total_in,
                "total_output_tokens": total_out,
                "total_cache_creation_input_tokens": total_cache_create,
                "total_cache_read_input_tokens": total_cache_read,
                "turns_used": turn,
                "tool_calls": n_tool_calls,
            },
            error=last_error,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _call_with_retries(
        self,
        *,
        system: str | list[dict[str, Any]],
        tools: list[dict],
        messages: list[dict],
    ):
        last_err = None
        for outer in range(3):
            try:
                return self._client.messages.create(
                    model=self._model,
                    max_tokens=self._max_tokens,
                    system=system,
                    tools=tools,
                    messages=messages,
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
                    "API error '%s: %s' — sleeping %ds (retry %d/3)",
                    type(e).__name__, e, wait, outer + 1,
                )
                time.sleep(wait)
        logger.error("giving up after 3 retries; last error: %s", last_err)
        return None

    @staticmethod
    def _log_response(
        response,
        usage,
        total_in,
        total_out,
        total_cache_create,
        total_cache_read,
    ):
        for block in response.content:
            if block.type == "text" and block.text.strip():
                logger.info("[claude] %s", block.text.strip())
            elif block.type == "tool_use":
                s = json.dumps(block.input, default=str)
                if len(s) > 250:
                    s = s[:250] + "...(+%d)" % (len(s) - 250)
                logger.info("[tool->] %s(%s)", block.name, s)
        logger.info(
            "[usage] in=%s cache_create=%s cache_read=%s out=%s stop=%s "
            "total_in=%s total_cache_create=%s total_cache_read=%s total_out=%s",
            usage.input_tokens,
            int(getattr(usage, "cache_creation_input_tokens", 0) or 0),
            int(getattr(usage, "cache_read_input_tokens", 0) or 0),
            usage.output_tokens,
            response.stop_reason,
            total_in,
            total_cache_create,
            total_cache_read,
            total_out,
        )

    @staticmethod
    def _execute_tools(
        response,
        tool_handler: Callable,
        tool_result_formatter: Callable,
    ):
        tool_results = []
        finish_result = None
        n = 0
        for block in response.content:
            if block.type != "tool_use":
                continue
            n += 1
            result = tool_handler(block.name, block.input)
            if isinstance(result, dict) and result.get("_finish"):
                finish_result = result
            summary = _summarise_result(result)
            s = json.dumps(summary, default=str)
            if len(s) > 350:
                s = s[:350] + "...(+%d)" % (len(s) - 350)
            logger.info("[tool<-] %s: %s", block.name, s)
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": tool_result_formatter(result),
            })
        return tool_results, finish_result, n


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


def _summarise_result(result: dict) -> dict:
    """Strip large fields from a tool result for console display."""
    summary = {
        k: v
        for k, v in result.items()
        if k not in ("state", "content", "log", "_image_path")
    }
    if isinstance(result, dict):
        if "state" in result:
            s = result["state"]
            summary["state_summary"] = {
                "eef": [round(x, 3) for x in s.get("robot0_eef_pos", [])][:3],
                "libero_terminated": result.get("libero_terminated"),
            }
        if "log" in result:
            lg = result["log"]
            if isinstance(lg, dict) and isinstance(lg.get("result"), dict):
                summary["log_result_keys"] = list(lg["result"].keys())
    return summary