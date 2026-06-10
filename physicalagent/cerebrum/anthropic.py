"""Anthropic API cerebrum — multi-turn tool-use agent loop."""

from __future__ import annotations

import json
import time
from typing import Any, Callable

import anthropic

from physicalagent.cerebrum.base import Cerebrum, CerebrumResult


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
        verbose: bool = True,
    ) -> CerebrumResult:
        messages: list[dict] = [{"role": "user", "content": user_message}]
        finish_result = None
        total_in = total_out = 0
        n_tool_calls = 0
        last_error = None

        for turn in range(1, max_turns + 1):
            if verbose:
                print(f"\n[agent] === turn {turn}/{max_turns} ===")

            response = self._call_with_retries(
                system=system_prompt,
                tools=tools_spec,
                messages=messages,
                verbose=verbose,
            )
            if response is None:
                break

            u = response.usage
            total_in += u.input_tokens
            total_out += u.output_tokens

            if verbose:
                self._log_response(response, u, total_in, total_out)

            messages.append({"role": "assistant", "content": response.content})

            if response.stop_reason == "tool_use":
                tool_results, finish_result, n = self._execute_tools(
                    response, tool_handler, tool_result_formatter, verbose,
                )
                n_tool_calls += n
                messages.append({"role": "user", "content": tool_results})
                if finish_result is not None:
                    if verbose:
                        print(f"\n[agent] FINISH called: {finish_result}")
                    break
            elif response.stop_reason == "end_turn":
                if verbose:
                    print("[agent] model ended turn without a tool call. Stopping.")
                break
            else:
                if verbose:
                    print(f"[agent] unexpected stop_reason: {response.stop_reason}")
                break

        return CerebrumResult(
            finish_result=finish_result,
            messages=messages,
            stats={
                "total_input_tokens": total_in,
                "total_output_tokens": total_out,
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
        system: str,
        tools: list[dict],
        messages: list[dict],
        verbose: bool,
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
                if verbose:
                    print(
                        f"[agent] API error '{type(e).__name__}: {e}' "
                        f"— sleeping {wait}s (retry {outer + 1}/3)"
                    )
                time.sleep(wait)
        if verbose:
            print(f"[agent] giving up after 3 retries; last error: {last_err}")
        return None

    @staticmethod
    def _log_response(response, usage, total_in, total_out):
        for block in response.content:
            if block.type == "text" and block.text.strip():
                print(f"[claude] {block.text.strip()}")
            elif block.type == "tool_use":
                s = json.dumps(block.input, default=str)
                if len(s) > 250:
                    s = s[:250] + "...(+%d)" % (len(s) - 250)
                print(f"[tool→] {block.name}({s})")
        print(
            f"[usage] in={usage.input_tokens}  out={usage.output_tokens}  "
            f"stop={response.stop_reason}  total_in={total_in}  total_out={total_out}"
        )

    @staticmethod
    def _execute_tools(
        response,
        tool_handler: Callable,
        tool_result_formatter: Callable,
        verbose: bool,
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
            if verbose:
                summary = _summarise_result(result)
                s = json.dumps(summary, default=str)
                if len(s) > 350:
                    s = s[:350] + "...(+%d)" % (len(s) - 350)
                print(f"[tool←] {block.name}: {s}")
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": tool_result_formatter(result),
            })
        return tool_results, finish_result, n


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
            if isinstance(lg, dict) and "result" in lg:
                summary["log_result_keys"] = list(lg["result"].keys())
    return summary