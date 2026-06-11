"""Shared provider-independent tool-use agent loop for API cerebrums."""
from __future__ import annotations

import json
from typing import Any, Callable

from physical_agent.cerebrum.adapters.base import ApiAdapter, ToolResult
from physical_agent.cerebrum.base import CerebrumResult
from physical_agent.utils.logging import get_logger

logger = get_logger("api_loop")


class ApiAgentLoop:
    """Cerebrum implementation that delegates API details to an adapter."""

    def __init__(self, adapter: ApiAdapter):
        self._adapter = adapter

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
        """Run the shared tool-calling loop until finish, stop, or budget."""
        state = self._adapter.start(
            system_prompt=system_prompt,
            user_message=user_message,
            tools_spec=tools_spec,
        )

        finish_result = None
        usage_totals: dict[str, int] = {}
        n_tool_calls = 0
        last_error = None
        turn = 0

        for turn in range(1, max_turns + 1):
            logger.info("=== turn %d/%d ===", turn, max_turns)

            model_turn = self._adapter.call(state)
            if model_turn is None:
                last_error = self._adapter.api_failure_error()
                break

            _accumulate_usage(usage_totals, model_turn.usage)
            self._adapter.log_model_turn(model_turn, usage_totals=usage_totals)
            self._adapter.append_assistant(state, model_turn)

            if model_turn.tool_calls:
                tool_results = []
                for tool_call in model_turn.tool_calls:
                    n_tool_calls += 1
                    if tool_call.parse_error is not None:
                        result = {
                            "error": tool_call.parse_error,
                            "raw_arguments": tool_call.raw_arguments,
                        }
                    else:
                        result = tool_handler(tool_call.name, tool_call.arguments)

                    if isinstance(result, dict) and result.get("_finish"):
                        finish_result = result

                    _log_tool_result(tool_call.name, result)
                    tool_results.append(ToolResult(
                        call_id=tool_call.id,
                        name=tool_call.name,
                        result=result,
                    ))

                self._adapter.append_tool_results(
                    state,
                    tool_results,
                    tool_result_formatter,
                )
                if finish_result is not None:
                    logger.info("FINISH called: %s", finish_result)
                    break
            elif self._adapter.is_normal_stop(model_turn):
                logger.info("model ended turn without a tool call. Stopping.")
                break
            else:
                logger.warning("unexpected stop_reason: %s", model_turn.stop_reason)
                break

        stats = {
            **usage_totals,
            "turns_used": turn,
            "tool_calls": n_tool_calls,
        }
        return CerebrumResult(
            finish_result=finish_result,
            messages=self._adapter.messages(state),
            stats=stats,
            error=last_error,
        )


def _accumulate_usage(totals: dict[str, int], usage: dict[str, int]) -> None:
    for key, value in usage.items():
        total_key = f"total_{key}"
        totals[total_key] = totals.get(total_key, 0) + int(value or 0)


def _log_tool_result(name: str, result: Any) -> None:
    summary = summarise_tool_result(result)
    s = json.dumps(summary, default=str)
    if len(s) > 350:
        s = s[:350] + "...(+%d)" % (len(s) - 350)
    logger.info("[tool<-] %s: %s", name, s)


def summarise_tool_result(result: Any) -> dict[str, Any]:
    """Strip large fields from a tool result for console display."""
    if not isinstance(result, dict):
        return {"result": str(result)}
    summary = {
        k: v
        for k, v in result.items()
        if k not in ("state", "content", "log", "_image_path", "_image_cam_path")
    }
    if "state" in result:
        state = result["state"]
        summary["state_summary"] = {
            "eef": [round(x, 3) for x in state.get("robot0_eef_pos", [])][:3],
            "libero_terminated": result.get("libero_terminated"),
        }
    if "log" in result:
        log = result["log"]
        if isinstance(log, dict) and isinstance(log.get("result"), dict):
            summary["log_result_keys"] = list(log["result"].keys())
    return summary
