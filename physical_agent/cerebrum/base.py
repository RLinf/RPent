"""Shared protocol for high-level reasoning backends."""
from __future__ import annotations

from typing import Any, Callable, Protocol


class CerebrumResult:
    """Result returned by a cerebrum invocation."""

    __slots__ = ("finish_result", "messages", "stats", "error")

    def __init__(
        self,
        *,
        finish_result: dict | None = None,
        messages: list[dict] | None = None,
        stats: dict | None = None,
        error: str | None = None,
    ):
        """Initialize a serializable cerebrum result."""
        self.finish_result = finish_result  # {"status": "success"/"failure"/"stuck", "summary": "..."}
        self.messages = messages or []       # serialisable conversation transcript
        self.stats = stats or {}             # {"total_input_tokens", "total_output_tokens", "turns_used", "tool_calls"}
        self.error = error                   # str | None  — set when the cerebrum raises


class Cerebrum(Protocol):
    """A cerebrum solves a task by conversing with an LLM/VLM backend.

    It is given one system prompt, one initial user message, and a set of
    tool definitions.  It returns a ``CerebrumResult`` after the task is
    finished or the turn budget is exhausted.
    """

    def solve(
        self,
        *,
        system_prompt: str,
        user_message: str,
        tools_spec: list[dict[str, Any]],
        tool_handler: Callable[[str, dict[str, Any]], dict[str, Any]],
        tool_result_formatter: Callable[[dict[str, Any]], list[dict[str, Any]]],
        max_turns: int,
    ) -> CerebrumResult:
        """Run the multi-turn agent loop until completion or budget.

        Args:
            system_prompt: System-level instructions (role, rules, workflow).
            user_message: Initial user message (task description, first steps).
            tools_spec: Anthropic-style tool definitions list.
            tool_handler: ``(name, input_dict) -> result_dict``.
            tool_result_formatter: ``result_dict -> list[content_block]``.
            max_turns: Maximum LLM turns before giving up.

        Returns:
            ``CerebrumResult`` with finish status, conversation transcript,
            token-usage stats, and optional error string.
        """
        ...
