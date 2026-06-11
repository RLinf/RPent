"""Normalised API adapter interface for tool-use model backends."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol


@dataclass
class ToolCall:
    """Provider-independent tool invocation requested by a model turn."""

    id: str
    name: str
    arguments: dict[str, Any]
    raw_arguments: Any = None
    parse_error: str | None = None


@dataclass
class ToolResult:
    """Result of executing one normalised tool call."""

    call_id: str
    name: str
    result: dict[str, Any]


@dataclass
class ModelTurn:
    """Provider-independent view of one assistant turn."""

    raw_response: Any
    assistant_payload: Any
    stop_reason: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=dict)


class ApiAdapter(Protocol):
    """Provider-specific bridge used by the shared API agent loop."""

    name: str

    def start(
        self,
        *,
        system_prompt: str,
        user_message: str,
        tools_spec: list[dict[str, Any]],
    ) -> Any:
        """Create provider-specific mutable conversation state."""
        ...

    def call(self, state: Any) -> ModelTurn | None:
        """Call the provider and return a normalised assistant turn."""
        ...

    def append_assistant(self, state: Any, turn: ModelTurn) -> None:
        """Append the provider-native assistant payload to conversation state."""
        ...

    def append_tool_results(
        self,
        state: Any,
        tool_results: list[ToolResult],
        tool_result_formatter: Callable[[dict[str, Any]], list[dict[str, Any]]],
    ) -> None:
        """Append provider-native tool-result messages to conversation state."""
        ...

    def messages(self, state: Any) -> list[dict[str, Any]]:
        """Return a serialisable transcript from provider state."""
        ...

    def is_normal_stop(self, turn: ModelTurn) -> bool:
        """Return whether a no-tool assistant turn should end the loop."""
        ...

    def log_model_turn(
        self,
        turn: ModelTurn,
        *,
        usage_totals: dict[str, int],
    ) -> None:
        """Emit provider-specific logs for one assistant turn."""
        ...

    def api_failure_error(self) -> str:
        """Return the error string used when retries are exhausted."""
        ...
