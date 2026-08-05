"""Lightweight event boundary for optional Dashboard updates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, TypeAlias


@dataclass(frozen=True, slots=True)
class TranscriptEvent:
    """Append one existing frontend transcript payload."""

    payload: dict[str, Any]


@dataclass(frozen=True, slots=True)
class UsageEvent:
    """Replace the cumulative planner usage counters."""

    inp: int
    out: int
    tool_calls: int


@dataclass(frozen=True, slots=True)
class RuntimeStatusEvent:
    """Update one environment-side runtime component."""

    component: str
    status: str
    error: BaseException | str | None = None


@dataclass(frozen=True, slots=True)
class ToolResultEvent:
    """Publish one raw tool result for Dashboard projection."""

    name: str
    result: Any


@dataclass(frozen=True, slots=True)
class RunStartedEvent:
    """Mark startup complete and the agent run active."""


DashboardEvent: TypeAlias = (
    TranscriptEvent
    | UsageEvent
    | RuntimeStatusEvent
    | ToolResultEvent
    | RunStartedEvent
)


class DashboardEventSink(Protocol):
    """Consumer used by planners, toolkits, and environment runtimes."""

    @property
    def enabled(self) -> bool:
        """Whether Dashboard-only projections and artifacts are needed."""
        ...

    def emit(self, event: DashboardEvent) -> None:
        """Consume one Dashboard event."""
        ...


@dataclass(frozen=True, slots=True)
class NullDashboardEventSink:
    """No-op sink used when the Dashboard is disabled."""

    @property
    def enabled(self) -> bool:
        return False

    def emit(self, event: DashboardEvent) -> None:
        return None
