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
    """Update one robot-side runtime component."""

    component: str
    status: str
    error: BaseException | str | None = None


@dataclass(frozen=True, slots=True)
class ToolResultEvent:
    """Publish one raw tool result for Dashboard projection."""

    name: str
    result: Any


@dataclass(frozen=True, slots=True)
class StepRecordEvent:
    """Publish one recorded robot step and its artifact context."""

    record: Any
    env_state: Any
    frame_artifacts: dict[str, str]


@dataclass(frozen=True, slots=True)
class RunStartedEvent:
    """Mark startup complete and the agent run active."""


DashboardEvent: TypeAlias = (
    TranscriptEvent
    | UsageEvent
    | RuntimeStatusEvent
    | ToolResultEvent
    | StepRecordEvent
    | RunStartedEvent
)


class DashboardEventSink(Protocol):
    """Consumer used by planners, toolkits, and robot runtimes."""

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
