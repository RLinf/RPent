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
"""Robot-independent contracts for a finite-action decision backend."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from rpent.tools.toolkit import ToolResult


@dataclass(frozen=True)
class ActionCandidate:
    """One action whose arguments are computed before the model is called."""

    id: str
    tool_name: str
    arguments: dict[str, Any]
    description: str

    def as_dict(self) -> dict[str, Any]:
        """Return the executable action and its model-visible description."""
        return {
            "id": self.id,
            "tool_name": self.tool_name,
            "arguments": self.arguments,
            "description": self.description,
        }


@dataclass(frozen=True)
class DecisionContext:
    """Current public observations, admissible actions, or a terminal outcome."""

    state: dict[str, Any]
    candidates: tuple[ActionCandidate, ...] = ()
    media: list[dict[str, str]] = field(default_factory=list)
    instructions: str = "Choose the feasible action that best advances the task."
    outcome: dict[str, str] | None = None


class DecisionAdapter(Protocol):
    """Robot-owned observation and candidate generation for one episode."""

    def prepare(self) -> DecisionContext:
        """Read the latest observation and compute fully specified actions."""
        ...

    def observe(self, action: ActionCandidate, result: ToolResult) -> None:
        """Update episode progress from the executed action and new observation."""
        ...
