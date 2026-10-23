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

"""Data contracts for tasks, planner calls, observations, actions, and predictions.

``missing`` keys are record field names or JSON Pointers into payload data.
Timestamps are timezone-aware. These containers do not copy nested data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal, TypeAlias

import numpy as np


@dataclass
class MissingInfo:
    """Reason for an absent field."""

    reason: Literal[
        "pending", "not_exposed", "not_recorded", "capture_failed", "not_applicable"
    ]
    detail: str


@dataclass
class ArtifactRef:
    """Array or byte attachment relative to the trajectory root."""

    path: str
    kind: Literal["ndarray", "bytes"]
    dtype: str | None = None
    shape: tuple[int, ...] | None = None
    media_type: str | None = None


@dataclass
class DataRef:
    """Reference immutable record content by a root-relative JSON Pointer."""

    record_type: Literal["observation", "tool_call"]
    record_id: str
    path: str


DataValue: TypeAlias = (
    bool
    | int
    | float
    | str
    | None
    | list["DataValue"]
    | dict[str, "DataValue"]
    | np.ndarray
    | np.generic
    | bytes
    | ArtifactRef
    | DataRef
)


@dataclass
class Payload:
    """Native content; ``missing`` distinguishes absent data from captured nulls."""

    format: str
    data: DataValue
    format_version: str | None = None  # Source format or SDK version.
    # Completeness is relative to this format's capture boundary.
    completeness: Literal["complete", "partial", "unknown"] = "unknown"
    missing: dict[str, MissingInfo] = field(default_factory=dict)
    # JSON Pointers into data; this storage hint is omitted from saved records.
    inline_array_paths: tuple[str, ...] = ()


@dataclass
class TaskRecord:
    """One task's metadata and execution outcome."""

    task_id: str
    instruction: str
    config: Payload
    status: Literal["running", "completed", "failed", "cancelled"]
    started_at: datetime
    ended_at: datetime | None = None
    stop_reason: str | None = None
    result: Payload | None = None


@dataclass
class PlannerRecord:
    """One model request/response and all tool calls issued by that response."""

    planner_id: str
    backend: str
    status: Literal["open", "completed", "failed", "cancelled"]  # Model call only.
    request: Payload | None = None  # Actual context, including messages and tools.
    response: Payload | None = None
    tool_calls: list[ToolCallRecord] = field(default_factory=list)  # Response order.
    source_ids: dict[str, str] = field(default_factory=dict)  # Namespaced backend IDs.
    started_at: datetime | None = None
    ended_at: datetime | None = None
    missing: dict[str, MissingInfo] = field(default_factory=dict)


@dataclass
class ToolCallRecord:
    """Tool invocation and execution with call-level observation boundaries."""

    tool_call_id: str
    planner_id: str | None
    origin: Literal["model", "program", "unknown"]
    name: str
    arguments: Payload
    status: Literal[
        "requested",
        "running",
        "completed",
        "failed",
        "cancelled",
        "not_executed",
        "unknown",
    ]
    source_ids: dict[str, str] = field(default_factory=dict)
    effective_instruction: str | None = None
    model_config: Payload | None = None
    started_at: datetime | None = None
    ended_at: datetime | None = None
    raw_result: Payload | None = None
    feedback: Payload | None = None  # Actual adapter return to the planner or SDK.
    before_observation_id: str | None = None
    after_observation_id: str | None = None
    missing: dict[str, MissingInfo] = field(default_factory=dict)


@dataclass
class ObservationRecord:
    """One collection of native environment data, shared by observation ID."""

    observation_id: str
    payload: Payload
    captured_at: datetime | None = None
    missing: dict[str, MissingInfo] = field(default_factory=dict)


@dataclass
class ActionRecord:
    """One confirmed control action with its final native value and result.

    ``action.format`` identifies the native protocol, interpreted with task config.
    """

    action_id: str
    tool_call_id: str
    source: Literal["vla", "tool_computed", "direct"]
    action: Payload
    before_observation_id: str | None
    after_observation_id: str | None
    executed_at: datetime | None = None
    result: Payload | None = None
    missing: dict[str, MissingInfo] = field(default_factory=dict)


@dataclass
class PredictionRecord:
    """Confirmed executed actions from one VLA prediction."""

    prediction_id: str
    planner_id: str | None
    tool_call_id: str
    input_observation_id: str | None
    instruction: str
    actions: Payload
    action_ids: list[str]  # Same order as actions.data.
    missing: dict[str, MissingInfo] = field(default_factory=dict)
