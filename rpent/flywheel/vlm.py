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

"""Planner-independent samples for training one VLM response at a time."""

from dataclasses import dataclass, field
from typing import Any, Literal, TypedDict

from rpent.session.records import ArtifactRef


class VLMText(TypedDict):
    """Text visible to the model, including serialized tool results and state."""

    type: Literal["text"]
    text: str


class VLMImage(TypedDict):
    """An image byte artifact; inline image data is not part of this format."""

    type: Literal["image"]
    # Path is relative to the trajectory root, or the dataset root after export.
    artifact: ArtifactRef
    observation_id: str | None


class VLMToolCall(TypedDict):
    """A normalized call ID, tool name, and decoded JSON arguments."""

    id: str
    name: str
    arguments: dict[str, Any]


class VLMToolDefinition(TypedDict):
    """Tool declaration using Toolkit's name, description, and schema fields."""

    name: str
    description: str
    input_schema: dict[str, Any]


@dataclass
class VLMMessage:
    """One message with ordered text/images and optional assistant tool calls."""

    role: Literal["system", "user", "assistant", "tool"]
    content: list[VLMText | VLMImage] = field(default_factory=list)
    tool_calls: list[VLMToolCall] = field(default_factory=list)
    tool_call_id: str | None = None


@dataclass
class VLMSample:
    """One planner turn, with history as input and only target supervised."""

    episode_id: str
    turn_id: str
    # Results of target's calls belong to later samples.
    messages: list[VLMMessage]
    tools: list[VLMToolDefinition]
    target: VLMMessage
    metadata: dict[str, Any] = field(default_factory=dict)
