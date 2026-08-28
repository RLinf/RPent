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

"""Lightweight types for robot-owned Dashboard configuration."""

from __future__ import annotations

from typing import Literal, TypedDict


class TaskFieldSpecRequired(TypedDict):
    name: str


class TaskFieldSpec(TaskFieldSpecRequired, total=False):
    kind: Literal["integer", "string"]
    minimum: int
    choices: tuple[str, ...]
    suggestions: tuple[str, ...]


class TaskSpec(TypedDict):
    command: str
    usage: str
    fields: tuple[TaskFieldSpec, ...]
    display: str
    output_slug: str


class LauncherFieldSpecRequired(TypedDict):
    name: str
    label: str
    kind: Literal["integer", "string"]


class LauncherFieldSpec(LauncherFieldSpecRequired, total=False):
    label_zh_cn: str
    minimum: int
    required: bool
    placeholder: str
    placeholder_zh_cn: str


class RuntimeComponentSpec(TypedDict):
    name: str
    label: str
    scope: Literal["shared", "unique"]


class FrameChannelSpec(TypedDict):
    name: str
    label: str
    artifact: str


class DashboardSpec(TypedDict):
    task: TaskSpec
    launcher_fields: tuple[LauncherFieldSpec, ...]
    runtime_components: tuple[RuntimeComponentSpec, ...]
    frame_channels: tuple[FrameChannelSpec, ...]
    primitives: tuple[str, ...]
