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
    kind: Literal["integer"]
    minimum: int
    choices: tuple[str, ...]
    suggestions: tuple[str, ...]


class TaskSpec(TypedDict):
    command: str
    usage: str
    fields: tuple[TaskFieldSpec, ...]
    display: str
    output_slug: str


class RuntimeComponentSpecRequired(TypedDict):
    name: str
    label: str
    scope: Literal["shared", "unique"]


class RuntimeComponentSpec(RuntimeComponentSpecRequired, total=False):
    planners: tuple[str, ...]


class DashboardSpec(TypedDict):
    task: TaskSpec
    runtime_components: tuple[RuntimeComponentSpec, ...]
    primitives: tuple[str, ...]
