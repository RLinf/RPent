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


class RuntimeComponentSpecRequired(TypedDict):
    name: str
    label: str


class RuntimeComponentSpec(RuntimeComponentSpecRequired, total=False):
    scope: Literal["shared", "task"]


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
