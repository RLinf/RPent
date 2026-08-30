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

"""BEHAVIOR prompt bundle assembly."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from robots.behavior.prompts import system as system_parts
from robots.behavior.prompts import user as user_parts
from rpent.prompt.utils import BulletList, PromptNode


@dataclass(frozen=True)
class _RuntimeText:
    """Opaque runtime text that must not be interpreted as a prompt template."""

    value: str

    def __str__(self) -> str:
        return self.value


def _value(
    variables: Mapping[str, object], *names: str, default: object = ""
) -> object:
    for name in names:
        value = variables.get(name)
        if value not in (None, ""):
            return value
    return default


def _text(value: object, *, default: str = "") -> str:
    if value is None:
        return default
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or default
    if isinstance(value, (Mapping, Sequence)) and not isinstance(
        value,
        (str, bytes, bytearray),
    ):
        return json.dumps(value, indent=2, sort_keys=True, default=str)
    return str(value)


def _optional_section(value: object, *, empty: str) -> _RuntimeText:
    rendered = _text(value)
    return _RuntimeText(rendered if rendered else empty)


def _cell_items(variables: Mapping[str, object]) -> BulletList:
    items: list[str] = []
    for label, names in (
        ("mode", ("behavior_mode", "behavior_phase", "mode")),
        ("task", ("task_name", "task")),
        ("task language", ("task_language",)),
        ("public seed", ("public_seed", "seed")),
        ("tag", ("recipe_tag",)),
        ("output root", ("output_dir",)),
        ("job", ("job_id",)),
        ("attempt", ("attempt_index",)),
        ("max episode steps", ("max_session_steps", "max_episode_steps")),
        ("tool budget", ("global_tool_budget", "tool_budget")),
        ("wall-clock seconds", ("wall_clock_seconds", "timeout_s")),
    ):
        value = _value(variables, *names)
        if value not in (None, ""):
            items.append(f"{label}: `{_text(value)}`")
    if not items:
        items.append("runtime metadata: supplied by RunConfig at execution time")
    return BulletList(items)


def system_prompt(variables: Mapping[str, object] | None = None) -> PromptNode:
    """Assemble the BEHAVIOR system prompt from runtime-provided variables."""
    vars_ = variables or {}
    return {
        "ROLE": system_parts.ROLE,
        "CURRENT INVOCATION": _cell_items(vars_),
        "INVOCATION MODEL": system_parts.INVOCATION_MODEL,
        "RUNTIME INJECTION": system_parts.RUNTIME_INJECTION,
        "TASK INSTRUCTION": _optional_section(
            _value(vars_, "task_instruction", "behavior_task_instruction"),
            empty="The runtime did not provide a task instruction in prompt variables.",
        ),
        "PUBLIC CAPABILITIES": _optional_section(
            _value(vars_, "capabilities", "public_capabilities", "tool_surface"),
            empty="Use the public tool schemas exposed by the active toolkit.",
        ),
        "EPISODE MEMORY": _optional_section(
            _value(vars_, "episode_memory", "memory", "prior_attempt_summaries"),
            empty="When enabled, episode memory is attached to the first public tool receipt.",
        ),
        "EVIDENCE": system_parts.EVIDENCE,
        "PLANNER TOOLS": system_parts.PLANNER_TOOLS,
        "TERMINATION": system_parts.TERMINATION,
        "OUTPUT DISCIPLINE": system_parts.OUTPUT_DISCIPLINE,
    }


def user_prompt(variables: Mapping[str, object] | None = None) -> PromptNode:
    """Assemble the BEHAVIOR user prompt tree."""
    vars_ = variables or {}
    instruction = _value(vars_, "user_instruction", "behavior_user_instructions")
    return {
        "CELL": _cell_items(vars_),
        "BEGIN": _RuntimeText(_text(instruction, default=user_parts.BEGIN)),
    }


__all__ = ["system_prompt", "user_prompt"]
