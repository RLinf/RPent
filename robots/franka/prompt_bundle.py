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

"""Prompt bundle assembly for the single-Franka environment."""

from __future__ import annotations

from collections.abc import Mapping

from robots.franka.prompts import system as system_parts
from robots.franka.prompts import user as user_parts
from rpent.prompt.utils import Numbered, PromptNode


def system_prompt(variables: Mapping[str, object] | None = None) -> PromptNode:
    """Assemble the Franka system prompt."""
    return {
        "ROLE": system_parts.ROLE,
        "RUNTIME": system_parts.RUNTIME,
        "SAFETY RULES": Numbered(system_parts.RULES),
        "WORKFLOW": Numbered(system_parts.WORKFLOW),
    }


def user_prompt(variables: Mapping[str, object] | None = None) -> PromptNode:
    """Assemble the task-specific initial user prompt."""
    return {
        "TASK": user_parts.TASK,
        "TASK CONSTRAINTS": user_parts.CONSTRAINTS,
        "BEGIN": user_parts.BEGIN,
    }
