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

"""LIBERO prompt bundle assembly."""

from __future__ import annotations

from collections.abc import Mapping

from robots.libero.prompts import evaluate as evaluate_parts
from robots.libero.prompts import explore as explore_parts
from robots.libero.prompts import user as user_parts
from rpent.prompt.utils import PromptNode


def system_prompt(
    variables: Mapping[str, object] | None = None,
) -> PromptNode:
    """Assemble the LIBERO system prompt for the selected run mode."""
    if (variables or {}).get("mode", "eval") == "explore":
        return explore_parts.system_prompt()
    return evaluate_parts.system_prompt(variables)


def user_prompt(variables: Mapping[str, object] | None = None) -> PromptNode:
    """Assemble the LIBERO user prompt tree."""
    sections = {
        "CELL": user_parts.CELL,
        "MODE": user_parts.MODE,
        "BEGIN": user_parts.BEGIN,
    }
    if (variables or {}).get("replay_config"):
        sections["TRAJECTORY TOOLS"] = (
            "You can call trajectory_catalog to select a demonstrated subtask, then trajectory_replay(segment, demo). "
            "The replay tool internally uses simulator object poses; using this tool is permitted. "
            "Inspect the current camera observation before deciding to replay, and inspect the resulting images and "
            "alignment residuals after each segment. You may choose replay or ordinary motor/VLA tools at each stage. "
            "If alignment is blocked, inspect the scene and recover within the same episode; do not reset. "
            "Only environment termination establishes task success."
        )
    return sections


__all__ = ["system_prompt", "user_prompt"]
