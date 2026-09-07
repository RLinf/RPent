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

from collections.abc import Mapping

from robots.behavior.prompts import eval as eval_parts
from robots.behavior.prompts import explore as explore_parts
from robots.behavior.prompts import user as user_parts
from rpent.prompt.utils import PromptNode


def system_prompt(variables: Mapping[str, object] | None = None) -> PromptNode:
    """Assemble the BEHAVIOR system prompt for the selected run mode."""
    mode = (variables or {}).get("behavior_mode", "eval")
    if mode == "eval":
        return eval_parts.system_prompt()
    if mode == "explore":
        return explore_parts.system_prompt()
    raise ValueError(f"unsupported BEHAVIOR prompt mode: {mode!r}")


def user_prompt(variables: Mapping[str, object] | None = None) -> PromptNode:
    """Assemble the BEHAVIOR user prompt tree."""
    return {
        "CELL": user_parts.CELL,
        "MODE": user_parts.MODE,
        "BEGIN": user_parts.BEGIN,
    }


__all__ = ["system_prompt", "user_prompt"]
