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

"""RoboDojo prompt bundle assembly."""

from __future__ import annotations

from collections.abc import Mapping

from robots.robodojo.prompts import evaluate as evaluate_parts
from robots.robodojo.prompts import explore as explore_parts
from robots.robodojo.prompts import user as user_parts
from rpent.prompt.utils import PromptNode


def system_prompt(
    variables: Mapping[str, object] | None = None,
) -> PromptNode:
    """Select the RoboDojo evaluation or exploration system prompt."""
    if (variables or {}).get("mode") == "explore":
        return explore_parts.system_prompt()
    return evaluate_parts.system_prompt(variables)


def user_prompt(
    variables: Mapping[str, object] | None = None,
) -> PromptNode:
    sections = {
        "TASK": user_parts.TASK,
        "BEGIN": user_parts.BEGIN,
    }
    if (variables or {}).get("mode") == "eval-fair":
        return {"TASK": "{{task}} / layout {{layout}}"}
    return sections


__all__ = ["system_prompt", "user_prompt"]
