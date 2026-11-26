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

from robots.robodojo.prompts import code_policy as code_parts
from robots.robodojo.prompts import evaluate as evaluate_parts
from robots.robodojo.prompts import explore as explore_parts
from robots.robodojo.prompts import user as user_parts
from robots.robodojo.prompts.policy import POLICY_PROMPTS
from rpent.prompt.utils import PromptNode


def system_prompt(
    variables: Mapping[str, object] | None = None,
) -> PromptNode:
    """Select the RoboDojo system prompt for the selected run mode."""
    variables = variables or {}

    if variables.get("mode") == "explore":
        return explore_parts.system_prompt(variables)

    return evaluate_parts.system_prompt(variables)


def code_system_prompt(variables: Mapping[str, object] | None = None) -> PromptNode:
    """Return operating guidance for Python cells."""
    return code_parts.system_prompt(backend=(variables or {}).get("vla_backend"))


def user_prompt(
    variables: Mapping[str, object] | None = None,
) -> PromptNode:
    variables = variables or {}
    if variables.get("policy_mode") == "code":
        return {"TASK": "{{task}} / layout {{layout}} / seed group {{seed}}"}

    backend = variables.get("vla_backend") or "vpp2"
    sections = {
        "TASK": user_parts.TASK.replace(
            "{{policy_control}}",
            POLICY_PROMPTS[backend]["policy_control"],
        ),
        "BEGIN": user_parts.BEGIN,
    }

    if variables.get("mode") == "eval-fair":
        return {"TASK": "{{task}} / layout {{layout}}"}

    return sections


__all__ = ["code_system_prompt", "system_prompt", "user_prompt"]
