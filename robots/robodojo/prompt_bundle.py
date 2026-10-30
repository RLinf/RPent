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

from robots.robodojo.prompts import system as system_parts
from robots.robodojo.prompts import user as user_parts
from rpent.prompt.utils import PromptNode


def system_prompt(
    variables: Mapping[str, object] | None = None,
) -> PromptNode:
    sections = {
        "ROLE AND RULES": system_parts.ROLE_AND_RULES,
        "TOOL ACCESS": system_parts.TOOL_ACCESS,
        "PERCEPTION": system_parts.PERCEPTION,
        "AVAILABLE OPERATIONS": system_parts.TOOLS
        + "\n\n"
        + (
            system_parts.AWOMO05_POLICY
            if (variables or {}).get("vla_backend") == "awomo05"
            else system_parts.VPP2_POLICY
        ),
        "OBSERVE AND ACT": system_parts.ACTION_LOOP,
        "SAFETY": system_parts.SAFETY,
        "REWARD AND SCORE": system_parts.REWARD,
        "OUTPUT DISCIPLINE": system_parts.OUTPUT_DISCIPLINE,
    }
    if (variables or {}).get("mode") == "eval-fair":
        sections["REWARD AND SCORE"] = (
            "Task success feedback is unavailable to this client."
        )
        sections["OUTPUT DISCIPLINE"] = (
            "Call finish with your observed outcome; do not claim a verified task verdict."
        )
    return sections


def user_prompt(
    variables: Mapping[str, object] | None = None,
) -> PromptNode:
    sections = {
        "TASK": user_parts.TASK.replace(
            "{{policy_control}}",
            "Awomo05 native joint targets"
            if (variables or {}).get("vla_backend") == "awomo05"
            else "VPP2 native EE16 targets",
        ),
        "BEGIN": user_parts.BEGIN,
    }
    if (variables or {}).get("mode") == "eval-fair":
        return {"TASK": "{{task}} / layout {{layout}}"}
    return sections


__all__ = ["system_prompt", "user_prompt"]
