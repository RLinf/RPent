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

from robots.robodojo.prompts import system as system_parts
from robots.robodojo.prompts import user as user_parts
from rpent.context.prompt_utils import PromptNode


def system_prompt() -> PromptNode:
    return {
        "ROLE AND RULES": system_parts.ROLE_AND_RULES,
        "TOOL ACCESS": system_parts.TOOL_ACCESS,
        "PERCEPTION": system_parts.PERCEPTION,
        "TOOLS AND MANIPULATION": system_parts.TOOLS,
        "SAFETY": system_parts.SAFETY,
        "REWARD AND SCORE": system_parts.REWARD,
        "OUTPUT DISCIPLINE": system_parts.OUTPUT_DISCIPLINE,
    }


def user_prompt() -> PromptNode:
    return {
        "TASK": user_parts.TASK,
        "BEGIN": user_parts.BEGIN,
    }


__all__ = ["system_prompt", "user_prompt"]
