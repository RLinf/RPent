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

"""RoboTwin prompt bundle assembly."""

from __future__ import annotations

from collections.abc import Mapping

from robots.robotwin.prompts import system as system_parts
from robots.robotwin.prompts import user as user_parts
from rpent.context.prompt_utils import Numbered, PromptNode


def system_prompt(
    variables: Mapping[str, object] | None = None,
) -> PromptNode:
    return {
        "PREAMBLE": system_parts.PREAMBLE,
        "GOAL": system_parts.GOAL,
        "RULES": system_parts.RULES,
        "AUTHORITY": system_parts.AUTHORITY,
        "HISTORICAL_CONTEXT": system_parts.HISTORICAL_CONTEXT,
        "PERCEPTION": system_parts.PERCEPTION,
        "CAMERA_ROLES": system_parts.CAMERA_ROLES,
        "EMBODIMENT": system_parts.EMBODIMENT,
        "PRIMITIVES": system_parts.PRIMITIVES,
        "VLA_RULES": system_parts.VLA_RULES,
        "GRIPPER_RULES": system_parts.GRIPPER_RULES,
        "PLANNER_RULES": system_parts.PLANNER_RULES,
        "BIMANUAL_RULES": system_parts.BIMANUAL_RULES,
        "RECOVERY": system_parts.RECOVERY,
        "BUDGET": system_parts.BUDGET,
        "WORKFLOW": Numbered(system_parts.WORKFLOW),
        "SUCCESS": system_parts.SUCCESS,
        "ACTION_COMMITMENT": system_parts.ACTION_COMMITMENT,
        "USER_MODE": system_parts.USER_MODE,
    }


def user_prompt(
    variables: Mapping[str, object] | None = None,
) -> PromptNode:
    return {
        "CELL": user_parts.CELL,
        "BEGIN": user_parts.BEGIN,
    }


__all__ = ["system_prompt", "user_prompt"]
