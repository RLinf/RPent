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

"""Select RoboTwin's local additions to the shared planner workflow."""

from collections.abc import Mapping

from robots.robotwin.prompts import action_models, system, user
from robots.robotwin.prompts import explore as exploration
from rpent.prompt import evaluate, explore
from rpent.prompt.workflow import user_prompt as build_user_prompt


def _backend(variables: Mapping[str, object]) -> str:
    default = "wam" if variables.get("policy_kind") == "wam" else "lingbot"
    return str(variables.get("policy_backend") or default)


def system_prompt(variables: Mapping[str, object] | None = None) -> dict[str, str]:
    variables = variables or {}
    backend = _backend(variables)
    guidance = action_models.GUIDANCE.get(backend, "")
    if backend not in {"lingbot", "uniwam"}:
        # Other WAM adapters use local memory and the compact evaluation protocol.
        return evaluate.system_prompt(
            role=system.ROLE,
            runtime=system.EVAL_ACCESS,
            success=system.SUCCESS,
            action_model_guidance=guidance,
        )

    exploring = variables.get("mode") == "explore"
    workflow = explore if exploring else evaluate
    read_order = exploration.READ_ORDER if exploring else system.READ_ORDER
    return workflow.system_prompt(
        role=system.ROLE,
        runtime="" if exploring else system.EVAL_ACCESS,
        initialization=read_order,
        perception=system.PERCEPTION,
        action=system.CONTROL,
        recovery=exploration.RECOVERY if exploring else system.BUDGET,
        success=exploration.SUCCESS if exploring else system.SUCCESS,
        action_model_guidance=guidance,
    )


def user_prompt(variables: Mapping[str, object] | None = None) -> dict[str, str]:
    backend = _backend(variables or {})
    if backend not in {"lingbot", "uniwam"}:
        return build_user_prompt(user.WAM_TASK, user.WAM_BEGIN)
    return build_user_prompt(user.TASK, user.BEGIN)
