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

"""Final Markdown prompts for the dual_franka environment."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from rpent.prompt.compose import COMMON, MODES, compose

PROMPTS = Path(__file__).parent / "prompts"


def system_prompt(variables: Mapping[str, object] | None = None) -> str:
    """Compose physical control rules and the current execution mode."""
    values = dict(variables or {})
    explore = values.get("mode") == "explore"
    paths = [
        COMMON / "base.md",
        COMMON / "physical.md",
        MODES / ("explore.md" if explore else "evaluate.md"),
        PROMPTS / "robot.md",
        PROMPTS / "tools.md",
    ]
    if explore:
        values["memory_sources"] = "\n".join(
            f"- {path}"
            for path in (
                f"{values['memory_dir']}/task-specific/",
                f"{values['memory_dir']}/task-family/",
                f"{values['memory_dir']}/MEMORY.md",
                f"{values['memory_dir']}/global/",
                f"{values['memory_inbox']}/wip/",
                f"{values['output_dir']}/attempts/",
            )
        )
        values["memory_kinds"] = "primitive|perception|strategy|failure|infra"
        paths.extend(
            [
                COMMON / "memory.md",
                PROMPTS / "exploration.md",
            ]
        )
    else:
        paths.append(PROMPTS / "evaluation.md")
    return compose(paths, values)


def user_prompt(variables: Mapping[str, object] | None = None) -> str:
    """Bind the current task once using the robot's Markdown template."""
    values = dict(variables or {})
    values["exploration_context"] = ""
    if values.get("mode") == "explore":
        values["exploration_context"] = (
            f"- exploration notes: {values['memory_inbox']}\n"
            f"- failed-attempt archives: {values['output_dir']}/attempts/"
        )
    return compose([PROMPTS / "user.md"], values)
