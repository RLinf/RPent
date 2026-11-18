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

"""Compose libero prompts from Markdown fragments."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from rpent.prompt.compose import COMMON, MODES, compose

PROMPTS = Path(__file__).with_name("prompts")


def system_prompt(variables: Mapping[str, object] | None = None) -> str:
    """Compose the selected policy, memory instructions and execution mode."""
    values = dict(variables or {})
    if values.get("policy_kind") == "wam":
        return compose([PROMPTS / "policies/wam.md"], values)

    explore = values.get("mode", "eval") == "explore"
    values["memory_sources"] = _memory_sources(values)
    values["memory_kinds"] = "primitive|perception|strategy|failure"
    values["allowed_primitives"] = _allowed_primitives(
        bool(values.get("enable_direct_action", False))
    )

    return compose(
        [
            COMMON / "base.md",
            MODES / ("explore.md" if explore else "evaluate.md"),
            PROMPTS / "robot.md",
            COMMON / "memory.md",
            PROMPTS / "tools.md",
            PROMPTS / "policies/pi05.md",
            *([COMMON / "memory-write.md"] if explore else []),
            PROMPTS / ("exploration.md" if explore else "evaluation.md"),
        ],
        values,
    )


def user_prompt(variables: Mapping[str, object] | None = None) -> str:
    """Bind the LIBERO task cell and optional WAM memory context once."""
    values = dict(variables or {})
    values["policy_context"] = (
        f"Local memory directory: {values['memory_dir']}."
        if values.get("policy_kind") == "wam"
        else ""
    )
    return compose([PROMPTS / "user.md"], values)


def _memory_sources(values: Mapping[str, object]) -> str:
    """List authorized memory paths in the mode's reading order."""
    explore = values.get("mode", "eval") == "explore"
    root = values["memory_dir"]
    pair = [
        f"{root}/task-specific/{values['reference_tag']}.json",
        f"{root}/task-specific/{values['reference_tag']}_recipe.jsonl",
    ]
    if explore:
        sources = [
            f"{root}/task-family/",
            f"{root}/MEMORY.md",
            f"{root}/global/",
            *pair,
            f"{values['memory_inbox']}/wip/",
        ]
    elif values.get("memory_profile", "hf") == "local":
        sources = [
            *pair,
            f"{root}/task-family/task-family_libero10_<regime>_t{values['task']}.md",
            f"{root}/task-family/",
            f"{root}/MEMORY.md",
            f"{root}/global/",
        ]
    else:
        sources = [
            f"{root}/MEMORY.md",
            f"{root}/global/",
            f"{root}/task-family/",
            *pair,
        ]
    return "\n".join(f"- {source}" for source in sources)


def _allowed_primitives(direct_action: bool) -> str:
    """Describe the standard controls and optional direct-action tool."""
    primitives = [
        "move_to",
        "pi0_pick",
        "pi0_doubled",
        "release",
        "set_gripper",
        "rotate_wrist",
        "rotate_pitch",
        "move_pose",
    ]
    if direct_action:
        primitives.append("execute_action")
    return ", ".join(f"`{name}`" for name in primitives)


__all__ = ["system_prompt", "user_prompt"]
