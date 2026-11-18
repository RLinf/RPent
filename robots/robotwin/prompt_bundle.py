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

"""Compose robotwin prompts from Markdown fragments."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from rpent.prompt.compose import COMMON, MODES, compose

PROMPTS = Path(__file__).with_name("prompts")
WAM_PROMPT_BACKENDS = {"cosmos-policy", "cosmos3", "fast-wam"}


def system_prompt(variables: Mapping[str, object] | None = None) -> str:
    """Select the control protocol, mode and local memory instructions."""
    values = dict(variables or {})
    backend = values.get("policy_backend", "lingbot")
    if backend in WAM_PROMPT_BACKENDS:
        return compose([PROMPTS / "policies/wam.md"], values)

    explore = values.get("mode", "eval") == "explore"
    values["memory_sources"] = _memory_sources(values)
    values["memory_kinds"] = "primitive|perception|strategy|failure|infra"

    return compose(
        [
            COMMON / "base.md",
            MODES / ("explore.md" if explore else "evaluate.md"),
            PROMPTS / "robot.md",
            PROMPTS / "tools.md",
            COMMON / "memory.md",
            PROMPTS / ("exploration.md" if explore else "evaluation.md"),
            PROMPTS / f"policies/{backend}.md",
        ],
        values,
    )


def user_prompt(variables: Mapping[str, object] | None = None) -> str:
    """Bind the RoboTwin task cell and selected backend context once."""
    values = dict(variables or {})
    backend = values.get("policy_backend", "lingbot")
    if backend in WAM_PROMPT_BACKENDS:
        values["policy_context"] = f"Local memory directory: {values['memory_dir']}."
    else:
        checkpoint = {
            "lingbot": "RLinf/LingBot-VLA-RoboTwin-EEF-ckpt1500",
            "uniwam": "UniWAM RoboTwin history-flow",
        }[backend]
        values["policy_context"] = f"- checkpoint: {checkpoint}"
    return compose([PROMPTS / "user.md"], values)


def _memory_sources(values: Mapping[str, object]) -> str:
    """List references for evaluation or prior evidence for exploration."""
    explore = values.get("mode", "eval") == "explore"
    root = values["memory_dir"]
    if explore:
        sources = [
            f"{root}/MEMORY.md",
            f"{root}/task-specific/",
            f"{root}/task-family/",
            f"{root}/global/",
            f"{values['memory_inbox']}/wip/",
        ]
    else:
        sources = [
            f"{root}/task-specific/{values['reference_tag']}.json",
            f"{root}/task-specific/{values['reference_tag']}_recipe.jsonl",
            f"{root}/task-family/",
            f"{root}/MEMORY.md",
            f"{root}/global/",
        ]
    return "\n".join(f"- {source}" for source in sources)


__all__ = ["system_prompt", "user_prompt"]
