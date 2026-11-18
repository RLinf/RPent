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

"""robodojo file-backed prompt composition."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from rpent.prompt.compose import COMMON, MODES, compose

PROMPTS = Path(__file__).with_name("prompts")


def system_prompt(variables: Mapping[str, object] | None = None) -> str:
    """Compose one backend and one execution mode with local memory rules."""
    values = dict(variables or {})
    backend = values.get("policy_backend") or "vpp2"
    if backend not in {"vpp2", "awomo05"}:
        raise KeyError(backend)
    explore = values.get("mode") == "explore"
    paths = [
        COMMON / "base.md",
        MODES / ("explore.md" if explore else "evaluate.md"),
        PROMPTS / "robot.md",
        PROMPTS / "tools.md",
        PROMPTS / f"policies/{backend}.md",
    ]
    if values.get("mode") == "eval-fair":
        return compose([*paths, PROMPTS / "evaluation-fair.md"], values)

    values["memory_sources"] = _memory_sources(values)
    values["memory_kinds"] = "primitive|perception|strategy|failure|infra"
    paths.extend(
        [
            COMMON / "memory.md",
            PROMPTS / ("exploration.md" if explore else "evaluation.md"),
        ]
    )
    return compose(paths, values)


def user_prompt(variables: Mapping[str, object] | None = None) -> str:
    """Return task-cell data without controller instructions."""
    values = variables or {}
    if values.get("mode") == "eval-fair":
        task = f"{values['task']} / layout {values['layout']}"
        context = ""
    else:
        task = f"Task: {values['task']}\nScene (static hint): {values['task_summary']}"
        context = (
            f"Seed group: {values['seed']}\nLayout: {values['layout']}\n"
            f"Environment configuration: {values['env_cfg_type']}\n"
            f"Output directory: {values['output_dir']}"
        )
    return compose(
        [PROMPTS / "user.md"], {"task_description": task, "run_context": context}
    )


def _memory_sources(values: Mapping[str, object]) -> str:
    """Resolve the current task's references or exploration memory directories."""
    root = values["memory_dir"]
    if values.get("mode") == "explore":
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
            f"{root}/task-family/task-family_robodojo_{values['env_cfg_type']}_t{values['task']}.md",
            f"{root}/MEMORY.md",
            f"{root}/global/",
        ]
    return "\n".join(f"- {source}" for source in sources)
