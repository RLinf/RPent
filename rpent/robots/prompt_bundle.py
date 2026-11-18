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

"""Planner-facing prompt factories and Markdown directory registration."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path

from rpent.prompt.compose import COMMON, MODES, compose

PromptFactory = Callable[[Mapping[str, object] | None], str]


@dataclass(frozen=True)
class PromptBundle:
    """Factories returning final, already-bound system and task strings."""

    system: PromptFactory
    user: PromptFactory

    @classmethod
    def for_robot(
        cls,
        robot: str,
        *,
        memory: bool = False,
        policy: str | Path | None = None,
        physical: str | Path | None = None,
    ) -> PromptBundle:
        """Build system and user factories from a robot's prompt directory.

        Args:
            robot: Package name under robots/. Relative Markdown paths are
                resolved under its prompts/ directory.
            memory: Include the standard TaskRun memory workflow.
            policy: Default policy Markdown path, overridable per run.
            physical: Onsite protocol Markdown path. Providing it also includes
                the shared hardware rules, even for standalone policies.
        """
        if not isinstance(memory, bool):
            raise TypeError("memory must be True or False")
        package = import_module(f"robots.{robot}")
        directory = Path(package.__file__).with_name("prompts")
        hardware = (
            [COMMON / "physical.md", directory / _markdown_path(physical)]
            if physical is not None
            else []
        )
        default_policy = _markdown_path(policy) if policy is not None else None

        def system(variables: Mapping[str, object] | None = None) -> str:
            values = dict(variables or {})
            selected = values.get("policy", default_policy)
            policy_paths = (
                [directory / _markdown_path(selected)] if selected is not None else []
            )
            if values.get("standalone_policy"):
                return compose([*hardware, *policy_paths], values)

            explore = values.get("mode") == "explore"
            paths = [
                COMMON / "base.md",
                MODES / ("explore.md" if explore else "evaluate.md"),
                directory / "robot.md",
                *hardware,
            ]
            if memory:
                if "memory_sources" not in values:
                    values["memory_sources"] = _memory_sources(values, explore)
                paths.append(COMMON / "memory.md")
            if explore:
                if not memory:
                    raise ValueError("Standard exploration requires memory=True")
                values.setdefault(
                    "memory_kinds", "primitive|perception|strategy|failure|infra"
                )
                paths.append(MODES / "exploration-artifacts.md")
            return compose([*paths, *policy_paths], values)

        def user(variables: Mapping[str, object] | None = None) -> str:
            return compose([directory / "user.md"], variables)

        return cls(system=system, user=user)


def _markdown_path(value: object) -> Path:
    """Return a policy or physical protocol file path."""
    if not isinstance(value, (str, Path)):
        raise TypeError("Expected a Markdown file path")
    path = Path(value)
    if path.suffix.lower() != ".md":
        raise ValueError(f"Expected a .md file: {path}")
    return path


def _memory_sources(values: Mapping[str, object], explore: bool) -> str:
    """Return references exposed by the standard TaskRun memory layout."""
    root = values["memory_dir"]
    if explore:
        sources = [f"{root}/task-specific/"]
    else:
        tag = values["reference_tag"]
        sources = [
            f"{root}/task-specific/{tag}.json",
            f"{root}/task-specific/{tag}_recipe.jsonl",
        ]
    sources.extend([f"{root}/task-family/", f"{root}/MEMORY.md", f"{root}/global/"])
    if explore:
        sources.append(f"{values['memory_inbox']}/wip/")
    return "\n".join(f"- {source}" for source in sources)
