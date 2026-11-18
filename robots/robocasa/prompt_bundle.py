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

"""robocasa file-backed prompt composition."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from robots.robocasa.memory import memory_from_variables
from rpent.prompt.compose import COMMON, MODES, compose

PROMPTS = Path(__file__).with_name("prompts")


def system_prompt(variables: Mapping[str, object] | None = None) -> str:
    """Compose the robot, authorized memory and current mode."""
    values = dict(variables or {})
    explore = values.get("mode") == "explore"
    values["memory_sources"] = _memory_sources(values)

    return compose(
        [
            COMMON / "base.md",
            MODES / ("explore.md" if explore else "evaluate.md"),
            PROMPTS / "robot.md",
            PROMPTS / "memory.md",
            PROMPTS / ("exploration.md" if explore else "evaluation.md"),
        ],
        values,
    )


def user_prompt(variables: Mapping[str, object] | None = None) -> str:
    """Return the concrete task and output locations."""
    return compose([PROMPTS / "user.md"], variables)


def _memory_sources(values: Mapping[str, object]) -> str:
    """Use the same evaluation allowlist as the toolkit."""
    explore = values.get("mode") == "explore"
    if explore:
        return (
            "\n".join(
                f"- `{values['memory_dir']}/{name}/`"
                for name in ("task-specific", "task-family", "global")
            )
            + f"\n- `{values['memory_inbox']}/wip/`"
        )
    else:
        selection = memory_from_variables(values)
        return "\n".join(f"- {selection.root / name}" for name in selection.selected)
