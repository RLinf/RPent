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

"""Fixed episode prompts for the selected robot and coordinate convention."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path


def system_prompt(variables: Mapping[str, object] | None = None) -> str:
    """Render the robot description with the fixed centimetre interface."""
    variables = variables or {}
    environment = variables.get("environment", "libero")
    prompts = Path(__file__).parent / "prompts"
    names = ("general.md", f"{environment}_general.md", "episode.md")
    sections = [(prompts / name).read_text().strip() for name in names]
    if not variables.get("reasoning", True):
        sections[-1] = sections[-1].replace(
            "Reason briefly about the next decision using observations, not hypothetical completed actions.",
            "Call the next tool directly without reasoning or explanatory text.",
        )
    limit = variables.get("max_actions_per_call", 3)
    sections.append(
        f"Action interface: execute_actions takes one to {limit} physical items in order, including single-item lists. "
        "Read each measured result. A partial result, error or terminal stops the remaining items."
    )
    sections.append(
        "Older reasoning or camera images may be removed from the conversation. "
        "Use a new view when you need pixels absent from the current history."
    )
    from robots.rl.units import CENTIMETERS

    return CENTIMETERS.description("\n\n".join(sections))


def user_prompt(variables: Mapping[str, object] | None = None) -> str:
    """Introduce the task and initial measured state."""
    return "The task and the measured initial state follow."
