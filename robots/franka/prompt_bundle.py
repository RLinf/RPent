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

"""Final Markdown prompts for the franka environment."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from rpent.prompt.compose import COMMON, MODES, compose

PROMPTS = Path(__file__).parent / "prompts"


def system_prompt(variables: Mapping[str, object] | None = None) -> str:
    """Compose and bind the current robot execution instructions once."""
    values = dict(variables or {})

    return compose(
        [
            COMMON / "base.md",
            COMMON / "physical.md",
            MODES / "evaluate.md",
            PROMPTS / "robot.md",
        ],
        values,
    )


def user_prompt(variables: Mapping[str, object] | None = None) -> str:
    """Bind the current task once using the robot's Markdown template."""
    return compose([PROMPTS / "user.md"], variables)
