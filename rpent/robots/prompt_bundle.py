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

"""Prompt bundle dataclass for robot-contributed LLM prompts.

Lives in :mod:`rpent.robots` so each robot's
``prompt_bundle.py`` (e.g. :mod:`robots.libero.prompt_bundle`)
can import it without depending on the RPC transport layer.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

from rpent.prompt.utils import PromptNode, format_prompt

PromptFactory = Callable[..., PromptNode]


@dataclass(frozen=True)
class PromptBundle:
    """Python-defined prompt factories for one robot."""

    system: PromptFactory
    user: PromptFactory

    def render(
        self,
        variant: str,
        *,
        variables: Mapping[str, object] | None = None,
    ) -> str:
        """Render one prompt variant (``"system"`` or ``"user"``)."""
        prompt = getattr(self, variant)(variables)
        return format_prompt(prompt, variables=variables)
