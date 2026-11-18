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

"""Compose YAM's local real-robot prompt fragments."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from rpent.prompt.compose import compose

PROMPTS = Path(__file__).with_name("prompts")


def system_prompt(variables: Mapping[str, object] | None = None) -> str:
    """Select the original YAM protocols for the current mode and policy."""
    values = variables or {}
    paths = [PROMPTS / "robot.md", PROMPTS / "memory.md"]
    if values.get("mode", "eval") == "explore":
        paths.append(PROMPTS / "exploration.md")
    backend = values.get("policy_backend") or "primitives-only"
    if backend not in {"pi05", "primitives-only"}:
        raise KeyError(backend)
    paths.append(PROMPTS / f"policies/{backend}.md")
    return compose(paths, values)


def user_prompt(variables: Mapping[str, object] | None = None) -> str:
    """Bind the current task in YAM's own message template."""
    return compose([PROMPTS / "user.md"], variables)


__all__ = ["system_prompt", "user_prompt"]
