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

"""Select concrete subagents without coupling planner runtimes to their roles."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from rpent.planner.api_loop import ApiAgentLoop, ApiSubAgent, _HarnessToolkit
    from rpent.planner.codex import CodexSubAgent
    from rpent.tools import Toolkit

SUBAGENT_NAMES = ("narrator",)
SUBAGENT_PLANNERS = ("api", "claude_code", "codex")


def validate_subagent(planner: str, name: str | None) -> None:
    """Reject unknown roles and backends that cannot launch subagents."""
    if name is None:
        return
    if name not in SUBAGENT_NAMES:
        raise ValueError(f"unknown subagent: {name}")
    if planner not in SUBAGENT_PLANNERS:
        raise ValueError(f"--subagent is not supported with the {planner} planner")


def build_api_subagent(
    name: str, planner: ApiAgentLoop, recorder: _HarnessToolkit
) -> ApiSubAgent:
    """Construct an API role using the session's Toolkit adapter."""
    validate_subagent("api", name)
    from rpent.planner.subagents.scene_narrator import build_api_narrator

    return build_api_narrator(planner, recorder)


def build_claude_subagent(
    name: str, sdk: Any, *, toolkit: Toolkit, model: str
) -> tuple[str, Any]:
    """Construct a named Claude Agent SDK role."""
    validate_subagent("claude_code", name)
    from rpent.planner.subagents.scene_narrator import (
        NARRATOR_NAME,
        build_claude_narrator,
    )

    return NARRATOR_NAME, build_claude_narrator(sdk, toolkit=toolkit, model=model)


def build_codex_subagent(
    name: str, *, toolkit: Toolkit, model: str | None
) -> CodexSubAgent:
    """Construct a Codex role using the existing RPent MCP endpoint."""
    validate_subagent("codex", name)
    from rpent.planner.subagents.scene_narrator import build_codex_narrator

    return build_codex_narrator(toolkit=toolkit, model=model)
