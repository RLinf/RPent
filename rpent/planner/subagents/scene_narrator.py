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

"""Shared scene narration instructions and factories for each planner backend."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from rpent.planner.api_loop import ApiAgentLoop, ApiSubAgent, _HarnessToolkit
    from rpent.planner.codex import CodexSubAgent
    from rpent.tools import Toolkit

NARRATOR_NAME = "scene_narrator"

#: API request budget, independent of the main conversation's ``max_turns``.
_NARRATOR_MAX_REQUESTS = 200

_DESCRIPTION = (
    "Keeps narrating the robot's camera view aloud until the episode ends. "
    "Launch it once, in the background, when asked to narrate; do not launch "
    "a second one while it runs. If it returns before the task is finished, "
    "launch it again. It cannot move the robot."
)

_PROMPT = (
    "You narrate a robot's camera view continuously. Repeat this round "
    "until told to stop:\n"
    "1. Call view_env_state(step=-1) or read_image for the latest "
    "observation and look at agentview_high.png.\n"
    "2. Write one Chinese sentence of at most 20 characters about the "
    "arm and the main objects, without Markdown. If the view has not "
    "changed since your last round, describe a different detail.\n"
    "3. Call a text-to-speech tool whose name ends with '__synthesize' "
    "once with exactly that sentence, and wait for it to return before "
    "starting the next round.\n"
    "Start the next round immediately, whether or not the view changed. "
    "Stop and reply with a one-line summary only when view_env_state "
    "reports terminated or truncated as true, or a tool reports that "
    "the session is closed. Do not call any other tool."
)


def select_narrator_tools(
    toolkit: Toolkit, *, include_read_image: bool = False
) -> tuple[str, ...]:
    """Select observations and registered MCP synthesis tools for narration.

    Args:
        toolkit: Toolkit after external MCP discovery and registration.
        include_read_image: Whether the adapter supplies its own image reader.

    Returns:
        Plain tool names to expose through the selected planner adapter.

    Raises:
        ValueError: If observation or synthesis tools are unavailable.
    """
    names = tuple(tool.name for tool in toolkit.list_tools())
    observe = tuple(
        name
        for name in ("view_env_state", "read_image")
        if name in names or (name == "read_image" and include_read_image)
    )
    if not observe:
        raise ValueError("the narrator needs a view_env_state or read_image tool")
    speak = tuple(name for name in names if name.partition("__")[2] == "synthesize")
    if not speak:
        raise ValueError(
            "the narrator needs a registered MCP synthesize tool; "
            "attach a server with tools: [synthesize] using --mcp-config"
        )
    return observe + speak


def build_api_narrator(planner: ApiAgentLoop, recorder: _HarnessToolkit) -> ApiSubAgent:
    """Build an API narrator with independent history and a fresh toolset per run."""
    from pydantic_ai import AbstractToolset, Agent, RunContext
    from pydantic_ai_harness.compaction import SlidingWindowCompaction

    from rpent.planner.api_loop import (
        ApiSubAgent,
        _build_model_settings,
        _HarnessToolkit,
    )

    # Define backend subclasses here so importing another role factory does not
    # load this backend's SDK or its planner adapter.
    class ApiSceneNarrator(ApiSubAgent):
        description = (
            "Start the background scene narrator. It repeatedly reads "
            "the latest robot observation and speaks one short Chinese "
            "sentence through the external text-to-speech MCP tool until "
            "the episode ends. Call it once when the user asks for "
            "spoken narration; do not wait for it."
        )

    allowed = select_narrator_tools(recorder.toolkit, include_read_image=True)
    agent = Agent(
        planner.subagent_model,
        name=NARRATOR_NAME,
        system_prompt=_PROMPT,
        output_type=str,
        model_settings=_build_model_settings(
            planner.subagent_model,
            min(planner.max_tokens, 1024),
            "none",
            parallel_tool_calls=False,
        ),
        capabilities=[
            SlidingWindowCompaction(
                max_messages=40, max_fraction=0.8, keep_messages=20
            ),
        ],
    )

    @agent.toolset(per_run_step=False)
    def narration_tools(_ctx: RunContext) -> AbstractToolset:
        adapter = _HarnessToolkit(
            recorder.toolkit,
            dashboard_events=planner.dashboard_events,
            no_images=False,
            sequential=True,
        )
        return adapter.filtered(lambda _ctx, tool: tool.name in allowed)

    return ApiSceneNarrator(
        agent,
        recorder,
        planner.dashboard_events,
        prompt="Start narrating the scene.",
        request_limit=_NARRATOR_MAX_REQUESTS,
    )


def build_claude_narrator(sdk: Any, *, toolkit: Toolkit, model: str) -> Any:
    """Build a background Claude role limited to observations and speech synthesis."""
    from rpent.planner.base import add_mcp_prefix

    return sdk.AgentDefinition(
        description=_DESCRIPTION,
        prompt=_PROMPT,
        tools=[add_mcp_prefix(name) for name in select_narrator_tools(toolkit)],
        model=model,
        maxTurns=200,
        background=True,
    )


def build_codex_narrator(*, toolkit: Toolkit, model: str | None) -> CodexSubAgent:
    """Build a Codex role using the session's filtered RPent MCP bridge."""
    from rpent.planner.codex import CodexSubAgent

    class CodexSceneNarrator(CodexSubAgent):
        name = NARRATOR_NAME
        description = (
            _DESCRIPTION + " Spawn it with "
            f'agent_type="{NARRATOR_NAME}" and fork_turns="none", and do not wait '
            "for it; continue the task."
        )

        def __init__(self) -> None:
            self.tools = select_narrator_tools(toolkit)
            self.model = model

        def render_config(
            self, mcp_url: str, *, model_provider: str | None = None
        ) -> str:
            """Allow only observation and synthesis tools on the RPent bridge."""
            lines = [
                'sandbox_mode = "read-only"',
                'approval_policy = "never"',
                "tools.web_search = false",
                f"developer_instructions = {json.dumps(_PROMPT)}",
                f"mcp_servers.rpent.url = {json.dumps(mcp_url)}",
                f"mcp_servers.rpent.enabled_tools = {json.dumps(self.tools)}",
                'mcp_servers.rpent.default_tools_approval_mode = "approve"',
            ]
            if self.model is not None:
                lines.append(f"model = {json.dumps(self.model)}")
            if model_provider is not None:
                lines.append(f"model_provider = {json.dumps(model_provider)}")
            return "\n".join(lines) + "\n"

    return CodexSceneNarrator()
