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

from __future__ import annotations

import asyncio
import json

import pytest
from mcp import types

from rpent.dashboard.events import NullDashboardEventSink
from rpent.planner.api_loop import (
    _ApiRunObserver,
    _build_tools,
)
from rpent.planner.claude_code import (
    _build_rpent_server,
)
from rpent.planner.claude_code import _Recorder as ClaudeRecorder
from rpent.planner.codex import _Recorder as CodexRecorder

from ._native_helpers import call_sdk_tool


def test_read_image_is_available_to_api_and_not_callable_through_mcp(make_toolkit):
    toolkit = make_toolkit()
    assert "read_image" in {definition.name for definition in _build_tools(toolkit)}

    async def scenario():
        server = _build_rpent_server(toolkit=toolkit)
        specs = await server["instance"].request_handlers[types.ListToolsRequest](
            types.ListToolsRequest(method="tools/list")
        )
        assert "read_image" not in {definition.name for definition in specs.root.tools}
        rejected = await call_sdk_tool(server, "read_image", {"name": "frame.png"})
        assert rejected.isError
        assert json.loads(rejected.content[0].text) == {
            "error": "Unknown tool: read_image"
        }
        assert toolkit.calls == []

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "recorder_type", [ClaudeRecorder, CodexRecorder, _ApiRunObserver]
)
def test_recorders_read_verified_finish_without_a_matching_provider_event(
    make_toolkit,
    recorder_type,
):
    accepted = {
        "status": "failure",
        "summary": "environment verification failed",
        "operator_aborted": True,
        "operator_finished": True,
        "operator_verdict": "abort",
        "operator_notes": "Stop the run.",
        "attempt": 2,
    }
    toolkit = make_toolkit({}, accepted=accepted)
    recorder = recorder_type(
        toolkit=toolkit,
        max_turns=3,
        dashboard_events=NullDashboardEventSink(),
        **({"messages": []} if recorder_type is _ApiRunObserver else {}),
    )
    assert recorder.finish_result is None
    result = toolkit.execute_tool(
        "finish", {"status": "success", "summary": "model claims success"}
    )
    assert not result.is_error
    assert result.data == {"_finish": True, **accepted}
    assert recorder.finish_result == accepted
    assert recorder.finish_result == toolkit.finish_result
    returned = recorder.finish_result
    returned["operator_aborted"] = False
    assert toolkit.finish_result["operator_aborted"] is True


def test_api_finish_accepted_during_interrupt_seals_dashboard():
    from types import SimpleNamespace

    from rpent.planner.api_loop import _ApiDashboardSession

    async def scenario():
        ended = asyncio.Event()
        control = SimpleNamespace(end=ended.set)
        session = _ApiDashboardSession(
            agent=None,
            control=control,
            observer=SimpleNamespace(
                finish_result={"status": "failure", "summary": "verified"}
            ),
            max_turns=3,
            no_images=False,
        )
        session._active_prompt = True
        session._run_task = asyncio.create_task(asyncio.Event().wait())
        assert await session.interrupt() == 1
        assert ended.is_set()
        assert session._run_task is None

    asyncio.run(scenario())
