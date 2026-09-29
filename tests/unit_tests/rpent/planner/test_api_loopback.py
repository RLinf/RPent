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

"""Exercise the GPU E2E planner fixture through real local HTTP."""

from __future__ import annotations

import json

import httpx
import pytest

from rpent.dashboard.events import NullDashboardEventSink
from rpent.memory.manager import MemoryManager
from rpent.planner.api_loop import ApiAgentLoop
from rpent.planner.base import build_api_model
from rpent.tools.toolkit import Toolkit, readonly
from rpent.utils.logging import init_output_dir
from tests.e2e_tests.offline_planner_server import (
    OFFLINE_MODEL_NAME,
    OfflinePlannerServer,
    ScriptedToolCall,
)


@pytest.mark.parametrize("stream", [True, False], ids=["api-planner", "json-client"])
def test_offline_planner_observe_then_finish(tmp_path, monkeypatch, stream):
    monkeypatch.setenv("OPENAI_API_KEY", OFFLINE_MODEL_NAME)
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")
    monkeypatch.setenv("PYDANTIC_AI_NO_BANNER", "1")
    init_output_dir(tmp_path)
    events = NullDashboardEventSink()
    toolkit = Toolkit(
        dashboard_events=events, memory=MemoryManager(tmp_path / "memory")
    )
    observations = []

    @readonly
    def observe():
        observations.append({"position": 1})
        return observations[-1]

    toolkit.add_tool(
        "observe",
        {
            "name": "observe",
            "description": "Read a synthetic observation.",
            "input_schema": {"type": "object", "properties": {}},
        },
        observe,
    )
    finish_args = {"status": "stuck", "summary": "offline observation complete"}
    script = (
        ScriptedToolCall("observe", {}),
        ScriptedToolCall("finish", finish_args),
    )
    with OfflinePlannerServer(script) as server:
        if stream:
            planner = ApiAgentLoop(
                model=build_api_model(
                    f"openai-chat:{OFFLINE_MODEL_NAME}", server.base_url
                ),
                dashboard_events=events,
                no_images=True,
                timeout_s=20,
            )
            result = planner.solve(
                system_prompt="Observe once and finish.",
                user_message="Read the observation.",
                toolkit=toolkit,
                max_turns=2,
            )
            assert result.error is None
            assert result.stats["turns_used"] == 2
            assert result.stats["tool_calls"] == 2
            assert result.stats["total_input_tokens"] == 2
            assert result.stats["total_output_tokens"] == 2
            finish_result = result.finish_result
        else:
            tools = [
                {
                    "type": "function",
                    "function": {
                        "name": spec["name"],
                        "parameters": spec["input_schema"],
                    },
                }
                for spec in toolkit.get_tools_spec()
            ]
            with httpx.Client(trust_env=False) as client:
                for expected in script:
                    response = client.post(
                        server.base_url + "/chat/completions",
                        json={
                            "model": OFFLINE_MODEL_NAME,
                            "stream": False,
                            "tools": tools,
                        },
                    )
                    response.raise_for_status()
                    assert response.headers["content-type"] == "application/json"
                    payload = response.json()
                    assert payload["object"] == "chat.completion"
                    call = payload["choices"][0]["message"]["tool_calls"][0]["function"]
                    assert call["name"] == expected.name
                    arguments = json.loads(call["arguments"])
                    assert arguments == expected.arguments
                    finish_result = toolkit.execute_tool(call["name"], arguments).result
        server.assert_complete()
        assert server.request_count == 2
    assert observations == [{"position": 1}]
    assert finish_result == {"_finish": True, **finish_args}
