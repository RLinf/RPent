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

"""Observable context boundaries, atomic tool validation and shared budgets."""

import copy
import json

import pytest
from pydantic_ai.models.function import DeltaThinkingPart, DeltaToolCall, FunctionModel

from tests.unit_tests.robots.rl.test_snapshot_contracts import (
    SnapshotToolkit,
    make_planner,
)


@pytest.fixture(autouse=True)
def no_cli_globals(monkeypatch):
    monkeypatch.setattr("rpent.tools.toolkit.substitute", lambda value: value)
    monkeypatch.setattr(
        "rpent.utils.templates.default_variables", lambda: {"output_dir": "/offline"}
    )


def run_script(tmp_path, workflow, script, max_turns=100):
    toolkit = SnapshotToolkit(tmp_path, workflow)
    requests = []

    async def stream(messages, info):
        requests.append(copy.deepcopy(messages))
        calls = script[len(requests) - 1]
        yield {0: DeltaThinkingPart(content=f"reasoning-secret-{len(requests)}")}
        yield {
            i + 1: DeltaToolCall(name=name, json_args=json.dumps(args))
            for i, (name, args) in enumerate(calls)
        }

    result = make_planner(FunctionModel(stream_function=stream)).solve(
        system_prompt="same-system",
        user_message="task",
        toolkit=toolkit,
        max_turns=min(max_turns, len(script)),
    )
    return toolkit, result, requests


def plan(text, status="in_progress"):
    return {"items": [{"text": text, "status": status}]}


def test_subtask_resets_only_after_valid_todo_and_keeps_queries(tmp_path):
    script = [
        [("change_query", {"queries": ["cube"]}), ("update_todo", plan("first-goal"))],
        [("move", {"x": 0.1})],
        [("update_todo", {"items": [{"text": "bad", "status": "invented"}]})],
        [("move", {"x": 0.2})],
        [("update_todo", plan("second-goal"))],
        [("finish", {})],
    ]
    toolkit, result, requests = run_script(tmp_path, "zap_subtask", script)
    assert result.error is None
    assert result.stats["turns_used"] == 6
    assert toolkit.executed == [0.1, 0.2]
    assert toolkit.workflow_stats["context_resets"] == 2
    assert toolkit.snapshot.queries == ["cube"]
    fresh = str(requests[-1])
    assert "second-goal" in fresh and "first-goal" not in fresh
    assert "reasoning-secret" not in fresh
    assert "reasoning-secret-2" in str(requests[3])  # invalid todo did not reset
    assert "reasoning-secret" not in str(requests[1])
    assert all("same-system" in str(request) for request in requests)


def test_initial_pair_is_validated_atomically_before_query_update(tmp_path):
    script = [
        [
            ("change_query", {"queries": ["cube"]}),
            ("update_todo", {"items": [{"text": "bad", "status": "invalid"}]}),
        ]
    ]
    toolkit, result, _ = run_script(tmp_path, "zap_subtask", script)
    assert result.stats["turns_used"] == 1
    assert result.stats["tool_calls"] == 0
    assert toolkit.snapshot.queries == [] and toolkit.snapshot.todo == []


def test_episode_zap_retains_history_without_registering_todo(tmp_path):
    script = [
        [("change_query", {"queries": ["cube"]})],
        [("move", {"x": 0.1})],
        [("finish", {})],
    ]
    toolkit, result, requests = run_script(tmp_path, "zap_episode", script)
    assert result.error is None
    assert "reasoning-secret-1" in str(requests[1])
    assert "reasoning-secret-2" in str(requests[2])
    assert "update_todo" not in {tool.name for tool in toolkit.list_tools()}


def test_reset_does_not_reset_global_request_limit(tmp_path):
    script = [[("change_query", {"queries": ["cube"]}), ("update_todo", plan("start"))]]
    script += [[("update_todo", plan(f"goal-{i}"))] for i in range(99)]
    toolkit, result, requests = run_script(tmp_path, "zap_subtask", script)
    assert result.error is None and len(requests) == 100
    assert result.stats["turns_used"] == 100
    assert result.stats["tool_calls"] == 101
    assert toolkit.workflow_stats["context_resets"] == 99


def test_subtask_reset_clears_even_compacted_history(tmp_path):
    from pydantic_ai.capabilities import AbstractCapability
    from pydantic_ai_harness.compaction import SlidingWindowCompaction

    toolkit = SnapshotToolkit(tmp_path, "zap_subtask")
    requests, compactions = [], []
    script = [
        [("change_query", {"queries": ["cube"]}), ("update_todo", plan("old-subtask"))]
    ]
    script += [[("move", {"x": float(i) / 100})] for i in range(45)]
    script += [
        [("update_todo", plan("new-subtask"))],
        [("move", {"x": 0.6})],
        [("finish", {})],
    ]

    async def stream(messages, info):
        requests.append(copy.deepcopy(messages))
        yield {0: DeltaThinkingPart(content=f"prior-secret-{len(requests)}")}
        yield {
            i + 1: DeltaToolCall(name=name, json_args=json.dumps(arguments))
            for i, (name, arguments) in enumerate(script[len(requests) - 1])
        }

    class CountCompaction(AbstractCapability):
        def __init__(self):
            self.window = SlidingWindowCompaction(
                max_messages=80,
                keep_messages=40,
                max_fraction=0.7,
                context_window=262144,
            )

        async def before_model_request(self, ctx, request_context):
            before = len(request_context.messages)
            result = await self.window.before_model_request(ctx, request_context)
            if len(result.messages) < before:
                compactions.append((before, len(result.messages)))
            return result

    planner = make_planner(FunctionModel(stream_function=stream))
    planner.compaction = CountCompaction
    result = planner.solve(
        system_prompt="same-system",
        user_message="task",
        toolkit=toolkit,
        max_turns=len(script),
    )
    assert result.error is None and compactions
    assert toolkit.workflow_stats["context_resets"] == 2
    assert "new-subtask" in str(requests[-2]) and "old-subtask" not in str(requests[-2])
    assert "prior-secret" not in str(requests[-2])
    assert "prior-secret-48" in str(requests[-1])
