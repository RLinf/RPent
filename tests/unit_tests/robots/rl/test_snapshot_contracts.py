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

"""Independent snapshot requests, explicit memory and exact-frame measurements."""

from __future__ import annotations

import io
import itertools
import json
import threading
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image
from pydantic_ai.messages import (
    BinaryContent,
    ModelRequest,
    ThinkingPart,
    UserPromptPart,
)
from pydantic_ai.models.function import DeltaThinkingPart, DeltaToolCall, FunctionModel

from robots.rl.snapshot import Snapshot, SubtaskPlan
from robots.rl.tools import LiberoRLPrimitives
from robots.rl.workflow import WorkflowCapability
from rpent.dashboard.events import NullDashboardEventSink
from rpent.planner.api_loop import ApiAgentLoop
from rpent.robots.components.sam3_client import Sam3Result
from rpent.session import EnvState
from rpent.tools import ToolResult, tool
from rpent.tools.toolkit import Toolkit


@pytest.fixture(autouse=True)
def tool_descriptions_without_cli_globals(monkeypatch):
    monkeypatch.setattr("rpent.tools.toolkit.substitute", lambda value: value)
    monkeypatch.setattr(
        "rpent.utils.templates.default_variables", lambda: {"output_dir": "/offline"}
    )


class Primitives:
    _audit_sam = LiberoRLPrimitives._audit_sam
    cameras = ("agentview", "wrist")

    def __init__(self, output):
        self._state = EnvState(output)
        self._sam3 = self
        self._sam_calls = itertools.count(1)
        self.env_steps = 0
        self._max_episode_steps = 10000
        self.queries = []
        self.observe()

    def observe(self):
        with self._state.record_step(
            state={"position": self.env_steps}, terminated=False, truncated=False
        ) as step:
            for camera in self.cameras:
                self._state.save(
                    f"{camera}.png", np.zeros((8, 8, 3), dtype=np.uint8), step=step
                )
                world = np.ones((8, 8, 3), dtype=np.float32)
                world[..., 2] = 0.02 + self.env_steps
                self._state.save(f"{camera}_world.npz", world, step=step)

    def _save(self, name, value, step):
        assert self._state.save(name, value, step=step)

    def images(self, step, cameras):
        return [
            self._state.load_bytes(f"{camera}.png", step=step) for camera in cameras
        ]

    def segment_queries(self, raw, queries):
        self.queries.append((raw, list(queries)))
        mask = np.ones((8, 8), dtype=bool)
        return [
            [
                Sam3Result(found=True, mask=mask, score=0.9),
                Sam3Result(found=True, mask=mask, score=0.8),
            ]
            for _ in queries
        ]


class SnapshotToolkit(Toolkit):
    planner_builtin_tools = False

    def __init__(self, output, workflow="zap_turn"):
        self.workflow = workflow
        self.output_dir = output
        self.image_size = 8
        self.workflow_stats = {}
        self.primitive = Primitives(output)
        self.snapshot = Snapshot(
            self.primitive, environment="libero", task="place the cube", image_size=8
        )
        self.executed = []
        super().__init__(
            state=self.primitive._state,
            memory=SimpleNamespace(),
            dashboard_events=NullDashboardEventSink(),
        )
        self.add_tool(self.snapshot.change_query)
        if workflow == "zap_subtask":
            self.add_tool(SubtaskPlan(self.snapshot).update_todo)
        elif workflow == "zap_turn":
            self.add_tool(self.snapshot.update_todo)
        self.add_tool(self.move)
        self.add_tool(self.finish)

    def _register_common_tools(self):
        pass

    def _count_images(self, observation):
        pass

    def initial_observation(self):
        return self.observe_snapshot() if self.workflow != "zap_turn" else None

    def observe_snapshot(self):
        return self.snapshot.observe({"position": self.primitive.env_steps})

    @tool(readonly=True)
    def move(self, x: float) -> ToolResult:
        """Move the test robot.

        Args:
            x: New coordinate.
        """
        self.executed.append(x)
        self.primitive.env_steps += 1
        self.primitive.observe()
        return ToolResult(data={"status": "reached", "x": x})

    @tool(readonly=True)
    def finish(self) -> ToolResult:
        """End the test episode."""
        return ToolResult(data={"_finish": True, "success": False})

    def execute_tool(self, name, arguments):
        result = super().execute_tool(name, arguments)
        self.snapshot.last_result = {
            "tool": name,
            "arguments": arguments,
            "result": dict(result.data),
        }
        if self.workflow != "zap_turn" and name in {"move", "change_query"}:
            observation = self.observe_snapshot()
            result.images = observation.images
            result.data["observation"] = observation.data
        return result


def make_planner(model):
    return ApiAgentLoop(
        model=model,
        dashboard_events=NullDashboardEventSink(),
        max_tokens=100,
        timeout_s=10,
        compaction=None,
        sequential_tools=True,
        capabilities_factory=lambda toolkit, system, user, turns: [
            WorkflowCapability(toolkit, system, user, turns, timeout_s=10)
        ],
    )


def test_plan_replacement_and_perception_use_one_recorded_frame(tmp_path):
    primitive = Primitives(tmp_path)
    snapshot = Snapshot(
        primitive, environment="libero", task="pick up the cube", image_size=8
    )
    first = snapshot.observe({})
    assert len(first.images) == 2 and first.data["perception"][0]["objects"] == []
    snapshot.change_query(["cube"])
    snapshot.update_todo(["old plan"])
    current = snapshot.observe({})
    assert len(current.data["perception"][0]["objects"]) == 2
    measured = current.data["perception"][0]["objects"][0]
    assert measured["box_rc"] == [0, 0, 8, 8]
    assert measured["surface"]["top_z_m"] == 0.02
    assert measured["box_surface"]["top_z_m"] == 0.02
    snapshot.update_todo(["replacement plan"])
    next_observation = snapshot.observe({})
    assert next_observation.data["todo"] == ["replacement plan"]
    assert len(primitive.queries) == 2  # same recorded views reuse exact-frame results
    primitive.env_steps = 1
    primitive.observe()
    latest = snapshot.observe({})
    assert len(primitive.queries) == 4
    assert latest.data["perception"][0]["objects"][0]["surface"]["top_z_m"] == 1.02
    for raw in latest.images:
        assert Image.open(io.BytesIO(raw)).size == (8, 8)


@pytest.mark.parametrize("bad", [[""], ["   "]])
def test_empty_plan_or_query_does_not_replace_current_memory(tmp_path, bad):
    snapshot = Snapshot(
        Primitives(tmp_path),
        environment="libero",
        task="pick up the cube",
        image_size=8,
    )
    snapshot.change_query(["cube"])
    snapshot.update_todo(["keep this"])
    assert snapshot.change_query(bad).is_error
    assert snapshot.update_todo(bad).is_error
    assert snapshot.queries == ["cube"] and snapshot.todo == ["keep this"]


def test_streamed_single_requests_do_not_replay_history_and_bootstrap_is_forced(
    tmp_path,
):
    toolkit = SnapshotToolkit(tmp_path)
    requests, tool_names = [], []
    script = [
        ("change_query", {"queries": ["cube"]}),
        ("update_todo", {"items": ["verify then place"]}),
        ("move", {"x": 0.1}),
        ("finish", {}),
    ]

    async def stream(messages, info):
        requests.append(messages)
        tool_names.append([definition.name for definition in info.function_tools])
        name, args = script[len(requests) - 1]
        yield {0: DeltaThinkingPart(content=f"private-reasoning-{len(requests)}")}
        yield {1: DeltaToolCall(name=name, json_args=json.dumps(args))}

    planner = make_planner(FunctionModel(stream_function=stream))
    result = planner.solve(
        system_prompt="one action", user_message="solve", toolkit=toolkit, max_turns=8
    )
    assert result.error is None and result.stats["turns_used"] == 4
    assert toolkit.executed == [0.1]
    assert tool_names[:2] == [["change_query"], ["update_todo"]]
    for index, messages in enumerate(requests):
        assert len(messages) == 1 and isinstance(messages[0], ModelRequest)
        assert not any(
            isinstance(part, ThinkingPart)
            for message in messages
            for part in message.parts
        )
        user = next(
            part for part in messages[0].parts if isinstance(part, UserPromptPart)
        )
        assert (
            len([part for part in user.content if isinstance(part, BinaryContent)]) == 2
        )
        text = user.content[0]
        assert "private-reasoning" not in text
        assert "old plan" not in text
        if index >= 2:
            assert "verify then place" in text
    assert len(list((tmp_path / "workflow_requests").glob("*.json"))) == 4


def test_multiple_calls_are_not_executed_and_do_not_advance_bootstrap(tmp_path):
    toolkit = SnapshotToolkit(tmp_path)
    calls = []

    async def stream(messages, info):
        calls.append(info.function_tools)
        if len(calls) == 1:
            yield {
                0: DeltaToolCall(name="change_query", json_args='{"queries":["cube"]}'),
                1: DeltaToolCall(name="move", json_args='{"x":1}'),
            }
        else:
            yield {
                0: DeltaToolCall(name="change_query", json_args='{"queries":["cube"]}')
            }

    planner = make_planner(FunctionModel(stream_function=stream))
    result = planner.solve(
        system_prompt="one action", user_message="solve", toolkit=toolkit, max_turns=2
    )
    assert result.error is None
    assert toolkit.executed == []
    assert [definition.name for definition in calls[1]] == ["change_query"]
    assert result.stats["tool_calls"] == 1


def test_cameras_overlap_but_observation_order_and_cache_remain_stable(tmp_path):
    primitive = Primitives(tmp_path)
    barrier = threading.Barrier(2)
    original = primitive.segment_queries

    def segment(raw, queries):
        barrier.wait(timeout=2)
        return original(raw, queries)

    primitive.segment_queries = segment
    snapshot = Snapshot(primitive, environment="libero", task="cube", image_size=8)
    snapshot.change_query(["cube"])
    observation = snapshot.observe({})
    assert [view["camera"] for view in observation.data["perception"]] == [
        "agentview",
        "wrist",
    ]
    assert [view["objects"][0]["id"] for view in observation.data["perception"]] == [
        "agentview:1",
        "wrist:1",
    ]
    assert len(primitive.queries) == 2
    snapshot.update_todo(["new plan"])
    cached = snapshot.observe({})
    assert len(primitive.queries) == 2
    assert cached.images == observation.images
    assert cached.data["perception"] == observation.data["perception"]
    assert cached.data["todo"] == ["new plan"]


def test_camera_error_does_not_publish_partial_snapshot_cache(tmp_path):
    primitive = Primitives(tmp_path)
    original = primitive.segment_queries
    barrier = threading.Barrier(2)

    def segment(raw, queries):
        position = barrier.wait(timeout=2)
        if position == 0:
            raise RuntimeError("camera inference failed")
        return original(raw, queries)

    primitive.segment_queries = segment
    snapshot = Snapshot(primitive, environment="libero", task="cube", image_size=8)
    snapshot.change_query(["cube"])
    with pytest.raises(RuntimeError, match="camera inference failed"):
        snapshot.observe({})
    assert snapshot._cache_key is None
    primitive.segment_queries = original
    recovered = snapshot.observe({})
    assert len(recovered.images) == 2
    assert all(view["objects"] for view in recovered.data["perception"])
