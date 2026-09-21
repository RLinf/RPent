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

"""Offline integration tests against the installed SDK and Harness, not loop mocks."""

from __future__ import annotations

import asyncio
import copy
import json
import queue
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest
from pydantic_ai import (
    BinaryContent,
    ModelRequest,
    TextPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import DeltaThinkingPart, DeltaToolCall, FunctionModel
from pydantic_ai.models.test import TestModel

from rpent.dashboard.events import RunStartedEvent, TranscriptEvent, UsageEvent
from rpent.dashboard.state import DashboardState
from rpent.planner.api_loop import ApiAgentLoop
from rpent.session import EnvState
from rpent.tools import common
from rpent.tools.toolkit import Toolkit, readonly

FINISH_ARGS = {"status": "success", "summary": "done"}


class Events:
    enabled = True

    def __init__(self):
        self.events = []

    def emit(self, event):
        self.events.append(event)


class RobotToolkit(Toolkit):
    def __init__(self, events, state=None):
        self.calls = []
        super().__init__(dashboard_events=events, memory=SimpleNamespace(), state=state)

    def _register_common_tools(self):
        self.add_tool("finish", common.TOOLS_SPEC[-1], self.finish)
        self.register("observe", lambda: {"position": 1, "_image_bytes": b"image"})

    def register(self, name, handler, schema=None):
        self.add_tool(
            name,
            {
                "name": name,
                "description": name,
                "input_schema": schema or {"type": "object", "properties": {}},
            },
            readonly(handler),
        )

    @readonly
    def finish(self, status, summary):
        return {"_finish": True, "status": status, "summary": summary}

    def execute_tool(self, name, input_dict):
        self.calls.append((name, input_dict))
        return super().execute_tool(name, input_dict)


@pytest.fixture(autouse=True)
def local_tools(monkeypatch):
    monkeypatch.setattr("rpent.tools.toolkit.substitute", lambda value: value)
    monkeypatch.setenv("PYDANTIC_AI_NO_BANNER", "1")


def tool(name, args=None, index=0):
    return {index: DeltaToolCall(name=name, json_args=json.dumps(args or {}))}


def finish():
    return tool("finish", FINISH_ARGS)


def solve(tmp_path, model, toolkit=None, events=None, **kwargs):
    events = events or Events()
    toolkit = toolkit or RobotToolkit(events)
    config = {
        key: kwargs.pop(key)
        for key in (
            "max_tokens",
            "timeout_s",
            "no_images",
            "reasoning_effort",
            "interactive",
        )
        if key in kwargs
    }
    planner = ApiAgentLoop(model=model, dashboard_events=events, **config)
    result = planner.solve(
        system_prompt="Use the robot toolkit. Finish when done.",
        user_message="Do the task.",
        toolkit=toolkit,
        max_turns=kwargs.pop("max_turns", 10),
        **kwargs,
    )
    return result, toolkit, events


def dashboard(tmp_path):
    state = DashboardState(
        output_dir=tmp_path,
        dashboard_spec={
            "task": {
                "command": "/rpent-task",
                "usage": "/rpent-task <seed>",
                "fields": ({"name": "seed", "kind": "integer", "minimum": 0},),
                "display": "{seed}",
                "output_slug": "s{seed}",
            },
            "runtime_components": (),
            "frame_channels": (),
            "primitives": (),
        },
    )
    state.shared_services_ready()
    state.submit_input("/rpent-task 0")
    assert state.wait_for_task(timeout=0) is not None
    state.emit(RunStartedEvent())
    return state


def user_texts(messages):
    return [
        part.content
        for message in messages
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, UserPromptPart)
    ]


def test_finish_returns_transcript_for_rpent_without_step_files(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    model = TestModel(call_tools=[], custom_output_args=FINISH_ARGS)
    result, toolkit, _ = solve(tmp_path, model)
    assert result.error is None
    assert result.finish_result == {"_finish": True, **FINISH_ARGS}
    assert toolkit.calls == [("finish", FINISH_ARGS)]
    assert result.stats["turns_used"] == 1
    assert [m["role"] for m in result.messages] == ["user", "assistant", "tool"]
    assert result.messages[0] == {"role": "user", "content": "Do the task."}
    call = result.messages[1]["content"][0]
    assert call["name"] == "finish"
    assert call["input"] == FINISH_ARGS
    returned = result.messages[2]
    assert returned["tool_call_id"] == call["id"]
    assert json.loads(returned["content"]) == {"_finish": True, **FINISH_ARGS}
    assert not list(tmp_path.iterdir())
    assert model.last_model_request_parameters.output_tools[0].name == "finish"
    assert "finish" not in [
        t.name for t in model.last_model_request_parameters.function_tools
    ]


def test_refused_finish_retries_without_claiming_success(tmp_path):
    events = Events()
    toolkit = RobotToolkit(events)
    attempts = []

    def guarded_finish(**args):
        attempts.append(args)
        if len(attempts) <= 2:
            return {"error": "finish refused; verify the environment"}
        return {"_finish": True, **args}

    toolkit.register("finish", guarded_finish, common.TOOLS_SPEC[-1]["input_schema"])
    histories = []

    async def stream(messages, info):
        histories.append(copy.deepcopy(messages))
        yield finish()

    result, _, _ = solve(
        tmp_path, FunctionModel(stream_function=stream), toolkit, events
    )
    assert result.error is None
    assert len(attempts) == 3
    assert result.stats["turns_used"] == 3
    assert result.finish_result == {"_finish": True, **FINISH_ARGS}
    assert "finish refused" in repr(histories[1])
    assert "finish refused" in json.dumps(result.messages)


def test_persistent_finish_refusal_stops_at_request_budget(tmp_path):
    events = Events()
    toolkit = RobotToolkit(events)
    toolkit.register(
        "finish",
        lambda **args: {"error": "finish refused"},
        common.TOOLS_SPEC[-1]["input_schema"],
    )

    async def stream(messages, info):
        yield finish()

    result, _, _ = solve(
        tmp_path, FunctionModel(stream_function=stream), toolkit, events, max_turns=3
    )
    assert result.finish_result is None
    assert "UsageLimitExceeded" in result.error
    assert result.stats["turns_used"] == 3
    assert len(toolkit.calls) == 3


def test_anthropic_request_retains_all_prompt_cache_controls(tmp_path, monkeypatch):
    from pydantic_ai.models.anthropic import AnthropicModel
    from pydantic_ai.providers.anthropic import AnthropicProvider

    model = AnthropicModel(
        "claude-sonnet-4-5", provider=AnthropicProvider(api_key="offline-test")
    )
    requests = []

    async def capture(**kwargs):
        requests.append(kwargs)
        raise RuntimeError("offline request captured")

    monkeypatch.setattr(model.client.beta.messages, "create", capture)
    result, _, _ = solve(tmp_path, model, max_tokens=321)
    assert result.error == "RuntimeError: offline request captured"
    assert len(requests) == 1
    request = requests[0]
    assert request["max_tokens"] == 321
    assert request["system"][0]["text"] == "Use the robot toolkit. Finish when done."
    assert request["system"][-1]["cache_control"]["type"] == "ephemeral"
    assert request["tools"][-1]["cache_control"]["type"] == "ephemeral"
    assert (
        request["messages"][-1]["content"][-1]["cache_control"]["type"] == "ephemeral"
    )
    assert request["tool_choice"]["disable_parallel_tool_use"] is True


@pytest.mark.parametrize("no_images", [False, True])
def test_multimodal_tool_results_and_dashboard_events(tmp_path, no_images):
    histories = []

    async def stream(messages, info):
        histories.append(copy.deepcopy(messages))
        if len(histories) == 1:
            yield {0: DeltaThinkingPart(content="Inspect the scene.")}
            yield tool("observe", index=1)
        else:
            yield "Ready."
            yield finish()

    result, _, events = solve(
        tmp_path, FunctionModel(stream_function=stream), no_images=no_images
    )
    assert result.error is None
    returns = [
        part
        for message in histories[1]
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, ToolReturnPart)
    ]
    images = [
        item
        for part in returns
        for item in part.content
        if isinstance(item, BinaryContent)
    ]
    assert bool(images) is not no_images
    if images:
        assert images[0].data == b"image"
    else:
        assert "Image omitted" in repr(returns)
    transcript = [e.payload for e in events.events if isinstance(e, TranscriptEvent)]
    assert {p["type"] for p in transcript} >= {
        "user",
        "thinking",
        "text",
        "tool_call",
        "tool_result",
    }
    assert all("aW1hZ2U=" not in str(p) for p in transcript)
    usage = [event for event in events.events if isinstance(event, UsageEvent)][-1]
    assert usage.tool_calls == 2
    assert result.stats["total_input_tokens"] == usage.inp > 0
    assert result.stats["total_output_tokens"] == usage.out > 0
    serialized = json.dumps(result.messages)
    assert "Inspect the scene." in serialized
    assert "Ready." in serialized
    assert "aW1hZ2U=" not in serialized
    assert any(m.get("name") == "observe" for m in result.messages)


@pytest.fixture
def image_state(tmp_path):
    state = EnvState(tmp_path / "state")
    for value in (0, 255):
        with state.record_step(state={}):
            for name in ("camera.png", "camera.jpg", "camera.jpeg"):
                image = np.full((4, 4, 3), value, dtype=np.uint8)
                assert state.save(name, image) == name
    assert state.save("notes.json", {"position": 1}, step=0) == "notes.json"
    return state


@pytest.mark.parametrize("no_images", [False, True])
@pytest.mark.parametrize(
    "name,step,media_type",
    [
        ("camera.png", None, "image/png"),
        ("camera.jpg", 0, "image/jpeg"),
        ("camera.jpeg", 1, "image/jpeg"),
    ],
)
def test_read_image_returns_saved_artifact_and_records_call(
    tmp_path, image_state, monkeypatch, no_images, name, step, media_type
):
    events = Events()
    toolkit = RobotToolkit(events, state=image_state)
    arguments = {"name": name}
    if step is not None:
        arguments["step"] = step
    resolved_step = image_state.latest_step if step is None else step
    expected_bytes = image_state.load_bytes(name, step=resolved_step)
    histories = []

    if no_images:

        def unexpected_read(*args, **kwargs):
            pytest.fail("--no-images should not read image bytes")

        monkeypatch.setattr(image_state, "load_bytes", unexpected_read)

    async def stream(messages, info):
        histories.append(copy.deepcopy(messages))
        yield tool("read_image", arguments) if len(histories) == 1 else finish()

    result, _, _ = solve(
        tmp_path,
        FunctionModel(stream_function=stream),
        toolkit,
        events,
        no_images=no_images,
    )
    assert result.error is None
    assert result.finish_result == {"_finish": True, **FINISH_ARGS}
    returned = next(
        part
        for message in histories[1]
        for part in message.parts
        if isinstance(part, ToolReturnPart) and part.tool_name == "read_image"
    )
    assert returned.content["artifact"] == name
    assert returned.content["step"] == resolved_step
    images = [
        item
        for message in histories[1]
        for part in message.parts
        if isinstance(part, (ToolReturnPart, UserPromptPart))
        and isinstance(part.content, list)
        for item in part.content
        if isinstance(item, BinaryContent)
    ]
    if no_images:
        assert not images
        assert "--no-images" in returned.content["notice"]
    else:
        assert len(images) == 1
        assert images[0].data == expected_bytes
        assert images[0].media_type == media_type
    assert result.stats["tool_calls"] == 2
    recorded = next(m for m in result.messages if m.get("name") == "read_image")
    assert recorded["tool_call_id"] == returned.tool_call_id
    assert json.loads(recorded["content"]) == returned.content
    transcript = [e.payload for e in events.events if isinstance(e, TranscriptEvent)]
    assert [p["type"] for p in transcript if p.get("tool") == "read_image"] == [
        "tool_call",
        "tool_result",
    ]


@pytest.mark.parametrize(
    "name,step,error",
    [
        ("missing.png", 0, "not available"),
        ("notes.json", 0, "not an image"),
        ("../camera.png", 0, "base filename"),
        ("camera.png", 99, "not present"),
        ("unregistered.png", 0, "not available"),
        ("camera.png", 0, "not available"),
    ],
)
def test_read_image_errors_reach_model_without_ending_run(
    tmp_path, image_state, name, step, error
):
    events = Events()
    toolkit = RobotToolkit(events, state=image_state)
    image_state.artifact_path("camera.png", step=0).unlink()
    unregistered = image_state.artifact_path("unregistered.png", step=0)
    unregistered.parent.mkdir()
    unregistered.write_bytes(b"not registered in the step")
    histories = []

    async def stream(messages, info):
        histories.append(copy.deepcopy(messages))
        if len(histories) == 1:
            yield tool("read_image", {"name": name, "step": step})
        else:
            yield finish()

    result, _, _ = solve(
        tmp_path, FunctionModel(stream_function=stream), toolkit, events
    )
    assert result.error is None
    assert result.finish_result == {"_finish": True, **FINISH_ARGS}
    returned = next(
        part
        for message in histories[1]
        for part in message.parts
        if isinstance(part, ToolReturnPart) and part.tool_name == "read_image"
    )
    assert error in returned.content["error"]


def test_schema_validation_and_sequential_physical_tools(tmp_path):
    events = Events()
    toolkit = RobotToolkit(events)
    positions = []
    toolkit.register(
        "move",
        lambda position: positions.append(position) or {"position": position},
        {
            "type": "object",
            "properties": {"position": {"type": "integer"}},
            "required": ["position"],
        },
    )
    calls = 0

    async def stream(messages, info):
        nonlocal calls
        calls += 1
        if calls == 1:
            yield tool("move", {"position": "bad"})
        elif calls == 2:
            yield tool("move", {"position": 1}) | tool("move", {"position": 2}, 1)
        else:
            yield finish()

    result, _, _ = solve(
        tmp_path, FunctionModel(stream_function=stream), toolkit, events
    )
    assert result.error is None
    assert positions == [1, 2]
    assert len(toolkit.calls) == 3  # Only validated calls reach the Toolkit.
    assert [m["name"] for m in result.messages if m["role"] == "tool"] == [
        "move",
        "move",
        "finish",
    ]


def test_finish_skips_sibling_actions_using_native_end_strategy(tmp_path):
    async def stream(messages, info):
        yield finish() | tool("observe", index=1)

    result, toolkit, _ = solve(tmp_path, FunctionModel(stream_function=stream))
    assert result.error is None
    assert [name for name, _ in toolkit.calls] == ["finish"]
    assert [m["name"] for m in result.messages if m["role"] == "tool"] == ["finish"]


def test_request_budget_is_enforced_by_sdk(tmp_path):
    async def stream(messages, info):
        yield tool("observe")

    result, toolkit, _ = solve(
        tmp_path, FunctionModel(stream_function=stream), max_turns=2
    )
    assert "UsageLimitExceeded" in result.error
    assert result.stats["turns_used"] == 2
    assert len(toolkit.calls) == 2
    assert len([m for m in result.messages if m["role"] == "tool"]) == 2


def test_harness_compacts_long_history(tmp_path):
    lengths = []

    async def stream(messages, info):
        lengths.append(len(messages))
        yield tool("observe") if len(lengths) < 45 else finish()

    result, _, _ = solve(tmp_path, FunctionModel(stream_function=stream), max_turns=50)
    assert result.error is None
    assert any(after < before for before, after in zip(lengths, lengths[1:]))
    assert result.messages[0] == {"role": "user", "content": "Do the task."}
    assert len([m for m in result.messages if m["role"] == "assistant"]) == 45
    assert len([m for m in result.messages if m.get("name") == "observe"]) == 44


def test_dashboard_steering_enters_next_request_in_same_native_run(tmp_path):
    state = dashboard(tmp_path)
    sent = []
    histories = []

    async def stream(messages, info):
        histories.append(copy.deepcopy(messages))
        if len(histories) == 1:
            sent.append(state.submit_input("Look left first."))
            await asyncio.sleep(0.05)
            yield tool("observe")
        else:
            assert "Look left first." in user_texts(messages)
            yield finish()

    result, _, _ = solve(
        tmp_path,
        FunctionModel(stream_function=stream),
        events=state,
        dashboard_interaction=state,
    )
    assert result.error is None
    assert [m["content"] for m in result.messages if m["role"] == "user"] == [
        "Do the task.",
        "Look left first.",
    ]
    assert len(histories) == 2
    assert state.planner_activity == "ended"
    assert state.snapshot()["interaction"]["messages"][0]["status"] == "sent"


def test_dashboard_message_during_finish_is_unsent(tmp_path):
    state = dashboard(tmp_path)
    toolkit = RobotToolkit(state)

    def finish_and_submit(**args):
        state.submit_input("This must not reopen the completed task.")
        time.sleep(0.05)
        return {"_finish": True, **args}

    toolkit.register("finish", finish_and_submit, common.TOOLS_SPEC[-1]["input_schema"])
    result, _, _ = solve(
        tmp_path,
        TestModel(call_tools=[], custom_output_args=FINISH_ARGS),
        toolkit,
        state,
        dashboard_interaction=state,
    )
    assert result.error is None
    assert result.stats["turns_used"] == 1
    assert state.snapshot()["interaction"]["messages"][0]["status"] == "unsent"
    assert "This must not reopen" not in json.dumps(result.messages)


def test_dashboard_interrupt_then_followup_preserves_history(tmp_path):
    state = dashboard(tmp_path)
    calls = 0

    async def stream(messages, info):
        nonlocal calls
        calls += 1
        if calls == 1:
            yield "Starting inspection."
            state.request_interrupt()
            state.submit_input("Continue with the new instruction.")
            await asyncio.sleep(5)
        else:
            assert "Do the task." in user_texts(messages)
            assert "Continue with the new instruction." in user_texts(messages)
            yield finish()

    result, _, _ = solve(
        tmp_path,
        FunctionModel(stream_function=stream),
        events=state,
        dashboard_interaction=state,
        timeout_s=3,
    )
    assert result.error is None
    assert result.finish_result["status"] == "success"
    assert [m["content"] for m in result.messages if m["role"] == "user"] == [
        "Do the task.",
        "Continue with the new instruction.",
    ]
    assert state.snapshot()["interaction"]["messages"][0]["status"] == "sent"


@pytest.fixture(params=[False, True], ids=["normal", "tool-stops-first"])
def dashboard_interrupt_order(request, monkeypatch):
    if not request.param:
        return
    from rpent.planner.api_loop import _Session

    interrupt = _Session.interrupt

    async def interrupt_after_tool_stop(self):
        # Let the SDK reach the queued sibling tool before Dashboard sends its
        # cancellation token, reproducing the physical-drain scheduling race.
        await asyncio.wait_for(self.run_done.wait(), timeout=1)
        return await interrupt(self)

    monkeypatch.setattr(_Session, "interrupt", interrupt_after_tool_stop)


def test_dashboard_interrupt_drains_tool_before_followup(
    tmp_path, dashboard_interrupt_order
):
    state = dashboard(tmp_path)
    toolkit = RobotToolkit(state)
    stopped = threading.Event()
    calls = 0

    def move():
        try:
            state.submit_input("Old instruction 1.")
            state.submit_input("Old instruction 2.")
            while any(
                m["status"] != "sending"
                for m in state.snapshot()["interaction"]["messages"]
            ):
                toolkit.raise_if_cancelled()
                state.wait_for_interaction_change(
                    state.interaction_version, timeout=0.05
                )
            state.request_interrupt()
            state.submit_input("Continue after stopping the move.")
            while True:
                toolkit.raise_if_cancelled()
                time.sleep(0.005)
        finally:
            stopped.set()

    toolkit.register("move", move)

    async def stream(messages, info):
        nonlocal calls
        calls += 1
        if calls == 1:
            yield tool("move") | tool("observe", index=1)
        else:
            assert stopped.is_set()
            assert user_texts(messages) == [
                "Do the task.",
                "Continue after stopping the move.",
            ]
            yield finish()

    result, _, _ = solve(
        tmp_path,
        FunctionModel(stream_function=stream),
        toolkit,
        state,
        dashboard_interaction=state,
        timeout_s=3,
    )
    assert result.error is None
    assert result.finish_result["status"] == "success"
    assert [name for name, _ in toolkit.calls] == ["move", "finish"]
    assert [m["status"] for m in state.snapshot()["interaction"]["messages"]] == [
        "unsent",
        "unsent",
        "sent",
    ]


def test_task_replacement_cancels_and_drains_physical_work(
    tmp_path, dashboard_interrupt_order
):
    state = dashboard(tmp_path)
    toolkit = RobotToolkit(state)
    stopped = threading.Event()

    def move():
        state.submit_input("/rpent-task 1")
        try:
            while True:
                toolkit.raise_if_cancelled()
                time.sleep(0.005)
        finally:
            stopped.set()

    toolkit.register("move", move)

    async def stream(messages, info):
        yield tool("move") | tool("observe", index=1)

    result, _, _ = solve(
        tmp_path,
        FunctionModel(stream_function=stream),
        toolkit,
        state,
        dashboard_interaction=state,
        timeout_s=3,
    )
    assert result.error is None
    assert stopped.is_set()
    assert [name for name, _ in toolkit.calls] == ["move"]
    assert state.planner_activity == "ended"
    assert result.finish_result is None


def test_timeout_drains_physical_work_and_records_partial_history(tmp_path):
    events = Events()
    toolkit = RobotToolkit(events)
    stopped = threading.Event()

    def move():
        try:
            while True:
                toolkit.raise_if_cancelled()
                time.sleep(0.005)
        finally:
            stopped.set()

    toolkit.register("move", move)

    async def stream(messages, info):
        yield tool("move")

    result, _, _ = solve(
        tmp_path, FunctionModel(stream_function=stream), toolkit, events, timeout_s=0.15
    )
    assert "timed out" in result.error
    assert stopped.is_set()
    assert result.messages[0] == {"role": "user", "content": "Do the task."}
    assert result.messages[1]["content"][0]["name"] == "move"
    json.dumps(result.messages)


def test_native_terminal_preserves_completed_actions_after_model_failure(
    tmp_path, monkeypatch
):
    import pydantic_ai._cli as cli

    requests = []
    replies = iter(["Inspect the scene.", "Continue after the error.", "/exit"])

    async def read_prompt(*args, **kwargs):
        assert requests, "The preset task must run before the first input prompt."
        return next(replies)

    async def stream(messages, info):
        requests.append(copy.deepcopy(messages))
        if len(requests) == 2:
            yield tool("observe")
        elif len(requests) == 3:
            raise RuntimeError("provider failed after observation")
        else:
            yield "Ready."

    monkeypatch.setattr(cli, "PYDANTIC_AI_HOME", tmp_path / "cli")
    monkeypatch.setattr(
        cli, "PromptSession", lambda **kwargs: SimpleNamespace(prompt_async=read_prompt)
    )
    result, toolkit, _ = solve(
        tmp_path, FunctionModel(stream_function=stream), interactive=True
    )
    assert result.error is None
    assert any(
        isinstance(part, TextPart) and part.content == "Ready."
        for message in requests[-1]
        for part in message.parts
    )
    assert user_texts(requests[-1]) == [
        "Do the task.",
        "Inspect the scene.",
        "Continue after the error.",
    ]
    returns = [
        part
        for message in requests[-1]
        for part in message.parts
        if isinstance(part, ToolReturnPart) and part.tool_name == "observe"
    ]
    assert len(returns) == 1
    assert "position" in str(returns[0].content)
    assert any(isinstance(item, BinaryContent) for item in returns[0].content)
    assert toolkit.calls == [("observe", {})]
    assert [m["content"] for m in result.messages if m["role"] == "user"] == user_texts(
        requests[-1]
    )


def test_legacy_terminal_queue_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="terminal input queue"):
        solve(tmp_path, TestModel(), input_queue=queue.Queue())


def test_native_terminal_and_dashboard_cannot_run_together(tmp_path):
    with pytest.raises(ValueError, match="clai and Dashboard cannot run together"):
        solve(tmp_path, TestModel(), interactive=True, dashboard_interaction=object())


@pytest.mark.parametrize("after_tool", [False, True])
def test_model_failure_preserves_rpent_transcript(tmp_path, after_tool):
    calls = 0

    async def stream(messages, info):
        nonlocal calls
        calls += 1
        if after_tool and calls == 1:
            yield tool("observe")
            return
        raise RuntimeError("provider unavailable")

    result, _, _ = solve(tmp_path, FunctionModel(stream_function=stream))
    assert result.error == "RuntimeError: provider unavailable"
    if after_tool:
        assert [m["role"] for m in result.messages] == ["user", "assistant", "tool"]
        assert result.messages[-1]["name"] == "observe"
    else:
        assert result.messages == [{"role": "user", "content": "Do the task."}]
    json.dumps(result.messages)


def test_factory_passes_interactive_mode(tmp_path, monkeypatch):
    from rpent.planner.base import Planner, build_planner

    model = TestModel(call_tools=[], custom_output_args=FINISH_ARGS)
    monkeypatch.setattr("rpent.planner.base.build_api_model", lambda *args: model)
    planner = build_planner(
        "api",
        output_dir=tmp_path,
        recipe_tag="test",
        robot_name="test",
        model="test:test",
        dashboard_events=Events(),
        interactive=True,
    )
    assert isinstance(planner, ApiAgentLoop)
    assert isinstance(planner, Planner)
    assert planner.interactive is True
