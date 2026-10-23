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
import math
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
from pydantic_ai.exceptions import UnexpectedModelBehavior, UsageLimitExceeded
from pydantic_ai.models.function import DeltaThinkingPart, DeltaToolCall, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_core import to_jsonable_python

from rpent.dashboard.events import RunStartedEvent, TranscriptEvent, UsageEvent
from rpent.dashboard.state import DashboardState
from rpent.planner.api_loop import ApiAgentLoop
from rpent.session import EnvState, Payload
from rpent.tools import Toolkit, ToolResult, common
from rpent.tools import tool as declare_tool

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
        self.add_tool(common.CommonTools.finish.with_handler(self.finish))
        self.register(
            "observe", lambda: ToolResult(data={"position": 1}, images=[b"image"])
        )

    def register(self, name, handler, declaration=None):
        declaration = declaration or declare_tool(name=name, readonly=True)(handler)

        def run(**kwargs):
            result = handler(**kwargs)
            return result if isinstance(result, ToolResult) else ToolResult(data=result)

        self.add_tool(declaration.with_handler(run), replace=name in self._tools)

    def finish(self, status, summary):
        return ToolResult(data={"_finish": True, "status": status, "summary": summary})

    def execute_tool(self, name, input_dict):
        self.calls.append((name, input_dict))
        return super().execute_tool(name, input_dict)


@pytest.fixture(autouse=True)
def local_tools(monkeypatch):
    monkeypatch.setattr("rpent.tools.toolkit.substitute", lambda value: value)
    monkeypatch.setattr("rpent.planner.api_loop.substitute", lambda value: value)
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

    toolkit.register("finish", guarded_finish, common.CommonTools.finish)
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
        common.CommonTools.finish,
    )

    async def stream(messages, info):
        yield finish()

    result, _, _ = solve(
        tmp_path, FunctionModel(stream_function=stream), toolkit, events, max_turns=3
    )
    assert result.finish_result is None
    assert result.error is None
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
    assert request["tool_choice"]["disable_parallel_tool_use"] is False


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
    image_state.start_task(instruction="Read the image.", config=Payload("test", {}))
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
    call = image_state.planner_records()[0].tool_calls[0]
    assert call.name == "read_image" and call.status == "completed"
    assert call.feedback.data["return_value"] == returned.content
    assert call.raw_result.data == call.feedback.data
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
    toolkit.close()
    image_state.end_task(status="completed", stop_reason=result.stop_reason)
    root = image_state.save_planner_trajectory()
    saved = [
        json.loads(line) for line in (root / "planners.jsonl").read_text().splitlines()
    ]
    restored = restore_byte_artifacts(saved, root)
    assert restored[0]["tool_calls"][0]["raw_result"]["data"] == call.raw_result.data
    assert restored[0]["tool_calls"][0]["feedback"]["data"] == call.feedback.data
    assert len(list((root / "artifacts").iterdir())) == (0 if no_images else 1)
    if not no_images:
        assert call.feedback.data["content"][0]["data"] == expected_bytes


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
    toolkit = RobotToolkit(events, state=EnvState(tmp_path))
    toolkit.state.start_task(instruction="Do the task.", config=Payload("test", {}))
    toolkit.get_env_state = lambda **kwargs: ToolResult(data=kwargs["result"])
    positions = []

    @declare_tool
    def move(position: int) -> ToolResult:
        positions.append(position)
        return ToolResult(data={"position": position})

    toolkit.add_tool(move)
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
    planners = toolkit.state.planner_records()
    assert [len(record.tool_calls) for record in planners] == [1, 2, 1]
    refused = planners[0].tool_calls[0]
    assert json.loads(refused.arguments.data) == {"position": "bad"}
    assert refused.status == "not_executed" and refused.started_at is None
    assert refused.feedback.format == "pydantic_ai.model_retry"
    assert all(call.status == "completed" for call in planners[1].tool_calls)
    assert len(toolkit.calls) == 3  # Only validated calls reach the Toolkit.
    assert [m["name"] for m in result.messages if m["role"] == "tool"] == [
        "move",
        "move",
        "finish",
    ]


def test_finish_skips_sibling_actions_using_native_end_strategy(tmp_path):
    toolkit = capturing_toolkit(tmp_path)

    async def stream(messages, info):
        yield finish() | tool("observe", index=1)

    result, toolkit, _ = solve(tmp_path, FunctionModel(stream_function=stream), toolkit)
    assert result.error is None
    calls = toolkit.state.planner_records()[0].tool_calls
    assert [call.name for call in calls] == ["finish", "observe"]
    assert [call.status for call in calls] == ["completed", "not_executed"]
    assert calls[1].started_at is None and calls[1].raw_result is None
    assert calls[1].missing["raw_result"].reason == "not_applicable"
    assert [name for name, _ in toolkit.calls] == ["finish"]
    assert [m["name"] for m in result.messages if m["role"] == "tool"] == ["finish"]


@pytest.mark.parametrize("mode", ["normal", "terminal", "dashboard"])
def test_request_budget_stops_normally_without_claiming_success(tmp_path, mode):
    async def stream(messages, info):
        yield tool("observe")

    state = dashboard(tmp_path) if mode == "dashboard" else None
    result, toolkit, _ = solve(
        tmp_path,
        FunctionModel(stream_function=stream),
        events=state,
        max_turns=2,
        interactive=mode == "terminal",
        dashboard_interaction=state,
    )
    assert result.error is None
    assert result.finish_result is None
    assert result.stats["turns_used"] == 2
    assert len(toolkit.calls) == 2
    assert len([m for m in result.messages if m["role"] == "tool"]) == 2


@pytest.mark.parametrize(
    "error",
    [
        UsageLimitExceeded("provider-specific limit"),
        UnexpectedModelBehavior("Exceeded max retries"),
    ],
)
def test_other_model_failures_remain_errors(tmp_path, error):
    async def stream(messages, info):
        raise error
        yield  # pragma: no cover

    result, _, _ = solve(tmp_path, FunctionModel(stream_function=stream), max_turns=1)
    assert result.error == f"{type(error).__name__}: {error}"
    assert result.finish_result is None


@pytest.mark.parametrize("interactive", [False, True])
def test_token_limit_truncation_is_marked_for_recovery(
    tmp_path, monkeypatch, interactive
):
    if interactive:
        import pydantic_ai._cli as cli

        replies = iter(["Inspect.", "/exit"])

        async def read_prompt(*args, **kwargs):
            return next(replies)

        monkeypatch.setattr(cli, "PYDANTIC_AI_HOME", tmp_path / "cli")
        monkeypatch.setattr(
            cli,
            "PromptSession",
            lambda **kwargs: SimpleNamespace(prompt_async=read_prompt),
        )

    requests = 0

    async def stream(messages, info):
        nonlocal requests
        requests += 1
        if interactive and requests == 1:
            yield "Ready."
            return
        yield {0: DeltaThinkingPart(content="Reasoning past the output budget.")}
        raise UnexpectedModelBehavior(
            "Model token limit (8192) exceeded before any response was "
            "generated. Increase the `max_tokens` model setting, or simplify "
            "the prompt to result in a shorter response that will fit within "
            "the limit."
        )

    result, _, _ = solve(
        tmp_path,
        FunctionModel(stream_function=stream),
        max_turns=2,
        interactive=interactive,
    )
    assert requests == (2 if interactive else 1)
    assert result.error is not None
    assert result.error.startswith("planner response truncated (length): ")
    assert (
        "Model token limit (8192) exceeded before any response was generated"
        in result.error
    )
    assert result.finish_result is None
    json.dumps(result.messages)


def test_agent_error_recovery_decisions():
    from rpent.planner.base import is_recoverable_agent_error

    truncation = (
        "planner response truncated (length): Model token limit (8192) "
        "exceeded before any response was generated"
    )
    assert is_recoverable_agent_error("planner timed out after 1800s")
    assert is_recoverable_agent_error(truncation)
    assert not is_recoverable_agent_error(None)
    assert not is_recoverable_agent_error("RuntimeError: provider unavailable")


def test_harness_compacts_long_history(tmp_path):
    toolkit = capturing_toolkit(tmp_path)
    lengths = []
    requests = []

    async def stream(messages, info):
        lengths.append(len(messages))
        from pydantic_ai.messages import ModelMessagesTypeAdapter

        requests.append(ModelMessagesTypeAdapter.dump_python(messages, mode="json"))
        yield tool("observe") if len(lengths) < 45 else finish()

    result, _, _ = solve(
        tmp_path, FunctionModel(stream_function=stream), toolkit, max_turns=50
    )
    assert result.error is None
    assert (
        to_jsonable_python(
            [p.request.data["messages"] for p in toolkit.state.planner_records()],
            bytes_mode="base64",
        )
        == requests
    )
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

    toolkit.register("finish", finish_and_submit, common.CommonTools.finish)
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
    toolkit = capturing_toolkit(tmp_path / "captures", state)
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
        toolkit=toolkit,
        events=state,
        dashboard_interaction=state,
        timeout_s=3,
    )
    assert result.error is None
    assert result.stop_reason == "finish"
    records = toolkit.state.planner_records()
    assert [p.status for p in records] == ["cancelled", "completed"]
    assert records[0].response.completeness == "partial"
    assert (
        records[0].source_ids["pydantic_ai.run_id"]
        != records[1].source_ids["pydantic_ai.run_id"]
    )
    assert toolkit.state.get_task().status == "running"
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
    toolkit = RobotToolkit(state, state=EnvState(tmp_path / "observations"))
    toolkit.state.start_task(instruction="Do the task.", config=Payload("test", {}))
    toolkit.get_env_state = lambda **kwargs: ToolResult(data=kwargs["result"])
    observations = []
    toolkit.register("observe", lambda: observations.append("ran") or ToolResult())
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
            with toolkit._scheduler.condition:
                assert toolkit._scheduler.condition.wait_for(
                    lambda: bool(toolkit._scheduler.pending), timeout=2
                )
            state.request_interrupt()
            state.submit_input("Continue after stopping the move.")
            while True:
                toolkit.raise_if_cancelled()
                time.sleep(0.005)
        finally:
            stopped.set()

    toolkit.register("move", move, declaration=declare_tool(name="move")(move))

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
    assert observations == []
    assert toolkit.calls[-1][0] == "finish"
    assert [m["status"] for m in state.snapshot()["interaction"]["messages"]] == [
        "unsent",
        "unsent",
        "sent",
    ]


def test_task_replacement_cancels_and_drains_physical_work(
    tmp_path, dashboard_interrupt_order
):
    state = dashboard(tmp_path)
    toolkit = RobotToolkit(state, state=EnvState(tmp_path / "observations"))
    toolkit.state.start_task(instruction="Do the task.", config=Payload("test", {}))
    toolkit.get_env_state = lambda **kwargs: ToolResult(data=kwargs["result"])
    observations = []
    toolkit.register("observe", lambda: observations.append("ran") or ToolResult())
    stopped = threading.Event()

    def move():
        with toolkit._scheduler.condition:
            assert toolkit._scheduler.condition.wait_for(
                lambda: bool(toolkit._scheduler.pending), timeout=2
            )
        state.submit_input("/rpent-task 1")
        try:
            while True:
                toolkit.raise_if_cancelled()
                time.sleep(0.005)
        finally:
            stopped.set()

    toolkit.register("move", move, declaration=declare_tool(name="move")(move))

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
    calls = toolkit.state.planner_records()[0].tool_calls
    assert [call.status for call in calls] == ["cancelled", "not_executed"]
    assert calls[0].raw_result.data["data"]["code"] == "tool_cancelled"
    assert calls[1].started_at is None
    assert observations == []
    assert state.planner_activity == "ended"
    assert result.finish_result is None


def test_timeout_drains_physical_work_and_records_partial_history(tmp_path):
    events = Events()
    toolkit = capturing_toolkit(tmp_path, events)
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
    assert result.stop_reason == "timeout"
    call = toolkit.state.planner_records()[0].tool_calls[0]
    assert call.status == "cancelled" and call.ended_at is not None
    assert call.raw_result.data["data"]["code"] == "tool_cancelled"
    assert call.feedback is None and call.missing["feedback"].reason == "not_recorded"
    assert stopped.is_set()
    assert result.messages[0] == {"role": "user", "content": "Do the task."}
    assert result.messages[1]["content"][0]["name"] == "move"
    json.dumps(result.messages)


def test_native_terminal_preserves_completed_actions_after_model_failure(
    tmp_path, monkeypatch
):
    import pydantic_ai._cli as cli

    toolkit = capturing_toolkit(tmp_path / "captures")
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
        tmp_path, FunctionModel(stream_function=stream), toolkit, interactive=True
    )
    assert result.error is None
    records = toolkit.state.planner_records()
    assert [p.status for p in records] == [
        "completed",
        "completed",
        "failed",
        "completed",
    ]
    assert records[1].tool_calls[0].status == "completed"
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


def test_native_terminal_followups_share_budget_and_clear_recovered_error(
    tmp_path, monkeypatch, capsys
):
    import pydantic_ai._cli as cli

    requests = []
    replies = iter(["Inspect.", "Retry.", "Try after the budget.", "/exit"])

    async def read_prompt(*args, **kwargs):
        return next(replies)

    async def stream(messages, info):
        requests.append(copy.deepcopy(messages))
        if len(requests) == 1:
            yield "Ready."
        elif len(requests) == 2:
            raise RuntimeError("temporary provider failure")
        else:
            yield tool("observe")

    monkeypatch.setattr(cli, "PYDANTIC_AI_HOME", tmp_path / "cli")
    monkeypatch.setattr(
        cli, "PromptSession", lambda **kwargs: SimpleNamespace(prompt_async=read_prompt)
    )
    result, toolkit, _ = solve(
        tmp_path, FunctionModel(stream_function=stream), interactive=True, max_turns=3
    )
    assert len(requests) == result.stats["turns_used"] == 3
    assert toolkit.calls == [("observe", {})]
    assert result.error is None
    assert result.finish_result is None
    assert any(m.get("name") == "observe" for m in result.messages)
    output = " ".join(capsys.readouterr().out.split())
    assert "Use /exit to close the session." in output


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


def test_offline_http_planner_executes_action_and_finish(tmp_path):
    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.openai import OpenAIProvider

    from tests.e2e_tests.offline_planner_server import (
        OFFLINE_MODEL_NAME,
        OfflinePlannerServer,
        ScriptedToolCall,
    )

    events = Events()
    toolkit = capturing_toolkit(tmp_path, events)
    toolkit.register("move", lambda: {"steps": 1})
    script = (
        ScriptedToolCall("move", {}),
        ScriptedToolCall("finish", FINISH_ARGS),
    )
    with OfflinePlannerServer(script) as server:
        model = OpenAIChatModel(
            OFFLINE_MODEL_NAME,
            provider=OpenAIProvider(base_url=server.base_url, api_key="offline-test"),
        )
        result, _, _ = solve(
            tmp_path, model, toolkit=toolkit, events=events, max_turns=2
        )
        assert result.error is None
        assert toolkit.calls == [("move", {}), ("finish", FINISH_ARGS)]
        assert result.finish_result == {"_finish": True, **FINISH_ARGS}
        assert len(toolkit.state.planner_records()) == 2
        assert all(
            p.source_ids.get("provider.response_id")
            for p in toolkit.state.planner_records()
        )
        server.assert_complete()


def test_offline_http_planner_preserves_non_streaming_responses():
    from openai import OpenAI

    from tests.e2e_tests.offline_planner_server import (
        OFFLINE_MODEL_NAME,
        OfflinePlannerServer,
        ScriptedToolCall,
    )

    call = ScriptedToolCall("finish", FINISH_ARGS)
    with OfflinePlannerServer((call,)) as server:
        with OpenAI(base_url=server.base_url, api_key="offline-test") as client:
            response = client.chat.completions.create(
                model=OFFLINE_MODEL_NAME,
                messages=[{"role": "user", "content": "Finish the task."}],
                tools=[
                    {
                        "type": "function",
                        "function": {
                            "name": "finish",
                            "parameters": common.CommonTools.finish.input_schema,
                        },
                    }
                ],
                stream=False,
            )
        returned = response.choices[0].message.tool_calls[0]
        assert returned.function.name == "finish"
        assert json.loads(returned.function.arguments) == FINISH_ARGS
        server.assert_complete()


@pytest.mark.parametrize("effort", ["none", "high"])
def test_reasoning_effort_reaches_openai_chat_request_body(tmp_path, effort):
    """Serialize ``--reasoning-effort`` into the outgoing Chat Completions body."""
    import httpx
    from openai import AsyncOpenAI
    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.openai import OpenAIProvider

    bodies = []

    def completion_for(body):
        return {
            "id": "chatcmpl-272",
            "object": "chat.completion",
            "created": 0,
            "model": body.get("model", "deepseek-v4.1-flash"),
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_272",
                                "type": "function",
                                "function": {
                                    "name": "finish",
                                    "arguments": json.dumps(FINISH_ARGS),
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }

    def handle(request):
        body = json.loads(request.content)
        bodies.append(body)
        completion = completion_for(body)
        if body.get("stream"):
            metadata = {key: completion[key] for key in ("id", "created", "model")}
            metadata["object"] = "chat.completion.chunk"
            message = completion["choices"][0]["message"]
            delta = {
                **message,
                "tool_calls": [
                    {"index": index, **call}
                    for index, call in enumerate(message["tool_calls"])
                ],
            }
            chunks = [
                {
                    **metadata,
                    "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
                },
                {
                    **metadata,
                    "choices": [
                        {"index": 0, "delta": {}, "finish_reason": "tool_calls"}
                    ],
                },
                {**metadata, "choices": [], "usage": completion["usage"]},
            ]
            sse = (
                "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks)
                + "data: [DONE]\n\n"
            )
            return httpx.Response(
                200, content=sse.encode(), headers={"Content-Type": "text/event-stream"}
            )
        return httpx.Response(200, json=completion)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handle))
    sdk = AsyncOpenAI(
        api_key="offline-only",
        base_url="https://api.deepseek.com/v1",
        http_client=client,
    )
    model = OpenAIChatModel(
        "deepseek-v4.1-flash", provider=OpenAIProvider(openai_client=sdk)
    )
    assert model.profile.get("supports_thinking") is not True

    events = Events()
    result, toolkit, _ = solve(
        tmp_path, model, events=events, reasoning_effort=effort, max_turns=2
    )
    assert result.error is None
    assert toolkit.calls == [("finish", FINISH_ARGS)]
    assert len(bodies) == 1
    assert bodies[0]["reasoning_effort"] == effort


def test_api_executes_readonly_calls_concurrently(tmp_path):
    from weakref import ref

    from rpent.session.capture import current_tool_call

    events = Events()
    toolkit = capturing_toolkit(tmp_path, events)
    rendezvous = threading.Barrier(2)
    completed = []
    captures = []

    @declare_tool(readonly=True)
    def sense(label: str) -> ToolResult:
        captures.append(ref(current_tool_call.get()))
        rendezvous.wait(timeout=5)
        completed.append(label)
        return ToolResult(data={"label": label})

    toolkit.add_tool(sense)
    requests = 0

    async def stream(messages, info):
        nonlocal requests
        requests += 1
        if requests == 1:
            yield tool("sense", {"label": "left"}) | tool(
                "sense", {"label": "right"}, 1
            )
        else:
            assert all(capture() is None for capture in captures)
            yield finish()

    result, _, _ = solve(
        tmp_path, FunctionModel(stream_function=stream), toolkit, events
    )
    assert result.error is None
    assert sorted(completed) == ["left", "right"]
    calls = toolkit.state.planner_records()[0].tool_calls
    assert [c.raw_result.data["data"]["label"] for c in calls] == ["left", "right"]
    assert all(c.status == "completed" for c in calls)
    assert result.finish_result == {"_finish": True, **FINISH_ARGS}
    assert result.stats["tool_calls"] == 3


def capturing_toolkit(tmp_path, events=None):
    state = EnvState(tmp_path)
    state.start_task(instruction="Do the task.", config=Payload("test", {}))
    return RobotToolkit(events or Events(), state=state)


def restore_byte_artifacts(value, root):
    if isinstance(value, list):
        return [restore_byte_artifacts(item, root) for item in value]
    if isinstance(value, dict):
        if value.get("$type") == "artifact_ref":
            assert value["kind"] == "bytes"
            return (root / value["path"]).read_bytes()
        return {key: restore_byte_artifacts(item, root) for key, item in value.items()}
    return value


@pytest.mark.parametrize("no_images", [False, True])
def test_capture_keeps_original_tool_result_and_actual_multimodal_feedback(
    tmp_path, no_images
):
    from pydantic_ai.messages import ModelMessagesTypeAdapter

    toolkit = capturing_toolkit(tmp_path)
    original = {"executed_steps": 2, "text": "a" * 61000}
    original_image = b"\x00\xfforiginal image"
    observation_image = b"\x00\xffobservation image"

    @declare_tool
    def move() -> ToolResult:
        return ToolResult(data=original, images=[original_image])

    toolkit.add_tool(move)
    toolkit.get_env_state = lambda **kwargs: ToolResult(
        data={"position": 5}, images=[observation_image]
    )
    seen = []

    async def stream(messages, info):
        seen.append(ModelMessagesTypeAdapter.dump_python(messages, mode="json"))
        yield tool("move") if len(seen) < 3 else finish()

    result, _, _ = solve(
        tmp_path, FunctionModel(stream_function=stream), toolkit, no_images=no_images
    )
    assert result.error is None
    planners = toolkit.state.planner_records()
    assert [p.status for p in planners] == ["completed"] * 3
    assert (
        to_jsonable_python(
            [p.request.data["messages"] for p in planners], bytes_mode="base64"
        )
        == seen
    )
    assert planners[0].request.data["model_request_parameters"]["function_tools"]
    assert planners[0].request.data["model_request_parameters"]["output_tools"]
    call = planners[0].tool_calls[0]
    assert call.raw_result.data == {"data": original, "images": [original_image]}
    assert json.loads(call.feedback.data["return_value"][0]) == {"position": 5}
    image_content = call.feedback.data["return_value"][1:]
    assert len(image_content) == 2
    assert all(
        isinstance(item, str) if no_images else item["kind"] == "binary"
        for item in image_content
    )
    if not no_images:
        assert [item["data"] for item in image_content] == [
            original_image,
            observation_image,
        ]
    assert call.before_observation_id is None
    assert call.missing["before_observation_id"].reason == "not_recorded"
    assert call.started_at <= call.ended_at
    assert "raw_result" not in call.missing and "feedback" not in call.missing
    assert toolkit.state.action_records() == []

    toolkit.close()
    toolkit.state.end_task(status="completed", stop_reason=result.stop_reason)
    root = toolkit.state.save_planner_trajectory()
    saved = [
        json.loads(line) for line in (root / "planners.jsonl").read_text().splitlines()
    ]
    restored = restore_byte_artifacts(saved, root)
    assert (
        to_jsonable_python(
            [p["request"]["data"]["messages"] for p in restored], bytes_mode="base64"
        )
        == seen
    )
    assert [p["response"]["data"] for p in restored] == [
        planner.response.data for planner in planners
    ]
    saved_call = saved[0]["tool_calls"][0]
    assert restored[0]["tool_calls"][0]["feedback"]["data"] == call.feedback.data
    assert saved_call["raw_result"]["data"]["data"] == original
    image_ref = saved_call["raw_result"]["data"]["images"][0]
    assert (root / image_ref["path"]).read_bytes() == original_image
    assert saved[1]["tool_calls"][0]["raw_result"]["data"]["images"][0] == image_ref
    if not no_images:
        assert saved_call["feedback"]["data"]["return_value"][1]["data"] == image_ref
    assert len(list((root / "artifacts").iterdir())) == (1 if no_images else 2)
    assert toolkit.state.planner_records() == planners


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_capture_saves_nonfinite_raw_results_and_serializable_feedback(tmp_path, value):
    toolkit = capturing_toolkit(tmp_path)

    @declare_tool(readonly=True)
    def measure() -> ToolResult:
        return ToolResult(data={"samples": [value, np.float32(value)], "count": 2})

    toolkit.add_tool(measure)
    requests = 0

    async def stream(messages, info):
        nonlocal requests
        requests += 1
        yield tool("measure") if requests == 1 else finish()

    result, _, _ = solve(tmp_path, FunctionModel(stream_function=stream), toolkit)
    assert result.error is None
    toolkit.close()
    toolkit.state.end_task(
        status="completed",
        stop_reason=result.stop_reason,
        result=Payload("test.result", {"score": value}),
    )

    root = toolkit.state.save_planner_trajectory()

    def reject_constant(token):
        pytest.fail(f"Nonstandard JSON constant: {token}")

    planners = [
        json.loads(line, parse_constant=reject_constant)
        for line in (root / "planners.jsonl").read_text().splitlines()
    ]
    saved_call = planners[0]["tool_calls"][0]
    raw = saved_call["raw_result"]["data"]["data"]
    assert raw["count"] == 2
    for sample in raw["samples"]:
        assert sample["$type"] == "float"
        restored = float(sample["value"])
        assert math.isnan(restored) if math.isnan(value) else restored == value
    call = toolkit.state.planner_records()[0].tool_calls[0]
    assert saved_call["status"] == "failed"
    assert saved_call["feedback"]["data"] == call.feedback.data
    assert "serialization failed" in saved_call["feedback"]["data"]["return_value"][0]
    assert not math.isfinite(call.raw_result.data["data"]["samples"][0])
    task = json.loads((root / "task.json").read_text(), parse_constant=reject_constant)
    assert task["task"]["result"]["data"]["score"] == raw["samples"][0]


@pytest.mark.parametrize("partial", [False, True])
def test_capture_keeps_model_failure_and_partial_response(tmp_path, partial):
    toolkit = capturing_toolkit(tmp_path)

    async def stream(messages, info):
        if partial:
            yield "Received part."
        raise RuntimeError("provider connection lost")
        yield "unreachable"

    result, _, _ = solve(tmp_path, FunctionModel(stream_function=stream), toolkit)
    assert "provider connection lost" in result.error
    (record,) = toolkit.state.planner_records()
    assert record.status == "failed" and record.ended_at is not None
    assert record.request is not None
    if partial:
        assert record.response.completeness == "partial"
        assert record.response.data["parts"][0]["content"] == "Received part."
    else:
        assert record.response is None
        assert record.missing["response"].reason == "capture_failed"


def test_capture_non_streaming_model_request(tmp_path):
    from pydantic_ai import ModelResponse, TextPart
    from pydantic_ai.messages import (
        AgentInstructionSource,
        FilePart,
        ImageUrl,
        InstructionId,
        InstructionPart,
        ModelMessagesTypeAdapter,
        ThinkingPart,
    )
    from pydantic_ai.models import ModelRequestParameters

    from rpent.session.capture import CapturingModel

    toolkit = capturing_toolkit(tmp_path)
    image = BinaryContent(
        b"\x00\xff\xfb\xef", media_type="image/png", vendor_metadata={"detail": "high"}
    )
    literal = "AP_77w=="
    messages = [
        ModelRequest(
            parts=[
                UserPromptPart(
                    ["hello", image, ImageUrl("https://example.com/image.png"), literal]
                )
            ],
            metadata={"kind": "binary", "data": literal, "media_type": "image/png"},
        )
    ]
    response = ModelResponse(
        parts=[
            TextPart("done"),
            FilePart(image),
            ThinkingPart("reasoning", signature=literal),
        ],
        provider_response_id="response-1",
    )
    parameters = ModelRequestParameters(
        instruction_parts=[
            InstructionPart("Look.", id=InstructionId(AgentInstructionSource()))
        ]
    )
    model = CapturingModel(
        FunctionModel(lambda messages, info: response),
        toolkit.state,
        "conversation-1",
    )
    asyncio.run(
        model.request(
            messages,
            {"max_tokens": 10},
            parameters,
        )
    )
    (record,) = toolkit.state.planner_records()
    assert record.status == "completed"
    assert record.request.data["model_settings"] == {"max_tokens": 10}
    assert (
        record.request.data["model_request_parameters"]["instruction_parts"][0]["id"]
        == "agent"
    )
    assert record.response.data["parts"][0]["content"] == "done"
    assert record.source_ids == {
        "pydantic_ai.conversation_id": "conversation-1",
        "provider.response_id": "response-1",
    }
    toolkit.state.end_task(status="completed", stop_reason="model_output")
    root = toolkit.state.save_planner_trajectory()
    saved = json.loads((root / "planners.jsonl").read_text())
    restored = restore_byte_artifacts(saved, root)
    restored_messages = ModelMessagesTypeAdapter.validate_python(
        restored["request"]["data"]["messages"]
    )
    restored_response = ModelMessagesTypeAdapter.validate_python(
        [restored["response"]["data"]]
    )
    assert ModelMessagesTypeAdapter.dump_python(
        restored_messages, mode="json"
    ) == ModelMessagesTypeAdapter.dump_python(messages, mode="json")
    assert ModelMessagesTypeAdapter.dump_python(
        restored_response, mode="json"
    ) == ModelMessagesTypeAdapter.dump_python([response], mode="json")
    assert len(list((root / "artifacts").iterdir())) == 1
    assert toolkit.state.planner_records() == [record]


@pytest.mark.parametrize("name,arguments", [("observe", "{"), ("missing_tool", "{}")])
def test_capture_keeps_calls_rejected_by_sdk(tmp_path, name, arguments):
    toolkit = capturing_toolkit(tmp_path)
    requests = 0

    async def stream(messages, info):
        nonlocal requests
        requests += 1
        if requests == 1:
            yield {0: DeltaToolCall(name=name, json_args=arguments)}
        else:
            yield finish()

    result, _, _ = solve(tmp_path, FunctionModel(stream_function=stream), toolkit)
    assert result.error is None
    call = toolkit.state.planner_records()[0].tool_calls[0]
    assert call.name == name and call.arguments.data == arguments
    assert call.status == "not_executed" and call.started_at is None
    assert call.raw_result is None and call.feedback is None
    assert call.missing["feedback"].reason == "not_applicable"
    retry_request = toolkit.state.planner_records()[1].request.data
    assert any(
        part["part_kind"] == "retry-prompt"
        for message in retry_request["messages"]
        for part in message["parts"]
    )


def test_image_reader_drains_before_capture_finishes_on_timeout(
    tmp_path, image_state, monkeypatch
):
    image_state.start_task(instruction="Read image.", config=Payload("test", {}))
    toolkit = RobotToolkit(Events(), state=image_state)
    load = image_state.load_bytes
    done = threading.Event()

    def slow_load(*args, **kwargs):
        time.sleep(0.2)
        result = load(*args, **kwargs)
        done.set()
        return result

    monkeypatch.setattr(image_state, "load_bytes", slow_load)

    async def stream(messages, info):
        yield tool("read_image", {"name": "camera.png"})

    result, _, _ = solve(
        tmp_path, FunctionModel(stream_function=stream), toolkit, timeout_s=0.1
    )
    assert result.stop_reason == "timeout" and done.is_set()
    call = image_state.planner_records()[0].tool_calls[0]
    assert call.status == "completed" and call.raw_result is not None
    assert call.feedback is None
    assert call.missing["feedback"].reason == "not_recorded"


def test_interrupted_tool_arguments_are_retained_without_execution(tmp_path):
    toolkit = capturing_toolkit(tmp_path)

    async def stream(messages, info):
        yield {0: DeltaToolCall(name="observe", json_args='{ "position":')}
        await asyncio.sleep(5)

    result, _, _ = solve(
        tmp_path, FunctionModel(stream_function=stream), toolkit, timeout_s=0.1
    )
    assert result.stop_reason == "timeout"
    (record,) = toolkit.state.planner_records()
    assert record.status == "cancelled"
    (call,) = record.tool_calls
    assert call.arguments.data == '{ "position":'
    assert call.arguments.completeness == "partial"
    assert call.status == "not_executed" and toolkit.calls == []
