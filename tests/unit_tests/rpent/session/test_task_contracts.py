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

"""Task ownership, record identity, and independent model/action trajectories."""

import json
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

from rpent.session import ArtifactRef, DataRef, EnvState, MissingInfo, Payload


@pytest.fixture
def state(tmp_path: Path) -> EnvState:
    state = EnvState(tmp_path)
    state.start_task(instruction="Move the cup.", config=Payload("test.config", {}))
    return state


@pytest.fixture
def tool_id(state: EnvState) -> str:
    planner_id = state.register_planner(
        backend="api",
        status="completed",
        response=Payload("test.response", {"tool_calls": ["move"]}),
    )
    return state.register_tool_call(
        planner_id=planner_id, name="move", arguments=Payload("json", {})
    )


@pytest.mark.parametrize(
    "status,reason,result",
    [
        ("completed", "request_limit", {"environment_success": False}),
        ("failed", "timeout", {"error": "Planner timed out."}),
        ("cancelled", "operator", None),
    ],
)
def test_task_end_preserves_collected_data_and_independent_call_statuses(
    state: EnvState,
    tool_id: str,
    status: str,
    reason: str,
    result: dict | None,
) -> None:
    observation_id = state.register_observation(
        payload=Payload("test.observation", [0.0])
    )
    action_id = state.register_action(
        tool_call_id=tool_id,
        source="vla",
        action=Payload("test.action", np.array([0.5], dtype=np.float32)),
        before_observation_id=observation_id,
        after_observation_id=None,
        missing={
            "after_observation_id": MissingInfo("capture_failed", "Camera timed out.")
        },
    )
    start = state.get_task()
    outcome = Payload("test.result", result) if result is not None else None

    state.end_task(status=status, stop_reason=reason, result=outcome)

    task = state.get_task()
    assert task.task_id == start.task_id
    assert task.status == status and task.stop_reason == reason
    assert task.started_at == start.started_at <= task.ended_at
    assert task.started_at.utcoffset().total_seconds() == 0
    assert task.result == outcome
    assert state.get_tool_call(tool_id).status == "requested"
    assert state.planner_records()[0].status == "completed"
    action = state.get_action(action_id)
    assert action.action.format == "test.action"
    assert action.before_observation_id == observation_id
    assert action.after_observation_id is None
    assert action.missing["after_observation_id"].reason == "capture_failed"
    assert state.get_observation(observation_id).payload.data == [0.0]
    np.testing.assert_array_equal(action.action.data, [0.5])


def test_task_is_explicit_and_cannot_be_overwritten(tmp_path: Path) -> None:
    state = EnvState(tmp_path)
    assert state.get_task() is None
    assert state.planner_records() == state.action_records() == []
    assert state.prediction_records() == []
    assert state.save("recipe.json", [], step=None) == "recipe.json"
    with state.record_step(state={"ready": True}):
        pass
    assert state.get_task() is None
    with pytest.raises(RuntimeError, match="not running"):
        state.register_planner(backend="api")
    with pytest.raises(RuntimeError, match="not running"):
        state.update_planner("P1", status="failed")
    with pytest.raises(RuntimeError, match="not running"):
        state.update_tool_call("T1", status="not_executed")
    with pytest.raises(RuntimeError, match="not running"):
        state.register_action(
            tool_call_id="T1",
            source="direct",
            action=Payload("test.action", [0.5]),
            before_observation_id=None,
            after_observation_id=None,
        )
    with pytest.raises(RuntimeError, match="not running"):
        state.end_task(status="completed", stop_reason="finish")
    with pytest.raises(RuntimeError, match="not running"):
        state.register_prediction(
            tool_call_id="T1",
            input_observation_id=None,
            instruction="Move.",
            actions=Payload("test.actions", []),
            action_ids=[],
        )

    task_id = state.start_task(instruction="Move.", config=Payload("json", {}))
    with pytest.raises(RuntimeError, match="already been started"):
        state.start_task(instruction="Replace.", config=Payload("json", {}))
    with pytest.raises(ValueError, match="invalid task end status"):
        state.end_task(status="running", stop_reason="finish")
    assert state.get_task().status == "running"

    state.end_task(status="completed", stop_reason="finish")
    with pytest.raises(RuntimeError, match="not running"):
        state.end_task(status="failed", stop_reason="error")
    with pytest.raises(RuntimeError, match="already been started"):
        state.start_task(instruction="Replace.", config=Payload("json", {}))
    assert state.get_task().task_id == task_id
    assert state.get_task().status == "completed"
    assert state.get_task().instruction == "Move."
    assert state.get().state == {"ready": True}

    other = EnvState(tmp_path / "other")
    assert other.start_task(instruction="Move.", config=Payload("json", {})) != task_id


def test_ended_task_rejects_record_writes(state: EnvState, tool_id: str) -> None:
    planner_id = state.get_tool_call(tool_id).planner_id
    planners = state.planner_records()
    state.end_task(status="cancelled", stop_reason="operator")

    writes = [
        lambda: state.register_planner(backend="api"),
        lambda: state.update_planner(planner_id, status="cancelled"),
        lambda: state.update_tool_call(tool_id, status="cancelled"),
        lambda: state.register_tool_call(
            planner_id=planner_id, name="move", arguments=Payload("json", {})
        ),
        lambda: state.register_observation(payload=Payload("json", {})),
        lambda: state.register_action(
            tool_call_id=tool_id,
            source="direct",
            action=Payload("test.action", [0.5]),
            before_observation_id=None,
            after_observation_id=None,
        ),
        lambda: state.register_prediction(
            tool_call_id=tool_id,
            input_observation_id=None,
            instruction="Move.",
            actions=Payload("test.actions", []),
            action_ids=[],
        ),
    ]
    for write in writes:
        with pytest.raises(RuntimeError, match="not running"):
            write()
    assert state.planner_records() == planners
    assert state.action_records() == []
    assert state.prediction_records() == []


def test_planner_update_preserves_request_tools_and_registration_order(
    state: EnvState,
) -> None:
    started = datetime(2026, 1, 1, tzinfo=timezone.utc)
    ended = started + timedelta(seconds=2)
    request = Payload("test.request", {"messages": ["Move."]})
    planner_id = state.register_planner(
        backend="api",
        request=request,
        started_at=started,
        source_ids={"pydantic_ai.run_id": "run_1"},
        missing={
            "response": MissingInfo("pending", "Waiting for the model."),
            "/source_ids/provider.request_id": MissingInfo(
                "not_exposed", "No provider request ID."
            ),
        },
    )
    next_planner = state.register_planner(backend="api")
    tool_id = state.register_tool_call(
        planner_id=planner_id, name="move", arguments=Payload("json", {})
    )
    before = state.get_planner(planner_id)
    response = Payload("test.response", {"tool_calls": ["move"]})

    state.update_planner(
        planner_id,
        status="completed",
        response=response,
        source_ids={"provider.response_id": "response_1"},
        ended_at=ended,
    )
    assert state.get_planner(planner_id).missing == before.missing
    state.update_planner(planner_id, missing={"response": None, "ended_at": None})
    state.update_planner(planner_id, status=None, response=None, ended_at=None)

    after = state.get_planner(planner_id)
    assert after.planner_id == before.planner_id
    assert after.backend == "api" and after.request == request
    assert after.started_at == started and after.ended_at == ended
    assert after.status == "completed" and after.response == response
    assert after.tool_calls == before.tool_calls == [state.get_tool_call(tool_id)]
    assert after.source_ids == {
        "pydantic_ai.run_id": "run_1",
        "provider.response_id": "response_1",
    }
    assert after.missing == {
        "/source_ids/provider.request_id": before.missing[
            "/source_ids/provider.request_id"
        ]
    }
    assert before.status == "open" and before.response is None
    assert before.ended_at is None
    assert [record.planner_id for record in state.planner_records()] == [
        planner_id,
        next_planner,
    ]
    assert state.register_planner(backend="api") == "P3"


@pytest.mark.parametrize("side", ["before", "after"])
def test_missing_tool_observation_clears_boundary_without_changing_actions(
    state: EnvState, tool_id: str, side: str
) -> None:
    observation_id = state.register_observation(payload=Payload("test.obs", [1]))
    state.update_tool_call(
        tool_id,
        before_observation_id=observation_id,
        after_observation_id=observation_id,
    )
    action_id = state.register_action(
        tool_call_id=tool_id,
        source="direct",
        action=Payload("test.action", [0.5]),
        before_observation_id=observation_id,
        after_observation_id=observation_id,
    )
    field = f"{side}_observation_id"
    reason = MissingInfo("capture_failed", "Environment response was lost.")
    state.update_tool_call(tool_id, missing={field: reason})
    assert getattr(state.get_tool_call(tool_id), field) is None
    assert state.get_tool_call(tool_id).missing[field] == reason
    assert getattr(state.get_action(action_id), field) == observation_id
    state.update_tool_call(tool_id, **{field: observation_id}, missing={field: None})
    assert getattr(state.get_tool_call(tool_id), field) == observation_id
    assert field not in state.get_tool_call(tool_id).missing


@pytest.mark.parametrize("status", ["completed", "failed", "cancelled"])
def test_tool_update_keeps_action_links_and_accepts_late_feedback(
    state: EnvState, tool_id: str, status: str
) -> None:
    before = state.get_tool_call(tool_id)
    planner_id = before.planner_id
    sibling_id = state.register_tool_call(
        planner_id=planner_id, name="observe", arguments=Payload("json", {})
    )
    observations = [
        state.register_observation(payload=Payload("test.obs", [value]))
        for value in [0, 1]
    ]
    started = datetime(2026, 1, 1, tzinfo=timezone.utc)
    ended = started + timedelta(seconds=3)
    config = Payload("test.config", {"model": "vla"})
    state.update_tool_call(
        tool_id,
        status="running",
        effective_instruction="Move the cup.",
        model_config=config,
        started_at=started,
        before_observation_id=observations[0],
    )
    action_id = state.register_action(
        tool_call_id=tool_id,
        source="vla",
        action=Payload("test.action", [0.5]),
        before_observation_id=observations[0],
        after_observation_id=observations[1],
    )
    state.update_tool_call(
        tool_id,
        status=status,
        raw_result=Payload("test.result", {"executed_steps": 1}),
        after_observation_id=observations[1],
        ended_at=ended,
        missing={"feedback": MissingInfo("pending", "Waiting for the adapter.")},
    )
    finished = state.get_tool_call(tool_id)
    feedback = Payload("test.feedback", DataRef("tool_call", tool_id, "/raw_result"))
    state.update_tool_call(tool_id, feedback=feedback, missing={"feedback": None})
    state.update_tool_call(
        tool_id,
        status=None,
        model_config=None,
        raw_result=None,
        feedback=None,
        before_observation_id=None,
        after_observation_id=None,
    )

    after = state.get_tool_call(tool_id)
    assert after.tool_call_id == before.tool_call_id
    assert after.planner_id == planner_id and after.origin == before.origin
    assert after.name == before.name and after.arguments == before.arguments
    assert after.status == status and after.effective_instruction == "Move the cup."
    assert after.model_config == config
    assert after.started_at == started and after.ended_at == ended
    assert after.before_observation_id == observations[0]
    assert after.after_observation_id == observations[1]
    assert after.raw_result == finished.raw_result
    assert after.feedback == feedback and after.missing == {}
    assert finished.feedback is None and "feedback" in finished.missing
    assert before.status == "requested" and before.started_at is None
    assert state.get_action(action_id).tool_call_id == tool_id
    planner = state.get_planner(planner_id)
    assert planner.status == "completed"
    assert planner.tool_calls == [after, state.get_tool_call(sibling_id)]
    assert state.planner_records()[0].tool_calls == planner.tool_calls
    assert (
        state.register_tool_call(
            planner_id=planner_id, name="finish", arguments=Payload("json", {})
        )
        == "T3"
    )


def test_updates_record_failures_without_inventing_results_or_times(
    state: EnvState, tool_id: str
) -> None:
    planner_id = state.register_planner(backend="api")
    state.update_planner(
        planner_id,
        status="failed",
        missing={"response": MissingInfo("capture_failed", "Connection closed.")},
    )
    state.update_tool_call(tool_id, status="not_executed")

    planner = state.get_planner(planner_id)
    tool = state.get_tool_call(tool_id)
    assert planner.status == "failed" and planner.response is None
    assert planner.started_at is None and planner.ended_at is None
    assert planner.missing["response"].reason == "capture_failed"
    assert tool.status == "not_executed" and tool.raw_result is None
    assert tool.started_at is None and tool.ended_at is None
    assert state.action_records() == []


def test_updates_isolate_nested_payloads_and_metadata(
    state: EnvState, tool_id: str
) -> None:
    planner_id = state.register_planner(backend="api")
    values = np.array([1.0, 2.0], dtype=np.float32)
    payload = Payload("test.data", {"values": values, "content": [b"image"]})
    source_ids = {"test.id": "original"}
    missing = {"ended_at": MissingInfo("pending", "Waiting for completion.")}
    state.update_planner(
        planner_id, response=payload, source_ids=source_ids, missing=missing
    )
    state.update_tool_call(
        tool_id,
        model_config=payload,
        raw_result=payload,
        feedback=payload,
        source_ids=source_ids,
        missing=missing,
    )
    values[:] = 99
    payload.data["content"].clear()
    source_ids.clear()
    missing["ended_at"].detail = "Changed"
    missing.clear()

    planner = state.get_planner(planner_id)
    tool = state.get_tool_call(tool_id)
    for captured in [
        planner.response,
        tool.model_config,
        tool.raw_result,
        tool.feedback,
    ]:
        np.testing.assert_array_equal(captured.data["values"], [1.0, 2.0])
        assert captured.data["content"] == [b"image"]
    for record in [planner, tool]:
        assert record.source_ids == {"test.id": "original"}
        assert record.missing == {
            "ended_at": MissingInfo("pending", "Waiting for completion.")
        }


@pytest.mark.parametrize(
    "kind,field",
    [
        ("planner", "response"),
        ("tool_call", "model_config"),
        ("tool_call", "raw_result"),
        ("tool_call", "feedback"),
    ],
)
@pytest.mark.parametrize("capture_at_registration", [False, True])
def test_captured_payloads_cannot_be_overwritten_even_when_null(
    state: EnvState, kind: str, field: str, capture_at_registration: bool
) -> None:
    captured = {field: Payload("test.null", None)}
    initial = captured if capture_at_registration else {}
    if kind == "planner":
        record_id = state.register_planner(backend="api", **initial)
    else:
        planner_id = state.register_planner(backend="api")
        record_id = state.register_tool_call(
            planner_id=planner_id, name="move", arguments=Payload("json", {}), **initial
        )
    update = getattr(state, f"update_{kind}")
    get = getattr(state, f"get_{kind}")
    if not capture_at_registration:
        update(record_id, **captured)
    before = state.planner_records()
    assert getattr(get(record_id), field) == captured[field]

    with pytest.raises(ValueError, match=f"{field} has already been captured"):
        update(
            record_id,
            status="failed",
            source_ids={"test.id": "changed"},
            missing={"ended_at": MissingInfo("not_recorded", "No timestamp.")},
            **{field: Payload("test.result", {"changed": True})},
        )

    assert state.planner_records() == before


@pytest.mark.parametrize(
    "kind,invalid_status", [("planner", "running"), ("tool_call", "open")]
)
def test_invalid_update_status_leaves_record_unchanged(
    state: EnvState, tool_id: str, kind: str, invalid_status: str
) -> None:
    record_id = (
        tool_id if kind == "tool_call" else state.get_tool_call(tool_id).planner_id
    )
    before = state.planner_records()
    with pytest.raises(ValueError, match="invalid .* status"):
        getattr(state, f"update_{kind}")(
            record_id,
            status=invalid_status,
            source_ids={"test.id": "changed"},
            missing={"ended_at": MissingInfo("not_recorded", "No timestamp.")},
        )
    assert state.planner_records() == before


def test_model_and_action_trajectories_share_explicit_links(state: EnvState) -> None:
    first = state.register_planner(backend="api", status="completed")
    second = state.register_planner(backend="api", status="completed")
    observations = [
        state.register_observation(
            payload=Payload("test.observation", {"position": position}),
        )
        for position in [0, 1, 1, 2]
    ]
    first_tool = state.register_tool_call(
        planner_id=first,
        name="move",
        arguments=Payload("json", {}),
        status="completed",
        before_observation_id=observations[0],
        after_observation_id=observations[2],
    )
    second_tool = state.register_tool_call(
        planner_id=second,
        name="move",
        arguments=Payload("json", {}),
        status="failed",
        before_observation_id=observations[2],
        after_observation_id=observations[3],
    )
    read_tool = state.register_tool_call(
        planner_id=first,
        name="observe",
        arguments=Payload("json", {}),
        status="completed",
    )
    action_ids = []
    for tool, before, after, source in [
        (first_tool, observations[0], observations[1], "vla"),
        (second_tool, observations[2], observations[3], "tool_computed"),
        (second_tool, observations[3], None, "direct"),
    ]:
        action_ids.append(
            state.register_action(
                tool_call_id=tool,
                source=source,
                action=Payload("test.action", [0.5]),
                before_observation_id=before,
                after_observation_id=after,
                missing=(
                    {
                        "after_observation_id": MissingInfo(
                            "capture_failed", "Camera failed."
                        ),
                    }
                    if after is None
                    else {}
                ),
            )
        )

    planners = state.planner_records()
    assert [record.planner_id for record in planners] == [first, second]
    assert [call.tool_call_id for call in planners[0].tool_calls] == [
        first_tool,
        read_tool,
    ]
    assert [call.tool_call_id for call in planners[1].tool_calls] == [second_tool]
    assert state.get_tool_call(read_tool).planner_id == first
    actions = state.action_records()
    assert [record.action_id for record in actions] == action_ids
    assert [record.source for record in actions] == ["vla", "tool_computed", "direct"]
    assert {record.action.format for record in actions} == {"test.action"}
    assert actions[1].after_observation_id == actions[2].before_observation_id
    assert (
        actions[0].after_observation_id
        != state.get_tool_call(first_tool).after_observation_id
    )
    assert observations[1] != observations[2]
    assert (
        state.get_observation(observations[1]).payload.data
        == state.get_observation(observations[2]).payload.data
    )
    assert state.get_tool_call(actions[1].tool_call_id).planner_id == second
    assert actions[2].missing["after_observation_id"].reason == "capture_failed"


def test_payload_keeps_native_data_without_copying_or_coercing():
    values = np.array([0.1, 0.2], dtype=np.float32)
    data = {
        "values": values,
        "image": bytes([0, 255]),
        "references": [
            ArtifactRef("artifacts/image.bin", "bytes"),
            DataRef("observation", "O1", "/payload/data/image"),
        ],
    }
    payload = Payload("test.native", data)

    assert payload.data is data
    assert payload.data["values"] is values
    values[0] = 0.5
    data["references"].clear()
    assert payload.data["values"][0] == 0.5
    assert payload.data["references"] == []


def test_registration_and_queries_isolate_nested_payloads(tmp_path: Path) -> None:
    state = EnvState(tmp_path)
    config = Payload("test.config", {"model": {"max_tokens": 1024}})
    state.start_task(instruction="Move.", config=config)
    request = Payload(
        "test.request", {"messages": [{"role": "user", "content": "Move."}]}
    )
    source_ids = {"pydantic_ai.run_id": "run_1"}
    planner = state.register_planner(
        backend="api", request=request, source_ids=source_ids
    )
    raw_result = Payload("test.result", {"done": True})
    tool = state.register_tool_call(
        planner_id=planner,
        name="move",
        arguments=Payload("json", {}),
        raw_result=raw_result,
    )
    values = np.array([0.1, 0.2], dtype=np.float32)
    obs = state.register_observation(payload=Payload("test.obs", values))
    missing = {"/gripper": MissingInfo("not_exposed", "Gripper command unavailable.")}
    action_payload = Payload(
        "test.joint_action",
        {"joints": values},
        format_version="1",
        completeness="partial",
        missing=missing,
    )
    action = state.register_action(
        tool_call_id=tool,
        source="vla",
        action=action_payload,
        before_observation_id=obs,
        after_observation_id=None,
        result=Payload("test.result", DataRef("tool_call", tool, "/raw_result/data")),
    )
    action_ids = [action]
    prediction_missing = {
        "input_observation_id": MissingInfo("capture_failed", "Camera failed.")
    }
    state.register_prediction(
        tool_call_id=tool,
        input_observation_id=None,
        instruction="Move the cup.",
        actions=Payload("test.actions", values[None, :], inline_array_paths=("",)),
        action_ids=action_ids,
        missing=prediction_missing,
    )

    action_ids.clear()
    prediction_missing["input_observation_id"].detail = "Changed"
    config.data["model"]["max_tokens"] = 1
    request.data["messages"].clear()
    source_ids.clear()
    raw_result.data["done"] = False
    action_payload.format = "test.changed"
    action_payload.format_version = "2"
    missing["/gripper"].detail = "Changed"
    missing.clear()
    values[:] = 99
    state.get_task().config.data.clear()
    state.get_planner(planner).request.data.clear()
    state.get_tool_call(tool).raw_result.data.clear()
    state.get_observation(obs).payload.data[:] = 88
    state.get_action(action).action.data["joints"][:] = 77
    state.planner_records()[0].tool_calls.clear()
    actions = state.action_records()
    actions[0].before_observation_id = None
    actions[0].action.format = "test.changed_again"
    actions[0].action.format_version = "3"
    actions[0].action.missing.clear()
    actions[0].action.data["joints"][:] = 66
    actions.clear()
    prediction = state.prediction_records()[0]
    prediction.actions.data[:] = 55
    prediction.action_ids.clear()
    prediction.missing.clear()
    stored_prediction = state.prediction_records()[0]
    assert (
        stored_prediction.planner_id == planner
        and stored_prediction.tool_call_id == tool
    )
    assert stored_prediction.instruction == "Move the cup."
    assert stored_prediction.input_observation_id is None
    assert stored_prediction.actions.inline_array_paths == ("",)
    assert stored_prediction.action_ids == [action]
    assert stored_prediction.missing["input_observation_id"].detail == "Camera failed."
    np.testing.assert_array_equal(
        stored_prediction.actions.data, np.array([[0.1, 0.2]], dtype=np.float32)
    )

    assert state.get_task().config.data == {"model": {"max_tokens": 1024}}
    assert state.get_planner(planner).request.data["messages"][0]["content"] == "Move."
    assert state.get_planner(planner).source_ids == {"pydantic_ai.run_id": "run_1"}
    assert state.get_planner(planner).tool_calls[0].raw_result.data == {"done": True}
    np.testing.assert_array_equal(
        state.get_observation(obs).payload.data, np.array([0.1, 0.2], dtype=np.float32)
    )
    stored = state.get_action(action)
    np.testing.assert_array_equal(
        stored.action.data["joints"], state.get_observation(obs).payload.data
    )
    assert stored.before_observation_id == obs
    assert stored.result.data == DataRef("tool_call", tool, "/raw_result/data")
    assert stored.action.format == "test.joint_action"
    assert stored.action.format_version == "1"
    assert stored.action.completeness == "partial"
    assert stored.action.missing == {
        "/gripper": MissingInfo("not_exposed", "Gripper command unavailable.")
    }
    outcome = Payload("test.result", {"environment_success": False})
    state.end_task(status="completed", stop_reason="finish", result=outcome)
    outcome.data.clear()
    state.get_task().result.data.clear()
    assert state.get_task().result.data == {"environment_success": False}


def test_interleaved_registration_keeps_ids_and_parent_links(state: EnvState) -> None:
    planners = [state.register_planner(backend="api") for _ in range(2)]

    def register(index: int) -> tuple[str, str, str, str]:
        tool = state.register_tool_call(
            planner_id=planners[index % 2],
            name="move",
            arguments=Payload("json", {"index": index}),
        )
        observation = state.register_observation(
            payload=Payload("test.obs", index),
        )
        action = state.register_action(
            tool_call_id=tool,
            source="vla",
            action=Payload("test.action", [index]),
            before_observation_id=observation,
            after_observation_id=None,
        )
        prediction = state.register_prediction(
            tool_call_id=tool,
            input_observation_id=observation,
            instruction="Move.",
            actions=Payload("test.actions", [[index]]),
            action_ids=[action],
        )
        return tool, observation, action, prediction

    with ThreadPoolExecutor(max_workers=8) as pool:
        identities = list(pool.map(register, range(32)))

    assert len({identity for group in identities for identity in group}) == 128
    assert [action.action_id for action in state.action_records()] == [
        f"A{index}" for index in range(1, 33)
    ]
    predictions = {
        record.prediction_id: record for record in state.prediction_records()
    }
    assert list(predictions) == [f"V{index}" for index in range(1, 33)]
    for index, (tool, observation, action, prediction) in enumerate(identities):
        assert state.get_tool_call(tool).planner_id == planners[index % 2]
        assert state.get_tool_call(tool).arguments.data == {"index": index}
        assert state.get_action(action).tool_call_id == tool
        assert state.get_action(action).before_observation_id == observation
        assert state.get_action(action).action.format == "test.action"
        assert state.get_action(action).action.data == [index]
        assert state.get_observation(observation).payload.data == index
        assert predictions[prediction].planner_id == planners[index % 2]
        assert predictions[prediction].tool_call_id == tool
        assert predictions[prediction].input_observation_id == observation
        assert predictions[prediction].action_ids == [action]
        assert predictions[prediction].actions.data == [[index]]
    for index, planner in enumerate(state.planner_records()):
        assert sorted(
            call.arguments.data["index"] for call in planner.tool_calls
        ) == list(range(index, 32, 2))


def test_legacy_reset_and_rollback_preserve_task_records(
    state: EnvState, tool_id: str
) -> None:
    task = state.get_task()
    original_planner = state.get_tool_call(tool_id).planner_id
    with state.record_step(state={"initial": True}):
        state.save("old.bin", b"initial")
    observation = state.register_observation(payload=Payload("test.obs", [0]))
    with pytest.raises(RuntimeError, match="failed to record step"):
        with state.record_step(state={"partial": True}) as step:
            state.save("partial.bin", b"partial")
            action = state.register_action(
                tool_call_id=tool_id,
                source="direct",
                action=Payload("test.action", [1]),
                before_observation_id=observation,
                after_observation_id=None,
            )
            raise ValueError("capture failed after execution")

    assert not state.exists("partial.bin", step=step)
    state.reset()
    assert state.records() == [] and state.latest_step is None
    assert state.load("old.bin", step=0) == b"initial"
    assert state.get_task() == task
    assert state.get_action(action).before_observation_id == observation
    assert state.get_action(action).action.format == "test.action"
    assert state.get_tool_call(tool_id).planner_id == original_planner
    assert state.get_observation(observation).payload.data == [0]
    assert state.register_planner(backend="api") != original_planner
    assert (
        state.register_tool_call(
            planner_id=original_planner, name="move", arguments=Payload("json", {})
        )
        != tool_id
    )
    assert state.register_observation(payload=Payload("test.obs", [0])) != observation
    assert (
        state.register_action(
            tool_call_id=tool_id,
            source="direct",
            action=Payload("test.action", [2]),
            before_observation_id=observation,
            after_observation_id=None,
        )
        != action
    )
    assert len(state.action_records()) == 2
    assert len(state.get_planner(original_planner).tool_calls) == 2


def test_unknown_parents_do_not_publish_child_records(state: EnvState) -> None:
    with pytest.raises(KeyError, match="unknown-planner"):
        state.update_planner("unknown-planner", status="failed")
    with pytest.raises(KeyError, match="unknown-tool"):
        state.update_tool_call("unknown-tool", status="not_executed")
    with pytest.raises(KeyError, match="unknown-planner"):
        state.register_tool_call(
            planner_id="unknown-planner", name="move", arguments=Payload("json", {})
        )
    with pytest.raises(KeyError, match="unknown-tool"):
        state.register_action(
            tool_call_id="unknown-tool",
            source="direct",
            action=Payload("test.action", [1]),
            before_observation_id=None,
            after_observation_id=None,
        )
    assert state.planner_records() == []
    assert state.action_records() == []
    for get in [
        state.get_planner,
        state.get_tool_call,
        state.get_observation,
        state.get_action,
    ]:
        with pytest.raises(KeyError):
            get("unknown")


@pytest.mark.parametrize("invalid", ["tool", "action", "owner"])
def test_prediction_rejects_invalid_action_links(
    state: EnvState, tool_id: str, invalid: str
) -> None:
    action_id = state.register_action(
        tool_call_id=tool_id,
        source="vla",
        action=Payload("test.action", [1]),
        before_observation_id=None,
        after_observation_id=None,
    )
    if invalid == "tool":
        tool_id = "unknown-tool"
    elif invalid == "action":
        action_id = "unknown-action"
    else:
        tool_id = state.register_tool_call(
            planner_id=state.get_tool_call(tool_id).planner_id,
            name="other",
            arguments=Payload("json", {}),
        )

    with pytest.raises(ValueError if invalid == "owner" else KeyError):
        state.register_prediction(
            tool_call_id=tool_id,
            input_observation_id=None,
            instruction="Move.",
            actions=Payload("test.actions", [[1]]),
            action_ids=[action_id],
        )
    assert state.prediction_records() == []


def test_one_tool_keeps_mixed_action_protocols_in_execution_order(
    state: EnvState, tool_id: str
) -> None:
    payloads = [
        Payload(
            "test.joints", np.array([0.1, 0.2], dtype=np.float32), format_version="1"
        ),
        Payload(
            "test.pose",
            {"position": np.array([0.3, 0.4, 0.5], dtype=np.float64)},
            format_version="2",
        ),
        Payload(
            "test.joints", np.array([0.6, 0.7], dtype=np.float32), format_version="1"
        ),
    ]
    action_ids = [
        state.register_action(
            tool_call_id=tool_id,
            source="tool_computed",
            action=payload,
            before_observation_id=None,
            after_observation_id=None,
        )
        for payload in payloads
    ]

    records = state.action_records()
    assert [record.action_id for record in records] == action_ids
    assert [record.action.format for record in records] == [
        "test.joints",
        "test.pose",
        "test.joints",
    ]
    assert [record.action.format_version for record in records] == ["1", "2", "1"]
    for index in (0, 2):
        np.testing.assert_array_equal(records[index].action.data, payloads[index].data)
        assert records[index].action.data.dtype == np.dtype("float32")
    np.testing.assert_array_equal(
        records[1].action.data["position"], payloads[1].data["position"]
    )
    assert records[1].action.data["position"].dtype == np.dtype("float64")


@pytest.mark.parametrize("started", [False, True])
def test_planner_trajectory_requires_an_ended_task(tmp_path, started):
    state = EnvState(tmp_path)
    if started:
        state.start_task(instruction="Move.", config=Payload("json", {}))
    with pytest.raises(RuntimeError, match=r"end_task\(\) must finish"):
        state.save_planner_trajectory()
    assert not (tmp_path / "trajectories").exists()


@pytest.mark.parametrize("status", ["completed", "failed", "cancelled"])
def test_planner_trajectory_preserves_records_and_registration_order(
    state, tmp_path, status
):
    timestamp = datetime.now(timezone.utc)
    planner_ids = [
        state.register_planner(
            backend="api",
            status="completed",
            request=Payload(
                "pydantic_ai.request",
                {"messages": [{"content": "移动杯子"}], "tools": [{"name": "move"}]},
                format_version="test-sdk",
                completeness="complete",
            ),
            response=Payload("pydantic_ai.response", {"parts": [{"text": str(index)}]}),
            source_ids={"provider.response_id": f"response-{index}"},
            started_at=timestamp,
            ended_at=timestamp,
        )
        for index in range(12)
    ]
    tool_ids = [
        state.register_tool_call(
            planner_id=planner_ids[0],
            name="move",
            arguments=Payload("json", {"position": index}),
            status="completed",
            source_ids={"pydantic_ai.tool_call_id": f"call-{index}"},
            effective_instruction="Move.",
            model_config=Payload("test.config", {"model": "test"}),
            started_at=timestamp,
            ended_at=timestamp,
            raw_result=Payload("rpent.tool_result", {"distance": index}),
            feedback=Payload("pydantic_ai.tool_result", f"Moved {index}."),
            missing={
                "before_observation_id": MissingInfo("not_recorded", "No capture")
            },
        )
        for index in range(12)
    ]
    state.register_planner(
        backend="api",
        status="failed",
        request=Payload("pydantic_ai.request", {"messages": []}),
        missing={"response": MissingInfo("capture_failed", "Request interrupted")},
    )
    state.end_task(
        status=status, stop_reason="test", result=Payload("test.result", {"score": 0})
    )

    root = state.save_planner_trajectory()

    task = state.get_task()
    assert root == tmp_path / "trajectories" / task.task_id
    expected = json.loads(
        json.dumps(
            [asdict(task), *map(asdict, state.planner_records())],
            default=datetime.isoformat,
        ),
        object_hook=lambda data: {
            key: value for key, value in data.items() if key != "inline_array_paths"
        },
    )
    saved_task = json.loads((root / "task.json").read_text())
    assert saved_task == {"schema_version": 3, "task": expected[0]}
    records = [
        json.loads(line) for line in (root / "planners.jsonl").read_text().splitlines()
    ]
    assert records == expected[1:]
    assert [record["planner_id"] for record in records[:-1]] == planner_ids
    assert [tool["tool_call_id"] for tool in records[0]["tool_calls"]] == tool_ids
    assert records[-1]["response"] is None
    assert records[-1]["missing"]["response"]["reason"] == "capture_failed"
    assert not (tmp_path / "states.json").exists()


def test_planner_trajectory_externalizes_nested_data_without_mutating_records(tmp_path):
    state = EnvState(tmp_path)
    shared = b"\x00\xff"
    duplicate = bytes(bytearray(shared))
    assert duplicate == shared and duplicate is not shared
    state.start_task(instruction="Move.", config=Payload("test.config", duplicate))
    image = np.arange(12, dtype=np.uint16).reshape(3, 4)
    existing = ArtifactRef("artifacts/existing.bin", "bytes", media_type="image/png")
    planner = state.register_planner(
        backend="api",
        status="completed",
        response=Payload("test.response", [shared, b"different"]),
    )
    tool = state.register_tool_call(
        planner_id=planner,
        name="move",
        arguments=Payload("json", {}),
        raw_result=Payload(
            "test.raw",
            {
                "nested": [{"image": image, "bytes": duplicate}],
                "existing": existing,
                "scalar": np.int64(42),
            },
        ),
    )
    reference = DataRef("tool_call", tool, "/raw_result/data/nested/0/image")
    state.update_tool_call(tool, feedback=Payload("test.feedback", reference))
    state.end_task(status="completed", stop_reason="finish")
    expected_root = tmp_path / "trajectories" / state.get_task().task_id
    (expected_root / "artifacts").mkdir(parents=True)
    (expected_root / existing.path).write_bytes(b"existing attachment")

    root = state.save_planner_trajectory()

    record = json.loads((root / "planners.jsonl").read_text())
    data = record["tool_calls"][0]["raw_result"]["data"]
    image_ref = data["nested"][0]["image"]
    assert image_ref["$type"] == "artifact_ref"
    assert image_ref["kind"] == "ndarray"
    assert image_ref["dtype"] == "uint16" and image_ref["shape"] == [3, 4]
    restored = np.load(root / image_ref["path"], allow_pickle=False)
    np.testing.assert_array_equal(restored, image)
    assert restored.dtype == image.dtype
    bytes_ref = data["nested"][0]["bytes"]
    assert bytes_ref["kind"] == "bytes"
    assert (root / bytes_ref["path"]).read_bytes() == shared
    assert record["response"]["data"][0] == bytes_ref
    different_ref = record["response"]["data"][1]
    assert different_ref["path"] != bytes_ref["path"]
    assert (root / different_ref["path"]).read_bytes() == b"different"
    assert len(list((root / "artifacts").iterdir())) == 4
    assert data["existing"] == {"$type": "artifact_ref", **asdict(existing)}
    assert (root / existing.path).read_bytes() == b"existing attachment"
    assert data["scalar"] == 42
    assert record["tool_calls"][0]["feedback"]["data"] == {
        "$type": "data_ref",
        **asdict(reference),
    }
    task_config = json.loads((root / "task.json").read_text())["task"]["config"]["data"]
    assert task_config == bytes_ref
    stored = state.get_tool_call(tool)
    np.testing.assert_array_equal(stored.raw_result.data["nested"][0]["image"], image)
    assert stored.raw_result.data["nested"][0]["bytes"] == shared
    assert stored.raw_result.data["existing"] == existing
    assert stored.feedback.data == reference
    assert state.get_task().config.data == shared

    files = {path: path.read_bytes() for path in root.rglob("*") if path.is_file()}
    assert state.save_planner_trajectory() == root
    assert {
        path: path.read_bytes() for path in root.rglob("*") if path.is_file()
    } == files


def test_planner_trajectory_saves_a_task_without_model_calls(state):
    state.end_task(status="failed", stop_reason="setup_error")
    root = state.save_planner_trajectory()
    assert (root / "planners.jsonl").read_text() == ""
    assert (root / "actions.jsonl").read_text() == ""
    assert (root / "predictions.jsonl").read_text() == ""
    assert json.loads((root / "task.json").read_text())["task"]["status"] == "failed"


@pytest.mark.parametrize(
    "dtype,values,expected",
    [
        ("bool", [True], [True]),
        ("int32", [3], [3]),
        (
            "float32",
            [np.nan, np.inf, -np.inf],
            [{"$type": "float", "value": value} for value in ("nan", "inf", "-inf")],
        ),
    ],
)
def test_trajectory_inlines_only_declared_array_paths(state, dtype, values, expected):
    native = np.array(values, dtype=dtype)
    planner_id = state.register_planner(
        backend="api",
        request=Payload(
            "test.request",
            {"nested/~": [native, None], "attachment": native},
            inline_array_paths=("/nested~1~0/0", "/nested~1~0/1", "/absent"),
        ),
    )
    state.end_task(status="completed", stop_reason="finish")
    root = state.save_planner_trajectory()
    payload = json.loads((root / "planners.jsonl").read_text())["request"]
    assert "inline_array_paths" not in payload
    assert payload["data"]["nested/~"] == [
        {"$type": "ndarray", "dtype": dtype, "shape": [len(values)], "value": expected},
        None,
    ]
    np.testing.assert_array_equal(
        np.load(root / payload["data"]["attachment"]["path"], allow_pickle=False),
        native,
    )
    assert len(list((root / "artifacts").iterdir())) == 1
    stored = state.get_planner(planner_id).request.data["nested/~"][0]
    np.testing.assert_array_equal(stored, native)
    assert stored.dtype == native.dtype


def test_planner_trajectory_distinguishes_references_and_literal_tags(state):
    reference = DataRef("tool_call", "T1", "/raw_result/data")
    artifact = ArtifactRef("artifacts/existing.bin", "bytes")
    literal = {"$type": "data_ref", **asdict(reference)}
    literal_artifact = {"$type": "artifact_ref", **asdict(artifact)}
    literal_float = {"$type": "float", "value": "nan"}
    literal_array = {"$type": "ndarray", "dtype": "bool", "shape": [1], "value": [True]}
    literal_dict = {"$type": "dict", "value": literal}
    payload = {
        "reference": reference,
        "artifact": artifact,
        "literals": [
            literal,
            literal_artifact,
            literal_float,
            literal_array,
            literal_dict,
        ],
    }
    planner_id = state.register_planner(
        backend="api", request=Payload("test.request", payload)
    )
    state.end_task(status="completed", stop_reason="finish")

    root = state.save_planner_trajectory()

    data = json.loads((root / "planners.jsonl").read_text())["request"]["data"]
    assert data["reference"] == literal
    assert data["artifact"] == literal_artifact
    assert data["literals"] == [
        {"$type": "dict", "value": literal},
        {"$type": "dict", "value": literal_artifact},
        {"$type": "dict", "value": literal_float},
        {"$type": "dict", "value": literal_array},
        {
            "$type": "dict",
            "value": {
                "$type": "dict",
                "value": {"$type": "dict", "value": literal},
            },
        },
    ]
    assert state.get_planner(planner_id).request.data == payload


@pytest.mark.parametrize(
    "filename", ["planners.jsonl", "actions.jsonl", "predictions.jsonl", "task.json"]
)
def test_planner_trajectory_write_failure_is_visible_and_retryable(
    state, tmp_path, monkeypatch, filename
):
    planner = state.register_planner(
        backend="api",
        status="completed",
        request=Payload("test.request", b"request"),
        response=Payload("test.response", [b"request", b"other"]),
    )
    tool = state.register_tool_call(
        planner_id=planner, name="move", arguments=Payload("json", {})
    )
    observation = state.register_observation(
        payload=Payload("test.obs", np.array([0], dtype=np.float32))
    )
    action_id = state.register_action(
        tool_call_id=tool,
        source="vla",
        action=Payload("test.action", np.array([1], dtype=np.float32)),
        before_observation_id=observation,
        after_observation_id=None,
    )
    prediction_id = state.register_prediction(
        tool_call_id=tool,
        input_observation_id=observation,
        instruction="Move.",
        actions=Payload(
            "test.actions", np.array([[1]], np.float32), inline_array_paths=("",)
        ),
        action_ids=[action_id],
    )
    state.end_task(status="completed", stop_reason="finish")
    root = tmp_path / "trajectories" / state.get_task().task_id
    root.mkdir(parents=True)
    destination = root / filename
    previous = b'{"previous": true}\n'
    destination.write_bytes(previous)
    replace = os.replace

    def fail_replace(source, destination):
        if destination.name == filename:
            raise OSError("disk full")
        replace(source, destination)

    with monkeypatch.context() as patch:
        patch.setattr("rpent.session.base.os.replace", fail_replace)
        with pytest.raises(OSError, match="disk full"):
            state.save_planner_trajectory()
    assert destination.read_bytes() == previous
    if filename != "task.json":
        assert not (root / "task.json").exists()
    assert not list(root.rglob(".*.tmp*"))
    assert state.planner_records()[0].request.data == b"request"

    assert state.save_planner_trajectory() == root
    assert (root / "task.json").is_file()
    saved = json.loads((root / "planners.jsonl").read_text())
    request = saved["request"]["data"]
    assert (root / request["path"]).read_bytes() == b"request"
    assert saved["response"]["data"][0] == request
    assert (root / saved["response"]["data"][1]["path"]).read_bytes() == b"other"
    action = json.loads((root / "actions.jsonl").read_text())
    np.testing.assert_array_equal(np.load(root / action["action"]["data"]["path"]), [1])
    reference = action["before_observation"]["payload"]["data"]
    np.testing.assert_array_equal(np.load(root / reference["path"]), [0])
    prediction = json.loads((root / "predictions.jsonl").read_text())
    assert prediction["prediction_id"] == prediction_id
    assert prediction["action_ids"] == [action_id]
    assert prediction["input_observation"] == action["before_observation"]
    assert prediction["actions"]["data"]["value"] == [[1]]
    assert len(list((root / "artifacts").iterdir())) == 4


@pytest.mark.parametrize(
    "attachment", ["planner", "action", "observation", "prediction"]
)
def test_trajectory_attachment_failure_marks_missing_and_can_be_retried(
    state, tmp_path, monkeypatch, attachment
):
    planner_id = state.register_planner(
        backend="api",
        request=Payload(
            "test.request",
            {"messages": [{"image/~": b"image", "text": "Move", "optional": None}]},
            completeness="partial",
            missing={"/metadata": MissingInfo("not_exposed", "No metadata.")},
        ),
    )
    tool_id = state.register_tool_call(
        planner_id=planner_id, name="move", arguments=Payload("json", {})
    )
    states = np.ones(8, np.float32)
    observation_id = state.register_observation(
        payload=Payload("test.obs", {"states": states}, completeness="complete")
    )
    actions = [
        state.register_action(
            tool_call_id=tool_id,
            source="vla",
            action=Payload("test.action", np.array([index]), completeness="complete"),
            before_observation_id=observation_id if index else None,
            after_observation_id=None if index else observation_id,
        )
        for index in range(2)
    ]
    prediction_actions = np.array([[1]])
    state.register_prediction(
        tool_call_id=tool_id,
        input_observation_id=observation_id,
        instruction="Move.",
        actions=Payload("test.actions", prediction_actions, completeness="complete"),
        action_ids=actions[1:],
    )
    failed_data = {
        "planner": b"image",
        "action": np.array([0]),
        "observation": states,
        "prediction": prediction_actions,
    }[attachment]
    replace = os.replace

    def fail_attachment(source, destination):
        if destination.parent.name == "artifacts":
            data = (
                source.read_bytes()
                if destination.suffix == ".bin"
                else np.load(source, allow_pickle=False)
            )
            if np.array_equal(data, failed_data):
                raise OSError("attachment write failed")
        replace(source, destination)

    state.end_task(status="failed", stop_reason="execution_error")
    with monkeypatch.context() as patch:
        patch.setattr("rpent.session.base.os.replace", fail_attachment)
        state.save_planner_trajectory()

    root = tmp_path / "trajectories" / state.get_task().task_id
    request = json.loads((root / "planners.jsonl").read_text())["request"]
    rows = [
        json.loads(line) for line in (root / "actions.jsonl").read_text().splitlines()
    ]
    assert [row["action_id"] for row in rows] == actions
    assert rows[0]["after_observation"] == rows[1]["before_observation"]
    prediction = json.loads((root / "predictions.jsonl").read_text())
    assert prediction["action_ids"] == actions[1:]
    assert prediction["input_observation"] == rows[1]["before_observation"]
    prediction_payload = prediction["actions"]
    action = rows[0]["action"]
    observation = rows[0]["after_observation"]["payload"]
    payload, pointer, absent = {
        "planner": (
            request,
            "/messages/0/image~1~0",
            request["data"]["messages"][0]["image/~"],
        ),
        "action": (action, "", action["data"]),
        "observation": (observation, "/states", observation["data"]["states"]),
        "prediction": (prediction_payload, "", prediction_payload["data"]),
    }[attachment]
    assert absent is None
    assert payload["completeness"] == "partial"
    assert payload["missing"][pointer] == {
        "reason": "capture_failed",
        "detail": "OSError: attachment write failed",
    }
    assert request["missing"]["/metadata"]["reason"] == "not_exposed"
    assert "/messages/0/optional" not in request["missing"]
    assert rows[1]["action"]["completeness"] == "complete"
    np.testing.assert_array_equal(
        np.load(root / rows[1]["action"]["data"]["path"], allow_pickle=False),
        [1],
    )
    assert not list(root.rglob(".*.tmp*"))

    state.save_planner_trajectory()
    restored = json.loads((root / "planners.jsonl").read_text())["request"]
    assert set(restored["missing"]) == {"/metadata"}
    for line in (root / "actions.jsonl").read_text().splitlines():
        row = json.loads(line)
        observation = row["before_observation"] or row["after_observation"]
        for restored in (row["action"], observation["payload"]):
            assert restored["completeness"] == "complete"
            assert restored["missing"] == {}
    restored = json.loads((root / "predictions.jsonl").read_text())["actions"]
    assert restored["completeness"] == "complete" and restored["missing"] == {}
    np.testing.assert_array_equal(
        np.load(root / restored["data"]["path"], allow_pickle=False), prediction_actions
    )


def test_action_trajectory_embeds_observations_and_reuses_attachments(state, tool_id):
    timestamp = datetime.now(timezone.utc)
    images = [np.full((2, 2, 3), index, dtype=np.uint8) for index in range(13)]
    observations = [
        state.register_observation(
            payload=Payload("test.obs", {"image": image, "state": [index]}),
            captured_at=timestamp,
        )
        for index, image in enumerate(images)
    ]
    state.update_tool_call(tool_id, raw_result=Payload("test.raw", b"shared"))
    sources = ["tool_computed", "vla", "direct"] * 4
    action_ids = [
        state.register_action(
            tool_call_id=tool_id,
            source=source,
            action=Payload("test.action", np.array([index], dtype=np.float32)),
            before_observation_id=observations[index],
            after_observation_id=observations[index + 1],
            executed_at=timestamp,
            result=Payload(
                "test.result", {"reward": np.float32(index), "info": b"shared"}
            ),
        )
        for index, source in enumerate(sources)
    ]
    prediction_ids = [
        state.register_prediction(
            tool_call_id=tool_id,
            input_observation_id=observations[index],
            instruction="Move.",
            actions=Payload(
                "test.actions",
                np.array([[index]], np.float32),
                inline_array_paths=("",),
            ),
            action_ids=[action_ids[index]],
        )
        for index in range(1, len(sources), 3)
    ]
    state.end_task(status="completed", stop_reason="finish")

    root = state.save_planner_trajectory()

    rows = [
        json.loads(line) for line in (root / "actions.jsonl").read_text().splitlines()
    ]
    planner = json.loads((root / "planners.jsonl").read_text())
    assert [row["action_id"] for row in rows] == action_ids
    assert [row["source"] for row in rows] == sources
    assert {row["tool_call_id"] for row in rows} == {
        planner["tool_calls"][0]["tool_call_id"]
    }
    for index, row in enumerate(rows):
        assert row["executed_at"] == timestamp.isoformat()
        action = np.load(root / row["action"]["data"]["path"], allow_pickle=False)
        np.testing.assert_array_equal(action, [index])
        assert action.dtype == np.float32
        assert row["result"]["data"]["reward"] == index
        assert (
            row["result"]["data"]["info"]
            == planner["tool_calls"][0]["raw_result"]["data"]
        )
        for side, position in (("before", index), ("after", index + 1)):
            observation = row[f"{side}_observation"]
            assert (
                observation["observation_id"]
                == row[f"{side}_observation_id"]
                == observations[position]
            )
            assert observation["captured_at"] == timestamp.isoformat()
            assert observation["payload"]["data"]["state"] == [position]
            reference = observation["payload"]["data"]["image"]
            assert reference["$type"] == "artifact_ref"
            assert reference["dtype"] == "uint8" and reference["shape"] == [2, 2, 3]
            np.testing.assert_array_equal(
                np.load(root / reference["path"], allow_pickle=False), images[position]
            )
    for previous, following in zip(rows, rows[1:]):
        assert previous["after_observation"] == following["before_observation"]
    predictions = [
        json.loads(line)
        for line in (root / "predictions.jsonl").read_text().splitlines()
    ]
    assert [row["prediction_id"] for row in predictions] == prediction_ids
    for prediction, action in zip(predictions, rows[1::3]):
        assert prediction["planner_id"] == planner["planner_id"]
        assert prediction["tool_call_id"] == tool_id
        assert prediction["input_observation"] == action["before_observation"]
        assert prediction["action_ids"] == [action["action_id"]]
        assert prediction["actions"]["data"] == {
            "$type": "ndarray",
            "dtype": "float32",
            "shape": [1, 1],
            "value": [[action["result"]["data"]["reward"]]],
        }
        assert "inline_array_paths" not in prediction["actions"]
    assert len(list((root / "artifacts").iterdir())) == 13 + 12 + 1
    assert not (root / "observations.jsonl").exists()
    assert isinstance(state.get_action(action_ids[0]).action.data, np.ndarray)
    np.testing.assert_array_equal(
        state.get_observation(observations[0]).payload.data["image"], images[0]
    )
    files = {path: path.read_bytes() for path in root.rglob("*") if path.is_file()}
    state.save_planner_trajectory()
    assert {
        path: path.read_bytes() for path in root.rglob("*") if path.is_file()
    } == files


def test_action_trajectory_preserves_missing_observations(state, tool_id):
    missing = {
        f"{side}_observation_id": MissingInfo("capture_failed", "Camera failed")
        for side in ("before", "after")
    }
    action_id = state.register_action(
        tool_call_id=tool_id,
        source="vla",
        action=Payload("test.action", [1]),
        before_observation_id=None,
        after_observation_id=None,
        missing=missing,
    )
    prediction_missing = {
        "input_observation_id": MissingInfo("capture_failed", "Camera failed")
    }
    state.register_prediction(
        tool_call_id=tool_id,
        input_observation_id=None,
        instruction="Move.",
        actions=Payload("test.actions", [[1]]),
        action_ids=[action_id],
        missing=prediction_missing,
    )
    state.end_task(status="failed", stop_reason="capture_failed")
    root = state.save_planner_trajectory()
    row = json.loads((root / "actions.jsonl").read_text())
    assert row["before_observation"] is row["after_observation"] is None
    assert row["missing"] == {key: asdict(value) for key, value in missing.items()}
    prediction = json.loads((root / "predictions.jsonl").read_text())
    assert prediction["input_observation_id"] is prediction["input_observation"] is None
    assert prediction["action_ids"] == [action_id]
    assert prediction["tool_call_id"] == tool_id
    assert prediction["planner_id"] == state.get_tool_call(tool_id).planner_id
    assert prediction["missing"] == {
        key: asdict(value) for key, value in prediction_missing.items()
    }


@pytest.mark.parametrize("kind", ["bytes", "ndarray"])
@pytest.mark.parametrize(
    "owner", ["task", "planner", "observation", "action", "prediction"]
)
@pytest.mark.parametrize("stem", ["payload-000001", ".payload-000001.tmp"])
def test_captured_artifacts_survive_trajectory_save(tmp_path, kind, owner, stem):
    value = (
        b"captured" if kind == "bytes" else np.arange(6, dtype=np.uint16).reshape(2, 3)
    )
    suffix = ".bin" if kind == "bytes" else ".npy"
    reference = ArtifactRef(
        path=f"artifacts/{stem}{suffix}",
        kind=kind,
        dtype=str(value.dtype) if kind == "ndarray" else None,
        shape=value.shape if kind == "ndarray" else None,
    )
    state = EnvState(tmp_path)
    state.start_task(
        instruction="Move.",
        config=Payload(
            "test.config", {"existing": [reference]} if owner == "task" else {}
        ),
    )
    assert state.save_trajectory_artifact(value, name=f"{stem}{suffix}") == reference
    planner_id = state.register_planner(
        backend="api",
        request=Payload("test.request", reference) if owner == "planner" else None,
    )
    tool_id = state.register_tool_call(
        planner_id=planner_id, name="move", arguments=Payload("json", {})
    )
    root = tmp_path / "trajectories" / state.get_task().task_id
    original = (root / reference.path).read_bytes()
    assert state.get_task().status == "running"
    assert not (root / "task.json").exists()
    if kind == "bytes":
        assert original == value
    else:
        np.testing.assert_array_equal(
            np.load(root / reference.path, allow_pickle=False), value
        )
        assert reference.dtype == "uint16" and reference.shape == (2, 3)
        value[:] = 99
    observation = state.register_observation(
        payload=Payload("test.obs", reference if owner == "observation" else {})
    )
    action_id = state.register_action(
        tool_call_id=tool_id,
        source="vla",
        action=Payload("test.action", reference if owner == "action" else [1]),
        before_observation_id=observation,
        after_observation_id=observation,
    )
    state.register_prediction(
        tool_call_id=tool_id,
        input_observation_id=observation,
        instruction="Move.",
        actions=Payload("test.actions", reference if owner == "prediction" else [[1]]),
        action_ids=[action_id],
    )
    state.register_planner(
        backend="api",
        request=Payload(
            "test.request", b"different" if kind == "bytes" else np.array([5])
        ),
    )
    state.end_task(status="completed", stop_reason="finish")
    state.save_planner_trajectory()

    row = json.loads((root / "actions.jsonl").read_text())
    prediction = json.loads((root / "predictions.jsonl").read_text())
    task = json.loads((root / "task.json").read_text())["task"]
    planners = [
        json.loads(line) for line in (root / "planners.jsonl").read_text().splitlines()
    ]
    if owner == "task":
        saved = task["config"]["data"]["existing"][0]
    elif owner == "planner":
        saved = planners[0]["request"]["data"]
    else:
        saved = {
            "observation": row["before_observation"]["payload"],
            "action": row["action"],
            "prediction": prediction["actions"],
        }[owner]["data"]
    assert saved == {
        "$type": "artifact_ref",
        **json.loads(json.dumps(asdict(reference))),
    }
    new_reference = planners[1]["request"]["data"]
    assert new_reference["path"] != reference.path
    if kind == "bytes":
        assert (root / new_reference["path"]).read_bytes() == b"different"
    else:
        np.testing.assert_array_equal(
            np.load(root / new_reference["path"], allow_pickle=False), [5]
        )
    assert (root / reference.path).read_bytes() == original
    assert len(list((root / "artifacts").iterdir())) == 2
    files = {path: path.read_bytes() for path in root.rglob("*") if path.is_file()}
    state.save_planner_trajectory()
    assert {
        path: path.read_bytes() for path in root.rglob("*") if path.is_file()
    } == files


def test_trajectory_artifact_requires_task_and_valid_data(tmp_path):
    state = EnvState(tmp_path)
    with pytest.raises(RuntimeError, match=r"start_task\(\)"):
        state.save_trajectory_artifact(b"data")
    state.start_task(instruction="Move", config=Payload("test.config", {}))
    with pytest.raises(TypeError, match="arrays or bytes"):
        state.save_trajectory_artifact([1])
    with pytest.raises(ValueError, match="base filename"):
        state.save_trajectory_artifact(b"data", name="../outside.bin")
    with pytest.raises(ValueError, match=r"end with \.bin"):
        state.save_trajectory_artifact(b"data", name="data.npy")
    first = state.save_trajectory_artifact(b"first")
    second = state.save_trajectory_artifact(b"second")
    assert first.path != second.path
