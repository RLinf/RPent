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

"""Native Codex trace imports preserve model evidence and execution identity."""

import json
import shutil
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import Mock

import openai_codex
import pytest

from rpent.planner.base import PlannerResult
from rpent.planner.codex import CodexPlanner
from rpent.session import EnvState, MissingInfo, Payload
from rpent.session.planner_capture import end_planner_task, import_codex_trace


@pytest.fixture
def state(tmp_path):
    state = EnvState(tmp_path / "run")
    state.start_task(instruction="Move.", config=Payload("test.config", {}))
    return state


@pytest.fixture
def native_trace(tmp_path):
    root = tmp_path / "native" / "bundle"
    (root / "payloads").mkdir(parents=True)
    payload_count = 0
    sequence = 0

    def payload(kind, data):
        nonlocal payload_count
        payload_count += 1
        name = f"payload-{payload_count}"
        path = f"payloads/{name}.json"
        (root / path).write_text(json.dumps(data), encoding="utf-8")
        return {"raw_payload_id": name, "kind": {"type": kind}, "path": path}

    def event(kind, *, thread_id="thread-1", **data):
        nonlocal sequence
        sequence += 1
        if kind == "inference_started":
            data = {
                "thread_id": thread_id,
                "codex_turn_id": "turn-1",
                "model": "gpt-5.5",
                "provider_name": "test",
                **data,
            }
        elif kind == "tool_call_started":
            data = {
                "model_visible_call_id": None,
                "code_mode_runtime_tool_id": None,
                "requester": {"type": "model"},
                "kind": {"type": "other", "name": "test_tool"},
                "summary": {
                    "type": "generic",
                    "label": "test_tool",
                    "input_preview": None,
                    "output_preview": None,
                },
                **data,
            }
        row = {
            "schema_version": 1,
            "seq": sequence,
            "wall_time_unix_ms": 1000 * sequence,
            "rollout_id": "rollout-1",
            "thread_id": thread_id,
            "codex_turn_id": "turn-1" if thread_id is not None else None,
            "payload": {"type": kind, **data},
        }
        with (root / "trace.jsonl").open("a", encoding="utf-8") as file:
            file.write(json.dumps(row) + "\n")

    return root, payload, event


@pytest.mark.parametrize("via_solve", [False, True], ids=["direct", "solve"])
@pytest.mark.parametrize("correlation_thread", ["thread-1", None])
def test_import_links_existing_mcp_calls_in_response_order(
    state, native_trace, monkeypatch, via_solve, correlation_thread
):
    root, payload, event = native_trace
    image = "data:image/png;base64,aW1hZ2U="
    request = {
        "model": "gpt-5.5",
        "input": [{"type": "input_image", "image_url": image}],
    }
    tool_ids = {}
    for name in ("right", "left"):
        tool_ids[name] = state.register_tool_call(
            planner_id=None,
            origin="unknown",
            name=name,
            arguments=Payload("mcp.tool_args", {"actual": name}),
            source_ids={"codex_bridge_mcp_call_id": f"bridge-{name}"},
            status="failed" if name == "right" else "completed",
            raw_result=Payload("rpent.tool_result", {"image": b"native"}),
            feedback=Payload("mcp.call_tool_result", {"image": image}),
            missing={"planner_id": MissingInfo("pending", "Waiting for trace.")},
        )
    prediction_id = state.register_prediction(
        tool_call_id=tool_ids["left"], instruction="Move left."
    )
    action_id = state.register_action(
        tool_call_id=tool_ids["left"],
        prediction_id=prediction_id,
        source="vla",
        action=Payload("test.action", [1]),
        before_observation_id=None,
        after_observation_id=None,
    )
    before = {name: state.get_tool_call(tool_id) for name, tool_id in tool_ids.items()}
    event(
        "inference_started",
        inference_call_id="inference-1",
        request_payload=payload("inference_request", request),
    )
    native_result = {
        "type": "direct_response",
        "response_item": {
            "type": "function_call_output",
            "call_id": "call-shell",
            "output": "shell result",
        },
    }
    for name in ("right", "shell", "left"):
        event(
            "tool_call_started",
            tool_call_id=f"native-{name}",
            model_visible_call_id=f"call-{name}",
            invocation_payload=payload(
                "tool_invocation",
                {
                    "tool_name": "exec_command"
                    if name == "shell"
                    else f"mcp__rpent__{name}",
                    "tool_namespace": None,
                    "payload": {"type": "function", "arguments": '{"different": true}'},
                },
            ),
        )
        if name != "shell":
            event(
                "mcp_tool_call_correlation_assigned",
                thread_id=correlation_thread,
                tool_call_id=f"native-{name}",
                mcp_call_id=f"bridge-{name}",
            )
        event(
            "tool_call_ended",
            tool_call_id=f"native-{name}",
            status="completed",
            result_payload=payload("tool_result", native_result),
        )
    response = {
        "response_id": "response-1",
        "upstream_request_id": "request-1",
        "token_usage": {"input_tokens": 10, "output_tokens": 20},
        "output_items": [
            {
                "type": "function_call",
                "name": name,
                "arguments": "{}",
                "call_id": f"call-{name}",
            }
            for name in ("left", "shell", "right")
        ],
    }
    event(
        "inference_completed",
        inference_call_id="inference-1",
        response_id="response-1",
        upstream_request_id="request-1",
        response_payload=payload("inference_response", response),
    )
    files = {path: path.read_bytes() for path in root.rglob("*") if path.is_file()}

    if via_solve:
        adapter = CodexPlanner(
            output_dir=str(root.parent),
            repo_root=root.parent,
            dashboard_events=Mock(),
        )

        def execute(**kwargs):
            shutil.copytree(root, kwargs["trace_root"] / root.name)
            return PlannerResult(finish_result=None, messages=[], stats={}, error=None)

        monkeypatch.setattr(adapter, "_solve_terminal", execute)
        result = adapter.solve(
            system_prompt="",
            user_message="Move.",
            toolkit=SimpleNamespace(state=state),
            max_turns=1,
        )
        stats = result.stats["codex_trace"]
    else:
        stats = import_codex_trace(state, root.parent)

    assert stats == {
        **({"sdk_version": openai_codex.__version__} if via_solve else {}),
        "bundles": 1,
        "planners": 1,
        "tool_calls": 1,
        "linked_tool_calls": 2,
        "unmatched_mcp_calls": 0,
        "unlinked_tool_calls": 0,
        "errors": [],
    }
    (planner,) = state.planner_records()
    assert planner.backend == "codex" and planner.status == "completed"
    assert planner.request == Payload(
        "codex.inference_request", request, format_version="1", completeness="complete"
    )
    assert planner.response == Payload(
        "codex.inference_response",
        response,
        format_version="1",
        completeness="complete",
    )
    assert planner.started_at == datetime.fromtimestamp(1, timezone.utc)
    assert planner.ended_at > planner.started_at and planner.missing == {}
    assert planner.source_ids["provider.response_id"] == "response-1"
    assert planner.source_ids["provider.request_id"] == "request-1"
    assert [call.name for call in planner.tool_calls] == [
        "left",
        "exec_command",
        "right",
    ]
    for name, tool_id in tool_ids.items():
        call = state.get_tool_call(tool_id)
        assert call.planner_id == planner.planner_id and call.origin == "model"
        assert call.source_ids["codex.tool_call_id"] == f"native-{name}"
        assert call.arguments == before[name].arguments
        assert call.raw_result == before[name].raw_result
        assert call.feedback == before[name].feedback
        assert call.status == before[name].status
        assert call.missing == {}
    native = planner.tool_calls[1]
    assert native.feedback.data == native_result
    assert (
        native.raw_result is None
        and native.missing["raw_result"].reason == "not_exposed"
    )
    assert native.before_observation_id is None and native.after_observation_id is None
    (prediction,) = state.prediction_records()
    assert prediction.planner_id == planner.planner_id
    assert prediction.tool_call_id == tool_ids["left"] and prediction.action_ids == [
        action_id
    ]
    assert state.get_action(action_id).tool_call_id == tool_ids["left"]
    assert {path: path.read_bytes() for path in files} == files
    if via_solve:
        assert state.get_task().status == "running"
        assert not (state.trajectory_dir / "task.json").exists()
        end_planner_task(
            state, result, finish_result=None, environment_success=False, exception=None
        )
        saved = json.loads((state.trajectory_dir / "planners.jsonl").read_text())
        assert saved["planner_id"] == planner.planner_id
        assert [call["name"] for call in saved["tool_calls"]] == [
            "left",
            "exec_command",
            "right",
        ]
        saved_prediction = json.loads(
            (state.trajectory_dir / "predictions.jsonl").read_text()
        )
        assert saved_prediction["planner_id"] == planner.planner_id
        assert not (state.trajectory_dir / "unlinked_tool_calls.jsonl").exists()
        saved_task = json.loads((state.trajectory_dir / "task.json").read_text())[
            "task"
        ]
        assert saved_task["status"] == "completed"
        assert saved_task["result"]["data"]["environment_success"] is False
        assert saved_task["result"]["data"]["stats"]["codex_trace"]["sdk_version"] == (
            openai_codex.__version__
        )


@pytest.mark.parametrize("terminal", ["failed", "cancelled"])
@pytest.mark.parametrize("partial", [False, True])
def test_import_keeps_retries_partial_responses_and_missing_terminal(
    state, native_trace, terminal, partial
):
    root, payload, event = native_trace
    response = {
        "response_id": None,
        "upstream_request_id": "upstream-1",
        "token_usage": None,
        "output_items": [
            {"type": "message", "content": [{"type": "output_text", "text": "partial"}]}
        ],
    }
    for index in range(3):
        request = {"input": [], "previous_response_id": "previous"}
        event(
            "inference_started",
            inference_call_id=f"inference-{index}",
            request_payload=payload("inference_request", request),
        )
        if index == 0:
            event(
                f"inference_{terminal}",
                inference_call_id="inference-0",
                upstream_request_id="upstream-1",
                **{"error" if terminal == "failed" else "reason": "stream interrupted"},
                partial_response_payload=payload("inference_response", response)
                if partial
                else None,
            )
        elif index == 1:
            event(
                "inference_completed",
                inference_call_id="inference-1",
                response_id="response-1",
                upstream_request_id=None,
                response_payload=payload(
                    "inference_response",
                    {
                        "response_id": "response-1",
                        "upstream_request_id": None,
                        "token_usage": None,
                        "output_items": [],
                    },
                ),
            )

    stats = import_codex_trace(state, root)

    assert stats["planners"] == 3 and stats["errors"] == []
    failed, retried, unfinished = state.planner_records()
    assert [record.status for record in (failed, retried, unfinished)] == [
        terminal,
        "completed",
        "open",
    ]
    assert all(
        record.request.data == request for record in (failed, retried, unfinished)
    )
    if partial:
        assert failed.response.data == response
        assert failed.response.completeness == "partial"
        assert failed.response.missing["completion"].detail == "stream interrupted"
    else:
        assert failed.response is None
        assert failed.missing["response"].detail == "stream interrupted"
    assert unfinished.response is None and unfinished.ended_at is None
    assert unfinished.missing["response"].reason == "not_recorded"
    assert unfinished.missing["ended_at"].reason == "not_recorded"


@pytest.mark.parametrize("parent_recorded", [False, True])
def test_import_distinguishes_code_mode_children_from_direct_calls(
    state, native_trace, parent_recorded
):
    root, payload, event = native_trace
    tool_id = state.register_tool_call(
        planner_id=None,
        origin="unknown",
        name="move",
        arguments=Payload("mcp.tool_args", {}),
        source_ids={"codex_bridge_mcp_call_id": "bridge-1"},
    )
    event(
        "inference_started",
        inference_call_id="inference-1",
        request_payload=payload("inference_request", {"input": []}),
    )
    event(
        "inference_completed",
        inference_call_id="inference-1",
        response_id="response-1",
        upstream_request_id=None,
        response_payload=payload(
            "inference_response",
            {
                "output_items": [
                    {
                        "type": "custom_tool_call",
                        "name": "exec",
                        "input": "await tools.move({})",
                        "call_id": "exec-1",
                    }
                ]
            },
        ),
    )
    if parent_recorded:
        event(
            "code_cell_started",
            runtime_cell_id="cell-1",
            model_visible_call_id="exec-1",
            source_js="await tools.move({})",
        )
    event(
        "tool_call_started",
        tool_call_id="native-1",
        requester={"type": "code_cell", "runtime_cell_id": "cell-1"},
        code_mode_runtime_tool_id="runtime-1",
        invocation_payload=None,
    )
    event(
        "mcp_tool_call_correlation_assigned",
        thread_id=None,
        tool_call_id="native-1",
        mcp_call_id="bridge-1",
    )

    stats = import_codex_trace(state, root)

    (planner,) = state.planner_records()
    call = state.get_tool_call(tool_id)
    assert call.origin == "program"
    assert call.source_ids["codex.runtime_cell_id"] == "cell-1"
    assert call.source_ids["codex.code_mode_runtime_tool_id"] == "runtime-1"
    assert "codex.model_visible_call_id" not in call.source_ids
    if parent_recorded:
        assert call.planner_id == planner.planner_id
        assert call.source_ids["codex.parent_model_visible_call_id"] == "exec-1"
    else:
        assert call.planner_id is None and planner.tool_calls == []
        assert call.missing["planner_id"].reason == "not_recorded"
    assert stats["linked_tool_calls"] == int(parent_recorded)
    assert stats["tool_calls"] == 0


def test_import_scopes_model_call_ids_to_their_threads(state, native_trace):
    root, payload, event = native_trace
    expected = {}
    for thread_id in ("parent", "child"):
        expected[thread_id] = state.register_tool_call(
            planner_id=None,
            name="move",
            arguments=Payload("mcp.tool_args", {}),
            source_ids={"codex_bridge_mcp_call_id": f"bridge-{thread_id}"},
        )
        event(
            "inference_started",
            thread_id=thread_id,
            inference_call_id=f"inference-{thread_id}",
            request_payload=payload("inference_request", {"input": []}),
        )
        event(
            "inference_completed",
            thread_id=thread_id,
            inference_call_id=f"inference-{thread_id}",
            response_id=f"response-{thread_id}",
            upstream_request_id=None,
            response_payload=payload(
                "inference_response",
                {
                    "output_items": [
                        {
                            "type": "function_call",
                            "name": "move",
                            "arguments": "{}",
                            "call_id": "reused-call-id",
                        }
                    ]
                },
            ),
        )
        event(
            "tool_call_started",
            thread_id=thread_id,
            tool_call_id=f"native-{thread_id}",
            model_visible_call_id="reused-call-id",
            invocation_payload=None,
        )
        event(
            "mcp_tool_call_correlation_assigned",
            thread_id=thread_id,
            tool_call_id=f"native-{thread_id}",
            mcp_call_id=f"bridge-{thread_id}",
        )

    stats = import_codex_trace(state, root)

    assert stats["planners"] == stats["linked_tool_calls"] == 2
    for planner in state.planner_records():
        assert [call.tool_call_id for call in planner.tool_calls] == [
            expected[planner.source_ids["codex.thread_id"]]
        ]


def test_import_does_not_match_rpent_tools_by_name_or_arguments(state, native_trace):
    root, payload, event = native_trace
    tool_id = state.register_tool_call(
        planner_id=None,
        origin="unknown",
        name="move",
        arguments=Payload("mcp.tool_args", {}),
    )
    event(
        "tool_call_started",
        tool_call_id="native-1",
        model_visible_call_id="call-1",
        invocation_payload=payload(
            "tool_invocation",
            {
                "tool_name": "mcp__rpent__move",
                "tool_namespace": None,
                "payload": {"type": "function", "arguments": "{}"},
            },
        ),
    )
    event(
        "mcp_tool_call_correlation_assigned",
        tool_call_id="native-1",
        mcp_call_id="unmatched-bridge",
    )

    stats = import_codex_trace(state, root)

    assert stats["unmatched_mcp_calls"] == 1 and stats["unlinked_tool_calls"] == 1
    assert [call.tool_call_id for call in state.tool_call_records()] == [tool_id]
    assert state.get_tool_call(tool_id).origin == "unknown"


@pytest.mark.parametrize("damage", ["missing", "invalid_json", "outside_bundle"])
def test_import_retains_records_when_payload_or_trace_tail_is_missing(
    state, native_trace, damage
):
    root, payload, event = native_trace
    reference = payload("inference_request", {"input": []})
    if damage == "missing":
        (root / reference["path"]).unlink()
    elif damage == "invalid_json":
        (root / reference["path"]).write_text("{", encoding="utf-8")
    else:
        (root.parent / "outside.json").write_text('{"private": true}', encoding="utf-8")
        reference["path"] = "../outside.json"
    event(
        "inference_started", inference_call_id="inference-1", request_payload=reference
    )
    with (root / "trace.jsonl").open("a", encoding="utf-8") as file:
        file.write('{"schema_version":')

    stats = import_codex_trace(state, root)

    assert len(stats["errors"]) == 2
    (planner,) = state.planner_records()
    assert planner.status == "open" and planner.response is None
    assert planner.request.data is None and planner.request.completeness == "partial"
    assert planner.request.missing[""].reason == "capture_failed"
    assert planner.missing["response"].reason == "not_recorded"


def test_import_preserves_response_without_inventing_a_missing_request(
    state, native_trace
):
    root, payload, event = native_trace
    event(
        "inference_completed",
        inference_call_id="inference-1",
        response_id="response-1",
        upstream_request_id=None,
        response_payload=payload("inference_response", {"output_items": []}),
    )

    import_codex_trace(state, root)

    (planner,) = state.planner_records()
    assert planner.status == "completed" and planner.response.data == {
        "output_items": []
    }
    assert planner.request is None and planner.started_at is None
    assert planner.missing["request"].reason == "not_recorded"


def test_import_reports_missing_trace_without_changing_existing_calls(state, tmp_path):
    tool_id = state.register_tool_call(
        planner_id=None, name="move", arguments=Payload("mcp.tool_args", {})
    )
    before = state.get_tool_call(tool_id)

    stats = import_codex_trace(state, tmp_path / "missing")

    assert stats["bundles"] == stats["planners"] == stats["tool_calls"] == 0
    assert stats["unlinked_tool_calls"] == 1 and len(stats["errors"]) == 1
    assert state.get_tool_call(tool_id) == before
