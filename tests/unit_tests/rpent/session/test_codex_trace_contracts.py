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

import base64
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

    def event(kind, *, thread_id="thread-1", codex_turn_id="turn-1", **data):
        nonlocal sequence
        sequence += 1
        if kind == "inference_started":
            data = {
                "thread_id": thread_id,
                "codex_turn_id": codex_turn_id,
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
            "codex_turn_id": codex_turn_id if thread_id is not None else None,
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
            kwargs["cleanup_done"].set()
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
        "codex.expanded_request",
        request,
        format_version="1",
        completeness="complete",
        base64_paths={"/input/0/image_url": "data:image/png;base64,"},
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
        reference = saved["request"]["data"]["input"][0]["image_url"]
        assert reference["base64_prefix"] == "data:image/png;base64,"
        assert (state.trajectory_dir / reference["path"]).read_bytes() == b"image"
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


def test_import_externalizes_protocol_images_and_restores_native_payloads(
    state, native_trace
):
    root, payload, event = native_trace
    prefix = "data:image/png;base64,"
    encoded = base64.b64encode(b"image").decode("ascii")
    image = {"type": "input_image", "image_url": prefix + encoded, "detail": "high"}
    request = {
        "model": "gpt-5.5",
        "instructions": "Initial instructions.",
        "tools": [{"type": "custom", "name": "first_tool"}],
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [
                    image,
                    {"type": "input_text", "text": prefix + encoded},
                    {"type": "input_image", "image_url": "https://example.com/a.png"},
                ],
            },
            {
                "type": "custom_tool_call_output",
                "call_id": "previous",
                "output": [image],
            },
            {"type": "additional_tools", "tools": [{"examples": [image]}]},
        ],
        "metadata": {"example": image},
    }
    arguments = json.dumps({"content": [image]})
    response = {
        "output_items": [
            {
                "type": "function_call",
                "name": "view_image",
                "call_id": "call-1",
                "arguments": arguments,
            },
            {"type": "reasoning", "encrypted_content": encoded},
            {
                "type": "message",
                "role": "assistant",
                "phase": "commentary",
                "content": [{"type": "output_text", "text": "Checking the image."}],
            },
        ]
    }
    invocation = {
        "tool_name": "view_image",
        "payload": {"type": "function", "arguments": arguments},
    }
    feedback = {
        "type": "direct_response",
        "response_item": {
            "type": "function_call_output",
            "call_id": "call-1",
            "output": [image],
        },
    }
    event(
        "inference_started",
        inference_call_id="inference-1",
        request_payload=payload("inference_request", request),
    )
    event(
        "inference_completed",
        inference_call_id="inference-1",
        response_id="response-1",
        response_payload=payload("inference_response", response),
    )
    event(
        "tool_call_started",
        tool_call_id="native-1",
        model_visible_call_id="call-1",
        invocation_payload=payload("tool_invocation", invocation),
    )
    event(
        "tool_call_ended",
        tool_call_id="native-1",
        status="completed",
        result_payload=payload("tool_result", feedback),
    )
    developer = {
        "type": "message",
        "role": "developer",
        "content": [{"type": "input_text", "text": "Updated guidance."}],
    }
    second_request = {
        "model": "gpt-5.6-sol",
        "tools": [],
        "client_metadata": {"turn": 1},
        "previous_response_id": "response-1",
        "input": [feedback["response_item"], developer],
    }
    second_response = {
        "output_items": [
            {
                "type": "message",
                "role": "assistant",
                "phase": "final_answer",
                "content": [{"type": "output_text", "text": "Done."}],
            }
        ]
    }
    followup = {"type": "message", "role": "user", "content": [image]}
    third_request = {"previous_response_id": "response-2", "input": [followup]}
    for index, data, output, turn in (
        (2, second_request, second_response, "turn-1"),
        (3, third_request, {"output_items": []}, "turn-2"),
    ):
        event(
            "inference_started",
            codex_turn_id=turn,
            inference_call_id=f"inference-{index}",
            request_payload=payload("inference_request", data),
        )
        event(
            "inference_completed",
            codex_turn_id=turn,
            inference_call_id=f"inference-{index}",
            response_id=f"response-{index}",
            response_payload=payload("inference_response", output),
        )
    native_files = {
        path: path.read_bytes() for path in root.rglob("*") if path.is_file()
    }

    stats = import_codex_trace(state, root)
    assert stats["errors"] == []
    planners = state.planner_records()
    planner, second, third = planners
    second_input = [
        *request["input"],
        *response["output_items"],
        feedback["response_item"],
        developer,
    ]
    expected_requests = [
        request,
        {
            "model": "gpt-5.6-sol",
            "tools": [],
            "client_metadata": {"turn": 1},
            "input": second_input,
        },
        {"input": [*second_input, second_response["output_items"][0], followup]},
    ]
    assert [record.request.data for record in planners] == expected_requests
    assert all(
        record.request.format == "codex.expanded_request"
        and record.request.format_version == "1"
        and record.request.completeness == "complete"
        and record.request.missing == {}
        for record in planners
    )
    assert "provider.previous_response_id" not in planner.source_ids
    assert second.source_ids["provider.previous_response_id"] == "response-1"
    assert third.source_ids["provider.previous_response_id"] == "response-2"
    assert third.source_ids["codex.codex_turn_id"] == "turn-2"
    assert planner.request.base64_paths == {
        "/input/0/content/0/image_url": prefix,
        "/input/1/output/0/image_url": prefix,
    }
    assert second.request.base64_paths == {
        "/input/0/content/0/image_url": prefix,
        "/input/1/output/0/image_url": prefix,
        "/input/6/output/0/image_url": prefix,
    }
    assert third.request.base64_paths == {
        **second.request.base64_paths,
        "/input/9/content/0/image_url": prefix,
    }
    assert planner.tool_calls[0].feedback.base64_paths == {
        "/response_item/output/0/image_url": prefix
    }
    state.end_task(status="completed", stop_reason="finish")
    trajectory = state.save_planner_trajectory()

    saved_records = [
        json.loads(line)
        for line in (trajectory / "planners.jsonl").read_text().splitlines()
    ]
    for record, saved_record in zip(planners, saved_records):
        for pointer in record.request.base64_paths:
            parts = pointer.lstrip("/").split("/")
            block = saved_record["request"]["data"]
            for key in parts[:-1]:
                block = block[int(key)] if isinstance(block, list) else block[key]
            reference = block[parts[-1]]
            assert reference["$type"] == "artifact_ref"
            assert reference["base64_prefix"] == prefix
            content = (trajectory / reference["path"]).read_bytes()
            assert content == b"image"
            block[parts[-1]] = prefix + base64.b64encode(content).decode("ascii")
    saved = saved_records[0]
    saved_feedback = saved["tool_calls"][0]["feedback"]["data"]
    block = saved_feedback["response_item"]["output"][0]
    reference = block["image_url"]
    assert reference["$type"] == "artifact_ref"
    assert reference["base64_prefix"] == prefix
    content = (trajectory / reference["path"]).read_bytes()
    assert content == b"image"
    block["image_url"] = prefix + base64.b64encode(content).decode("ascii")
    assert len(list((trajectory / "artifacts").iterdir())) == 1
    assert [record["request"]["data"] for record in saved_records] == expected_requests
    assert saved["response"]["data"] == response
    assert saved_records[1]["response"]["data"] == second_response
    assert saved["tool_calls"][0]["arguments"]["data"] == invocation
    assert saved_feedback == feedback
    assert state.planner_records() == planners
    assert {path: path.read_bytes() for path in native_files} == native_files


@pytest.mark.parametrize("terminal", ["failed", "cancelled"])
@pytest.mark.parametrize("partial", [False, True])
def test_import_keeps_retries_partial_responses_and_missing_terminal(
    state, native_trace, terminal, partial
):
    root, payload, event = native_trace
    initial_input = [{"role": "user", "content": "Move."}]
    initial_output = [{"type": "reasoning", "encrypted_content": "initial"}]
    event(
        "inference_started",
        inference_call_id="initial",
        request_payload=payload("inference_request", {"input": initial_input}),
    )
    event(
        "inference_completed",
        inference_call_id="initial",
        response_id="previous",
        response_payload=payload(
            "inference_response", {"output_items": initial_output}
        ),
    )
    response = {
        "response_id": None,
        "upstream_request_id": "upstream-1",
        "token_usage": None,
        "output_items": [
            {"type": "message", "content": [{"type": "output_text", "text": "partial"}]}
        ],
    }
    for index in range(3):
        request = {
            "input": [],
            "previous_response_id": "response-1" if index == 2 else "previous",
        }
        event(
            "inference_started",
            inference_call_id=f"inference-{index}",
            request_payload=payload("inference_request", request),
        )
        if index == 0:
            event(
                f"inference_{terminal}",
                inference_call_id="inference-0",
                response_id="failed-response",
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

    assert stats["planners"] == 4 and stats["errors"] == []
    initial, failed, retried, unfinished = state.planner_records()
    assert initial.request.data == {"input": initial_input}
    assert [record.status for record in (failed, retried, unfinished)] == [
        terminal,
        "completed",
        "open",
    ]
    assert all(
        record.request.data == {"input": [*initial_input, *initial_output]}
        and record.request.completeness == "complete"
        for record in (failed, retried, unfinished)
    )
    assert retried.source_ids["provider.previous_response_id"] == "previous"
    assert unfinished.source_ids["provider.previous_response_id"] == "response-1"
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


@pytest.mark.parametrize(
    "damage",
    ["missing_parent", "missing_request", "missing_response", "partial_response"],
)
def test_import_marks_incomplete_context_until_a_full_request(
    state, native_trace, damage
):
    root, payload, event = native_trace
    if damage != "missing_parent":
        if damage != "missing_request":
            event(
                "inference_started",
                inference_call_id="parent",
                request_payload=payload("inference_request", {"input": []}),
            )
        response = payload("inference_response", {"output_items": []})
        event(
            "inference_failed"
            if damage == "partial_response"
            else "inference_completed",
            inference_call_id="parent",
            response_id="parent",
            response_payload=None if damage == "missing_response" else response,
            error="stream interrupted" if damage == "partial_response" else None,
        )
    deltas = [
        {
            "input": [{"role": "user", "content": "First."}],
            "previous_response_id": "parent",
        },
        {
            "input": [{"role": "user", "content": "Next."}],
            "previous_response_id": "response-0",
        },
    ]
    compacted = {"type": "compaction", "encrypted_content": "new-context"}
    followup = {"role": "user", "content": "Continue."}
    outputs = [
        {"type": "message", "role": "assistant", "content": f"Answer {index}."}
        for index in range(4)
    ]
    requests = [
        *deltas,
        {"input": [compacted], "previous_response_id": None, "instructions": "Reset."},
        {"input": [followup], "previous_response_id": "response-2"},
    ]
    for index, request in enumerate(requests):
        event(
            "inference_started",
            inference_call_id=f"inference-{index}",
            request_payload=payload("inference_request", request),
        )
        event(
            "inference_completed",
            inference_call_id=f"inference-{index}",
            response_id=f"response-{index}",
            response_payload=payload(
                "inference_response", {"output_items": [outputs[index]]}
            ),
        )

    stats = import_codex_trace(state, root)

    first, second, reset, continued = state.planner_records()[-4:]
    assert len(stats["errors"]) == 2
    for record, request in zip((first, second), deltas):
        previous = request["previous_response_id"]
        assert record.request.data == {"input": request["input"]}
        assert record.request.completeness == "partial"
        assert record.request.missing["/input"].reason == "not_recorded"
        assert previous in record.request.missing["/input"].detail
        assert record.source_ids["provider.previous_response_id"] == previous
        assert record.response.completeness == "complete"
        assert any(
            record.request.missing["/input"].detail in error
            for error in stats["errors"]
        )
    assert reset.request.data == {"input": [compacted], "instructions": "Reset."}
    assert continued.request.data == {"input": [compacted, outputs[2], followup]}
    assert reset.request.completeness == continued.request.completeness == "complete"
    assert reset.request.missing == continued.request.missing == {}
    assert "provider.previous_response_id" not in reset.source_ids


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


def test_import_scopes_payloads_and_call_ids_to_each_bundle(state, native_trace):
    root, payload, event = native_trace
    event(
        "inference_started",
        inference_call_id="inference-1",
        request_payload=payload("inference_request", {"input": ["first-bundle"]}),
    )
    event(
        "inference_completed",
        inference_call_id="inference-1",
        response_payload=payload(
            "inference_response",
            {
                "output_items": [
                    {
                        "type": "function_call",
                        "call_id": "call-1",
                        "name": "exec_command",
                        "arguments": "first-bundle",
                    }
                ]
            },
        ),
    )
    event(
        "tool_call_started",
        tool_call_id="native-1",
        model_visible_call_id="call-1",
        invocation_payload=payload(
            "tool_invocation",
            {"tool_name": "exec_command", "arguments": "first-bundle"},
        ),
    )
    event(
        "tool_call_ended",
        tool_call_id="native-1",
        status="completed",
        result_payload=payload("tool_result", {"output": "first-bundle"}),
    )
    second = shutil.copytree(root, root.with_name("bundle-2"))
    for path in (second / "payloads").iterdir():
        path.write_text(
            path.read_text(encoding="utf-8").replace("first-bundle", "second-bundle"),
            encoding="utf-8",
        )

    stats = import_codex_trace(state, root.parent)

    assert stats["bundles"] == stats["planners"] == stats["tool_calls"] == 2
    assert stats["unlinked_tool_calls"] == 0 and stats["errors"] == []
    planners = state.planner_records()
    assert len(planners) == 2
    for planner, label in zip(planners, ("first-bundle", "second-bundle")):
        assert planner.request.data == {"input": [label]}
        assert planner.response.data["output_items"][0]["arguments"] == label
        (call,) = planner.tool_calls
        assert call.planner_id == planner.planner_id
        assert call.arguments.data == {"tool_name": "exec_command", "arguments": label}
        assert call.feedback.data == {"output": label}
        assert call.status == "completed"
    assert len(state.tool_call_records()) == 2


def test_import_does_not_expand_context_from_another_bundle(state, native_trace):
    root, payload, event = native_trace
    event(
        "inference_started",
        inference_call_id="parent",
        request_payload=payload(
            "inference_request", {"input": [{"role": "user", "content": "Move."}]}
        ),
    )
    event(
        "inference_completed",
        inference_call_id="parent",
        response_id="previous",
        response_payload=payload("inference_response", {"output_items": []}),
    )
    parent_events = (root / "trace.jsonl").read_text()
    event(
        "inference_started",
        inference_call_id="child",
        request_payload=payload(
            "inference_request", {"input": [], "previous_response_id": "previous"}
        ),
    )
    second = shutil.copytree(root, root.with_name("bundle-2"))
    (second / "trace.jsonl").write_text(
        (root / "trace.jsonl").read_text()[len(parent_events) :]
    )
    (root / "trace.jsonl").write_text(parent_events)

    stats = import_codex_trace(state, root.parent)

    assert stats["bundles"] == stats["planners"] == 2
    assert len(stats["errors"]) == 1
    parent, child = state.planner_records()
    assert parent.request.completeness == "complete"
    assert child.request.data == {"input": []}
    assert child.request.completeness == "partial"
    assert child.request.missing["/input"].reason == "not_recorded"


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
