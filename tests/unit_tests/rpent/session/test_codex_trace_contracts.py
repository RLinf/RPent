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
from itertools import count

import pytest

from rpent.session import EnvState, Payload
from rpent.session.codex_capture import import_codex_trace


@pytest.fixture
def state(tmp_path):
    state = EnvState(tmp_path / "run")
    state.start_task(instruction="Move.", config=Payload("test.config", {}))
    return state


@pytest.fixture
def native_trace(tmp_path):
    return make_trace(tmp_path / "native" / "bundle")


def make_trace(root):
    payload_ids = count(1)
    (root / "payloads").mkdir(parents=True)
    sequence = 0

    def payload(kind, data):
        name = f"payload-{next(payload_ids)}"
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


def record_inference(
    trace, call_id, request, output, *, thread_id="thread-1", turn="turn-1"
):
    _, payload, event = trace
    event(
        "inference_started",
        thread_id=thread_id,
        codex_turn_id=turn,
        inference_call_id=call_id,
        request_payload=payload("inference_request", request),
    )
    event(
        "inference_completed",
        thread_id=thread_id,
        codex_turn_id=turn,
        inference_call_id=call_id,
        response_id=f"response-{call_id}",
        response_payload=payload("inference_response", {"output_items": output}),
    )


def test_import_links_executed_mcp_calls_in_model_order(state, native_trace):
    root, _, event = native_trace
    tools = {}
    for name in ("right", "left"):
        tools[name] = state.register_tool_call(
            planner_id=None,
            origin="unknown",
            name=name,
            arguments=Payload("mcp.tool_args", {"arm": name}),
            raw_result=Payload("test.result", name),
            status="failed" if name == "right" else "completed",
            source_ids={"codex_bridge_mcp_call_id": f"bridge-{name}"},
        )
    prediction_id = state.register_prediction(
        tool_call_id=tools["left"], instruction="Move."
    )
    action = state.register_action(
        tool_call_id=tools["left"],
        prediction_id=prediction_id,
        source="vla",
        action=Payload("test.action", [1]),
        before_observation_id=None,
        after_observation_id=None,
    )
    output = [
        {"type": "function_call", "call_id": name, "name": name, "arguments": "{}"}
        for name in ("left", "right")
    ]
    record_inference(native_trace, "plan", {"input": []}, output)
    for name in tools:
        event(
            "tool_call_started",
            tool_call_id=f"native-{name}",
            model_visible_call_id=name,
            invocation_payload=None,
        )
        event(
            "mcp_tool_call_correlation_assigned",
            thread_id=None,
            tool_call_id=f"native-{name}",
            mcp_call_id=f"bridge-{name}",
        )

    stats = import_codex_trace(state, root)
    (plan,) = state.planner_records()
    assert stats["errors"] == [] and stats["linked_tool_calls"] == 2
    left, right = plan.tool_calls
    assert [left.tool_call_id, right.tool_call_id] == [tools["left"], tools["right"]]
    assert left.status == "completed" and right.status == "failed"
    for call in plan.tool_calls:
        assert call.origin == "model" and call.planner_id == plan.planner_id
        assert call.raw_result.data == call.name
        assert call.arguments.data == {"arm": call.name}
    (prediction,) = state.prediction_records()
    assert prediction.planner_id == plan.planner_id and prediction.action_ids == [
        action
    ]


def test_import_expands_incremental_requests_and_externalizes_only_protocol_images(
    state, native_trace
):
    root, payload, event = native_trace
    prefix = "data:image/png;base64,"
    uri = prefix + base64.b64encode(b"camera").decode()
    image = {"type": "input_image", "image_url": uri}
    user = {
        "type": "message",
        "role": "user",
        "content": [image, {"type": "input_text", "text": uri}],
    }
    request = {
        "input": [user],
        "tools": [{"name": "camera", "example": image}],
        "instructions": "First.",
    }
    call = {
        "type": "function_call",
        "call_id": "camera",
        "name": "view_image",
        "arguments": "{}",
    }
    result = {"type": "function_call_output", "call_id": "camera", "output": [image]}
    reply = {
        "type": "message",
        "role": "assistant",
        "content": [{"type": "output_text", "text": "Done."}],
    }
    record_inference(native_trace, "one", request, [call])
    event(
        "tool_call_started",
        tool_call_id="native-1",
        model_visible_call_id="camera",
        invocation_payload=payload(
            "tool_invocation", {"tool_name": "view_image", "arguments": "{}"}
        ),
    )
    event(
        "tool_call_ended",
        tool_call_id="native-1",
        status="completed",
        result_payload=payload(
            "tool_result", {"type": "direct_response", "response_item": result}
        ),
    )
    record_inference(
        native_trace,
        "two",
        {"previous_response_id": "response-one", "input": [result], "tools": []},
        [reply],
    )
    record_inference(
        native_trace,
        "three",
        {"previous_response_id": "response-two", "input": [user]},
        [],
        turn="turn-2",
    )

    assert import_codex_trace(state, root)["errors"] == []
    first, second, third = state.planner_records()
    assert first.request.data == request
    assert second.request.data == {"input": [user, call, result], "tools": []}
    assert third.request.data == {"input": [user, call, result, reply, user]}
    assert all(
        record.request.completeness == "complete" for record in (first, second, third)
    )
    assert third.source_ids["provider.previous_response_id"] == "response-two"

    state.end_task(status="completed", stop_reason="finish")
    saved = state.save_planner_trajectory()
    first, _, third = map(
        json.loads, (saved / "planners.jsonl").read_text().splitlines()
    )
    initial = first["request"]["data"]
    assert initial["tools"] == request["tools"]
    assert initial["input"][0]["content"][1]["text"] == uri
    ref = initial["input"][0]["content"][0]["image_url"]
    assert (
        ref["base64_prefix"] == prefix
        and (saved / ref["path"]).read_bytes() == b"camera"
    )
    history = third["request"]["data"]["input"]
    assert (
        history[0]["content"][0]["image_url"]
        == history[-1]["content"][0]["image_url"]
        == ref
    )
    assert history[2]["output"][0]["image_url"] == ref
    feedback = first["tool_calls"][0]["feedback"]["data"]["response_item"]
    assert feedback["output"][0]["image_url"] == ref


@pytest.mark.parametrize(
    "terminal,has_partial", [("failed", True), ("cancelled", False)]
)
def test_import_preserves_partial_responses_and_unfinished_retries(
    state, native_trace, terminal, has_partial
):
    root, payload, event = native_trace
    request = {"input": [{"role": "user", "content": "Move."}]}
    reference = payload("inference_request", request)
    partial = {"output_items": [{"type": "message", "content": "partial"}]}
    event("inference_started", inference_call_id="first", request_payload=reference)
    event(
        f"inference_{terminal}",
        inference_call_id="first",
        **{"error" if has_partial else "reason": "stream interrupted"},
        partial_response_payload=payload("inference_response", partial)
        if has_partial
        else None,
    )
    event("inference_started", inference_call_id="retry", request_payload=reference)

    import_codex_trace(state, root)
    failed, retry = state.planner_records()
    assert failed.status == terminal and retry.status == "open"
    assert failed.request.data == retry.request.data == request
    if has_partial:
        assert (
            failed.response.data == partial
            and failed.response.completeness == "partial"
        )
    else:
        assert (
            failed.response is None
            and failed.missing["response"].detail == "stream interrupted"
        )
    assert retry.response is None and retry.ended_at is None


def test_missing_parent_context_stays_partial_until_a_full_request(state, native_trace):
    root, _, _ = native_trace
    first = {"role": "user", "content": "First."}
    followup = {"role": "user", "content": "Next."}
    compacted = {"type": "compaction", "encrypted_content": "new-context"}
    record_inference(
        native_trace, "one", {"input": [first], "previous_response_id": "lost"}, []
    )
    record_inference(
        native_trace,
        "two",
        {"input": [followup], "previous_response_id": "response-one"},
        [],
    )
    record_inference(native_trace, "reset", {"input": [compacted]}, [])
    record_inference(
        native_trace,
        "next",
        {"input": [followup], "previous_response_id": "response-reset"},
        [],
    )

    import_codex_trace(state, root)
    one, two, reset, continued = state.planner_records()
    for record, message in ((one, first), (two, followup)):
        assert record.request.data == {"input": [message]}
        assert record.request.completeness == "partial"
        assert record.request.missing["/input"].reason == "not_recorded"
    assert reset.request.completeness == continued.request.completeness == "complete"
    assert continued.request.data == {"input": [compacted, followup]}


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
    record_inference(
        native_trace,
        "plan",
        {"input": []},
        [
            {
                "type": "custom_tool_call",
                "name": "exec",
                "input": "await tools.move({})",
                "call_id": "exec-1",
            }
        ],
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
        record_inference(
            native_trace,
            thread_id,
            {"input": []},
            [
                {
                    "type": "function_call",
                    "name": "move",
                    "arguments": "{}",
                    "call_id": "reused-call-id",
                }
            ],
            thread_id=thread_id,
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


def test_bundles_with_reused_ids_cannot_share_payloads_or_parent_context(
    state, tmp_path
):
    for label in ("first", "second"):
        trace = make_trace(tmp_path / label)
        _, payload, event = trace
        request = {"input": [label]}
        if label == "second":
            request["previous_response_id"] = "response-first"
        call = {
            "type": "function_call",
            "call_id": "same-id",
            "name": "inspect",
            "arguments": "{}",
        }
        record_inference(trace, label, request, [call])
        event(
            "tool_call_started",
            tool_call_id="native-1",
            model_visible_call_id="same-id",
            invocation_payload=payload(
                "tool_invocation", {"tool_name": "inspect", "arguments": label}
            ),
        )

    stats = import_codex_trace(state, tmp_path)
    first, second = state.planner_records()
    assert stats["bundles"] == 2
    for label, plan in (("first", first), ("second", second)):
        assert plan.request.data == {"input": [label]}
        (tool,) = plan.tool_calls
        assert tool.arguments.data["arguments"] == label
    assert first.request.completeness == "complete"
    assert (
        second.request.completeness == "partial" and "/input" in second.request.missing
    )


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
