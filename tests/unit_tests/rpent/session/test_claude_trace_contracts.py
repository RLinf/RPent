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

"""Claude Messages API recordings retain call identity and native image content."""

import base64
import json
from datetime import datetime, timezone

import pytest

from rpent.session import EnvState, MissingInfo, Payload
from rpent.session.claude_capture import import_claude_trace


@pytest.fixture
def state(tmp_path):
    state = EnvState(tmp_path)
    state.start_task(
        instruction="Inspect the camera.", config=Payload("test.config", {})
    )
    return state


def test_import_preserves_calls_and_externalizes_only_image_blocks(state):
    run = state.trajectory_dir / "claude" / "run1"
    run.mkdir(parents=True)
    encoded = base64.b64encode(b"camera image").decode()
    image = {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/png", "data": encoded},
    }
    result = {"type": "tool_result", "tool_use_id": "tool-1", "content": [image]}
    request = {
        "messages": [
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "tool-1", "input": {"example": image}}
                ],
            },
            {"role": "user", "content": [result, image]},
        ],
        "tools": [{"name": "camera"}],
    }
    response = {"id": "msg-main", "content": [{"type": "text", "text": "Done."}]}
    entries = []
    for name, source in (("title", "generate_session_title"), ("main", "sdk")):
        entry = {
            "request_file": f"{name}.request.json",
            "response_file": f"{name}.response.json",
            "session_id": "session",
            "request_id": f"req-{name}",
            "message_id": f"msg-{name}",
            "query_source": source,
            "timestamp": "2026-11-09T03:29:37.308Z",
        }
        (run / entry["request_file"]).write_text(
            json.dumps(request if name == "main" else {"messages": []})
        )
        (run / entry["response_file"]).write_text(
            json.dumps(response if name == "main" else {"id": "msg-title"})
        )
        entries.append(json.dumps(entry))
    (run / "index.jsonl").write_text("\n".join([*entries, entries[-1]]))

    assert import_claude_trace(state, run) == {"planners": 2, "errors": []}
    assert import_claude_trace(state, run) == {"planners": 0, "errors": []}
    title, main = state.planner_records()
    assert title.source_ids["claude.query_source"] == "generate_session_title"
    assert main.backend == "claude_code" and main.status == "completed"
    assert main.source_ids["claude.query_source"] == "sdk"
    assert main.source_ids["claude.session_id"] == "session"
    assert main.source_ids["provider.request_id"] == "req-main"
    assert main.source_ids["provider.response_id"] == "msg-main"
    assert main.ended_at == datetime(2026, 11, 9, 3, 29, 37, 308000, timezone.utc)
    assert main.request.data == request and main.response.data == response

    state.end_task(status="completed", stop_reason="model_output")
    root = state.save_planner_trajectory()
    saved = json.loads((root / "planners.jsonl").read_text().splitlines()[1])
    assistant, user = saved["request"]["data"]["messages"]
    assert assistant["content"][0]["input"]["example"] == image
    nested, direct = user["content"]
    ref = nested["content"][0]["source"]["data"]
    assert ref == direct["source"]["data"] and ref["$type"] == "artifact_ref"
    assert (root / ref["path"]).read_bytes() == b"camera image"
    assert len(list((root / "artifacts").iterdir())) == 1


def test_import_keeps_unindexed_attempt_and_survives_a_damaged_index(state, tmp_path):
    run = tmp_path / "native"
    run.mkdir()
    (run / "retry.request.json").write_text('{"messages": []}')
    assert import_claude_trace(state, run)["planners"] == 1
    (retry,) = state.planner_records()
    assert retry.status == "open" and retry.response is None and retry.ended_at is None
    assert retry.missing["response"].reason == "not_recorded"

    (run / "retry.response.json").write_text(
        '{"content": [{"type": "thinking", "thinking": "[REDACTED]"}]}'
    )
    entry = {
        "request_file": "retry.request.json",
        "response_file": "retry.response.json",
    }
    (run / "another.request.json").write_text('{"messages": []}')
    missing = {
        "request_file": "another.request.json",
        "response_file": "missing.response.json",
    }
    (run / "index.jsonl").write_text(
        f"{{\n{json.dumps(entry)}\n{json.dumps(missing)}\n"
    )
    stats = import_claude_trace(state, run)
    assert stats["planners"] == 1 and len(stats["errors"]) == 2
    assert (
        "index.jsonl:1:" in stats["errors"][0]
        and "missing.response.json" in stats["errors"][1]
    )
    updated, damaged = state.planner_records()
    assert updated.planner_id == retry.planner_id and updated.status == "completed"
    assert "response" not in updated.missing
    assert updated.response.completeness == "partial"
    assert updated.response.missing["/content/0"].reason == "not_exposed"
    assert damaged.status == "completed" and damaged.request.data == {"messages": []}
    assert (
        damaged.response is None
        and damaged.missing["response"].reason == "capture_failed"
    )
    (run / "missing.response.json").write_text('{"content": []}')
    assert import_claude_trace(state, run)["planners"] == 0
    repaired = state.get_planner(damaged.planner_id)
    assert (
        repaired.response.data == {"content": []} and "response" not in repaired.missing
    )


def write_recording(run, stem, response, messages=()):
    run.mkdir(parents=True, exist_ok=True)
    request_file, response_file = f"{stem}.request.json", f"{stem}.response.json"
    (run / request_file).write_text(json.dumps({"messages": list(messages)}))
    (run / response_file).write_text(json.dumps({"content": response}))
    with (run / "index.jsonl").open("a") as stream:
        stream.write(
            json.dumps({"request_file": request_file, "response_file": response_file})
            + "\n"
        )


def tool_use(key, name, **arguments):
    return {"type": "tool_use", "id": key, "name": name, "input": arguments}


def record_camera(state, run_id, key):
    ids = {"claude.run_id": run_id}
    if key is not None:
        ids["claude.tool_use_id"] = key
    return state.register_tool_call(
        planner_id=None,
        origin="unknown",
        name="camera",
        status="completed",
        arguments=Payload("mcp.tool_args", {"camera": "front"}),
        raw_result=Payload("rpent.tool_result", {"images": [b"camera"]}),
        feedback=Payload(
            "claude.sdk_tool_result",
            {"content": [{"type": "text", "text": "SDK feedback"}]},
        ),
        source_ids=ids,
        missing={"planner_id": MissingInfo("pending", "Waiting for import.")},
    )


def test_import_links_executions_in_model_order_and_fills_native_results(state):
    run = state.trajectory_dir / "claude" / "solve-1"
    second = record_camera(state, run.name, "b")
    first = record_camera(state, run.name, "a")
    original = state.get_tool_call(first)
    prediction = state.register_prediction(tool_call_id=first, instruction="Inspect.")
    action = state.register_action(
        tool_call_id=first,
        prediction_id=prediction,
        source="vla",
        action=Payload("test.action", [1]),
        before_observation_id=None,
        after_observation_id=None,
    )
    uses = [
        tool_use("a", "mcp__rpent__camera", camera="front"),
        tool_use("read", "Read", file_path="image.png"),
        tool_use("b", "mcp__rpent__camera", camera="front"),
        tool_use("bash", "Bash", command="pwd"),
        tool_use("rejected", "mcp__rpent__camera"),
    ]
    stale = {"type": "tool_result", "tool_use_id": "read", "content": "Old history."}
    write_recording(run, "plan", uses, [{"role": "user", "content": [stale]}])
    assert import_claude_trace(state, run)["errors"] == []
    (plan,) = state.planner_records()
    assert [call.name for call in plan.tool_calls] == [
        "camera",
        "Read",
        "camera",
        "Bash",
    ]
    left, read, right, bash = plan.tool_calls
    assert [left.tool_call_id, right.tool_call_id] == [first, second]
    assert left.origin == "model" and left.status == original.status
    assert (
        left.arguments == original.arguments and left.raw_result == original.raw_result
    )
    assert left.feedback == original.feedback and "planner_id" not in left.missing
    (linked_prediction,) = state.prediction_records()
    assert (
        linked_prediction.planner_id == plan.planner_id
        and linked_prediction.action_ids == [action]
    )
    assert read.status == bash.status == "requested" and read.feedback is None
    assert (
        read.arguments.data == uses[1]["input"]
        and read.missing["feedback"].reason == "not_recorded"
    )

    image = {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": "image/png",
            "data": base64.b64encode(b"disk image").decode(),
        },
    }
    read_result = {"type": "tool_result", "tool_use_id": "read", "content": [image]}
    bash_result = {
        "type": "tool_result",
        "tool_use_id": "bash",
        "is_error": True,
        "content": "Denied.",
    }
    messages = [
        {"role": "user", "content": [stale]},
        {"role": "assistant", "content": uses},
        {
            "role": "user",
            "content": [
                bash_result,
                read_result,
                {
                    "type": "tool_result",
                    "tool_use_id": "a",
                    "content": "Model-visible feedback.",
                },
            ],
        },
    ]
    write_recording(run, "followup", [], messages)
    write_recording(run, "repeat-history", [], messages)
    assert import_claude_trace(state, run) == {"planners": 2, "errors": []}
    tools = state.get_planner(plan.planner_id).tool_calls
    assert [call.tool_call_id for call in tools] == [
        call.tool_call_id for call in plan.tool_calls
    ]
    assert tools[0] == left and tools[2] == right
    read, bash = tools[1], tools[3]
    assert read.status == "completed" and read.feedback.data == read_result
    assert bash.status == "failed" and bash.feedback.data == bash_result
    assert (
        read.raw_result is None and read.missing["raw_result"].reason == "not_exposed"
    )
    assert read.started_at is None and read.ended_at is None
    assert "feedback" not in read.missing
    assert import_claude_trace(state, run) == {"planners": 0, "errors": []}
    assert state.get_planner(plan.planner_id).tool_calls == tools
    assert len(state.tool_call_records()) == 4

    state.end_task(status="completed", stop_reason="finish")
    root = state.save_planner_trajectory()
    saved, followup, _ = map(
        json.loads, (root / "planners.jsonl").read_text().splitlines()
    )
    ref = saved["tool_calls"][1]["feedback"]["data"]["content"][0]["source"]["data"]
    assert (
        ref
        == followup["request"]["data"]["messages"][-1]["content"][1]["content"][0][
            "source"
        ]["data"]
    )
    assert (root / ref["path"]).read_bytes() == b"disk image"
    assert not (root / "unlinked_tool_calls.jsonl").exists()


def test_import_leaves_ambiguous_executions_unlinked_and_isolates_runs(state, tmp_path):
    run, other = tmp_path / "one" / "native", tmp_path / "two" / "native"
    valid = record_camera(state, "one", "shared")
    other_call = record_camera(state, "two", "shared")
    keys = [None, "twice", "twice", "repeat", "wrong", "absent"]
    unlinked = [record_camera(state, "one", key) for key in keys]
    uses = [
        tool_use(key, "mcp__rpent__camera") for key in ("shared", "twice", "repeat")
    ]
    uses.append(tool_use("wrong", "mcp__rpent__other"))
    write_recording(run, "one", uses)
    write_recording(run, "two", [tool_use("repeat", "mcp__rpent__camera")])
    write_recording(other, "one", [tool_use("shared", "mcp__rpent__camera")])

    import_claude_trace(state, run, run_id="one")
    first, second = state.planner_records()
    assert [call.tool_call_id for call in first.tool_calls] == [valid]
    assert second.tool_calls == []
    assert state.get_tool_call(other_call).missing["planner_id"].reason == "pending"
    for call_id, detail in zip(
        unlinked,
        ["provide", "Ambiguous", "Ambiguous", "Ambiguous", "name", "No Claude"],
        strict=True,
    ):
        call = state.get_tool_call(call_id)
        assert call.planner_id is None and call.status == "completed"
        assert detail in call.missing["planner_id"].detail
    import_claude_trace(state, other, run_id="two")
    assert (
        state.get_tool_call(other_call).planner_id
        == state.planner_records()[-1].planner_id
    )
    assert state.get_tool_call(valid).planner_id == first.planner_id
    state.end_task(status="completed", stop_reason="finish")
    root = state.save_planner_trajectory()
    saved = list(
        map(json.loads, (root / "unlinked_tool_calls.jsonl").read_text().splitlines())
    )
    assert [row["tool_call_id"] for row in saved] == unlinked
    assert all(row["missing"]["planner_id"]["reason"] != "pending" for row in saved)


def test_retry_corrects_a_conflicting_link_and_inserts_late_execution_in_order(
    state, tmp_path
):
    run = tmp_path / "native"
    conflicted = record_camera(state, "solve", "reused")
    stable = record_camera(state, "solve", "stable")
    state.register_prediction(tool_call_id=conflicted, instruction="Inspect.")
    uses = [tool_use(key, "mcp__rpent__camera") for key in ("reused", "stable", "late")]
    uses.append(tool_use("read", "Read", file_path="test.txt"))
    write_recording(run, "plan", uses)
    # A copied native directory has no solve ID until the caller supplies it.
    assert import_claude_trace(state, run) == {"planners": 1, "errors": []}
    (offline,) = state.planner_records()
    assert [call.name for call in offline.tool_calls] == ["Read"]
    assert state.get_tool_call(stable).planner_id is None
    assert import_claude_trace(state, run, run_id="solve")["errors"] == []
    (plan,) = state.planner_records()
    assert [call.tool_call_id for call in plan.tool_calls[:2]] == [conflicted, stable]
    assert state.prediction_records()[0].planner_id == plan.planner_id
    read_id = plan.tool_calls[2].tool_call_id
    assert read_id == offline.tool_calls[0].tool_call_id
    assert (
        plan.source_ids["claude.run_id"]
        == plan.tool_calls[2].source_ids["claude.run_id"]
        == "solve"
    )

    late = record_camera(state, "solve", "late")
    write_recording(run, "retry", [tool_use("reused", "mcp__rpent__camera")])
    assert import_claude_trace(state, run)["errors"] == []
    assert [
        call.tool_call_id for call in state.get_planner(plan.planner_id).tool_calls
    ] == [stable, late, read_id]
    unlinked = state.get_tool_call(conflicted)
    assert unlinked.planner_id is None and unlinked.origin == "unknown"
    assert "Ambiguous" in unlinked.missing["planner_id"].detail
    assert unlinked.raw_result == plan.tool_calls[0].raw_result
    assert state.prediction_records()[0].planner_id is None
    before = state.planner_records()
    assert import_claude_trace(state, run, run_id="other")["errors"]
    assert state.planner_records() == before


def test_native_results_require_unambiguous_evidence_in_the_request(state):
    run = state.trajectory_dir / "claude" / "run"
    uses = [
        tool_use("conflict", "Read"),
        {"type": "tool_use", "name": "Bash", "input": {}},
        tool_use("reused", "Read"),
    ]
    write_recording(run, "plan", uses)
    confirmed = {
        "type": "tool_result",
        "tool_use_id": "reused",
        "content": "first result",
    }
    conflicting = {"type": "tool_result", "tool_use_id": "conflict", "content": "one"}
    write_recording(
        run,
        "followup",
        [tool_use("reused", "Bash")],
        [
            {"role": "assistant", "content": uses},
            {
                "role": "user",
                "content": [conflicting, {**conflicting, "content": "two"}, confirmed],
            },
        ],
    )
    assert import_claude_trace(state, run)["errors"] == []
    plan, followup = state.planner_records()
    first, no_id, read = plan.tool_calls
    assert first.feedback is None and "Conflicting" in first.missing["feedback"].detail
    assert no_id.status == "requested" and no_id.feedback is None
    assert read.status == "completed" and read.feedback.data == confirmed
    (last,) = followup.tool_calls
    assert last.status == "requested" and last.feedback is None
    assert import_claude_trace(state, run)["planners"] == 0
    assert state.planner_records() == [plan, followup]
