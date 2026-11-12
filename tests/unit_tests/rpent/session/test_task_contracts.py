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

import base64
import json
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest

from rpent.session import (
    DataRef,
    EnvState,
    MissingInfo,
    Payload,
)


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


def test_task_can_only_be_started_and_finished_once(tmp_path):
    state = EnvState(tmp_path)
    config = Payload("json", {})
    with pytest.raises(RuntimeError, match="not running"):
        state.register_planner(backend="api")
    task_id = state.start_task(instruction="Move.", config=config)
    with pytest.raises(RuntimeError, match="already been started"):
        state.start_task(instruction="Replace.", config=config)
    with pytest.raises(RuntimeError, match=r"end_task\(\) must finish"):
        state.save_planner_trajectory()
    state.end_task(status="completed", stop_reason="finish")
    with pytest.raises(RuntimeError, match="not running"):
        state.register_planner(backend="api")
    with pytest.raises(RuntimeError, match="not running"):
        state.end_task(status="failed", stop_reason="error")
    with pytest.raises(RuntimeError, match="already been started"):
        state.start_task(instruction="Replace.", config=config)
    assert state.get_task().task_id == task_id


def test_rejected_update_does_not_partially_link_or_overwrite_a_tool(state):
    planner = state.register_planner(backend="api")
    tool = state.register_tool_call(
        planner_id=None,
        name="move",
        arguments=Payload("json", {}),
        raw_result=Payload("test.result", None),
    )
    state.register_prediction(tool_call_id=tool, instruction="Move.")
    with pytest.raises(ValueError, match="raw_result has already been captured"):
        state.update_tool_call(
            tool,
            planner_id=planner,
            status="failed",
            raw_result=Payload("test.result", [1]),
        )

    call = state.get_tool_call(tool)
    assert call.raw_result == Payload("test.result", None)
    assert call.status == "requested" and call.planner_id is None
    assert state.get_planner(planner).tool_calls == []
    assert state.prediction_records()[0].planner_id is None


def test_record_snapshots_are_isolated_from_callers(state, tool_id):
    positions = np.array([1, 2], dtype=np.float32)
    observation = state.register_observation(payload=Payload("test.obs", positions))
    state.update_tool_call(
        tool_id, raw_result=Payload("test.result", {"positions": positions})
    )

    positions[:] = 99
    state.get_observation(observation).payload.data[:] = 0
    state.get_tool_call(tool_id).raw_result.data.clear()
    np.testing.assert_array_equal(
        state.get_observation(observation).payload.data, [1, 2]
    )
    np.testing.assert_array_equal(
        state.get_tool_call(tool_id).raw_result.data["positions"], [1, 2]
    )


def test_concurrent_registrations_keep_distinct_ids_and_their_planner(state):
    planners = [state.register_planner(backend="api") for _ in range(2)]

    def register(index):
        return state.register_tool_call(
            planner_id=planners[index % 2],
            name="move",
            arguments=Payload("json", index),
        )

    with ThreadPoolExecutor(max_workers=4) as pool:
        tools = list(pool.map(register, range(16)))
    assert len(set(tools)) == len(tools)
    for index, planner in enumerate(planners):
        assert {
            call.tool_call_id for call in state.get_planner(planner).tool_calls
        } == set(tools[index::2])


def test_legacy_rollback_and_reset_preserve_executed_actions(state, tool_id):
    with pytest.raises(RuntimeError, match="failed to record step"):
        with state.record_step(state={}):
            action = state.register_action(
                tool_call_id=tool_id,
                source="direct",
                action=Payload("test.action", [1]),
                before_observation_id=None,
                after_observation_id=None,
            )
            raise ValueError("failure after execution")
    state.reset()
    assert state.records() == []
    assert state.get_task().status == "running"
    assert state.get_action(action).tool_call_id == tool_id


def test_prediction_cannot_claim_another_tools_action(state, tool_id):
    other = state.register_tool_call(
        planner_id=None, name="other", arguments=Payload("json", {})
    )
    action = state.register_action(
        tool_call_id=tool_id,
        source="vla",
        action=Payload("test.action", [1]),
        before_observation_id=None,
        after_observation_id=None,
    )
    with pytest.raises(ValueError, match="does not belong"):
        state.register_prediction(
            tool_call_id=other, instruction="Move.", action_ids=[action]
        )
    assert state.prediction_records() == []


def test_base64_and_bytes_share_an_attachment_and_keep_their_encoding(state):
    image = b"image"
    encoded = base64.b64encode(image).decode("ascii")
    prefix = "data:image/png;base64,"
    values = [image, encoded, prefix + encoded, encoded]
    state.register_planner(
        backend="api",
        request=Payload(
            "test.request",
            {"images/~": values},
            base64_paths={
                "/images~1~0/1": "",
                "/images~1~0/2": prefix,
            },
        ),
    )
    state.end_task(status="completed", stop_reason="finish")
    root = state.save_planner_trajectory()
    saved = json.loads((root / "planners.jsonl").read_text())["request"]["data"][
        "images/~"
    ]
    raw, bare, uri, text = saved
    assert raw["path"] == bare["path"] == uri["path"]
    assert [ref["base64_prefix"] for ref in (raw, bare, uri)] == [None, "", prefix]
    assert (root / raw["path"]).read_bytes() == image
    assert text == encoded


@pytest.mark.parametrize(
    "value,prefix",
    [
        ("%%%", ""),
        ("非ASCII", ""),
        ("x" * len("data:image/png;base64,") + "aW1hZ2U=", "data:image/png;base64,"),
    ],
)
def test_trajectory_base64_decode_failure_marks_only_the_affected_field(
    state, value, prefix
):
    planner_id = state.register_planner(
        backend="codex",
        request=Payload(
            "test.request",
            {"images/~": [value, "aW1hZ2U="]},
            completeness="complete",
            missing={"/metadata": MissingInfo("not_exposed", "No metadata.")},
            base64_paths={"/images~1~0/0": prefix, "/images~1~0/1": ""},
        ),
    )
    before = state.get_planner(planner_id)
    state.end_task(status="completed", stop_reason="finish")

    root = state.save_planner_trajectory()

    request = json.loads((root / "planners.jsonl").read_text())["request"]
    assert request["data"]["images/~"][0] is None
    assert request["completeness"] == "partial"
    assert set(request["missing"]) == {"/metadata", "/images~1~0/0"}
    assert request["missing"]["/images~1~0/0"]["reason"] == "capture_failed"
    assert request["missing"]["/images~1~0/0"]["detail"]
    assert request["missing"]["/metadata"]["reason"] == "not_exposed"
    reference = request["data"]["images/~"][1]
    assert reference["base64_prefix"] == ""
    assert (root / reference["path"]).read_bytes() == b"image"
    assert state.get_planner(planner_id) == before


def test_unlinked_execution_can_be_recovered_from_a_failed_trajectory(state):
    tool = state.register_tool_call(
        planner_id=None, name="move", arguments=Payload("json", {}), status="completed"
    )
    prediction = state.register_prediction(tool_call_id=tool, instruction="Move.")
    action = state.register_action(
        tool_call_id=tool,
        prediction_id=prediction,
        source="vla",
        action=Payload("test.action", [1]),
        before_observation_id=None,
        after_observation_id=None,
    )
    state.end_task(status="failed", stop_reason="planner_error")
    root = state.save_planner_trajectory()
    saved_tool = json.loads((root / "unlinked_tool_calls.jsonl").read_text())
    saved_prediction = json.loads((root / "predictions.jsonl").read_text())
    saved_action = json.loads((root / "actions.jsonl").read_text())
    assert saved_tool["planner_id"] is saved_prediction["planner_id"] is None
    assert (
        saved_tool["tool_call_id"]
        == saved_action["tool_call_id"]
        == saved_prediction["tool_call_id"]
        == tool
    )
    assert saved_prediction["action_ids"] == [saved_action["action_id"]] == [action]


def test_literal_reference_tags_remain_distinguishable_from_references(state):
    reference = DataRef("tool_call", "T1", "/raw_result/data")
    literal = {"$type": "data_ref", **asdict(reference)}
    state.register_planner(
        backend="api", request=Payload("test.request", [reference, literal])
    )
    state.end_task(status="completed", stop_reason="finish")
    root = state.save_planner_trajectory()
    data = json.loads((root / "planners.jsonl").read_text())["request"]["data"]
    assert data == [literal, {"$type": "dict", "value": literal}]


def test_record_write_failure_preserves_file_and_can_be_retried(
    state, tool_id, monkeypatch
):
    state.end_task(status="completed", stop_reason="finish")
    root = state.trajectory_dir
    root.mkdir(parents=True)
    destination = root / "planners.jsonl"
    destination.write_text("previous")
    replace_file = os.replace

    def fail_replace(src, dest):
        if dest == destination:
            raise OSError("disk full")
        replace_file(src, dest)

    with monkeypatch.context() as patch:
        patch.setattr(os, "replace", fail_replace)
        with pytest.raises(OSError, match="disk full"):
            state.save_planner_trajectory()
    assert destination.read_text() == "previous"
    assert not (root / "task.json").exists()
    assert not list(root.rglob(".*.tmp*"))

    state.save_planner_trajectory()
    assert (
        json.loads(destination.read_text())["tool_calls"][0]["tool_call_id"] == tool_id
    )
    assert (root / "task.json").is_file()


def test_attachment_failure_marks_only_the_missing_field_and_can_be_retried(
    state, monkeypatch
):
    state.register_planner(
        backend="api",
        request=Payload(
            "test.request",
            {"camera": b"image", "text": "Move."},
            completeness="complete",
        ),
    )
    state.end_task(status="failed", stop_reason="execution_error")
    with monkeypatch.context() as patch:
        patch.setattr(
            state, "save_trajectory_artifact", Mock(side_effect=OSError("disk full"))
        )
        root = state.save_planner_trajectory()
    path = root / "planners.jsonl"
    request = json.loads(path.read_text())["request"]
    assert request["data"] == {"camera": None, "text": "Move."}
    assert request["completeness"] == "partial"
    assert request["missing"] == {
        "/camera": {"reason": "capture_failed", "detail": "OSError: disk full"}
    }

    state.save_planner_trajectory()
    restored = json.loads(path.read_text())["request"]
    assert restored["completeness"] == "complete" and restored["missing"] == {}
    assert (root / restored["data"]["camera"]["path"]).read_bytes() == b"image"


@pytest.mark.parametrize("name", ["payload-000001.bin", ".payload-000001.tmp.bin"])
def test_trajectory_save_preserves_live_artifacts_even_on_retry(state, name):
    camera = state.save_trajectory_artifact(b"live", name=name)
    state.register_tool_call(
        planner_id=None,
        name="inspect",
        arguments=Payload("test.args", [camera, b"other"]),
    )
    state.end_task(status="completed", stop_reason="finish")
    for _ in range(2):
        root = state.save_planner_trajectory()
        call = json.loads((root / "unlinked_tool_calls.jsonl").read_text())
        existing, other = call["arguments"]["data"]
        assert existing["path"] == camera.path
        assert (root / existing["path"]).read_bytes() == b"live"
        assert (root / other["path"]).read_bytes() == b"other"
    assert len(list((root / "artifacts").iterdir())) == 2
