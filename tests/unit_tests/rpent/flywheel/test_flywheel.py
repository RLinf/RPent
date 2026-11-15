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

from copy import deepcopy

import pytest

from rpent.flywheel.flywheel import (
    select_vla_segments,
)


def _ranges(result):
    return [
        (segment["start_index"], segment["stop_index"])
        for segment in result["segments"]
    ]


def test_selects_executed_predictions_across_chunks_without_loading_images(trajectory):
    trajectory["predictions"][1]["tool_call_id"] = "second-tool"
    for action in trajectory["actions"][4:7]:
        action["tool_call_id"] = "second-tool"
    for path in trajectory["root"].rglob("*.npy"):
        path.unlink()
    result = select_vla_segments(trajectory)

    assert result["segments"] == [
        {"start_index": 1, "stop_index": 7, "instruction": "pick up the bowl"}
    ]
    assert result["rejected"] == []


@pytest.mark.parametrize("success", [False, "true"])
def test_planner_completion_requires_explicit_environment_success(trajectory, success):
    trajectory["task"]["result"]["data"]["environment_success"] = success
    result = select_vla_segments(trajectory)
    assert result["segments"] == []
    assert result["rejected"][0]["reason"] == "no_confirmed_environment_success"


def test_missing_task_success_excludes_the_trajectory(trajectory):
    trajectory["task"]["result"]["missing"] = {
        "/environment_success": {"reason": "capture_failed"}
    }
    selected = select_vla_segments(trajectory)
    assert selected["segments"] == []
    assert selected["rejected"][0]["reason"] == "no_confirmed_environment_success"


@pytest.mark.parametrize("success,done", [(True, False), (False, True)])
def test_robocasa_selection_uses_task_success_instead_of_done(
    robocasa_trajectory, success, done
):
    robocasa_trajectory["task"]["result"]["data"]["environment_success"] = success
    robocasa_trajectory["actions"][2]["result"]["data"]["done"] = done
    assert _ranges(select_vla_segments(robocasa_trajectory)) == (
        [(1, 7)] if success else []
    )


@pytest.mark.parametrize("index", [6, 11], ids=["gripper", "control_mode"])
def test_robocasa_rejects_an_unmapped_control_value(robocasa_trajectory, index):
    action = robocasa_trajectory["actions"][3]
    action["action"]["data"]["value"][index] = 0
    result = select_vla_segments(robocasa_trajectory)
    assert _ranges(result) == [(1, 3), (4, 7)]
    assert result["rejected"][0]["action_id"] == action["action_id"]


def test_includes_success_action_but_excludes_later_chunk_padding(trajectory):
    trajectory["actions"][4]["result"]["data"]["terminated"] = True
    result = select_vla_segments(trajectory)
    assert _ranges(result) == [(1, 5)]
    assert [
        row["action_id"]
        for row in result["rejected"]
        if row["reason"] == "after_environment_success"
    ] == [action["action_id"] for action in trajectory["actions"][5:7]]


@pytest.mark.parametrize("source", ["direct", "tool_computed"])
def test_non_vla_action_splits_the_segment_even_when_prediction_claims_it(
    trajectory, source
):
    trajectory["actions"][2]["source"] = source
    assert _ranges(select_vla_segments(trajectory)) == [(1, 2), (3, 7)]


@pytest.mark.parametrize("claim", ["no_prediction", "another_tool"])
def test_only_actions_owned_by_the_prediction_are_selected(trajectory, claim):
    if claim == "no_prediction":
        trajectory["predictions"].pop(0)
    else:
        trajectory["predictions"][0]["tool_call_id"] = "other-tool"
    assert _ranges(select_vla_segments(trajectory)) == [(4, 7)]


def test_instruction_change_starts_a_new_segment(trajectory):
    trajectory["predictions"][1]["instruction"] = "different"
    assert _ranges(select_vla_segments(trajectory)) == [(1, 4), (4, 7)]


@pytest.mark.parametrize("index,expected", [(1, [(1, 6)]), (2, [(1, 2), (2, 6)])])
def test_missing_action_breaks_continuity_without_discarding_other_actions(
    trajectory, index, expected
):
    missing_id = trajectory["actions"].pop(index)["action_id"]
    result = select_vla_segments(trajectory)
    assert _ranges(result) == expected
    assert any(
        row.get("action_id") == missing_id and row["reason"] == "missing_action_record"
        for row in result["rejected"]
    )


@pytest.mark.parametrize("change", ["id", "payload"])
def test_disconnected_observations_are_not_joined(trajectory, change):
    action = trajectory["actions"][2]
    if change == "id":
        action["before_observation_id"] = "O-disconnected"
        action["before_observation"]["observation_id"] = "O-disconnected"
    else:
        action["before_observation"]["payload"]["data"]["states"]["value"][0] += 1
    assert _ranges(select_vla_segments(trajectory)) == [(1, 2), (2, 7)]


def test_action_record_cannot_contain_an_entire_chunk(trajectory):
    payload = trajectory["actions"][2]["action"]
    payload["data"]["shape"] = [1, 7]
    payload["data"]["value"] = [payload["data"]["value"]]
    result = select_vla_segments(trajectory)
    assert _ranges(result) == [(1, 2), (3, 7)]
    assert any("expected shape (7,)" in row["reason"] for row in result["rejected"])


@pytest.mark.parametrize("field", ["image", "state"])
def test_invalid_before_observation_splits_the_segment(trajectory, field):
    action = trajectory["actions"][2]
    data = action["before_observation"]["payload"]["data"]
    if field == "image":
        data["main_images"]["dtype"] = "float32"
    else:
        data["states"]["value"][0] = float("nan")
    result = select_vla_segments(trajectory)
    assert _ranges(result) == [(1, 2), (3, 7)]
    assert result["rejected"][0]["action_id"] == action["action_id"]


def test_selection_only_requires_training_fields_in_before_observations(trajectory):
    last = trajectory["actions"][-2]
    last["after_observation"]["payload"]["data"] = {}
    last["before_observation"]["payload"]["missing"] = {
        "/extra_view_images": {"reason": "not_exposed"}
    }
    trajectory["actions"][-3]["after_observation"] = deepcopy(
        last["before_observation"]
    )
    assert _ranges(select_vla_segments(trajectory)) == [(1, 7)]


def test_invalid_prediction_order_excludes_its_actions(trajectory):
    prediction = trajectory["predictions"][0]
    prediction["action_ids"].reverse()
    result = select_vla_segments(trajectory)
    assert _ranges(result) == [(4, 7)]
    assert result["rejected"][0]["prediction_id"] == prediction["prediction_id"]


def test_two_predictions_cannot_claim_the_same_executed_action(trajectory):
    duplicate = deepcopy(trajectory["predictions"][0])
    duplicate["prediction_id"] = "V-duplicate"
    trajectory["predictions"].append(duplicate)
    result = select_vla_segments(trajectory)
    assert _ranges(result) == [(4, 7)]
    assert {
        (row["prediction_id"], row["action_id"])
        for row in result["rejected"]
        if row["reason"] == "multiple_predictions_claim_action"
    } == {
        (prediction["prediction_id"], action_id)
        for prediction in (trajectory["predictions"][0], duplicate)
        for action_id in prediction["action_ids"]
    }


@pytest.mark.parametrize("bimanual_trajectory", [("robotwin", "ee")], indirect=True)
def test_robotwin_truncation_before_success_excludes_the_trajectory(
    bimanual_trajectory,
):
    bimanual_trajectory["actions"][3]["result"]["data"]["truncated"] = True
    selected = select_vla_segments(bimanual_trajectory)
    assert selected["segments"] == []
    assert selected["rejected"][0]["reason"] == "no_confirmed_environment_success"


@pytest.mark.parametrize("bimanual_trajectory", [("robodojo", "joint")], indirect=True)
def test_a_missing_native_success_flag_excludes_the_trajectory(bimanual_trajectory):
    result = bimanual_trajectory["actions"][-1]["result"]
    result["missing"] = {"/done": {"reason": "capture_failed"}}
    assert select_vla_segments(bimanual_trajectory)["segments"] == []


@pytest.mark.parametrize(
    "path",
    ["/robot_state", "/observation/left_camera"],
)
@pytest.mark.parametrize("bimanual_trajectory", [("robotwin", "ee")], indirect=True)
def test_marking_a_parent_field_missing_invalidates_the_training_frame(
    bimanual_trajectory, path
):
    action = bimanual_trajectory["actions"][3]
    action["before_observation"]["payload"]["missing"] = {
        path: {"reason": "capture_failed"}
    }
    selected = select_vla_segments(bimanual_trajectory)
    assert _ranges(selected) == [(1, 3), (4, 7)]
    assert selected["rejected"][0]["action_id"] == action["action_id"]
