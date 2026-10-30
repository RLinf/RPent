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

import numpy as np
import pytest

from rpent.flywheel.prepare import select_vla_segments
from rpent.flywheel.reader import read_array


def _ranges(result):
    return [
        (segment["start_index"], segment["stop_index"])
        for segment in result["segments"]
    ]


def test_selects_executed_predictions_across_chunks_without_loading_images(trajectory):
    original = deepcopy(trajectory)
    for path in trajectory["root"].rglob("*.npy"):
        path.unlink()
    result = select_vla_segments(trajectory)

    assert result["segments"] == [
        {"start_index": 1, "stop_index": 7, "instruction": "pick up the bowl"}
    ]
    assert result["rejected"] == []
    assert trajectory == original


@pytest.mark.parametrize("success", [False, None, "true", 1])
@pytest.mark.parametrize("robot", ["libero", "robocasa", "robodojo", "robotwin"])
def test_planner_completion_does_not_establish_environment_success(
    trajectory, success, robot
):
    trajectory["task"]["config"]["data"]["arguments"]["robot_name"] = robot
    trajectory["task"]["result"]["data"]["environment_success"] = success
    result = select_vla_segments(trajectory)
    assert result["segments"] == []
    assert result["rejected"][0]["reason"] == "no_confirmed_environment_success"


@pytest.mark.parametrize("robot", ["libero", "robocasa", "robodojo", "robotwin"])
@pytest.mark.parametrize("missing", ["", "/environment_success"])
def test_missing_task_success_excludes_the_trajectory(trajectory, robot, missing):
    trajectory["task"]["config"]["data"]["arguments"]["robot_name"] = robot
    trajectory["task"]["result"]["missing"] = {missing: {"reason": "capture_failed"}}
    selected = select_vla_segments(trajectory)
    assert selected["segments"] == []
    assert selected["rejected"][0]["reason"] == "no_confirmed_environment_success"


def test_unrelated_missing_task_result_field_does_not_exclude_the_trajectory(
    trajectory,
):
    trajectory["task"]["result"]["missing"] = {"/reward": {"reason": "not_exposed"}}
    assert _ranges(select_vla_segments(trajectory)) == [(1, 7)]


@pytest.mark.parametrize("success", [True, False])
@pytest.mark.parametrize("done", [True, False])
def test_robocasa_selection_uses_task_success_instead_of_done(
    robocasa_trajectory, success, done
):
    robocasa_trajectory["task"]["result"]["data"]["environment_success"] = success
    robocasa_trajectory["actions"][2]["result"]["data"]["done"] = done
    assert _ranges(select_vla_segments(robocasa_trajectory)) == (
        [(1, 7)] if success else []
    )


@pytest.mark.parametrize(
    "corruption",
    ["action", "gripper", "control_mode", "state", "image", "image_kind", "image_path"],
)
def test_robocasa_selection_breaks_at_invalid_training_data(
    robocasa_trajectory, corruption
):
    action = robocasa_trajectory["actions"][3]
    fields = action["before_observation"]["payload"]["data"]
    if corruption == "action":
        action["action"]["data"]["value"][0] = float("nan")
    elif corruption in ("gripper", "control_mode"):
        action["action"]["data"]["value"][6 if corruption == "gripper" else 11] = 0
    elif corruption == "state":
        fields["robot0_base_to_eef_quat"]["shape"] = [3]
    elif corruption in ("image_kind", "image_path"):
        fields["images"]["robot0_eye_in_hand"][corruption.removeprefix("image_")] = (
            "bytes" if corruption == "image_kind" else ""
        )
    else:
        fields["images"].pop("robot0_eye_in_hand")
    selected = select_vla_segments(robocasa_trajectory)
    assert _ranges(selected) == [(1, 3), (4, 7)]
    assert len(selected["rejected"]) == 1


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


def test_vla_actions_without_a_prediction_are_excluded(trajectory):
    trajectory["predictions"].pop(0)
    result = select_vla_segments(trajectory)
    assert _ranges(result) == [(4, 7)]
    assert result["rejected"] == []


def test_unexecuted_prediction_does_not_supply_training_actions(trajectory):
    prediction = deepcopy(trajectory["predictions"][0])
    prediction.update(prediction_id="V-unexecuted", action_ids=[])
    trajectory["predictions"].append(prediction)
    assert _ranges(select_vla_segments(trajectory)) == [(1, 7)]


def test_instruction_change_starts_a_new_segment(trajectory):
    trajectory["predictions"][1]["instruction"] = "different"
    assert _ranges(select_vla_segments(trajectory)) == [(1, 4), (4, 7)]


def test_action_format_change_starts_a_new_segment(trajectory, monkeypatch):
    monkeypatch.setattr(
        "rpent.flywheel.prepare.read_actions",
        lambda root, payload, config: read_array(root, payload["data"]),
    )
    for action in trajectory["actions"][4:7]:
        action["action"]["format"] = "test.other_action"
    result = select_vla_segments(trajectory)
    assert _ranges(result) == [(1, 4), (4, 7)]
    assert result["rejected"] == []


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


@pytest.mark.parametrize("field", ["before_observation", "after_observation", "action"])
def test_missing_required_action_data_excludes_the_step_and_breaks_continuity(
    trajectory, field
):
    trajectory["actions"][2][field] = None
    assert _ranges(select_vla_segments(trajectory)) == [(1, 2), (3, 7)]


def test_action_record_cannot_contain_an_entire_chunk(trajectory):
    payload = trajectory["actions"][2]["action"]
    payload["data"]["shape"] = [1, 7]
    payload["data"]["value"] = [payload["data"]["value"]]
    result = select_vla_segments(trajectory)
    assert _ranges(result) == [(1, 2), (3, 7)]
    assert any("expected shape (7,)" in row["reason"] for row in result["rejected"])


@pytest.mark.parametrize("index,expected", [(1, [(2, 7)]), (2, [(1, 2), (3, 7)])])
def test_incomplete_before_observation_excludes_the_step(trajectory, index, expected):
    observation = trajectory["actions"][index]["before_observation"]
    observation["payload"]["data"].pop("main_images")
    trajectory["actions"][index - 1]["after_observation"] = deepcopy(observation)
    assert _ranges(select_vla_segments(trajectory)) == expected


@pytest.mark.parametrize("trajectory", [35], indirect=True)
@pytest.mark.parametrize("field", ["main_images", "states"])
def test_final_after_observation_data_is_not_required_for_selection(trajectory, field):
    observation = trajectory["actions"][35]["after_observation"]
    observation["payload"]["data"].pop(field)
    trajectory["actions"][36]["before_observation"] = deepcopy(observation)
    result = select_vla_segments(trajectory)

    assert _ranges(result) == [(1, 36)]
    assert result["rejected"] == []


def test_action_from_another_tool_call_is_excluded(trajectory):
    trajectory["actions"][2]["tool_call_id"] = "T-other"
    assert _ranges(select_vla_segments(trajectory)) == [(1, 2), (3, 7)]


def test_missing_observation_mark_is_respected_even_if_reference_exists(trajectory):
    trajectory["actions"][2]["missing"]["before_observation_id"] = {
        "reason": "capture_failed"
    }
    assert _ranges(select_vla_segments(trajectory)) == [(1, 2), (3, 7)]


@pytest.mark.parametrize(
    "corruption",
    [
        "empty_instruction",
        "reverse_actions",
        "repeated_action",
        "missing_instruction",
    ],
)
def test_invalid_prediction_does_not_supply_training_actions(trajectory, corruption):
    prediction = trajectory["predictions"][0]
    if corruption == "empty_instruction":
        prediction["instruction"] = " "
    elif corruption == "reverse_actions":
        prediction["action_ids"][1:] = reversed(prediction["action_ids"][1:])
    elif corruption == "repeated_action":
        prediction["action_ids"][1] = prediction["action_ids"][0]
    else:
        prediction["missing"]["instruction"] = {"reason": "capture_failed"}
    result = select_vla_segments(trajectory)
    assert _ranges(result) == [(4, 7)]
    assert any(
        row.get("prediction_id") == prediction["prediction_id"]
        for row in result["rejected"]
    )


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


@pytest.mark.parametrize("order", ["reversed", "interleaved"])
def test_predictions_follow_execution_order_when_records_arrive_out_of_order(
    trajectory, order
):
    if order == "interleaved":
        for offset, prediction in enumerate(trajectory["predictions"]):
            prediction["action_ids"] = [
                action["action_id"]
                for action in trajectory["actions"][1 + offset : 7 : 2]
            ]
    trajectory["predictions"].reverse()
    result = select_vla_segments(trajectory)
    assert _ranges(result) == [(1, 7)]
    assert result["rejected"] == []


def test_script_between_predictions_breaks_the_segment_without_being_a_candidate(
    trajectory,
):
    script = deepcopy(trajectory["actions"][0])
    observation = trajectory["actions"][4]["before_observation"]
    script.update(
        action_id="A-script",
        before_observation_id=observation["observation_id"],
        after_observation_id=observation["observation_id"],
        before_observation=observation,
        after_observation=observation,
    )
    trajectory["actions"].insert(4, script)
    result = select_vla_segments(trajectory)
    assert _ranges(result) == [(1, 4), (5, 8)]
    assert result["rejected"] == []


@pytest.mark.parametrize(
    "records,field", [("actions", "action_id"), ("predictions", "prediction_id")]
)
def test_duplicate_record_ids_fail_with_a_useful_error(trajectory, records, field):
    trajectory[records][1][field] = trajectory[records][0][field]
    with pytest.raises(ValueError, match=f"[Dd]uplicate {field}"):
        select_vla_segments(trajectory)


@pytest.mark.parametrize(
    "robot,message",
    [
        ("not_installed", "unknown robot: 'not_installed'"),
        ("franka", "not supported for franka"),
    ],
)
def test_robot_without_data_support_has_a_clear_error(trajectory, robot, message):
    trajectory["task"]["config"]["data"]["arguments"]["robot_name"] = robot
    with pytest.raises(ValueError, match=message):
        select_vla_segments(trajectory)


@pytest.mark.parametrize("success", [True, False])
def test_bimanual_selection_requires_confirmed_task_success(
    bimanual_trajectory, success
):
    trajectory = bimanual_trajectory
    trajectory["task"]["result"]["data"]["environment_success"] = success
    selected = select_vla_segments(trajectory)
    assert selected["segments"] == (
        [{"start_index": 1, "stop_index": 7, "instruction": "pick up the bowl"}]
        if success
        else []
    )
    if not success:
        assert selected["rejected"][0]["reason"] == "no_confirmed_environment_success"


def test_bimanual_selection_stops_after_first_native_success(bimanual_trajectory):
    trajectory = bimanual_trajectory
    result = trajectory["actions"][4]["result"]
    field = "done" if result["format"] == "robodojo.step_result" else "terminated"
    result["data"][field] = True
    selected = select_vla_segments(trajectory)
    assert _ranges(selected) == [(1, 5)]
    assert [row["action_id"] for row in selected["rejected"]] == [
        action["action_id"] for action in trajectory["actions"][5:7]
    ]
    assert all(
        row["reason"] == "after_environment_success" for row in selected["rejected"]
    )


@pytest.mark.parametrize("bimanual_trajectory", ["robotwin"], indirect=True)
def test_robotwin_truncation_before_success_excludes_the_trajectory(
    bimanual_trajectory,
):
    bimanual_trajectory["actions"][3]["result"]["data"]["truncated"] = True
    selected = select_vla_segments(bimanual_trajectory)
    assert selected["segments"] == []
    assert selected["rejected"][0]["reason"] == "no_confirmed_environment_success"


@pytest.mark.parametrize(
    "corruption", ["no_success", "format", "missing_root", "missing_field"]
)
def test_bimanual_selection_requires_native_step_success(
    bimanual_trajectory, corruption
):
    trajectory = bimanual_trajectory
    result = trajectory["actions"][-1]["result"]
    if corruption == "no_success":
        result["data"] = {}
    elif corruption == "format":
        result["format"] = "other.step_result"
    else:
        field = "done" if result["format"] == "robodojo.step_result" else "terminated"
        result["missing"] = {
            "" if corruption == "missing_root" else f"/{field}": {
                "reason": "capture_failed"
            }
        }
    assert select_vla_segments(trajectory)["segments"] == []


@pytest.mark.parametrize("bimanual_trajectory", ["robotwin"], indirect=True)
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_robotwin_state_keeps_pose_dtype_for_grippers(bimanual_trajectory, dtype):
    from robots.robotwin.robot_spec import FLYWHEEL_CONFIG
    from rpent.flywheel.flywheel import validate_observation

    trajectory = bimanual_trajectory
    observation = trajectory["actions"][1]["before_observation"]
    robot = observation["payload"]["data"]["robot_state"]
    for side in ("left", "right"):
        robot[f"{side}_eef_pose"]["dtype"] = dtype
        robot[f"{side}_gripper"] = 0.123456789

    state = validate_observation(trajectory["root"], observation, FLYWHEEL_CONFIG)
    assert state.dtype == np.dtype(dtype)
    np.testing.assert_array_equal(
        state[[7, 15]], np.asarray([0.123456789, 0.123456789], dtype=dtype)
    )


@pytest.mark.parametrize(
    "corruption",
    [
        "action_type",
        "missing_action",
        "state_shape",
        "nonfinite",
        "missing_state_parent",
        "missing_image_parent",
        "missing_camera",
        "missing_image",
        "image_kind",
        "image_path",
    ],
)
def test_bimanual_selection_breaks_at_invalid_required_data(
    bimanual_trajectory, corruption
):
    trajectory = bimanual_trajectory
    robot = trajectory["task"]["config"]["data"]["arguments"]["robot_name"]
    action = trajectory["actions"][3]
    fields = action["before_observation"]["payload"]["data"]
    if corruption == "action_type":
        action["action"]["format"] = (
            "robodojo.action.ee" if robot == "robodojo" else "robotwin.action.qpos"
        )
    elif corruption == "missing_action":
        action["action"] = None
    elif corruption in ("missing_state_parent", "missing_image_parent"):
        if corruption == "missing_state_parent":
            path = "/state" if robot == "robodojo" else "/robot_state"
        else:
            path = (
                "/vision/cam_left_wrist"
                if robot == "robodojo"
                else "/observation/left_camera"
            )
        action["before_observation"]["payload"]["missing"] = {
            path: {"reason": "capture_failed"}
        }
    elif corruption in ("state_shape", "nonfinite"):
        state = (
            fields["state"]["left_arm_joint_state"]
            if robot == "robodojo"
            else fields["robot_state"]["left_eef_pose"]
        )
        if corruption == "state_shape":
            state["value"].pop()
            state["shape"][0] -= 1
        else:
            state["value"][0] = float("inf")
    else:
        cameras = fields["vision"] if robot == "robodojo" else fields["observation"]
        camera = "cam_left_wrist" if robot == "robodojo" else "left_camera"
        if corruption == "missing_image":
            cameras[camera]["color" if robot == "robodojo" else "rgb"] = None
        elif corruption in ("image_kind", "image_path"):
            image = cameras[camera]["color" if robot == "robodojo" else "rgb"]
            image[corruption.removeprefix("image_")] = (
                "bytes" if corruption == "image_kind" else ""
            )
        else:
            del cameras[camera]
    selected = select_vla_segments(trajectory)
    assert [(s["start_index"], s["stop_index"]) for s in selected["segments"]] == [
        (1, 3),
        (4, 7),
    ]
    assert selected["rejected"][0]["action_id"] == action["action_id"]
    assert selected["rejected"][0]["reason"].startswith("invalid_action:")
