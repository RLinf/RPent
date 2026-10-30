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

import json
from pathlib import Path

import numpy as np
import pytest

from rpent.flywheel.reader import iter_trajectories, read_array, validate_image
from rpent.session import EnvState, Payload


def _save_trajectory(output: Path) -> Path:
    """Build a trajectory using the current capture format."""
    state = EnvState(output)
    state.start_task(instruction="Move the cup", config=Payload("test.config", {}))
    planner = state.register_planner(backend="api")
    tool = state.register_tool_call(
        planner_id=planner, name="pick", arguments=Payload("json", {})
    )
    observations = [
        state.register_observation(
            payload=Payload(
                "test.observation",
                {
                    "camera": np.full((3, 4, 3), index, dtype=np.uint8),
                    "state": np.array([index, 0, 1], dtype=np.float32),
                },
                inline_array_paths=("/state",),
            )
        )
        for index in range(3)
    ]
    actions = np.array([[1, 2], [3, 4]], dtype=np.float32)
    action_ids = [
        state.register_action(
            tool_call_id=tool,
            source="vla",
            action=Payload("test.action", action, inline_array_paths=("",)),
            before_observation_id=observations[index],
            after_observation_id=observations[index + 1],
        )
        for index, action in enumerate(actions)
    ]
    state.register_prediction(
        tool_call_id=tool,
        instruction="拿起杯子",
        action_ids=action_ids,
    )
    state.end_task(
        status="completed",
        stop_reason="finish",
        result=Payload("test.result", {"environment_success": False}),
    )
    return state.save_planner_trajectory()


def test_reads_saved_records_and_arrays_without_robot_specific_fields(tmp_path):
    root = _save_trajectory(tmp_path)
    trajectory = next(iter_trajectories(root))

    assert trajectory["root"] == root.resolve()
    assert trajectory["schema_version"] == 5
    assert trajectory["task"]["result"]["data"]["environment_success"] is False
    actions = trajectory["actions"]
    prediction = trajectory["predictions"][0]
    assert prediction["instruction"] == "拿起杯子"
    assert prediction["action_ids"] == [action["action_id"] for action in actions]
    assert "input_observation" not in prediction
    assert "input_observation_id" not in prediction
    assert actions[0]["after_observation"] == actions[1]["before_observation"]

    for index, action in enumerate(actions):
        obs = action["before_observation"]["payload"]["data"]
        image = read_array(root, obs["camera"])
        assert image.dtype == np.uint8 and image.shape == (3, 4, 3)
        np.testing.assert_array_equal(image, np.full((3, 4, 3), index))
        np.testing.assert_array_equal(read_array(root, obs["state"]), [index, 0, 1])
    assert "actions" not in prediction
    np.testing.assert_array_equal(
        np.stack([read_array(root, action["action"]["data"]) for action in actions]),
        [[1, 2], [3, 4]],
    )


def test_discovers_nested_attempts_and_filters_jobs_in_any_parent(tmp_path):
    parent = tmp_path / "collected-runs"
    first = _save_trajectory(parent / "job-a" / "attempt_01")
    second = _save_trajectory(parent / "job-a" / "attempt_02")
    third = _save_trajectory(parent / "job-b")
    (parent / "empty-job").mkdir()
    (parent / "notes.txt").write_text("Not a trajectory", encoding="utf-8")

    assert [t["root"] for t in iter_trajectories(parent)] == sorted(
        path.resolve() for path in (first, second, third)
    )
    assert [t["root"] for t in iter_trajectories(parent, "job-b*")] == [third.resolve()]
    # Recursive globs can discover the same trajectory through several parents.
    assert len(list(iter_trajectories(parent, "**"))) == 3


def test_reads_metadata_without_planner_log_or_image_files(tmp_path):
    root = _save_trajectory(tmp_path)
    trajectory = next(iter_trajectories(root))
    image_ref = trajectory["actions"][0]["before_observation"]["payload"]["data"][
        "camera"
    ]
    (root / image_ref["path"]).unlink()
    (root / "planners.jsonl").unlink()

    validate_image(image_ref, "camera", expected_shape=(3, 4, 3))
    assert next(iter_trajectories(root)) == trajectory
    with pytest.raises(FileNotFoundError) as error:
        read_array(root, image_ref)
    assert image_ref["path"] in str(error.value)


@pytest.mark.parametrize(
    "field,value",
    [
        ("$type", "bytes"),
        ("shape", [3, 4]),
        ("shape", [3, 4, 4]),
        ("dtype", "float32"),
        ("kind", "bytes"),
        ("path", ""),
    ],
)
def test_invalid_image_metadata_reports_the_camera(tmp_path, field, value):
    trajectory = next(iter_trajectories(_save_trajectory(tmp_path)))
    image = trajectory["actions"][0]["before_observation"]["payload"]["data"]["camera"]
    image[field] = value
    with pytest.raises(ValueError, match="Invalid camera"):
        validate_image(image, "camera")


def test_inline_image_metadata_requires_no_artifact_reference():
    image = {
        "$type": "ndarray",
        "shape": [2, 4, 3],
        "dtype": "uint8",
        "value": np.zeros((2, 4, 3), dtype=np.uint8).tolist(),
    }
    validate_image(image, "camera")
    validate_image(image, "camera", expected_shape=(2, 4, 3))
    with pytest.raises(ValueError, match="Invalid camera"):
        validate_image(image, "camera", expected_shape=(256, 256, 3))


def test_preserves_empty_failed_trajectory(tmp_path):
    state = EnvState(tmp_path)
    state.start_task(instruction="Move", config=Payload("test.config", {}))
    state.end_task(status="failed", stop_reason="setup_error")
    root = state.save_planner_trajectory()
    trajectory = next(iter_trajectories(root))
    assert trajectory["task"]["status"] == "failed"
    assert trajectory["actions"] == trajectory["predictions"] == []


def test_preserves_jsonl_order_extra_fields_and_blank_lines(tmp_path):
    root = _save_trajectory(tmp_path)
    rows = [{"action_id": "A10", "extra": {"x": 1}}, {"action_id": "A2"}]
    (root / "actions.jsonl").write_text(
        "\n" + "\n\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )
    assert next(iter_trajectories(root))["actions"] == rows


@pytest.mark.parametrize("filename", ["actions.jsonl", "predictions.jsonl"])
@pytest.mark.parametrize("bad_line", ["{broken", "[]"])
def test_invalid_jsonl_reports_filename_and_physical_line(tmp_path, filename, bad_line):
    root = _save_trajectory(tmp_path)
    path = root / filename
    path.write_text("{}\n\n" + bad_line + "\n", encoding="utf-8")
    with pytest.raises(ValueError) as error:
        list(iter_trajectories(root))
    assert f"{path.resolve()}:3:" in str(error.value)


@pytest.mark.parametrize(
    "contents",
    [
        "{\ninvalid",
        "[]",
        '{"schema_version": 5}',
    ],
)
def test_invalid_task_reports_source_file(tmp_path, contents):
    root = _save_trajectory(tmp_path)
    (root / "task.json").write_text(contents, encoding="utf-8")
    with pytest.raises(ValueError) as error:
        list(iter_trajectories(root))
    assert str(root.resolve() / "task.json") in str(error.value)


@pytest.mark.parametrize("version", [2, 3, 4, 6])
def test_unsupported_trajectory_schema_reports_source_file(tmp_path, version):
    root = _save_trajectory(tmp_path)
    path = root / "task.json"
    document = json.loads(path.read_text())
    document["schema_version"] = version
    path.write_text(json.dumps(document))
    with pytest.raises(
        ValueError, match="expected trajectory schema_version 5"
    ) as error:
        list(iter_trajectories(root))
    assert str(path.resolve()) in str(error.value)


@pytest.mark.parametrize("filename", ["actions.jsonl", "predictions.jsonl"])
def test_missing_record_file_is_not_treated_as_an_empty_trajectory(tmp_path, filename):
    root = _save_trajectory(tmp_path)
    (root / filename).unlink()
    with pytest.raises(FileNotFoundError) as error:
        list(iter_trajectories(root))
    assert filename in str(error.value)


def test_missing_input_and_no_matches_are_visible(tmp_path):
    with pytest.raises(FileNotFoundError):
        list(iter_trajectories(tmp_path / "missing"))
    with pytest.raises(ValueError, match="No trajectories found"):
        list(iter_trajectories(tmp_path))
    root = _save_trajectory(tmp_path / "job")
    with pytest.raises(NotADirectoryError):
        list(iter_trajectories(root / "task.json"))
    with pytest.raises(ValueError, match="No trajectories found"):
        list(iter_trajectories(tmp_path, "other-*"))


@pytest.mark.parametrize(
    "array",
    [
        np.array([[np.nan, np.inf], [-np.inf, 1]], dtype=np.float32),
        np.empty((2, 0, 3), dtype=np.int16),
        np.array(True),
    ],
)
def test_reads_inline_arrays_using_the_current_writer(tmp_path, array):
    state = EnvState(tmp_path)
    state.start_task(
        instruction="Move",
        config=Payload("test.array", array, inline_array_paths=("",)),
    )
    state.end_task(status="failed", stop_reason="setup_error")
    root = state.save_planner_trajectory()
    task = json.loads((root / "task.json").read_text())["task"]
    restored = read_array(root, task["config"]["data"])
    np.testing.assert_array_equal(restored, array)
    assert restored.dtype == array.dtype and restored.shape == array.shape


@pytest.mark.parametrize(
    "field,wrong_value", [("shape", [12, 3]), ("dtype", "float32")]
)
def test_artifact_metadata_must_match_stored_array(tmp_path, field, wrong_value):
    root = _save_trajectory(tmp_path)
    trajectory = next(iter_trajectories(root))
    reference = trajectory["actions"][0]["before_observation"]["payload"]["data"][
        "camera"
    ]
    reference[field] = wrong_value
    with pytest.raises(ValueError, match=f"Array {field} mismatch"):
        read_array(root, reference)


def test_array_references_cannot_leave_the_trajectory_directory(tmp_path):
    root = tmp_path / "trajectory"
    root.mkdir()
    outside = tmp_path / "outside.npy"
    np.save(outside, np.zeros(3))
    (root / "linked.npy").symlink_to(outside)
    for path in (str(outside), "../outside.npy", "linked.npy"):
        with pytest.raises(ValueError, match="must stay inside"):
            read_array(root, {"$type": "artifact_ref", "kind": "ndarray", "path": path})


def test_rejects_pickle_arrays_and_non_array_references(tmp_path):
    np.save(tmp_path / "objects.npy", np.array([{}], dtype=object))
    with pytest.raises(ValueError, match="Object arrays"):
        read_array(
            tmp_path,
            {"$type": "artifact_ref", "kind": "ndarray", "path": "objects.npy"},
        )
    for value in (
        {"$type": "artifact_ref", "kind": "bytes", "path": "image.bin"},
        {"$type": "dict", "value": {"$type": "ndarray"}},
    ):
        with pytest.raises(ValueError, match="Expected an encoded ndarray"):
            read_array(tmp_path, value)
