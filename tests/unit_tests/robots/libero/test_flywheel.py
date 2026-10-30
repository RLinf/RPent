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

from robots.libero.robot_spec import FLYWHEEL_CONFIG
from rpent.flywheel.flywheel import (
    read_actions,
    read_frame,
    successful_prefix,
    validate_observation,
)


def _array(values):
    array = np.asarray(values, dtype=np.float32)
    return {
        "$type": "ndarray",
        "dtype": "float32",
        "shape": list(array.shape),
        "value": array.tolist(),
    }


def test_success_prefix_requires_native_step_success():
    actions = [
        {"result": {"format": "libero.step_result", "data": {"terminated": False}}},
        {"result": {"format": "libero.step_result", "data": {"terminated": True}}},
    ]
    assert successful_prefix(actions, FLYWHEEL_CONFIG) == 2
    assert successful_prefix(actions[:1], FLYWHEEL_CONFIG) == 0
    actions[0]["result"]["data"]["truncated"] = True
    assert successful_prefix(actions, FLYWHEEL_CONFIG) == 0
    actions[0]["result"]["data"]["truncated"] = False
    actions[1]["result"]["missing"] = {"/terminated": {"reason": "capture_failed"}}
    assert successful_prefix(actions, FLYWHEEL_CONFIG) == 0
    actions[1]["result"]["missing"] = {"/reward": {"reason": "not_exposed"}}
    assert successful_prefix(actions, FLYWHEEL_CONFIG) == 2


def test_native_actions_are_preserved(tmp_path):
    values = np.arange(7, dtype=np.float32)
    restored = read_actions(
        tmp_path,
        {
            "format": "libero.action",
            "data": _array(values),
            "completeness": "complete",
        },
        FLYWHEEL_CONFIG,
    )
    np.testing.assert_array_equal(restored, values)
    assert restored.dtype == values.dtype


@pytest.mark.parametrize("values", [[0] * 6, [np.inf] * 7, [np.nan] * 7, [[0] * 7]])
def test_invalid_native_action_is_rejected(tmp_path, values):
    with pytest.raises(ValueError, match=r"expected shape \(7,\) with finite values"):
        read_actions(
            tmp_path,
            {
                "format": "libero.action",
                "data": _array(values),
                "completeness": "complete",
            },
            FLYWHEEL_CONFIG,
        )


@pytest.fixture
def observation():
    return {
        "payload": {
            "format": "libero.observation",
            "data": {
                "states": _array([0] * 8),
                **{
                    name: {
                        "$type": "artifact_ref",
                        "kind": "ndarray",
                        "path": f"artifacts/{name}.npy",
                        "dtype": "uint8",
                        "shape": [256, 256, 3],
                    }
                    for name in ("main_images", "wrist_images")
                },
            },
        },
    }


@pytest.fixture
def action_record(tmp_path, observation):
    fields = observation["payload"]["data"]
    fields["states"] = _array(np.arange(8))
    fields["task_descriptions"] = "complete the whole task"
    for index, name in enumerate(("main_images", "wrist_images"), 1):
        path = tmp_path / fields[name]["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        image = np.full((256, 256, 3), index, dtype=np.uint8)
        image[0, 0] = [10, 20, 30]
        np.save(path, image)
    return {
        "before_observation": observation,
        "after_observation": {"payload": None},
        "action": {
            "format": "libero.action",
            "data": _array(np.arange(7) / 10),
            "completeness": "complete",
        },
    }


def test_read_frame_uses_before_observation_executed_action_and_local_instruction(
    tmp_path, action_record
):
    original = deepcopy(action_record)
    frame = read_frame(tmp_path, action_record, "pick up the bowl", FLYWHEEL_CONFIG)

    assert set(frame) == {
        "image",
        "wrist_image",
        "observation.state",
        "action",
        "prompt",
    }
    for name, field in (("image", "main_images"), ("wrist_image", "wrist_images")):
        expected = np.load(tmp_path / f"artifacts/{field}.npy", allow_pickle=False)
        np.testing.assert_array_equal(frame[name], expected)
        assert frame[name].shape == (256, 256, 3)
        assert frame[name].dtype == np.uint8
    np.testing.assert_array_equal(
        frame["observation.state"], np.arange(8, dtype=np.float32)
    )
    np.testing.assert_array_equal(
        frame["action"], np.asarray(np.arange(7) / 10, dtype=np.float32)
    )
    assert frame["observation.state"].dtype == frame["action"].dtype == np.float32
    assert frame["prompt"] == "pick up the bowl"
    assert action_record == original


def test_read_frame_rejects_an_action_chunk(tmp_path, action_record):
    action_record["action"]["data"] = _array(np.zeros((2, 7)))
    with pytest.raises(ValueError, match=r"expected shape \(7,\) with finite values"):
        read_frame(tmp_path, action_record, "pick up the bowl", FLYWHEEL_CONFIG)


@pytest.mark.parametrize("field", ["state", "image", "action"])
def test_read_frame_checks_required_data(tmp_path, action_record, field):
    fields = action_record["before_observation"]["payload"]["data"]
    if field == "state":
        fields["states"] = _array([0] * 7)
    elif field == "image":
        fields["main_images"]["shape"] = [128, 128, 3]
    else:
        action_record["action"]["data"] = _array([np.inf] * 7)
    with pytest.raises(ValueError):
        read_frame(tmp_path, action_record, "pick up the bowl", FLYWHEEL_CONFIG)


@pytest.mark.parametrize("corruption", ["missing", "wrong_shape"])
def test_read_frame_checks_image_files(tmp_path, action_record, corruption):
    path = tmp_path / "artifacts/main_images.npy"
    if corruption == "missing":
        path.unlink()
        error, message = FileNotFoundError, "main_images.npy"
    else:
        np.save(path, np.zeros((128, 128, 3), dtype=np.uint8))
        error, message = ValueError, "Array shape mismatch"
    with pytest.raises(error, match=message):
        read_frame(tmp_path, action_record, "pick up the bowl", FLYWHEEL_CONFIG)


def test_missing_unused_observation_fields_do_not_drop_training_data(
    tmp_path, observation
):
    observation["payload"]["completeness"] = "partial"
    observation["payload"]["missing"] = {
        "/extra_view_images": {"reason": "not_exposed"}
    }
    validate_observation(tmp_path, observation, FLYWHEEL_CONFIG)


@pytest.mark.parametrize("field", ["states", "main_images", "wrist_images"])
def test_missing_required_observation_field_is_rejected(tmp_path, observation, field):
    observation["payload"]["data"][field] = None
    with pytest.raises(ValueError):
        validate_observation(tmp_path, observation, FLYWHEEL_CONFIG)


@pytest.mark.parametrize("state", [[0] * 7, [np.inf] * 8, [np.nan] * 8])
def test_invalid_proprioception_is_rejected(tmp_path, observation, state):
    observation["payload"]["data"]["states"] = _array(state)
    with pytest.raises(ValueError, match=r"expected shape \(8,\) with finite values"):
        validate_observation(tmp_path, observation, FLYWHEEL_CONFIG)


@pytest.mark.parametrize(
    "field,value",
    [("shape", [128, 128, 3]), ("dtype", "float32"), ("kind", "bytes"), ("path", "")],
)
def test_invalid_image_metadata_is_rejected(tmp_path, observation, field, value):
    observation["payload"]["data"]["main_images"][field] = value
    with pytest.raises(ValueError, match="Invalid image"):
        validate_observation(tmp_path, observation, FLYWHEEL_CONFIG)
