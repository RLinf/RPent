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

import numpy as np
import pytest

from rpent.robots.components.pi05_vla_client import Pi05VLAClient


@pytest.fixture
def observation():
    return {
        "vision": {
            camera: {"color": np.full((2, 4, 3), i, dtype=np.float32)}
            for i, camera in enumerate(
                ("cam_head", "cam_left_wrist", "cam_right_wrist"), start=1
            )
        },
        "state": {
            "left_arm_joint_state": list(range(6)),
            "right_arm_joint_state": list(range(6, 12)),
            "left_ee_joint_state": [1.0],
            "right_ee_joint_state": [0.0],
        },
        "instruction": "pick the bottle",
    }


def test_robodojo_wire_keys_views_and_unflipped_state(observation):
    encoded = Pi05VLAClient(None, embodiment="robodojo").encode_obs(observation)
    assert set(encoded) == {
        "main_images",
        "wrist_images",
        "extra_view_images",
        "states",
        "task_descriptions",
    }
    for i, key in enumerate(
        ("main_images", "wrist_images", "extra_view_images"), start=1
    ):
        assert encoded[key].dtype == np.uint8
        np.testing.assert_array_equal(encoded[key], np.full((1, 2, 4, 3), i))
    assert encoded["states"].dtype == np.float32
    np.testing.assert_array_equal(encoded["states"], [list(range(12)) + [1.0, 0.0]])
    assert encoded["task_descriptions"] == ["pick the bottle"]


@pytest.mark.parametrize("instruction,expected", [(None, ""), ("", ""), (123, "123")])
def test_robodojo_instruction_conversion(observation, instruction, expected):
    observation["instruction"] = instruction
    client = Pi05VLAClient(None, embodiment="robodojo")
    assert client.encode_obs(observation)["task_descriptions"] == [expected]
    del observation["instruction"]
    assert client.encode_obs(observation)["task_descriptions"] == [""]


@pytest.mark.parametrize("camera", ["cam_head", "cam_left_wrist", "cam_right_wrist"])
@pytest.mark.parametrize("missing", ["camera", "color"])
def test_robodojo_missing_camera_fails(observation, camera, missing):
    if missing == "camera":
        del observation["vision"][camera]
    else:
        del observation["vision"][camera]["color"]
    with pytest.raises(
        ValueError, match=f"missing RoboDojo camera: vision/{camera}/color"
    ):
        Pi05VLAClient(None, embodiment="robodojo").encode_obs(observation)


@pytest.mark.parametrize(
    "key",
    [
        "left_arm_joint_state",
        "right_arm_joint_state",
        "left_ee_joint_state",
        "right_ee_joint_state",
    ],
)
def test_robodojo_missing_state_fails(observation, key):
    del observation["state"][key]
    with pytest.raises(ValueError, match=f"missing RoboDojo state: state/{key}"):
        Pi05VLAClient(None, embodiment="robodojo").encode_obs(observation)


@pytest.mark.parametrize("section", ["vision", "state"])
def test_robodojo_missing_section_fails(observation, section):
    del observation[section]
    with pytest.raises(ValueError, match="missing RoboDojo"):
        Pi05VLAClient(None, embodiment="robodojo").encode_obs(observation)


@pytest.mark.parametrize("shape", [(1, 2, 4, 3), (2, 4), (2, 4, 4)])
def test_robodojo_rejects_non_rgb_single_view(observation, shape):
    observation["vision"]["cam_head"]["color"] = np.zeros(shape)
    with pytest.raises(ValueError, match="cam_head: expected"):
        Pi05VLAClient(None, embodiment="robodojo").encode_obs(observation)


@pytest.mark.parametrize("value", [[0] * 5, [[0] * 6], None])
def test_robodojo_rejects_invalid_state_shape(observation, value):
    observation["state"]["left_arm_joint_state"] = value
    with pytest.raises(ValueError, match="state/left_arm_joint_state: expected"):
        Pi05VLAClient(None, embodiment="robodojo").encode_obs(observation)
