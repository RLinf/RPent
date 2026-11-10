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

from unittest.mock import Mock

import numpy as np
import pytest

from rpent.robots.components.pi05_vla_client import Pi05VLAClient


@pytest.mark.parametrize(
    "embodiment,field,shape",
    [
        ("libero", "wrist_images", None),
        ("libero", "wrist_images", (4, 5, 3)),
        ("libero", "extra_view_images", (4, 5, 3)),
        ("franka", "extra_view_images", None),
        ("franka", "extra_view_images", (4, 5, 3)),
        ("franka", "extra_view_images", (2, 4, 5, 3)),
    ],
)
def test_prediction_preserves_optional_camera_wire_format(embodiment, field, shape):
    rpc = Mock(call=Mock(return_value=np.zeros((1, 5, 7))))
    client = Pi05VLAClient(rpc, embodiment=embodiment)
    view = None if shape is None else np.full(shape, 127, dtype=np.float32)
    result = client.predict(
        {"main_images": np.zeros((4, 5, 3)), "states": np.zeros(8), field: view}
    )
    assert rpc.call.call_args.args == ("vla.predict",)
    assert rpc.call.call_args.kwargs["args"][1] is None
    assert rpc.call.call_args.kwargs["timeout_s"] == 120.0
    assert result.shape == (5, 7)
    encoded = rpc.call.call_args.kwargs["args"][0][field]
    if view is None:
        assert encoded is None
    else:
        assert encoded.shape == (1, *shape)
        assert encoded.dtype == np.uint8
        np.testing.assert_array_equal(encoded[0], view)
        assert view.dtype == np.float32


@pytest.mark.parametrize(
    "embodiment,field,shape",
    [
        ("libero", "wrist_images", (2, 4, 5, 3)),
        ("libero", "extra_view_images", (2, 4, 5, 3)),
        ("franka", "extra_view_images", (4, 5)),
    ],
)
def test_invalid_camera_rank_fails_before_rpc(embodiment, field, shape):
    rpc = Mock()
    client = Pi05VLAClient(rpc, embodiment=embodiment)
    with pytest.raises(ValueError, match="image, got shape"):
        client.predict(
            {
                "main_images": np.zeros((4, 5, 3)),
                "states": np.zeros(8),
                field: np.zeros(shape),
            }
        )
    rpc.call.assert_not_called()


def _behavior_frame():
    return {
        "r1::proprio": np.arange(61, dtype=np.float32),
        "r1::zed_link:Camera:0::rgb": np.full((5, 6, 4), 17, dtype=np.uint8),
        "r1::left_realsense_link:Camera:0::rgb": np.full((5, 6, 3), 31, dtype=np.uint8),
        "r1::right_realsense_link:Camera:0::rgb": np.full(
            (5, 6, 3), 49, dtype=np.uint8
        ),
        "task_descriptions": "Turn on the radio.",
    }


def test_behavior_prediction_uses_public_state_and_ordered_views():
    native = np.arange(16 * 23, dtype=np.float32).reshape(1, 16, 23)
    rpc = Mock(call=Mock(return_value=native))
    client = Pi05VLAClient(rpc, embodiment="behavior")
    result = client.predict(_behavior_frame())
    assert rpc.call.call_args.args == ("vla.predict",)
    encoded, options = rpc.call.call_args.kwargs["args"]
    assert options is None
    expected = np.zeros(256, dtype=np.float32)
    expected[253:256] = [0, 1, 2]
    expected[158:165] = np.arange(3, 10)
    expected[193:195] = [24, 25]
    expected[197:204] = np.arange(28, 35)
    expected[232:234] = [49, 50]
    expected[236:240] = [53, 54, 55, 56]
    np.testing.assert_array_equal(encoded["states"], expected[None])
    assert encoded["main_images"].shape == (1, 5, 6, 3)
    assert encoded["wrist_images"].shape == (1, 2, 5, 6, 3)
    np.testing.assert_array_equal(encoded["main_images"], 17)
    np.testing.assert_array_equal(encoded["wrist_images"][0, 0], 31)
    np.testing.assert_array_equal(encoded["wrist_images"][0, 1], 49)
    assert encoded["extra_view_images"] is None
    assert encoded["task_descriptions"] == ["Turn on the radio."]
    np.testing.assert_array_equal(result, native[0])


def test_behavior_keeps_existing_state_layout_and_converts_float_chw():
    obs = _behavior_frame()
    obs[b"r1::proprio"] = np.arange(256, dtype=np.float32)
    del obs["r1::proprio"]
    obs["r1::zed_link:Camera:0::rgb"] = np.ones((3, 5, 6), dtype=np.float32)
    encoded = Pi05VLAClient(None, embodiment="behavior").encode_obs(obs)
    np.testing.assert_array_equal(encoded["states"][0], obs[b"r1::proprio"])
    np.testing.assert_array_equal(encoded["main_images"], 255)
    assert encoded["main_images"].shape == (1, 5, 6, 3)


@pytest.mark.parametrize("shape", [(60,), (255,), (257,), (1, 61)])
def test_behavior_rejects_unknown_state_layout_before_rpc(shape):
    rpc = Mock()
    obs = _behavior_frame()
    obs["r1::proprio"] = np.zeros(shape, dtype=np.float32)
    with pytest.raises(ValueError, match="R1Pro proprio") as error:
        Pi05VLAClient(rpc, embodiment="behavior").predict(obs)
    assert str(shape) in str(error.value)
    rpc.call.assert_not_called()


@pytest.mark.parametrize(
    "missing", ["task_descriptions", "r1::left_realsense_link:Camera:0::rgb"]
)
def test_behavior_missing_sensor_or_instruction_fails_before_rpc(missing):
    rpc = Mock()
    obs = _behavior_frame()
    del obs[missing]
    with pytest.raises(ValueError):
        Pi05VLAClient(rpc, embodiment="behavior").predict(obs)
    rpc.call.assert_not_called()
