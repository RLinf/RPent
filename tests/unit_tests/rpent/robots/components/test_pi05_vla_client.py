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
    client.predict(
        {"main_images": np.zeros((4, 5, 3)), "states": np.zeros(8), field: view}
    )
    assert rpc.call.call_args.args == ("vla.predict",)
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
