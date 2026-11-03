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

from __future__ import annotations

import numpy as np
import pytest

from robots.r2.camera_geometry import UrdfCameraChains


def test_camera_pose_follows_the_measured_joints(tiny_urdf):
    chains = UrdfCameraChains(tiny_urdf, {"head": "cam"})

    assert chains.joint_names == {"J_NeckYaw"}
    straight = chains.pose("head", {"J_NeckYaw": 0.0})
    np.testing.assert_allclose(straight[:3, 3], [0.1, 0.0, 1.0], atol=1e-9)
    # Optical frame: z forward along +x, x right along -y, y down along -z.
    np.testing.assert_allclose(straight[:3, 2], [1.0, 0.0, 0.0], atol=1e-9)
    np.testing.assert_allclose(straight[:3, 0], [0.0, -1.0, 0.0], atol=1e-9)

    turned = chains.pose("head", {"J_NeckYaw": np.pi / 2})
    np.testing.assert_allclose(turned[:3, 3], [0.0, 0.1, 1.0], atol=1e-9)
    np.testing.assert_allclose(turned[:3, 2], [0.0, 1.0, 0.0], atol=1e-9)

    with pytest.raises(KeyError):
        chains.pose("head", {})


def test_a_frame_outside_base_link_is_rejected(tiny_urdf):
    with pytest.raises(ValueError, match="does not descend from 'base_link'"):
        UrdfCameraChains(tiny_urdf, {"head": "missing_link"})
