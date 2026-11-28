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

"""Adapt recorded MetaWorld camera calibration to the shared RGB-D tools."""

from uuid import uuid4

import numpy as np

from rpent.session import EnvState
from toolkits.observation import GeometryObservation


def geometry_observation(rgb, depth, meta, step, frame) -> GeometryObservation:
    # MuJoCo has Y up and Z backward; the shared pinhole convention has Y
    # down and Z forward. Both use the same pixel-center intrinsics.
    pose = np.eye(4)
    pose[:3, :3] = np.asarray(meta["rotation_world_from_camera"]) @ np.diag([1, -1, -1])
    pose[:3, 3] = meta["position_world"]
    return GeometryObservation(
        camera="primary",
        step=step,
        world_frame_id=frame,
        rgb=rgb,
        depth_m=depth,
        intrinsics=meta["intrinsics"],
        camera_to_world=pose,
        valid_mask=depth < meta["clip_range"][1] * (1 - 1e-5),
        up_direction=[0, 0, 1],
    )


class MetaWorldGeometryObservations:
    def __init__(self, state: EnvState):
        self.state = state
        self.frame = f"metaworld_world/{uuid4().hex}"

    def get_observation(self, *, camera: str, step: int) -> GeometryObservation:
        if camera != "primary":
            raise ValueError("MetaWorld geometry camera is primary")
        record = self.state.get(step)
        index = record.step_idx
        return geometry_observation(
            self.state.load("camera.png", step=index),
            self.state.load("depth.npy", step=index),
            record.state["camera"],
            index,
            self.frame,
        )
