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


"""Adapt recorded LIBERO camera views to the shared RGB-D observation interface."""

from __future__ import annotations

from uuid import uuid4

import numpy as np

from rpent.session import EnvState
from toolkits.observation import GeometryObservation


class LiberoGeometryObservations:
    """Read only RGB, metric points and calibration at the requested recorded step."""

    def __init__(self, state: EnvState):
        self.state = state
        self.world_frame = f"libero_world/{uuid4().hex}"

    def get_observation(self, *, camera: str, step: int) -> GeometryObservation:
        if camera not in {"agentview", "wrist"}:
            raise ValueError("LIBERO geometry cameras are agentview and wrist")
        record = self.state.get(step)
        nn = record.step_idx
        for suffix in ("_high", ""):
            rgb_name = f"{camera}{suffix}.png"
            world_name = f"{camera}_world{suffix}.npz"
            meta_name = f"{camera}_metadata{suffix}.json"
            if meta_name not in record.artifacts:
                meta_name = f"{camera}_metadata.json"
            names = [rgb_name, world_name, meta_name]
            if not set(names).issubset(record.artifacts):
                continue
            rgb = self.state.load(rgb_name, step=nn)
            meta = self.state.load(meta_name, step=nn)
            h, w = rgb.shape[:2]
            if w * meta["height"] != h * meta["width"]:
                raise ValueError("recorded calibration and RGB aspect ratios differ")
            scale = np.diag([w / meta["width"], h / meta["height"], 1.0])
            return GeometryObservation(
                camera=camera,
                step=nn,
                world_frame_id=self.world_frame,
                rgb=rgb,
                world_xyz=self.state.load(world_name, step=nn),
                intrinsics=scale @ meta["intrinsic_K"],
                camera_to_world=meta["extrinsic_cam2world"],
                up_direction=np.array([0.0, 0.0, 1.0]),
                source_refs={
                    name: str(self.state.artifact_path(name, step=nn)) for name in names
                },
            )
        raise ValueError(f"no matching calibrated RGB-D pair for {camera} at step {nn}")
