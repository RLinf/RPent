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

"""Adapt recorded VLABench camera views to the shared RGB-D observation interface."""

from __future__ import annotations

from collections.abc import Iterable
from uuid import uuid4

from robots.vlabench.sim.contracts import VLABENCH_CAMERA_NAMES
from rpent.session import EnvState
from toolkits.observation import GeometryObservation


def artifact_names(camera: str) -> tuple[str, str, str]:
    """Return the RGB, depth and calibration artifact names of one camera."""
    return (
        f"{camera}_rgb.png",
        f"{camera}_depth.npy",
        f"{camera}_camera_meta.json",
    )


class VLABenchGeometryObservations:
    """Read the RGB, metric depth and calibration recorded at one step."""

    def __init__(self, state: EnvState) -> None:
        self.state = state
        self.world_frame = f"vlabench_world/{uuid4().hex}"

    def get_observation(self, *, camera: str, step: int) -> GeometryObservation:
        """Return one calibrated snapshot; ``step=-1`` selects the latest record."""
        if camera not in VLABENCH_CAMERA_NAMES:
            choices = ", ".join(VLABENCH_CAMERA_NAMES)
            raise ValueError(f"VLABench cameras are {choices}")
        record = self.state.get(step)
        index = record.step_idx
        names = artifact_names(camera)
        if not set(names).issubset(record.artifacts):
            raise ValueError(f"no calibrated RGB-D pair for {camera} at step {index}")
        rgb_name, depth_name, meta_name = names
        meta = self.state.load(meta_name, step=index)
        return GeometryObservation(
            camera=camera,
            step=index,
            world_frame_id=self.world_frame,
            rgb=self.state.load(rgb_name, step=index),
            depth_m=self.state.load(depth_name, step=index),
            intrinsics=meta["intrinsic_K"],
            camera_to_world=meta["extrinsic_cam2world"],
            up_direction=[0.0, 0.0, 1.0],
            source_refs={
                name: str(self.state.artifact_path(name, step=index)) for name in names
            },
        )

    def additional_observations(
        self, primary: GeometryObservation
    ) -> Iterable[GeometryObservation]:
        """Yield the other cameras recorded at the same step."""
        record = self.state.get(primary.step)
        for camera in VLABENCH_CAMERA_NAMES:
            if camera != primary.camera and artifact_names(camera)[0] in (
                record.artifacts
            ):
                yield self.get_observation(camera=camera, step=primary.step)
