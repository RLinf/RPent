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

import io
from collections.abc import Iterable
from uuid import uuid4

import numpy as np
from PIL import Image

from rpent.session import EnvState
from rpent.tools import ToolResult
from toolkits.observation import GeometryObservation


def view_geometry_state(step: int = -1, *, state: EnvState) -> ToolResult:
    """Keep calibrated full-frame coordinates while sending only the two working views."""
    try:
        record = state.get(step)
    except (LookupError, ValueError) as exc:
        return ToolResult(data={"error": str(exc)})
    payload = {
        "step": record.step_idx,
        "state": record.state,
        "terminated": record.terminated,
        "truncated": record.truncated,
        "task_language": record.extras.get("task_language"),
        "result": record.result,
        "views": [],
    }
    images = []
    for camera in ("agentview", "wrist"):
        name = next(
            (
                n
                for n in (f"{camera}_high.png", f"{camera}.png")
                if n in record.artifacts
            ),
            None,
        )
        if name is None:
            continue
        with Image.open(
            io.BytesIO(state.load_bytes(name, step=record.step_idx))
        ) as img:
            full_shape = [img.height, img.width]
            img.thumbnail((512, 512), Image.Resampling.LANCZOS)
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            images.append(buf.getvalue())
            payload["views"].append(
                {
                    "camera": camera,
                    "image_shape": [img.height, img.width],
                    "full_image": name,
                    "full_image_shape": full_shape,
                }
            )
    return ToolResult(data=payload, images=images)


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

    def additional_observations(
        self, primary: GeometryObservation
    ) -> Iterable[GeometryObservation]:
        """Read the other calibrated camera from this exact state record."""
        other = "wrist" if primary.camera == "agentview" else "agentview"
        if not any(
            name in self.state.get(primary.step).artifacts
            for name in (f"{other}_high.png", f"{other}.png")
        ):
            return
        yield self.get_observation(camera=other, step=primary.step)
