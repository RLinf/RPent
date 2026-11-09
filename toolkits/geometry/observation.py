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

"""Explicit camera observations for geometry, independent of sensor file layouts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np


def _snapshot(value: np.ndarray, dtype: type | None = None) -> np.ndarray:
    result = np.array(value, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class GeometryObservation:
    """One aligned RGB/3-D observation in a caller-owned metric coordinate frame.

    Supply either ``world_xyz`` or optical-axis ``depth_m`` with pinhole
    ``intrinsics`` and ``camera_to_world``. Camera coordinates use positive Z
    forward. The output frame need not use Z as its up direction. ``step`` is
    an existing shared EnvState record; the provider resolves latest-step
    requests before returning this snapshot. Arrays are copied and made read-only.
    """

    camera: str
    step: int
    world_frame_id: str
    rgb: np.ndarray
    world_xyz: np.ndarray | None = None
    depth_m: np.ndarray | None = None
    intrinsics: np.ndarray | None = None
    camera_to_world: np.ndarray | None = None
    up_direction: np.ndarray | None = None
    valid_mask: np.ndarray | None = None
    source_refs: dict[str, str] = field(default_factory=dict)
    source_dtype: str = field(init=False)

    def __post_init__(self) -> None:
        if not self.camera.strip() or not self.world_frame_id.strip() or self.step < 0:
            raise ValueError(
                "observation requires a camera, frame ID and recorded step"
            )
        rgb = _snapshot(self.rgb)
        if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError("RGB must be a uint8 array with shape [H,W,3]")
        if min(rgb.shape[:2]) < 1:
            raise ValueError("geometry images must have nonempty dimensions")
        object.__setattr__(self, "rgb", rgb)
        shape = rgb.shape[:2]
        depth = None if self.depth_m is None else _snapshot(self.depth_m, float)
        if depth is not None and depth.shape != shape:
            raise ValueError("metric depth must match the RGB resolution")
        transform = self.camera_to_world
        if transform is not None:
            transform = _snapshot(transform, float)
            if (
                transform.shape != (4, 4)
                or not np.isfinite(transform).all()
                or not np.allclose(transform[3], [0, 0, 0, 1])
                or not np.allclose(transform[:3, :3].T @ transform[:3, :3], np.eye(3))
                or not np.isclose(np.linalg.det(transform[:3, :3]), 1)
            ):
                raise ValueError("camera_to_world must be a finite rigid 4x4 transform")
        intrinsics = self.intrinsics
        if intrinsics is not None:
            intrinsics = _snapshot(intrinsics, float)
            if (
                intrinsics.shape != (3, 3)
                or not np.isfinite(intrinsics).all()
                or abs(np.linalg.det(intrinsics)) < 1e-12
                or not np.allclose(intrinsics[2], [0, 0, 1])
            ):
                raise ValueError("intrinsics must be an invertible pinhole 3x3 matrix")
        world = self.world_xyz
        if world is None:
            if depth is None or intrinsics is None or transform is None:
                raise ValueError("supply world_xyz or depth_m with intrinsics and pose")
            rows, cols = np.indices(shape)
            pixels = np.stack((cols, rows, np.ones(shape)), axis=-1)
            camera_points = (pixels @ np.linalg.inv(intrinsics).T) * depth[..., None]
            world = camera_points @ transform[:3, :3].T + transform[:3, 3]
        object.__setattr__(self, "source_dtype", str(np.asarray(world).dtype))
        world = _snapshot(world, float)
        if world.shape != (*shape, 3):
            raise ValueError("world_xyz must match RGB with shape [H,W,3]")
        valid = np.isfinite(world).all(axis=-1)
        if depth is not None:
            valid &= np.isfinite(depth) & (depth > 0)
        elif transform is not None:
            optical_depth = (world - transform[:3, 3]) @ transform[:3, 2]
            valid &= np.isfinite(optical_depth) & (optical_depth > 0)
        if self.valid_mask is not None:
            supplied = np.asarray(self.valid_mask)
            if supplied.shape != shape or supplied.dtype != np.bool_:
                raise ValueError("valid_mask must be a boolean array matching RGB")
            valid &= supplied
        up = self.up_direction
        if up is not None:
            up = np.asarray(up, dtype=float)
            if up.shape != (3,) or not np.isfinite(up).all() or np.linalg.norm(up) == 0:
                raise ValueError("up_direction must be a finite nonzero 3-vector")
            up = _snapshot(up / np.linalg.norm(up))
        object.__setattr__(self, "world_xyz", world)
        object.__setattr__(self, "valid_mask", _snapshot(valid))
        object.__setattr__(self, "depth_m", depth)
        object.__setattr__(self, "intrinsics", intrinsics)
        object.__setattr__(self, "camera_to_world", transform)
        object.__setattr__(self, "up_direction", up)
        object.__setattr__(self, "source_refs", dict(self.source_refs))

    def optical_depth(self) -> np.ndarray | None:
        """Return optical-axis depths when supplied or derivable from the pose."""
        if self.depth_m is not None:
            return self.depth_m
        if self.camera_to_world is not None:
            transform = self.camera_to_world
            return (self.world_xyz - transform[:3, 3]) @ transform[:3, 2]
        return None

    def metadata(self) -> dict:
        """Describe the snapshot without embedding full sensor arrays in tool text."""
        return {
            "camera": self.camera,
            "step": self.step,
            "world_frame_id": self.world_frame_id,
            "units": "m",
            "image_shape": list(self.rgb.shape[:2]),
            "source_dtype": self.source_dtype,
            "camera_to_world": None
            if self.camera_to_world is None
            else self.camera_to_world.tolist(),
            "intrinsics": None if self.intrinsics is None else self.intrinsics.tolist(),
            "up_direction": None
            if self.up_direction is None
            else self.up_direction.tolist(),
            "source_refs": self.source_refs,
        }


class ObservationProvider(Protocol):
    """Resolve camera names and steps without exposing environment file conventions."""

    def get_observation(self, *, camera: str, step: int) -> GeometryObservation:
        """Return one aligned snapshot; resolve step=-1 to its recorded source step."""
        ...
