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

import io
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Annotated, Protocol

import numpy as np
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, model_validator


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
                # Native renderers may publish a rotation computed in float32.
                or not np.allclose(
                    transform[:3, :3].T @ transform[:3, :3], np.eye(3), atol=1e-6
                )
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

    def png_bytes(self) -> bytes:
        """Encode the original RGB raster for the injected perception models."""
        buffer = io.BytesIO()
        Image.fromarray(self.rgb).save(buffer, format="PNG")
        return buffer.getvalue()


class ObservationProvider(Protocol):
    """Resolve camera names and steps without exposing environment file conventions."""

    def get_observation(self, *, camera: str, step: int) -> GeometryObservation:
        """Return one aligned snapshot; resolve step=-1 to its recorded source step."""
        ...


class MultiViewObservationProvider(ObservationProvider, Protocol):
    """Optionally enumerate additional calibrated snapshots; acquisition belongs to the caller."""

    def additional_observations(
        self, primary: GeometryObservation
    ) -> Iterable[GeometryObservation]:
        """Yield other actual cameras in the same metric frame and identify their recorded step."""
        ...


Index = Annotated[int, Field(strict=True, ge=0)]
Size = Annotated[int, Field(strict=True, gt=0)]


class ImageSelection(BaseModel):
    """Native or resized full-frame pixels, with exclusive bottom/right box edges."""

    model_config = ConfigDict(extra="forbid")
    image_shape: Annotated[
        list[Size],
        Field(
            min_length=2,
            max_length=2,
            description="Height and width of the image used to select these pixels.",
        ),
    ]
    points_rc: (
        Annotated[
            list[Annotated[list[Index], Field(min_length=2, max_length=2)]],
            Field(min_length=1, max_length=16),
        ]
        | None
    ) = None
    box_rc: Annotated[list[Index], Field(min_length=4, max_length=4)] | None = None
    height_m: (
        Annotated[list[FiniteFloat | None], Field(min_length=2, max_length=2)] | None
    ) = None

    @model_validator(mode="after")
    def check_extent(self):
        if (self.points_rc is None) == (self.box_rc is None):
            raise ValueError("Supply points_rc or box_rc.")
        h, w = self.image_shape
        for i, (r, c) in enumerate(self.points_rc or ()):
            if r >= h or c >= w:
                raise ValueError(
                    f"Pixel [{r}, {c}] at index {i} exceeds image_shape [{h}, {w}]. "
                    f"Choose rows 0..{h - 1} and columns 0..{w - 1} on the same "
                    "image used to declare image_shape."
                )
        if self.box_rc:
            r0, c0, r1, c1 = self.box_rc
            if not (r0 < r1 <= h and c0 < c1 <= w):
                raise ValueError(
                    f"box_rc={self.box_rc} is invalid for image_shape [{h}, {w}]. "
                    f"Use 0 <= top < bottom <= {h} and 0 <= left < right <= {w}. "
                    "Select the visible box on the same image used to declare image_shape."
                )
        if self.height_m is not None:
            low, high = self.height_m
            if low is not None and high is not None and low > high:
                raise ValueError(
                    "height_m is [low, high] in metres and requires low <= high."
                )
        return self

    def native(self, obs: GeometryObservation):
        """Map pixel centers and half-open boxes to the measured raster."""
        h, w = self.image_shape
        nh, nw = obs.rgb.shape[:2]
        if h * nw != w * nh:
            raise ValueError(
                "Selection must use the full calibrated image at its original aspect ratio."
            )
        if self.points_rc is not None:
            return [
                (
                    min(nh - 1, (2 * r + 1) * nh // (2 * h)),
                    min(nw - 1, (2 * c + 1) * nw // (2 * w)),
                )
                for r, c in self.points_rc
            ], None
        r0, c0, r1, c1 = self.box_rc
        return [], (
            r0 * nh // h,
            c0 * nw // w,
            (r1 * nh + h - 1) // h,
            (c1 * nw + w - 1) // w,
        )
