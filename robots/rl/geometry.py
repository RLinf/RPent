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

"""Measured mounting frames, RGB-D surface geometry, and image coordinate mapping."""

from __future__ import annotations

import io
import math
import warnings
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

#: Minimum number of valid points for surface statistics.
MIN_SURFACE_POINTS = 10
#: Major/minor xy variance ratio required to report a surface direction.
_AXIS_RATIO = 2.0

IMAGE_SIZES = (1024, 512, 256)
NATIVE_SIZE = 1024


def grasp_axes(closing: np.ndarray, approach: np.ndarray) -> np.ndarray:
    """Build canonical tool axes from measured jaw and wrist-to-finger directions."""
    x, z = (np.asarray(value, dtype=np.float64).copy() for value in (closing, approach))
    if any(
        value.shape != (3,)
        or not np.isfinite(value).all()
        or np.linalg.norm(value) < 1e-6
        for value in (x, z)
    ):
        raise ValueError("general requires measurable jaw and approach axes")
    x /= np.linalg.norm(x)
    if x[np.argmax(np.abs(x))] < 0:
        x *= -1
    z /= np.linalg.norm(z)
    if abs(x @ z) > 0.1:
        raise ValueError("measured jaw and approach axes are not perpendicular")
    z -= x * (x @ z)
    z /= np.linalg.norm(z)
    return np.column_stack((x, np.cross(z, x), z))


@dataclass(frozen=True)
class Coordinates:
    """Map simulator-world positions and rotations into the measured robot mounting frame."""

    origin: np.ndarray
    rotation: np.ndarray
    table_height_m: float | None = None
    floor_height_m: float | None = None
    version: int = 0
    tcp: str = "native_grasp_tcp"

    @classmethod
    def from_control(cls, control: dict[str, Any]) -> Coordinates:
        """Validate native calibration once, before any general-frame action."""
        try:
            calibration = control["coordinate_calibration"]
            origin = np.asarray(calibration["base_xyz"], dtype=np.float64)
            base_rotation = np.asarray(calibration["base_rotation"], dtype=np.float64)
            kind = "floor" if "floor_top_xyz" in calibration else "table"
            surface = np.asarray(calibration[f"{kind}_top_xyz"], dtype=np.float64)
        except KeyError as exc:
            raise ValueError(
                "general requires native mounting-base and support-surface calibration"
            ) from exc
        if any(v.shape != (3,) or not np.isfinite(v).all() for v in (origin, surface)):
            raise ValueError("general calibration positions must be finite xyz vectors")
        if (
            base_rotation.shape != (3, 3)
            or not np.isfinite(base_rotation).all()
            or not np.allclose(base_rotation.T @ base_rotation, np.eye(3), atol=1e-6)
            or not np.isclose(np.linalg.det(base_rotation), 1, atol=1e-6)
        ):
            raise ValueError("general calibration needs a proper base rotation")
        forward = base_rotation[:, 0].copy()
        forward[2] = 0
        if np.linalg.norm(forward) < 1e-6:
            raise ValueError(
                "robot base forward axis must have a horizontal projection"
            )
        forward /= np.linalg.norm(forward)
        up = np.array([0.0, 0.0, 1.0])
        rotation = np.column_stack((forward, np.cross(up, forward), up))
        return cls(
            origin.copy(),
            rotation,
            **{f"{kind}_height_m": float(surface[2] - origin[2])},
            tcp=str(calibration.get("tcp", "native_grasp_tcp")),
        )

    def after(self, previous: Coordinates) -> Coordinates:
        """Advance the version only when the measured mount frame changes."""
        changed = not (
            np.allclose(self.origin, previous.origin, rtol=0, atol=1e-6)
            and np.allclose(self.rotation, previous.rotation, rtol=0, atol=1e-6)
        )
        return replace(self, version=previous.version + int(changed))

    def points(self, world: np.ndarray) -> np.ndarray:
        return (np.asarray(world, dtype=np.float64) - self.origin) @ self.rotation

    def world_position(self, position: np.ndarray) -> np.ndarray:
        return self.origin + self.rotation @ position

    def orientation(self, world_rotation: np.ndarray) -> np.ndarray:
        return self.rotation.T @ world_rotation

    def world_orientation(self, rotation: np.ndarray) -> np.ndarray:
        return self.rotation @ rotation

    def metadata(self) -> dict[str, Any]:
        return {
            "tcp": self.tcp,
            "units": {"position": "metres", "orientation": "radians"},
            "axes": {"x": "forward", "y": "robot_left", "z": "up"},
            "origin_in_sim_world": self.origin.tolist(),
            "rotation_to_sim_world": self.rotation.tolist(),
            "raw_point_map_frame": "sim_world",
            "raw_point_map_unit": "m",
            "contract_version": "episode_coordinates_v2",
            "frame_version": self.version,
            "table_height_above_base_m": self.table_height_m,
            **(
                {
                    "support_surface": "floor",
                    "floor_height_above_base_m": self.floor_height_m,
                }
                if self.floor_height_m is not None
                else {}
            ),
        }

    def description(self) -> dict[str, Any]:
        metadata = self.metadata()
        return {
            "tcp": metadata["tcp"],
            "units": metadata["units"],
            "axes": metadata["axes"],
            "contract_version": metadata["contract_version"],
            "frame_version": self.version,
            "orientation": "fixed-axis XYZ RPY radians; tool +x closes the jaws, +z approaches",
            "table_height_above_base_m": None
            if self.table_height_m is None
            else round(self.table_height_m, 6),
            **(
                {
                    "support_surface": "floor",
                    "floor_height_above_base_m": round(self.floor_height_m, 6),
                }
                if self.floor_height_m is not None
                else {}
            ),
        }


def pose_matrix(pose: Any) -> np.ndarray:
    """Convert a world xyz/wxyz pose to a rigid 4x4 transform."""
    value = np.asarray(pose, dtype=np.float64)
    if value.shape != (7,) or not np.isfinite(value).all():
        raise ValueError("Expected seven finite xyz/wxyz values")
    if not np.isclose(np.linalg.norm(value[3:]), 1, atol=1e-4):
        raise ValueError("Expected a unit wxyz quaternion")
    result = np.eye(4)
    result[:3, :3] = Rotation.from_quat(value[[4, 5, 6, 3]]).as_matrix()
    result[:3, 3] = value[:3]
    return result


def mounting_calibration(
    left: Any, right: Any, surface: Any, *, support_surface: str = "table"
) -> dict[str, Any]:
    """Construct the shared arm frame and record the measured support surface."""
    if support_surface not in ("table", "floor"):
        raise ValueError("Support surface must be a table or a floor")
    left, right, surface = (
        np.asarray(v, dtype=np.float64) for v in (left, right, surface)
    )
    if any(v.shape != (3,) or not np.isfinite(v).all() for v in (left, right, surface)):
        raise ValueError("Mount and support positions must be finite xyz vectors")
    lateral = left - right
    lateral[2] = 0
    if np.linalg.norm(lateral) < 1e-6:
        raise ValueError("Two distinguishable arm mounting centres are required")
    lateral /= np.linalg.norm(lateral)
    up = np.array([0.0, 0.0, 1.0])
    rotation = np.column_stack((np.cross(lateral, up), lateral, up))
    return {
        "base_xyz": ((left + right) / 2).tolist(),
        "base_rotation": rotation.tolist(),
        f"{support_surface}_top_xyz": surface.tolist(),
        "left_mount_xyz": left.tolist(),
        "right_mount_xyz": right.tolist(),
    }


def finger_contact_centres(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Use the opposed inner faces in native axes where the jaws slide along Y."""
    first_below = (first[:, 1].min() + first[:, 1].max()) < (
        second[:, 1].min() + second[:, 1].max()
    )
    inward_limits = (np.max, np.min) if first_below else (np.min, np.max)
    centres = []
    for vertices, inward in zip((first, second), inward_limits, strict=True):
        lateral = vertices[:, 1]
        face = vertices[np.isclose(lateral, inward(lateral), rtol=0, atol=1e-5)]
        if len(face) < 3:
            raise ValueError("Missing flat inner contact face on a finger")
        centres.append((face.min(axis=0) + face.max(axis=0)) / 2)
    return np.asarray(centres)


def rpy_of(rotation: np.ndarray) -> list[float]:
    """Canonical fixed-axis XYZ angles."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Gimbal lock detected")
        angles = Rotation.from_matrix(rotation).as_euler("xyz")
    return [round(float(value), 6) for value in angles]


def metric_depth(depth: np.ndarray, camera_meta: dict[str, Any]) -> np.ndarray:
    """Convert a normalized MuJoCo depth buffer to metres."""
    buffer = np.asarray(depth, dtype=np.float64)
    if buffer.ndim == 3:
        buffer = buffer[..., 0]
    near, far = camera_meta["depth_near"], camera_meta["depth_far"]
    return near / (1.0 - buffer * (1.0 - near / far))


def world_from_depth(depth_m: np.ndarray, camera_meta: dict[str, Any]) -> np.ndarray:
    """Back-project a metric depth image to world-frame points."""
    intrinsic = np.asarray(camera_meta["intrinsic_K"], dtype=np.float64)
    extrinsic = np.asarray(camera_meta["extrinsic_cam2world"], dtype=np.float64)
    rows, cols = np.mgrid[0 : depth_m.shape[0], 0 : depth_m.shape[1]]
    camera_points = np.stack(
        [
            (cols - intrinsic[0, 2]) * depth_m / intrinsic[0, 0],
            (rows - intrinsic[1, 2]) * depth_m / intrinsic[1, 1],
            depth_m,
            np.ones_like(depth_m),
        ],
        axis=-1,
    )
    return (camera_points @ extrinsic.T)[..., :3]


def valid_points(points: np.ndarray, *, zero_is_invalid: bool = True) -> np.ndarray:
    """Keep finite points; optionally reject the legacy zero-depth sentinel."""
    flat = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    keep = np.isfinite(flat).all(axis=1)
    if zero_is_invalid:
        keep &= np.abs(flat).sum(axis=1) > 1e-6
    return flat[keep]


def surface_geometry(
    points: np.ndarray, *, center: np.ndarray | None = None
) -> dict[str, Any] | None:
    """Summarize visible surface points, or return None when too few exist."""
    if len(points) < MIN_SURFACE_POINTS:
        return None
    low, high = np.percentile(points, [5, 95], axis=0)
    xy = points[:, :2] - points[:, :2].mean(axis=0)
    variances, axes = np.linalg.eigh(xy.T @ xy / len(xy))
    axis_yaw = None
    if variances[1] > _AXIS_RATIO * variances[0]:
        heading = math.atan2(axes[1, 1], axes[0, 1])
        axis_yaw = round((heading + math.pi / 2) % math.pi - math.pi / 2, 3)
    return {
        "world_xyz": _rounded(np.median(points, axis=0) if center is None else center),
        "top_z_m": round(float(high[2]), 4),
        "size_xyz_m": _rounded(high - low),
        "axis_yaw": axis_yaw,
    }


def box_from_xyxy(box: list[float], shape: tuple[int, ...]) -> list[int]:
    """Convert a SAM3 [x0, y0, x1, y1] box to an exclusive row/col box."""
    height, width = shape[:2]
    x0, y0, x1, y1 = box
    return [
        min(max(math.floor(y0), 0), height),
        min(max(math.floor(x0), 0), width),
        min(max(math.ceil(y1), 0), height),
        min(max(math.ceil(x1), 0), width),
    ]


def box_from_mask(mask: np.ndarray) -> list[int]:
    """Return the exclusive row/col bounds of a nonempty mask."""
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    return [int(rows[0]), int(cols[0]), int(rows[-1]) + 1, int(cols[-1]) + 1]


def encode_png(image: np.ndarray, *, size: int | None = None) -> bytes:
    """Encode an RGB numpy array as lossless PNG (fast compress_level=1)."""
    pil = Image.fromarray(np.asarray(image, dtype=np.uint8))
    if size is not None and pil.size != (size, size):
        pil = pil.resize((size, size), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    pil.save(buf, format="PNG", compress_level=1)
    return buf.getvalue()


def resize_image(data: bytes, size: int) -> bytes:
    """Resize one planner image without changing the recorded native image."""
    with Image.open(io.BytesIO(data)) as image:
        if image.size == (size, size):
            return data
        output = io.BytesIO()
        image.resize((size, size), Image.Resampling.LANCZOS).save(
            output, format="PNG", compress_level=1
        )
        return output.getvalue()


def map_box(
    box: list[int], source: tuple[int, int], target: tuple[int, int]
) -> list[int]:
    """Map validated half-open rectangle edges outward."""
    return [
        (math.floor if index < 2 else math.ceil)(
            value * target[index % 2] / source[index % 2]
        )
        for index, value in enumerate(box)
    ]


def map_pixel(
    pixel: list[int], source: tuple[int, int], target: tuple[int, int]
) -> list[int]:
    """Map a validated displayed pixel centre to the native observation."""
    return [
        min(to - 1, math.floor((value + 0.5) * to / from_))
        for value, from_, to in zip(pixel, source, target, strict=True)
    ]


def _rounded(values: np.ndarray) -> list[float]:
    return [round(float(value), 4) for value in values]
