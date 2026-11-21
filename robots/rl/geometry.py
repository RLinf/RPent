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

"""End-effector angles and visible-surface geometry for the LIBERO RL tools."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

#: Minimum number of valid points for surface statistics.
MIN_SURFACE_POINTS = 10
#: Major/minor xy variance ratio required to report a surface direction.
_AXIS_RATIO = 2.0


def rpy_of(rotation: np.ndarray) -> list[float]:
    """Canonical fixed-axis XYZ angles; rotation matrices remain authoritative."""
    import warnings

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Gimbal lock detected")
        angles = Rotation.from_matrix(rotation).as_euler("xyz")
    return [round(float(value), 6) for value in angles]


def metric_depth(depth: np.ndarray, camera_meta: dict[str, Any]) -> np.ndarray:
    """Convert a normalized MuJoCo depth buffer to meters."""
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


def valid_points(points: np.ndarray) -> np.ndarray:
    """Keep finite, nonzero world points as an ``(N, 3)`` float array."""
    flat = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    keep = np.isfinite(flat).all(axis=1) & (np.abs(flat).sum(axis=1) > 1e-6)
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
    """Convert a SAM3 ``[x0, y0, x1, y1]`` box to an exclusive row/col box."""
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


def _rounded(values: np.ndarray) -> list[float]:
    return [round(float(value), 4) for value in values]
