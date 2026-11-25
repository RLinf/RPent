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

"""Assess visible support and clearance for a local placement footprint."""

from typing import Annotated, Any

import numpy as np
from pydantic import Field, FiniteFloat

from rpent.tools import ToolResult, tool
from toolkits.geometry.geometry import FitOptions, fit_plane
from toolkits.observation import GeometryObservation, ObservationProvider

_GRID_SIZE = 8
_MIN_PLANE_POINTS = 12
_MAX_PLANE_POINTS = 1024


@tool(readonly=True, exclude=("observations",))
def check_free_space(
    camera: Annotated[str, Field(min_length=1, max_length=128)],
    center_xyz: Annotated[list[FiniteFloat], Field(min_length=3, max_length=3)],
    world_frame_id: Annotated[str, Field(min_length=1)],
    footprint_radius_m: Annotated[FiniteFloat, Field(gt=0)],
    clearance_m: Annotated[FiniteFloat, Field(gt=0)],
    step: Annotated[int, Field(ge=-1)] = -1,
    max_slope_deg: Annotated[FiniteFloat, Field(ge=0, le=45)] = 8.0,
    plane_tolerance_m: Annotated[FiniteFloat, Field(gt=0)] = 0.003,
    *,
    observations: ObservationProvider,
) -> ToolResult:
    """Check visible support and vertical clearance around a placement point.

    Free requires a calibrated view with visible support pixels across the
    footprint and no measured protrusion. Blocked means a protrusion or
    excessive slope was observed. Unknown means the observation is insufficient.
    Evaluate motion and collision separately.

    Args:
        camera: Recorded camera to inspect.
        center_xyz: [x,y,z] in metres at the expected support surface.
        world_frame_id: World frame of center_xyz, as reported by the observation.
        footprint_radius_m: Radius of the horizontal circle that encloses the object.
        clearance_m: Height to inspect above the fitted support, measured along up.
        step: Recorded step, or -1 for the latest record.
        max_slope_deg: Maximum angle from horizontal, 8 degrees by default.
        plane_tolerance_m: Plane fit and vertical height tolerance, 0.003 m by default.
        observations: Provider for recorded camera observations.
    """
    try:
        if plane_tolerance_m >= clearance_m:
            raise ValueError("plane_tolerance_m must be smaller than clearance_m.")
        view = observations.get_observation(camera=camera, step=step)
        if view.camera != camera or (step >= 0 and view.step != step):
            raise ValueError("Provider returned a different camera or step.")
        if view.world_frame_id != world_frame_id:
            raise ValueError(
                f"Expected world frame {view.world_frame_id!r}, got {world_frame_id!r}."
            )
        return ToolResult(
            data=_check_placement(
                view,
                np.asarray(center_xyz, dtype=float),
                radius=footprint_radius_m,
                clearance=clearance_m,
                max_slope=max_slope_deg,
                tolerance=plane_tolerance_m,
            )
        )
    except (
        ValueError,
        LookupError,
        OSError,
        RuntimeError,
        np.linalg.LinAlgError,
    ) as exc:
        return ToolResult(data={"status": "error", "error": str(exc)})


def _check_placement(
    view: GeometryObservation,
    center: np.ndarray,
    *,
    radius: float,
    clearance: float,
    max_slope: float,
    tolerance: float,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "camera": view.camera,
        "step": view.step,
        "world_frame_id": view.world_frame_id,
        "center_xyz": center.tolist(),
        "footprint_radius_m": radius,
        "clearance_m": clearance,
    }
    up = view.up_direction
    if up is None:
        return {**result, "status": "unknown", "reason": "missing_up_direction"}

    delta = view.world_xyz[view.valid_mask] - center
    axis = np.eye(3)[np.argmin(np.abs(up))]
    axis -= (axis @ up) * up
    axis /= np.linalg.norm(axis)
    horizontal = np.column_stack((axis, np.cross(up, axis)))
    xy = delta @ horizontal
    distance = np.linalg.norm(xy, axis=1)
    context_radius = 1.5 * radius
    context = (distance <= context_radius) & (
        np.abs(delta @ up) <= context_radius + 3 * tolerance
    )
    points = delta[context]
    if len(points) < _MIN_PLANE_POINTS:
        return {**result, "status": "unknown", "reason": "insufficient_support_points"}
    if len(points) > _MAX_PLANE_POINTS:
        indices = np.random.default_rng(0).choice(
            len(points), _MAX_PLANE_POINTS, replace=False
        )
        points = points[indices]

    fit = fit_plane(
        points,
        FitOptions(
            distance_threshold_m=tolerance,
            min_points=_MIN_PLANE_POINTS,
            max_points=_MAX_PLANE_POINTS,
            ransac_iterations=96,
        ),
        anchor=np.zeros(3),
    )
    if not fit.accepted or fit.quality["n_inliers"] < _MIN_PLANE_POINTS:
        return {
            **result,
            "status": "unknown",
            "reason": "unreliable_support_plane",
            "fit_reason": fit.reason or "insufficient_plane_consensus",
        }
    origin = np.asarray(fit.parameters["point"])
    normal = np.asarray(fit.parameters["normal"])
    if normal @ up < 0:
        normal = -normal
    alignment = float(np.clip(normal @ up, 0.0, 1.0))
    slope = float(
        np.degrees(np.arctan2(np.linalg.norm(np.cross(up, normal)), alignment))
    )
    result.update(
        slope_deg=slope,
        plane={"point_xyz": (origin + center).tolist(), "normal_xyz": normal.tolist()},
    )
    if abs(float(origin @ normal)) > tolerance * alignment:
        return {**result, "status": "unknown", "reason": "support_height_mismatch"}

    inside = distance <= radius
    perpendicular = (delta[inside] - origin) @ normal
    support = np.abs(perpendicular) <= tolerance * alignment
    visible, expected = _support_coverage(xy[inside][support] / radius)
    result.update(
        support_coverage=visible / expected,
        visible_support_cells=visible,
        expected_support_cells=expected,
    )
    if slope > max_slope + 1e-8:
        if visible == expected:
            return {**result, "status": "blocked", "reason": "support_too_steep"}
        return {
            **result,
            "status": "unknown",
            "reason": "insufficient_support_coverage",
        }

    heights = perpendicular / alignment
    obstacles = (heights > tolerance) & (heights <= clearance)
    count = int(obstacles.sum())
    result["obstacle_points"] = count
    if count:
        result["max_obstacle_height_m"] = float(heights[obstacles].max())
        return {**result, "status": "blocked", "reason": "obstacle_in_clearance"}
    if (heights > clearance).any():
        return {**result, "status": "unknown", "reason": "surface_above_clearance"}
    if (heights < -tolerance).any():
        return {**result, "status": "unknown", "reason": "support_discontinuity"}
    if visible < expected:
        return {
            **result,
            "status": "unknown",
            "reason": "insufficient_support_coverage",
        }
    footprint_origin = center + up * ((origin @ normal) / alignment)
    issue = _visibility_issue(
        view, footprint_origin, normal, horizontal, up, radius, clearance, tolerance
    )
    if issue is not None:
        return {**result, "status": "unknown", "reason": issue}
    return {**result, "status": "free"}


def _visibility_issue(
    view: GeometryObservation,
    origin: np.ndarray,
    normal: np.ndarray,
    horizontal: np.ndarray,
    up: np.ndarray,
    radius: float,
    clearance: float,
    tolerance: float,
) -> str | None:
    intrinsic, pose = view.intrinsics, view.camera_to_world
    if intrinsic is None or pose is None:
        return "missing_camera_calibration"
    rotation, camera = pose[:3, :3], pose[:3, 3]
    alignment = float(normal @ up)
    camera_offset = camera - origin
    camera_height = camera_offset @ normal
    if camera_height <= clearance * alignment:
        return "camera_below_clearance"

    # Lift the horizontal disc basis onto the support plane, then project it.
    plane_basis = horizontal - np.outer(up, (normal @ horizontal) / alignment)
    world_to_pixel = intrinsic @ rotation.T
    pixel_origin = world_to_pixel @ -camera_offset
    pixel_basis = world_to_pixel @ plane_basis
    height, width = view.rgb.shape[:2]
    frustum = np.array(
        [
            [0, 0, 1],
            [1, 0, 0.5],
            [-1, 0, width - 0.5],
            [0, 1, 0.5],
            [0, -1, height - 0.5],
        ]
    )
    # The clearance volume lies between two discs with the same horizontal footprint.
    extent = radius * np.linalg.norm(frustum @ pixel_basis, axis=1)
    top = clearance * (frustum @ world_to_pixel @ up)
    if np.any(frustum @ pixel_origin + np.minimum(0, top) <= extent):
        return "footprint_outside_camera_view"

    supported = view.valid_mask & (
        np.abs(np.einsum("ijk,k->ij", view.world_xyz, normal) - origin @ normal)
        <= tolerance * alignment
    )
    rows, cols = np.where(~supported)
    if not len(rows):
        return None
    rays = (
        np.column_stack((cols, rows, np.ones(len(rows))))
        @ np.linalg.inv(world_to_pixel).T
    )
    ray_dot = rays @ normal
    forward = ray_dot < 0
    if not forward.any():
        return None
    rays, ray_dot = rays[forward], ray_dot[forward]
    rows, cols = rows[forward], cols[forward]
    horizontal_origin = camera_offset @ horizontal
    horizontal_rays = rays @ horizontal
    far = -camera_height / ray_dot
    support_hits = horizontal_origin + far[:, None] * horizontal_rays
    if np.any(np.hypot(*support_hits.T) <= radius):
        return "unobserved_support_pixels"

    # The part of each ray behind its first return remains unobserved.
    near = (clearance * alignment - camera_height) / ray_dot
    observed = view.valid_mask[rows, cols]
    measured = (view.world_xyz[rows[observed], cols[observed]] - camera) @ rotation[
        :, 2
    ]
    near[observed] = np.maximum(near[observed], measured)
    speed_squared = np.einsum("ij,ij->i", horizontal_rays, horizontal_rays)
    closest = np.divide(
        -(horizontal_rays @ horizontal_origin),
        speed_squared,
        out=far.copy(),
        where=speed_squared > 0,
    )
    closest = np.clip(closest, near, far)
    clearance_hits = horizontal_origin + closest[:, None] * horizontal_rays
    if np.any((near < far) & (np.hypot(*clearance_hits.T) <= radius)):
        return "unobserved_clearance_pixels"
    return None


def _support_coverage(normalized_xy: np.ndarray) -> tuple[int, int]:
    centers = (np.arange(_GRID_SIZE) + 0.5) * 2 / _GRID_SIZE - 1
    nearest = np.maximum(np.abs(centers) - 1 / _GRID_SIZE, 0)
    footprint = nearest[:, None] ** 2 + nearest[None, :] ** 2 < 1
    indices = ((normalized_xy + 1) * _GRID_SIZE / 2).astype(int)
    indices = np.clip(indices, 0, _GRID_SIZE - 1)
    present = np.zeros(footprint.shape, dtype=bool)
    present[indices[:, 0], indices[:, 1]] = True
    return int((present & footprint).sum()), int(footprint.sum())
