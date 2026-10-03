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

"""Model-free Franka geometry proposals from public RGB-D and robot state."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from robots.franka.perception import project_depth_pixels_to_base


@dataclass(frozen=True)
class WorkspaceBounds:
    """Axis-aligned robot-base bounds for measured points and motion targets."""

    minimum: tuple[float, float, float]
    maximum: tuple[float, float, float]

    def __post_init__(self) -> None:
        lower = np.asarray(self.minimum, dtype=float)
        upper = np.asarray(self.maximum, dtype=float)
        if (
            lower.shape != (3,)
            or upper.shape != (3,)
            or not np.isfinite(lower).all()
            or not np.isfinite(upper).all()
            or not np.all(lower < upper)
        ):
            raise ValueError("workspace bounds must be finite increasing 3-vectors")


@dataclass(frozen=True)
class PerceptionConfig:
    """Sampling and connected-region thresholds for the classical detector."""

    pixel_stride: int = 2
    min_table_points: int = 100
    min_component_points: int = 20

    def __post_init__(self) -> None:
        if (
            min(self.pixel_stride, self.min_table_points, self.min_component_points)
            <= 0
        ):
            raise ValueError(
                "perception sampling and point thresholds must be positive"
            )


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    """Group adjacent elevated samples without depending on a vision package."""
    remaining = mask.copy()
    groups: list[list[tuple[int, int]]] = []
    for row, col in np.argwhere(mask):
        if not remaining[row, col]:
            continue
        remaining[row, col] = False
        stack = [(int(row), int(col))]
        group: list[tuple[int, int]] = []
        while stack:
            current_row, current_col = stack.pop()
            group.append((current_row, current_col))
            for neighbor_row, neighbor_col in (
                (current_row - 1, current_col),
                (current_row + 1, current_col),
                (current_row, current_col - 1),
                (current_row, current_col + 1),
            ):
                if (
                    0 <= neighbor_row < mask.shape[0]
                    and 0 <= neighbor_col < mask.shape[1]
                    and remaining[neighbor_row, neighbor_col]
                ):
                    remaining[neighbor_row, neighbor_col] = False
                    stack.append((neighbor_row, neighbor_col))
        groups.append(group)
    return groups


def _coarse_color(rgb: np.ndarray) -> str:
    """Name only obvious dominant RGB colors; do not claim object identity."""
    red, green, blue = np.median(rgb, axis=0).astype(float)
    if red > 1.5 * max(green, blue) and red > 80:
        return "red"
    if green > 1.5 * max(red, blue) and green > 80:
        return "green"
    if blue > 1.5 * max(red, green) and blue > 80:
        return "blue"
    return "unknown"


class ClassicalWorldStateBuilder:
    """Track measured elevated regions and freeze bounded motion proposals."""

    def __init__(
        self,
        *,
        workspace: WorkspaceBounds,
        config: PerceptionConfig,
        calibration: dict[str, Any],
        calibration_id: str,
    ) -> None:
        self.workspace = workspace
        self.config = config
        self.calibration = calibration
        self.calibration_id = calibration_id
        self._frame = 0
        self._next_track = 1
        self._previous: list[dict[str, Any]] = []

    def update(
        self,
        *,
        rgb: np.ndarray,
        depth: np.ndarray,
        camera_meta: dict[str, Any],
        robot_state: dict[str, Any],
        captured_at_s: float,
    ) -> dict[str, Any]:
        """Build one state from a synchronized external RGB-D/robot snapshot."""
        image = np.asarray(rgb)
        depth_array = np.asarray(depth)
        if (
            image.ndim != 3
            or image.shape[2] != 3
            or depth_array.shape != image.shape[:2]
        ):
            raise ValueError("RGB must be HxWx3 and depth must share its HxW shape")
        if not math.isfinite(captured_at_s):
            raise ValueError("capture time must be finite")
        tcp_pose = np.asarray(robot_state["raw_base_state"]["tcp_pose"], dtype=float)
        if tcp_pose.shape != (7,) or not np.isfinite(tcp_pose).all():
            raise ValueError("raw_base_state.tcp_pose must be a finite 7-vector")

        stride = self.config.pixel_stride
        grid_rows, grid_cols = np.meshgrid(
            np.arange(0, image.shape[0], stride),
            np.arange(0, image.shape[1], stride),
            indexing="ij",
        )
        points, valid = project_depth_pixels_to_base(
            depth=depth_array,
            rows=grid_rows.ravel(),
            cols=grid_cols.ravel(),
            meta=camera_meta,
            camera="third_person",
            tcp_pose=tcp_pose,
            calibration=self.calibration,
        )
        points = points.reshape(*grid_rows.shape, 3)
        valid = valid.reshape(grid_rows.shape)
        lower = np.asarray(self.workspace.minimum)
        upper = np.asarray(self.workspace.maximum)
        valid &= np.all((points >= lower) & (points <= upper), axis=2)

        self._frame += 1
        state_id = f"{self.calibration_id}/{self._frame:06d}"
        objects: list[dict[str, Any]] = []
        status = "insufficient_geometry"
        if valid.sum() >= self.config.min_table_points:
            table_z = float(np.quantile(points[valid, 2], 0.25))
            table_points = valid & (np.abs(points[:, :, 2] - table_z) <= 0.02)
            if table_points.sum() >= self.config.min_table_points:
                status = "no_objects"
                elevated = valid & (points[:, :, 2] > table_z + 0.03)
                for group in _components(elevated):
                    if len(group) < self.config.min_component_points:
                        continue
                    rows, cols = np.asarray(group).T
                    center = np.median(points[rows, cols], axis=0)
                    pixels = image[grid_rows[rows, cols], grid_cols[rows, cols]]
                    objects.append(
                        {
                            "position_base_m": center.tolist(),
                            "semantic_label": None,
                            "appearance": {"coarse_color": _coarse_color(pixels)},
                            "sample_count": len(group),
                        }
                    )
                if objects:
                    status = "ok"

        self._assign_tracks(objects)
        fallback = [
            {
                "candidate_id": f"{state_id}/hold",
                "tool": "finish",
                "args": {
                    "status": "stuck",
                    "summary": "Hold: no reliable object geometry for motion",
                },
            },
            {
                "candidate_id": f"{state_id}/refresh",
                "tool": "view_env_state",
                "args": {},
            },
            {
                "candidate_id": f"{state_id}/retreat-up",
                "tool": "move_delta",
                "args": {
                    "delta_xyz": [0.0, 0.0, max(0.0, min(0.02, upper[2] - tcp_pose[2]))]
                },
            },
        ]
        candidates = fallback
        for obj in objects:
            target = np.asarray(obj["position_base_m"], dtype=float).copy()
            target[2] = min(upper[2], target[2] + 0.08)
            delta = target - tcp_pose[:3]
            distance = float(np.linalg.norm(delta))
            if distance > 0.02:
                delta *= 0.02 / distance
            candidates.append(
                {
                    "candidate_id": f"{state_id}/approach-{obj['track_id']}",
                    "tool": "move_delta",
                    "args": {"delta_xyz": delta.tolist()},
                    "subject_track_id": obj["track_id"],
                    "based_on_state_id": state_id,
                    "resolver_version": "classical-v1",
                }
            )
        return {
            "state_id": state_id,
            "captured_at_s": captured_at_s,
            "source": {
                "camera": "third_person",
                "calibration_id": self.calibration_id,
                "simulator_object_state_used": False,
                "learned_model_used": False,
            },
            "objects": objects,
            "perception_quality": {"status": status, "camera_skew_ms": None},
            "parameter_candidates": candidates,
        }

    def _assign_tracks(self, objects: list[dict[str, Any]]) -> None:
        """Retain IDs for nearby visible regions across successive snapshots."""
        unmatched = {item["track_id"]: item for item in self._previous}
        for obj in objects:
            center = np.asarray(obj["position_base_m"])
            nearest = min(
                unmatched.values(),
                key=lambda item: np.linalg.norm(
                    center - np.asarray(item["position_base_m"])
                ),
                default=None,
            )
            if (
                nearest is not None
                and np.linalg.norm(center - np.asarray(nearest["position_base_m"]))
                <= 0.1
            ):
                obj["track_id"] = nearest["track_id"]
                unmatched.pop(nearest["track_id"])
            else:
                obj["track_id"] = f"obj-{self._next_track}"
                self._next_track += 1
        self._previous = objects
