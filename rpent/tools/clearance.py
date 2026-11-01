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

"""Judge whether a short straight drive is clear, using a built map.

Localization answers "where is the camera"; this module answers "is the floor
ahead empty enough to drive on". It projects the map's world points into a
corridor in front of a localized camera pose, measured against the map's
gravity-up direction and estimated floor:

1. points are rotated so gravity-up is ``+z`` and the floor is ``z = 0``;
2. a corridor of ``length`` by ``2 * half_width`` starts at the camera centre
   and runs along the camera's optical axis projected onto the floor, rotated
   by ``heading_deg``;
3. a point in the corridor between ``min_height`` and ``max_height`` above the
   floor is an obstacle; three or more such points block the segment;
4. a segment with no obstacle is ``clear`` only when every along-path bin has
   observed geometry. Otherwise the map never covered part of the path and the
   result is ``unmapped`` with the distance at which coverage ends.

This is an advisory check, not a planner: the caller still decides how to
drive. Distances are reported in camera heights by default because a map built
without base poses has arbitrary scale; a map that records a metric anchor (or
a caller-supplied camera height) also reports metres.

Check a pose archived in a built map::

    python -m rpent.tools.clearance \\
        --map-path /path/to/maps/workspace \\
        --frame-index 40 --heading-deg 15
"""

from __future__ import annotations

import argparse
import json
import math
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from rpent.tools.ground_map import LocalMapSource, select_serialized_frame
from rpent.tools.map_geometry import gravity_up, meters_per_unit, rotation_to_z
from rpent.utils.logging import get_logger
from toolkits.mapping.server import (
    OBJECTS_FILE,
    SEMANTIC_POINTS_FILE,
)

logger = get_logger("clearance")

#: Default corridor geometry, in camera heights.
DEFAULT_LENGTH = 2.0
DEFAULT_HALF_WIDTH = 0.75
DEFAULT_MIN_HEIGHT = 0.05
DEFAULT_MAX_HEIGHT = 4.0
DEFAULT_UNIT = "camera_heights"
#: Distance units a request may use. ``m`` needs a metric scale.
UNITS = ("camera_heights", "map_units", "m")
#: Along-path bins that must all have observed geometry for a clear verdict.
DEFAULT_COVERAGE_BINS = 12
#: Corridor points at or above this count block the segment.
DEFAULT_BLOCKING_POINTS = 3
#: Pixel stride while scanning each frame of the stored point grid.
DEFAULT_PIXEL_STRIDE = 4
#: Upper bound on loaded points, keeping the check bounded on large maps.
DEFAULT_MAX_POINTS = 2_000_000
#: Minimum stored depth confidence for a point to be considered.
DEFAULT_MIN_CONFIDENCE = 1.0
#: Voxel size for deduplicating the loaded cloud, in camera heights.
DEFAULT_VOXEL_FRACTION = 0.05
#: Semantic lookup cell size, in camera heights.
DEFAULT_SEMANTIC_CELL_FRACTION = 0.1


@dataclass(frozen=True)
class ClearanceGeometry:
    """Gravity and floor geometry needed to judge a corridor.

    ``rotation`` maps model-frame points into a gravity-aligned frame where up
    is ``+z``; ``floor_offset`` is the floor height along that axis. Both come
    from the map metadata and stay in the model frame: localization poses and
    ``world_points.npy`` share it.
    """

    status: str
    rotation: np.ndarray
    floor_offset: float
    camera_height: float | None
    meters_per_unit: float | None
    reason: str | None = None

    @property
    def usable(self) -> bool:
        """Whether gravity, the floor, and the camera height are all known."""
        return self.status == "ok"

    @classmethod
    def from_metadata(
        cls,
        metadata: Mapping[str, Any],
        *,
        camera_height_m: float | None = None,
    ) -> ClearanceGeometry:
        """Read gravity, floor, and scale from ``map.json``.

        ``camera_height_m`` supplies a metric anchor when the map has none;
        one model unit then equals ``camera_height_m / floor.camera_height``
        metres for this checker. It does not change the saved map.
        """
        scale = meters_per_unit(dict(metadata))
        up = gravity_up(dict(metadata))
        if up is None:
            return cls(
                status="unusable",
                rotation=np.eye(3),
                floor_offset=0.0,
                camera_height=None,
                meters_per_unit=scale,
                reason="the map records no gravity estimate",
            )
        rotation = rotation_to_z(up)
        floor = metadata.get("floor")
        if not isinstance(floor, Mapping) or floor.get("status") != "estimated":
            return cls(
                status="unusable",
                rotation=rotation,
                floor_offset=0.0,
                camera_height=None,
                meters_per_unit=scale,
                reason="the map records no floor estimate",
            )
        offset = floor.get("offset")
        height = floor.get("camera_height")
        try:
            offset_value = float(offset)
            height_value = float(height)
        except (TypeError, ValueError):
            offset_value = height_value = math.nan
        if not math.isfinite(offset_value) or not math.isfinite(height_value):
            return cls(
                status="unusable",
                rotation=rotation,
                floor_offset=0.0,
                camera_height=None,
                meters_per_unit=scale,
                reason="the map's floor offset or camera height is not finite",
            )
        if height_value <= 0.0:
            return cls(
                status="unusable",
                rotation=rotation,
                floor_offset=offset_value,
                camera_height=None,
                meters_per_unit=scale,
                reason="the map's camera height above the floor is not positive",
            )
        if camera_height_m is not None:
            anchor = float(camera_height_m)
            if not math.isfinite(anchor) or anchor <= 0.0:
                raise ValueError("camera_height_m must be a positive number")
            if scale is None:
                scale = anchor / height_value
        return cls(
            status="ok",
            rotation=rotation,
            floor_offset=offset_value,
            camera_height=height_value,
            meters_per_unit=scale,
        )


@dataclass(frozen=True)
class ClearanceRequest:
    """Corridor geometry to test, with distances in ``unit``."""

    heading_deg: float = 0.0
    length: float = DEFAULT_LENGTH
    half_width: float = DEFAULT_HALF_WIDTH
    min_height: float = DEFAULT_MIN_HEIGHT
    max_height: float = DEFAULT_MAX_HEIGHT
    unit: str = DEFAULT_UNIT
    bins: int = DEFAULT_COVERAGE_BINS
    blocking_points: int = DEFAULT_BLOCKING_POINTS

    def __post_init__(self) -> None:
        values = {
            "heading_deg": self.heading_deg,
            "length": self.length,
            "half_width": self.half_width,
            "min_height": self.min_height,
            "max_height": self.max_height,
        }
        for name, value in values.items():
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")
        if self.length <= 0.0:
            raise ValueError("length must be positive")
        if self.half_width <= 0.0:
            raise ValueError("half_width must be positive")
        if self.min_height < 0.0:
            raise ValueError("min_height must not be negative")
        if self.max_height <= self.min_height:
            raise ValueError("max_height must be greater than min_height")
        if self.unit not in UNITS:
            raise ValueError(f"unit must be one of {list(UNITS)}, got {self.unit!r}")
        if int(self.bins) < 1:
            raise ValueError("bins must be positive")
        if int(self.blocking_points) < 1:
            raise ValueError("blocking_points must be positive")


@dataclass(frozen=True)
class CorridorScan:
    """Verdict of one corridor test on gravity-aligned points."""

    status: str
    observed_bins: np.ndarray
    coverage: float
    observed_distance: float | None
    blocked_points: int
    nearest_index: int | None
    nearest: dict[str, float] | None


@dataclass(frozen=True)
class ClearanceResult:
    """Outcome of one clearance check, in the requested unit."""

    status: str
    unit: str
    length: float
    half_width: float
    heading_deg: float
    min_height: float
    max_height: float
    frame: str
    pose: np.ndarray | None
    keyframe: int | None
    camera_height: float | None
    meters_per_unit: float | None
    unit_scale: float | None
    nearest: dict[str, Any] | None
    observed_distance: float | None
    coverage: float | None
    point_count: int
    blocked_points: int
    coverage_bins: list[bool]
    latency_ms: float
    reason: str | None = None
    localization: dict[str, Any] | None = None

    @property
    def ok(self) -> bool:
        """Whether the check produced a clear, blocked, or unmapped verdict."""
        return self.status in {"clear", "blocked", "unmapped"}

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable record."""
        payload: dict[str, Any] = {
            "status": self.status,
            "frame": self.frame,
            "unit": self.unit,
            "length": self.length,
            "half_width": self.half_width,
            "heading_deg": self.heading_deg,
            "min_height": self.min_height,
            "max_height": self.max_height,
            "pose": None if self.pose is None else self.pose.tolist(),
            "keyframe": self.keyframe,
            "camera_height": self.camera_height,
            "meters_per_unit": self.meters_per_unit,
            "unit_scale": self.unit_scale,
            "nearest": self.nearest,
            "observed_distance": self.observed_distance,
            "coverage": self.coverage,
            "point_count": self.point_count,
            "blocked_points": self.blocked_points,
            "coverage_bins": self.coverage_bins,
            "latency_ms": self.latency_ms,
            "reason": self.reason,
            "localization": self.localization,
        }
        return payload

    @classmethod
    def from_dict(cls, payload: Any) -> ClearanceResult:
        """Decode one RPC record, rejecting a malformed response."""
        if not isinstance(payload, Mapping):
            raise ValueError("clearance response must be an object")
        status = payload.get("status")
        if not isinstance(status, str) or not status:
            raise ValueError("clearance response has no status")
        unit = payload.get("unit")
        if unit not in UNITS:
            raise ValueError(f"clearance response has an unknown unit: {unit!r}")
        numbers: dict[str, float] = {}
        for name in (
            "length",
            "half_width",
            "heading_deg",
            "min_height",
            "max_height",
            "latency_ms",
        ):
            value = payload.get(name)
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise ValueError(f"clearance response has no numeric {name}")
            numbers[name] = float(value)
        frame = payload.get("frame")
        if not isinstance(frame, str) or not frame:
            raise ValueError("clearance response has no frame")
        pose = payload.get("pose")
        pose_array: np.ndarray | None = None
        if pose is not None:
            pose_array = np.asarray(pose, dtype=np.float64)
            if pose_array.shape != (4, 4) or not np.isfinite(pose_array).all():
                raise ValueError("clearance response pose must be a finite 4x4")
        keyframe = payload.get("keyframe")
        if keyframe is not None and (
            isinstance(keyframe, bool) or not isinstance(keyframe, int)
        ):
            raise ValueError("clearance response keyframe must be an integer")
        point_count = payload.get("point_count")
        blocked_points = payload.get("blocked_points")
        for name, value in (
            ("point_count", point_count),
            ("blocked_points", blocked_points),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"clearance response has no non-negative {name}")
        nearest = payload.get("nearest")
        if nearest is not None and not isinstance(nearest, Mapping):
            raise ValueError("clearance response nearest must be an object or null")
        localization = payload.get("localization")
        if localization is not None and not isinstance(localization, Mapping):
            raise ValueError("clearance response localization must be an object")
        coverage_bins = payload.get("coverage_bins", [])
        if not isinstance(coverage_bins, list) or not all(
            isinstance(value, bool) for value in coverage_bins
        ):
            raise ValueError("clearance response coverage_bins must be booleans")
        optional_float = {}
        for name in (
            "camera_height",
            "meters_per_unit",
            "unit_scale",
            "observed_distance",
            "coverage",
        ):
            value = payload.get(name)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int | float)
            ):
                raise ValueError(f"clearance response {name} must be numeric or null")
            optional_float[name] = None if value is None else float(value)
        reason = payload.get("reason")
        if reason is not None and not isinstance(reason, str):
            raise ValueError("clearance response reason must be a string or null")
        return cls(
            status=status,
            unit=unit,
            length=numbers["length"],
            half_width=numbers["half_width"],
            heading_deg=numbers["heading_deg"],
            min_height=numbers["min_height"],
            max_height=numbers["max_height"],
            frame=frame,
            pose=pose_array,
            keyframe=keyframe,
            camera_height=optional_float["camera_height"],
            meters_per_unit=optional_float["meters_per_unit"],
            unit_scale=optional_float["unit_scale"],
            nearest=None if nearest is None else dict(nearest),
            observed_distance=optional_float["observed_distance"],
            coverage=optional_float["coverage"],
            point_count=point_count,
            blocked_points=blocked_points,
            coverage_bins=[bool(value) for value in coverage_bins],
            latency_ms=numbers["latency_ms"],
            reason=reason,
            localization=None if localization is None else dict(localization),
        )


class _SemanticLookup:
    """Nearest semantic point lookup on a voxel grid, for obstacle labels."""

    def __init__(
        self,
        points: np.ndarray,
        labels: np.ndarray,
        object_ids: np.ndarray,
        label_names: Sequence[str],
        object_labels: Sequence[str],
        *,
        cell_size: float,
    ) -> None:
        self._cell_size = float(cell_size)
        self._cells: dict[tuple[int, int, int], tuple[np.ndarray, str, str | None]] = {}
        if points.size == 0:
            return
        keys = np.floor(points / self._cell_size).astype(np.int64)
        _, first = np.unique(keys, axis=0, return_index=True)
        for index in first:
            label_index = int(labels[index])
            object_index = int(object_ids[index])
            if not 0 <= label_index < len(label_names):
                continue
            object_id = (
                object_labels[object_index]
                if 0 <= object_index < len(object_labels)
                else None
            )
            self._cells[tuple(int(value) for value in keys[index])] = (
                points[index],
                label_names[label_index],
                object_id,
            )

    @classmethod
    def from_map(
        cls,
        map_dir: Path,
        *,
        cell_size: float,
        rotation: np.ndarray,
        floor_offset: float,
    ) -> _SemanticLookup | None:
        """Load a grounded map's semantic cloud, or ``None`` when absent.

        The cloud is stored in the model frame like ``world_points.npy``, so it
        is aligned to the same gravity-up frame as the corridor points.
        """
        points_path = map_dir / SEMANTIC_POINTS_FILE
        if not points_path.is_file():
            return None
        with np.load(points_path) as data:
            points = np.asarray(data["points"], dtype=np.float64) @ rotation.T
            points[:, 2] -= floor_offset
            labels = np.asarray(data["labels"], dtype=np.int64)
            object_ids = np.asarray(data["object_ids"], dtype=np.int64)
            label_names = [str(name) for name in data["label_names"]]
        objects_path = map_dir / OBJECTS_FILE
        object_labels: list[str] = []
        if objects_path.is_file():
            document = json.loads(objects_path.read_text())
            entries = document.get("objects", []) if isinstance(document, dict) else []
            if isinstance(entries, list):
                object_labels = [
                    str(entry.get("id")) for entry in entries if isinstance(entry, dict)
                ]
        return cls(
            points,
            labels,
            object_ids,
            label_names,
            object_labels,
            cell_size=cell_size,
        )

    def nearest(self, point: np.ndarray) -> tuple[str, str | None] | None:
        """Return the label and object id of the closest semantic point."""
        if not self._cells:
            return None
        cell = np.floor(np.asarray(point, dtype=np.float64) / self._cell_size)
        base = tuple(int(value) for value in cell)
        best: tuple[str, str | None] | None = None
        best_distance = self._cell_size * math.sqrt(3.0)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    entry = self._cells.get((base[0] + dx, base[1] + dy, base[2] + dz))
                    if entry is None:
                        continue
                    distance = float(np.linalg.norm(entry[0] - point))
                    if distance <= best_distance:
                        best_distance = distance
                        best = (entry[1], entry[2])
        return best


def check_corridor(
    points: np.ndarray,
    *,
    origin: np.ndarray,
    forward: np.ndarray,
    length: float,
    half_width: float,
    min_height: float,
    max_height: float,
    bins: int = DEFAULT_COVERAGE_BINS,
    blocking_points: int = DEFAULT_BLOCKING_POINTS,
) -> CorridorScan:
    """Classify a straight corridor on gravity-aligned points.

    ``points`` are ``[N,3]`` with ``+z`` up and the floor at ``z = 0``.
    ``origin`` is the camera centre in that frame; ``forward`` is the heading
    in the ``x``/``y`` plane. Returned distances and ``nearest_index`` refer to
    the input rows.
    """
    array = np.asarray(points, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError(f"points must be [N,3], got {array.shape}")
    heading = np.asarray(forward, dtype=np.float64).reshape(-1)
    if heading.shape != (2,):
        raise ValueError(f"forward must be a 2-vector, got {forward!r}")
    norm = float(np.linalg.norm(heading))
    if not math.isfinite(norm) or norm <= 0.0:
        raise ValueError(f"forward must be finite and non-zero, got {forward!r}")
    heading = heading / norm
    origin_xy = np.asarray(origin, dtype=np.float64).reshape(-1)
    if origin_xy.shape[0] < 2:
        raise ValueError(f"origin must have at least two coordinates: {origin!r}")

    finite = np.isfinite(array).all(axis=1)
    relative = array[:, :2] - origin_xy[:2]
    along = relative @ heading
    lateral = relative[:, 0] * (-heading[1]) + relative[:, 1] * heading[0]
    height = array[:, 2]
    inside = (
        finite
        & (along > 0.0)
        & (along <= float(length))
        & (np.abs(lateral) <= float(half_width))
    )
    observed = inside & (height <= float(max_height))
    obstacle = observed & (height >= float(min_height))
    bin_count = int(bins)
    bin_index = np.clip(
        np.floor(along / float(length) * bin_count).astype(np.int64),
        0,
        bin_count - 1,
    )
    observed_bins = np.zeros(bin_count, dtype=bool)
    observed_bins[bin_index[observed]] = True
    blocked_count = int(obstacle.sum())
    coverage = float(observed_bins.mean())
    observed_distance: float | None = None
    if blocked_count >= int(blocking_points):
        status = "blocked"
    elif observed_bins.all():
        status = "clear"
    else:
        status = "unmapped"
        observed_distance = float(int(np.argmin(observed_bins)) * length / bin_count)
    nearest_index: int | None = None
    nearest: dict[str, float] | None = None
    if blocked_count:
        candidates = np.nonzero(obstacle)[0]
        nearest_index = int(candidates[np.argmin(along[candidates])])
        nearest = {
            "distance": float(along[nearest_index]),
            "lateral": float(lateral[nearest_index]),
            "height": float(height[nearest_index]),
        }
    return CorridorScan(
        status=status,
        observed_bins=observed_bins,
        coverage=coverage,
        observed_distance=observed_distance,
        blocked_points=blocked_count,
        nearest_index=nearest_index,
        nearest=nearest,
    )


class MapClearance:
    """A built map's points and floor geometry, prepared for corridor checks."""

    def __init__(
        self,
        map_path: str | Path,
        *,
        metadata: Mapping[str, Any] | None = None,
        camera_height_m: float | None = None,
        pixel_stride: int = DEFAULT_PIXEL_STRIDE,
        max_points: int = DEFAULT_MAX_POINTS,
        min_confidence: float = DEFAULT_MIN_CONFIDENCE,
        voxel_fraction: float = DEFAULT_VOXEL_FRACTION,
        semantic_cell_fraction: float = DEFAULT_SEMANTIC_CELL_FRACTION,
    ) -> None:
        self.map_path = str(Path(map_path))
        self._map_dir = Path(map_path)
        if metadata is None:
            metadata = json.loads((self._map_dir / "map.json").read_text())
        self.metadata = dict(metadata)
        self.geometry = ClearanceGeometry.from_metadata(
            self.metadata, camera_height_m=camera_height_m
        )
        self._points_path = self._map_dir / "world_points.npy"
        self._confidence_path = self._map_dir / "world_points_conf.npy"
        self._pixel_stride = max(1, int(pixel_stride))
        self._max_points = max(1, int(max_points))
        self._min_confidence = float(min_confidence)
        self._voxel_fraction = float(voxel_fraction)
        self._semantic_cell_fraction = float(semantic_cell_fraction)
        self._points: np.ndarray | None = None
        self._lookup: _SemanticLookup | None = None
        self._semantic_checked = False
        self._lock = threading.Lock()

    @property
    def has_points(self) -> bool:
        """Whether the map stored a world-point cloud."""
        return self._points_path.is_file()

    @property
    def usable(self) -> bool:
        """Whether a corridor check can run on this map."""
        return self.geometry.usable and self.has_points

    def info(self) -> dict[str, Any]:
        """Report availability, geometry, and default request units."""
        return {
            "available": self.usable,
            "status": self.geometry.status,
            "reason": self.geometry.reason,
            "frame": "model",
            "has_points": self.has_points,
            "camera_height": self.geometry.camera_height,
            "meters_per_unit": self.geometry.meters_per_unit,
            "units": list(UNITS),
            "defaults": {
                "unit": DEFAULT_UNIT,
                "length": DEFAULT_LENGTH,
                "half_width": DEFAULT_HALF_WIDTH,
                "min_height": DEFAULT_MIN_HEIGHT,
                "max_height": DEFAULT_MAX_HEIGHT,
            },
        }

    @property
    def points(self) -> np.ndarray:
        """Gravity-aligned points with the floor at ``z = 0``."""
        with self._lock:
            if self._points is None:
                self._points = self._load_points()
            return self._points

    def _load_points(self) -> np.ndarray:
        if not self.has_points:
            return np.empty((0, 3), dtype=np.float64)
        serialized = np.load(self._points_path, mmap_mode="r")
        frames = int(self.metadata.get("frames", 0))
        if frames <= 0:
            frames = int(serialized.shape[0])
        confidence = (
            np.load(self._confidence_path, mmap_mode="r")
            if self._confidence_path.is_file()
            else None
        )
        rotation = self.geometry.rotation
        collected: list[np.ndarray] = []
        for index in range(frames):
            frame = np.asarray(
                select_serialized_frame(serialized, index, frames, "world_points")
            )
            frame = frame[:: self._pixel_stride, :: self._pixel_stride].reshape(-1, 3)
            finite = np.isfinite(frame).all(axis=1)
            if confidence is not None:
                support = np.asarray(
                    select_serialized_frame(confidence, index, frames, "confidence")
                )[:: self._pixel_stride, :: self._pixel_stride].reshape(-1)
                finite &= np.isfinite(support) & (support >= self._min_confidence)
            selected = frame[finite]
            if selected.size:
                collected.append(selected @ rotation.T)
        if not collected:
            return np.empty((0, 3), dtype=np.float64)
        points = np.concatenate(collected, axis=0)
        points[:, 2] -= self.geometry.floor_offset
        camera_height = self.geometry.camera_height or 1.0
        points = _voxel_dedupe(points, self._voxel_fraction * camera_height)
        if points.shape[0] > self._max_points:
            choice = (
                np.linspace(0, points.shape[0] - 1, self._max_points)
                .round()
                .astype(np.int64)
            )
            points = points[choice]
        logger.info(
            "loaded %d clearance points from %s", points.shape[0], self.map_path
        )
        return points

    def _semantic_lookup(self) -> _SemanticLookup | None:
        with self._lock:
            if not self._semantic_checked:
                self._semantic_checked = True
                cell = self._semantic_cell_fraction * (
                    self.geometry.camera_height or 1.0
                )
                try:
                    self._lookup = _SemanticLookup.from_map(
                        self._map_dir,
                        cell_size=cell,
                        rotation=self.geometry.rotation,
                        floor_offset=self.geometry.floor_offset,
                    )
                except Exception as exc:
                    logger.warning(
                        "could not read semantic points for attribution: %s", exc
                    )
            return self._lookup

    def check(
        self,
        pose: np.ndarray | None,
        request: ClearanceRequest,
        *,
        keyframe: int | None = None,
        localization: Mapping[str, Any] | None = None,
    ) -> ClearanceResult:
        """Check one camera pose, or explain why the check cannot run."""
        start = time.perf_counter()
        pose_array: np.ndarray | None = None
        if pose is not None:
            pose_array = np.asarray(pose, dtype=np.float64)
            if pose_array.shape != (4, 4) or not np.isfinite(pose_array).all():
                raise ValueError("pose must be a finite 4x4 camera-to-world matrix")
            pose_array = pose_array.copy()
        localization_record = None if localization is None else dict(localization)
        if not self.geometry.usable:
            return self._result(
                "unusable",
                request,
                pose=None,
                keyframe=keyframe,
                localization=localization_record,
                start=start,
                reason=self.geometry.reason,
            )
        if not self.has_points:
            return self._result(
                "unusable",
                request,
                pose=None,
                keyframe=keyframe,
                localization=localization_record,
                start=start,
                reason="the map has no world points",
            )
        if pose_array is None:
            return self._result(
                "unlocalized",
                request,
                pose=None,
                keyframe=keyframe,
                localization=localization_record,
                start=start,
                reason="no camera pose is available for the clearance check",
            )
        try:
            unit_scale = self._unit_scale(request.unit)
        except ValueError as exc:
            return self._result(
                "unusable",
                request,
                pose=pose_array,
                keyframe=keyframe,
                localization=localization_record,
                start=start,
                reason=str(exc),
            )
        rotation = self.geometry.rotation
        camera = pose_array[:3, 3] @ rotation.T
        camera[2] -= self.geometry.floor_offset
        axis = pose_array[:3, 2] @ rotation.T
        horizontal = axis[:2]
        norm = float(np.linalg.norm(horizontal))
        if not math.isfinite(norm) or norm < 0.1:
            return self._result(
                "unusable",
                request,
                pose=pose_array,
                keyframe=keyframe,
                localization=localization_record,
                start=start,
                reason=(
                    "the camera optical axis is too close to vertical to define "
                    "a drive heading"
                ),
            )
        forward = horizontal / norm
        angle = math.radians(float(request.heading_deg))
        forward = np.asarray(
            [
                math.cos(angle) * forward[0] - math.sin(angle) * forward[1],
                math.sin(angle) * forward[0] + math.cos(angle) * forward[1],
            ]
        )
        length = float(request.length) / unit_scale
        half_width = float(request.half_width) / unit_scale
        min_height = float(request.min_height) / unit_scale
        max_height = float(request.max_height) / unit_scale
        points = self.points
        scan = check_corridor(
            points,
            origin=camera,
            forward=forward,
            length=length,
            half_width=half_width,
            min_height=min_height,
            max_height=max_height,
            bins=int(request.bins),
            blocking_points=int(request.blocking_points),
        )
        nearest: dict[str, Any] | None = None
        if scan.nearest is not None and scan.nearest_index is not None:
            nearest = {name: value * unit_scale for name, value in scan.nearest.items()}
            match = None
            lookup = self._semantic_lookup()
            if lookup is not None:
                match = lookup.nearest(points[scan.nearest_index])
            if match is not None:
                nearest["label"], nearest["object_id"] = match
            else:
                nearest["label"] = None
                nearest["object_id"] = None
            if self.geometry.meters_per_unit is not None:
                nearest["distance_m"] = (
                    scan.nearest["distance"] * self.geometry.meters_per_unit
                )
        return self._result(
            scan.status,
            request,
            pose=pose_array,
            keyframe=keyframe,
            localization=localization_record,
            start=start,
            unit_scale=unit_scale,
            nearest=nearest,
            observed_distance=(
                None
                if scan.observed_distance is None
                else scan.observed_distance * unit_scale
            ),
            coverage=scan.coverage,
            blocked_points=scan.blocked_points,
            coverage_bins=[bool(value) for value in scan.observed_bins],
        )

    def _unit_scale(self, unit: str) -> float:
        """Return how many ``unit``s one model unit spans."""
        if unit == "map_units":
            return 1.0
        if unit == "camera_heights":
            if self.geometry.camera_height is None:
                raise ValueError(
                    "camera height is unknown; pass unit='map_units' instead"
                )
            return 1.0 / self.geometry.camera_height
        if self.geometry.meters_per_unit is None:
            raise ValueError(
                "the map has no metric scale; pass --camera-height-m when "
                "starting the service or use unit='camera_heights'"
            )
        return self.geometry.meters_per_unit

    def _result(
        self,
        status: str,
        request: ClearanceRequest,
        *,
        pose: np.ndarray | None,
        keyframe: int | None,
        localization: dict[str, Any] | None,
        start: float,
        reason: str | None = None,
        unit_scale: float | None = None,
        nearest: dict[str, Any] | None = None,
        observed_distance: float | None = None,
        coverage: float | None = None,
        blocked_points: int = 0,
        coverage_bins: Sequence[bool] | None = None,
    ) -> ClearanceResult:
        return ClearanceResult(
            status=status,
            unit=request.unit,
            length=float(request.length),
            half_width=float(request.half_width),
            heading_deg=float(request.heading_deg),
            min_height=float(request.min_height),
            max_height=float(request.max_height),
            frame="model",
            pose=pose,
            keyframe=keyframe,
            camera_height=self.geometry.camera_height,
            meters_per_unit=self.geometry.meters_per_unit,
            unit_scale=unit_scale,
            nearest=nearest,
            observed_distance=observed_distance,
            coverage=coverage,
            point_count=0 if self._points is None else int(self._points.shape[0]),
            blocked_points=int(blocked_points),
            coverage_bins=[
                bool(value)
                for value in ([] if coverage_bins is None else coverage_bins)
            ],
            latency_ms=(time.perf_counter() - start) * 1000.0,
            reason=reason,
            localization=localization,
        )


def _voxel_dedupe(points: np.ndarray, size: float) -> np.ndarray:
    """Keep one point per voxel of ``size``."""
    if size <= 0.0 or points.shape[0] == 0:
        return points
    keys = np.floor(points / size).astype(np.int64)
    _, first = np.unique(keys, axis=0, return_index=True)
    return points[np.sort(first)]


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Check whether a short drive is clear on a built map"
    )
    parser.add_argument("--map-path", required=True, help="built map directory")
    parser.add_argument(
        "--frame-index",
        type=int,
        required=True,
        help="archived frame whose camera pose starts the corridor",
    )
    parser.add_argument("--heading-deg", type=float, default=0.0)
    parser.add_argument("--length", type=float, default=DEFAULT_LENGTH)
    parser.add_argument("--half-width", type=float, default=DEFAULT_HALF_WIDTH)
    parser.add_argument("--min-height", type=float, default=DEFAULT_MIN_HEIGHT)
    parser.add_argument("--max-height", type=float, default=DEFAULT_MAX_HEIGHT)
    parser.add_argument("--unit", choices=UNITS, default=DEFAULT_UNIT)
    parser.add_argument(
        "--camera-height-m",
        type=float,
        default=None,
        help="metric camera height; anchors scale on a map without one",
    )
    parser.add_argument(
        "--output", default=None, help="write the JSON result here instead of stdout"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Run one clearance check from an archived camera pose."""
    args = _build_argparser().parse_args(argv)
    source = LocalMapSource(args.map_path)
    poses = source.camera_to_world
    if poses is None:
        raise SystemExit(f"{args.map_path} has no camera poses")
    if not 0 <= args.frame_index < len(poses):
        raise SystemExit(
            f"--frame-index must be within [0, {len(poses) - 1}], "
            f"got {args.frame_index}"
        )
    request = ClearanceRequest(
        heading_deg=args.heading_deg,
        length=args.length,
        half_width=args.half_width,
        min_height=args.min_height,
        max_height=args.max_height,
        unit=args.unit,
    )
    clearance = MapClearance(args.map_path, camera_height_m=args.camera_height_m)
    result = clearance.check(
        poses[args.frame_index], request, keyframe=args.frame_index
    )
    payload = result.to_dict()
    if args.output is not None:
        Path(args.output).write_text(json.dumps(payload, indent=2))
    else:
        print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
