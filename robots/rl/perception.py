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

"""Shared RGB-D observation store and perception tools for all RL simulators."""

from __future__ import annotations

import itertools
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Annotated, Any

import numpy as np
from pydantic import Field

from robots.rl import geometry
from robots.rl.geometry import encode_png, map_box, map_pixel
from rpent.session import EnvState, StepRecord
from rpent.tools import ToolResult, tool

Pixel = Annotated[list[int], Field(min_length=2, max_length=2)]
Step = Annotated[int, Field(ge=0)]
MIN_REGION_POINTS = 8


@dataclass
class CameraCapture:
    """In-memory RGB-D capture with lazy lossless PNG encoding."""

    raw_rgb: np.ndarray
    annotated_rgb: np.ndarray
    world: np.ndarray
    calibration: dict[str, Any]
    _raw_png: bytes | None = None
    _annotated_png: bytes | None = None

    @property
    def shape(self) -> tuple[int, int]:
        return (int(self.annotated_rgb.shape[0]), int(self.annotated_rgb.shape[1]))

    def raw_png(self) -> bytes:
        if self._raw_png is None:
            self._raw_png = encode_png(self.raw_rgb)
        return self._raw_png

    def annotated_png(self, display_size: int | None = None) -> bytes:
        if display_size is None:
            if self._annotated_png is None:
                self._annotated_png = encode_png(self.annotated_rgb)
            return self._annotated_png
        return encode_png(self.annotated_rgb, size=display_size)


@dataclass
class StepSnapshot:
    """Immutable per-step state and camera captures stored in RAM."""

    step_idx: int
    state: dict[str, Any]
    terminated: bool
    truncated: bool
    frame_version: int
    cameras: dict[str, CameraCapture] = field(default_factory=dict)


class RecordedPerception:
    """Base class for all simulator adapters with lazy in-memory RGB-D queries."""

    cameras: tuple[str, ...]
    display_size: int | None = None
    coordinates: geometry.Coordinates

    def _init_perception(
        self,
        *,
        env: Any,
        sam3_client: Any,
        check_cancelled: Callable[[], None],
        state: EnvState | None = None,
        record_artifacts: bool = False,
        max_episode_steps: int = 10000,
        segment_min_score: float = 0.2,
    ) -> None:
        self.env = env
        self._sam3 = sam3_client
        self._check_cancelled = check_cancelled
        self._state = state
        self.record_artifacts = record_artifacts and state is not None
        self._max_episode_steps = max_episode_steps
        self._segment_min_score = segment_min_score
        self.env_steps = 0
        self._success = False
        self.infrastructure_error: dict[str, Any] | None = None
        self.sam3_service_errors: list[dict[str, Any]] = []
        self.frames: list[np.ndarray] = []
        self._steps: list[StepSnapshot] = []
        self._masks: dict[str, tuple[int, str, np.ndarray]] = {}
        self._mask_ids = itertools.count(1)
        self._sam_calls = itertools.count(1)

    @property
    def latest_step(self) -> int:
        return len(self._steps) - 1

    @property
    def latest_snapshot(self) -> StepSnapshot:
        return self._steps[-1]

    @property
    def ended(self) -> bool:
        return self.native_success() or self.env_steps >= self._max_episode_steps

    def native_success(self) -> bool:
        raise NotImplementedError

    def _ended_reason(self) -> str:
        return "terminated" if self.native_success() else "truncated"

    def robot_state(self) -> dict[str, Any]:
        raise NotImplementedError

    def capture(self, camera: str) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        raise NotImplementedError

    def capture_all(self) -> dict[str, tuple[np.ndarray, np.ndarray, dict[str, Any]]]:
        return {camera: self.capture(camera) for camera in self.cameras}

    def annotate(
        self,
        image: np.ndarray,
        calibration: dict[str, Any],
        state: dict[str, Any] | None = None,
    ) -> np.ndarray:
        return image

    def _coordinates_metadata(self) -> dict[str, Any]:
        return self.coordinates.metadata()

    def _save_artifact(self, name: str, value: Any, step: int | None) -> None:
        if self.record_artifacts and self._state is not None:
            self._state.save(name, value, step=step)

    def record_observation(
        self,
        *,
        command: dict[str, Any] | None = None,
        result: dict[str, Any] | None = None,
        elapsed_s: float | None = None,
    ) -> StepRecord | None:
        """Capture robot state and RGB-D views; persist to disk only if requested."""
        state = self.robot_state()
        terminated = self.native_success()
        truncated = (not terminated) and bool(
            state.get("truncated", self.env_steps >= self._max_episode_steps)
        )
        step_idx = len(self._steps)
        frame_version = getattr(getattr(self, "coordinates", None), "version", 0)
        captures: dict[str, CameraCapture] = {}
        rendered = self.capture_all()
        for camera in self.cameras:
            raw_rgb, world, calibration = rendered[camera]
            raw_arr = np.asarray(raw_rgb, dtype=np.uint8)
            annotated = np.asarray(
                self.annotate(raw_arr, calibration, state), dtype=np.uint8
            )
            captures[camera] = CameraCapture(
                raw_rgb=raw_arr,
                annotated_rgb=annotated,
                world=np.asarray(world, dtype=np.float32),
                calibration=calibration,
            )
            if self.record_artifacts and camera == self.cameras[0]:
                self.frames.append(annotated)

        self._steps.append(
            StepSnapshot(
                step_idx=step_idx,
                state=state,
                terminated=terminated,
                truncated=truncated,
                frame_version=frame_version,
                cameras=captures,
            )
        )
        if not self.record_artifacts or self._state is None:
            return None

        with self._state.record_step(
            state=state,
            terminated=terminated,
            truncated=truncated,
            command=command,
            result=result,
            elapsed_s=elapsed_s,
            extras={"env_steps": self.env_steps},
        ) as step:
            self._state.save(
                "step_coordinates.json", self._coordinates_metadata(), step=step
            )
            for camera, cap in captures.items():
                if not np.array_equal(cap.raw_rgb, cap.annotated_rgb):
                    self._state.save(f"{camera}_raw.png", cap.raw_rgb, step=step)
                self._state.save(f"{camera}.png", cap.annotated_rgb, step=step)
                self._state.save(f"{camera}_world.npz", cap.world, step=step)
                self._state.save(
                    f"{camera}_calibration.json", cap.calibration, step=step
                )
        return self._state.get(step)

    def _model_points(self, native: np.ndarray) -> np.ndarray:
        return self.coordinates.points(native)

    def _surface_sample(
        self, samples: np.ndarray, z_range_m: list[float] | None = None
    ) -> tuple[np.ndarray, np.ndarray | None]:
        native = geometry.valid_points(samples, zero_is_invalid=False)
        points = self._model_points(native)
        if z_range_m is not None:
            low, high = z_range_m
            keep = (low <= points[:, 2]) & (points[:, 2] <= high)
            points = points[keep]
        center = np.median(points, axis=0) if len(points) else None
        return points, center

    def _frame_metadata(self, step: int, camera: str) -> dict[str, Any]:
        snap = self._steps[step]
        return {
            "camera": camera,
            "step": step,
            "observation_step": step,
            "position_unit": "m",
            "capture_frame_version": snap.frame_version,
            "coordinate_step": self.latest_step,
            "frame_version": getattr(getattr(self, "coordinates", None), "version", 0),
            "historical_geometry": step != self.latest_step,
        }

    def _query_view(
        self, camera: str | None, step: int | None
    ) -> tuple[
        int, str, CameraCapture, tuple[int, int], tuple[int, int], dict[str, Any]
    ]:
        camera = self.cameras[0] if camera is None else camera
        if camera not in self.cameras:
            raise ValueError(f"available cameras: {self.cameras}")
        idx = self.latest_step if step is None else step
        if not 0 <= idx < len(self._steps):
            raise ValueError(f"unknown step {step}")
        cap = self._steps[idx].cameras[camera]
        native_shape = cap.shape
        displayed = (
            native_shape
            if self.display_size is None
            else (self.display_size, self.display_size)
        )
        metadata = self._frame_metadata(idx, camera)
        metadata["size"] = list(displayed)
        return idx, camera, cap, native_shape, displayed, metadata

    def _audit_sam(
        self,
        *,
        step: int,
        camera: str,
        query: dict[str, Any],
        min_score: float,
        result: Any = None,
        error: Exception | None = None,
    ) -> None:
        if error is not None:
            self.sam3_service_errors.append(
                {
                    "step": step,
                    "camera": camera,
                    "error": {"type": type(error).__name__, "message": str(error)},
                }
            )
        if not self.record_artifacts:
            return
        name = f"sam3_detection_{next(self._sam_calls)}"
        data: dict[str, Any] = {
            "step": step,
            "camera": camera,
            "query": query,
            "min_score": min_score,
            "category": "service_error" if error is not None else "not_found",
            "found": False if result is None else result.found,
            "score": None if result is None else result.score,
            "box_xyxy": None if result is None else result.box,
            "reason": None if result is None else result.reason,
        }
        if error is not None:
            data["error"] = {"type": type(error).__name__, "message": str(error)}
        if result is not None and result.mask is not None:
            mask_name = f"{name}-mask.png"
            self._save_artifact(mask_name, result.mask.astype(np.uint8) * 255, step)
            data.update(mask=mask_name, mask_shape=list(result.mask.shape))
        if result is not None and result.found:
            data["category"] = "detected"
        self._save_artifact(f"{name}.json", data, step)

    @tool(readonly=True)
    def view_env(self, camera: str = "all") -> ToolResult:
        """Read the latest recorded RGB and image dimensions. Request a named camera or all. Each view uses its own height and width; use the returned step and camera for perception."""
        names = self.cameras if camera == "all" else (camera,)
        step = self.latest_step
        images, summaries = [], []
        for name in names:
            try:
                _, _, cap, _, _, metadata = self._query_view(name, step)
            except ValueError as exc:
                return _error(str(exc))
            images.append(cap.annotated_png(self.display_size))
            summaries.append({"camera": name, "size": metadata["size"]})
        return ToolResult(
            data={**self._frame_metadata(step, camera), "images": summaries},
            images=images,
        )

    @tool(readonly=True)
    def segment(
        self,
        prompt: str | None = None,
        point: Pixel | None = None,
        camera: str | None = None,
        step: Step | None = None,
    ) -> ToolResult:
        """Segment one object with SAM3 in a recorded image and measure its visible surface. Provide exactly one non-empty prompt or point, not both. Returns mask_id, box [row_min, col_min, row_max, col_max] (max exclusive), and geometry in the current result frame of the visible points: world_xyz (per-axis median), top_z_m (95th-percentile z), size_xyz_m (5th to 95th percentile extent), and axis_yaw (major xy direction in [-pi/2, pi/2), null when the surface is not elongated).

        Args:
            prompt: Non-empty text naming one visible object; omit point. Example: {"prompt":"red cup"}.
            point: One pixel [row, col] on the target in the named image; omit prompt and provide that image's step.
            camera: Named camera; defaults to the overview.
            step: Recorded observation step matching the image; required for point. Text queries default to latest when omitted.
        """
        if (prompt is None) == (point is None):
            return _error(
                "Provide exactly one non-empty prompt or one point [row,col], not both. Correct the query and try again."
            )
        if prompt is not None and not prompt.strip():
            return _error(
                "Provide a non-empty object description in prompt, or a point [row,col]."
            )
        if point is not None and step is None:
            return _error("point requires step")
        try:
            index, camera, cap, native_shape, displayed, metadata = self._query_view(
                camera, step
            )
        except ValueError as exc:
            return _error(str(exc))
        if point is not None:
            if not _inside(point, displayed):
                return _error("point outside the image", **metadata)
            point = map_pixel(point, displayed, native_shape)
        world = cap.world
        if world.shape != (*native_shape, 3):
            raise ValueError("recorded image and depth dimensions differ")
        try:
            segmentation = self._sam3.segment(
                cap.raw_png(),
                text_prompt=prompt,
                point=point,
                min_score=self._segment_min_score,
            )
        except Exception as exc:
            self._audit_sam(
                step=index,
                camera=camera,
                query={"prompt": prompt, "point": point},
                min_score=self._segment_min_score,
                error=exc,
            )
            return _error(f"segmentation_failed: {exc}", **metadata)
        self._audit_sam(
            step=index,
            camera=camera,
            query={"prompt": prompt, "point": point},
            min_score=self._segment_min_score,
            result=segmentation,
        )
        if not segmentation.found:
            return ToolResult(
                data={
                    **metadata,
                    "code": "not_found",
                    "message": "Try a clearer object description, a point on the object, or another camera.",
                }
            )
        mask = segmentation.mask
        if mask is None or mask.shape != native_shape or not mask.any():
            return _error(
                "SAM3 mask dimensions differ from the selected image", **metadata
            )
        mask_id = f"m_{index}_{camera}_{next(self._mask_ids)}"
        self._masks[mask_id] = (index, camera, mask)
        data = dict(metadata)
        data["mask_id"] = mask_id
        native_box = (
            geometry.box_from_xyxy(segmentation.box, mask.shape)
            if segmentation.box is not None
            else geometry.box_from_mask(mask)
        )
        data["box"] = (
            None if native_box is None else map_box(native_box, native_shape, displayed)
        )
        points, center = self._surface_sample(world[mask])
        surface = geometry.surface_geometry(points, center=center)
        if surface is None:
            data.update(
                dict.fromkeys(("world_xyz", "top_z_m", "size_xyz_m", "axis_yaw")),
                geometry_unavailable="insufficient depth points",
            )
        else:
            data.update(surface)
        if self.record_artifacts:
            self._save_artifact(
                f"{mask_id}.png", _overlay(cap.annotated_rgb, mask), index
            )
        return ToolResult(data=data)

    @tool(readonly=True)
    def back_project(
        self,
        pixel: Pixel | None = None,
        box: Annotated[list[int], Field(min_length=4, max_length=4)] | None = None,
        mask_id: str | None = None,
        camera: str | None = None,
        step: Step | None = None,
        z_range_m: Annotated[list[float], Field(min_length=2, max_length=2)]
        | None = None,
    ) -> ToolResult:
        """Return the current-frame position of a pixel's surface point, or the per-axis median of the valid points in a box or segment mask. Box regions include any background they cover.

        Args:
            pixel: [row, col]; requires step.
            box: [row_min, col_min, row_max, col_max], max exclusive; requires step.
            mask_id: A mask_id from segment; it fixes the step and camera.
            camera: Named camera for pixel and box; defaults to the overview.
            step: Observation step for pixel and box.
            z_range_m: For box or mask, keep points with result-frame z in [low, high].
        """
        if sum(value is not None for value in (pixel, box, mask_id)) != 1:
            return _error("provide exactly one of pixel, box, or mask_id")
        mask = None
        if mask_id is not None:
            if camera is not None or step is not None:
                return _error("mask_id fixes the step and camera")
            if mask_id not in self._masks:
                return _error(f"unknown mask_id {mask_id}")
            step, camera, mask = self._masks[mask_id]
        elif step is None:
            return _error("pixel and box require step")
        try:
            step, camera, cap, native_shape, displayed, metadata = self._query_view(
                camera, step
            )
        except ValueError as exc:
            return _error(str(exc))
        world = cap.world
        if world.shape != (*native_shape, 3):
            raise ValueError("recorded image and depth dimensions differ")

        if pixel is not None:
            if z_range_m is not None:
                return _error("z_range_m applies to box and mask")
            if not _inside(pixel, displayed):
                return _error("pixel outside the image", **metadata)
            row, col = map_pixel(pixel, displayed, native_shape)
            samples = world[row, col]
        elif box is not None:
            if not (
                0 <= box[0] < box[2] <= displayed[0]
                and 0 <= box[1] < box[3] <= displayed[1]
            ):
                return _error("box outside the image or empty", **metadata)
            r0, c0, r1, c1 = map_box(box, displayed, native_shape)
            samples = world[r0:r1, c0:c1]
        else:
            samples = world[mask]

        points, center = self._surface_sample(samples, z_range_m)
        needed = 1 if pixel is not None else MIN_REGION_POINTS
        if center is None or len(points) < needed:
            return _error(f"insufficient depth: {len(points)} valid points", **metadata)
        return ToolResult(
            data={
                **metadata,
                "world_xyz": [round(float(v), 4) for v in center],
            }
        )


def _error(message: str, **metadata: Any) -> ToolResult:
    return ToolResult(data={**metadata, "error": message})


def _inside(pixel: list[int], shape: tuple[int, ...]) -> bool:
    return 0 <= pixel[0] < shape[0] and 0 <= pixel[1] < shape[1]


def _overlay(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    overlay = np.asarray(image)[..., :3].copy()
    overlay[mask] = (0.55 * overlay[mask] + [0.45 * 255, 0, 0]).astype(np.uint8)
    return overlay
