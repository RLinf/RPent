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

"""Recorded RGB-D queries shared by simulator adapters."""

from __future__ import annotations

import io
from typing import Annotated, Any

import numpy as np
from PIL import Image
from pydantic import Field

from robots.rl import geometry
from robots.rl.images import map_box, map_pixel, resize_image
from rpent.session import StepRecord
from rpent.tools import ToolResult, tool

Pixel = Annotated[list[int], Field(min_length=2, max_length=2)]
Step = Annotated[int, Field(ge=0)]
MIN_REGION_POINTS = 8


class RecordedPerception:
    """Query the scene's immutable observations in the current coordinate frame."""

    cameras: tuple[str, ...]
    display_size: int | None = None

    def surface_points(self, native_points: np.ndarray) -> np.ndarray:
        """Filter invalid native depth, then convert before statistics or z filters."""
        return self._model_points(
            geometry.valid_points(
                native_points,
                zero_is_invalid=False,
            )
        )

    def _model_points(self, native: np.ndarray) -> np.ndarray:
        return self.coordinates.points(native)

    def _surface_center(self, native: np.ndarray, points: np.ndarray) -> np.ndarray:
        return np.median(points, axis=0)

    def _surface_sample(
        self, samples: np.ndarray, z_range_m: list[float] | None = None
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Select corresponding native and model-frame points before aggregation."""
        native = geometry.valid_points(
            samples,
            zero_is_invalid=False,
        )
        points = self._model_points(native)
        if z_range_m is not None:
            low, high = z_range_m
            keep = (low <= points[:, 2]) & (points[:, 2] <= high)
            native, points = native[keep], points[keep]
        center = self._surface_center(native, points) if len(points) else None
        return points, center

    def _audit_sam(
        self,
        *,
        step: int,
        camera: str,
        query,
        min_score: float,
        result=None,
        error=None,
    ) -> None:
        """Archive exact RPC detections independently of the model-visible result."""
        name = f"sam3_detection_{next(self._sam_calls)}"
        data = {
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
            self._save(mask_name, result.mask.astype(np.uint8) * 255, step)
            data.update(mask=mask_name, mask_shape=list(result.mask.shape))
        if result is not None and result.found:
            data["category"] = "detected"
        self._save(f"{name}.json", data, step)

    def images(self, step: int, cameras: list[str]) -> list[bytes]:
        """Return the recorded PNG bytes for ``cameras`` at ``step``."""
        return [
            self._state.load_bytes(f"{camera}.png", step=step) for camera in cameras
        ]

    def _frame_metadata(self, step: int, camera: str) -> dict[str, Any]:
        capture = self._state.get(step)
        capture_version = (
            self._state.load("step_coordinates.json", step=step)["frame_version"]
            if "step_coordinates.json" in capture.artifacts
            else 0
        )
        return {
            "camera": camera,
            "step": step,
            "observation_step": step,
            "position_unit": "m",
            "capture_frame_version": capture_version,
            "coordinate_step": self._state.latest_step,
            "frame_version": getattr(getattr(self, "coordinates", None), "version", 0),
            "historical_geometry": step != self._state.latest_step,
        }

    def _image(self, step: int, camera: str) -> tuple[bytes, tuple[int, int]]:
        data = self._state.load_bytes(f"{camera}.png", step=step)
        with Image.open(io.BytesIO(data)) as im:
            return data, (im.height, im.width)

    def _query_view(self, camera, step):
        camera = self.cameras[0] if camera is None else camera
        if camera not in self.cameras:
            raise ValueError(f"available cameras: {self.cameras}")
        record = self._record(step)
        if record is None:
            raise ValueError(f"unknown step {step}")
        rgb, native_shape = self._image(record.step_idx, camera)
        if self.display_size is None:
            displayed_shape = native_shape
        else:
            displayed_shape = (self.display_size, self.display_size)
        metadata = self._frame_metadata(record.step_idx, camera)
        metadata["size"] = list(displayed_shape)
        return record.step_idx, camera, rgb, native_shape, displayed_shape, metadata

    @staticmethod
    def _depth_aligned(world, shape):
        if world.shape != (*shape, 3):
            raise ValueError("recorded image and depth dimensions differ")
        return world

    @tool(readonly=True)
    def view_env(self, camera: str = "all") -> ToolResult:
        """Read the latest recorded RGB and image dimensions. Request a named camera or all. Each view uses its own height and width; use the returned step and camera for perception."""
        names = self.cameras if camera == "all" else (camera,)
        step = self._state.latest_step
        images, summaries = [], []
        for name in names:
            try:
                _, _, rgb, _, _, metadata = self._query_view(name, step)
            except ValueError as exc:
                return _error(str(exc))
            if self.display_size is not None:
                rgb = resize_image(rgb, self.display_size)
            images.append(rgb)
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
            point: One pixel [row, col] on the target in the named image; omit prompt and provide that image\'s step.
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
            index, camera, rgb, native_shape, displayed, metadata = self._query_view(
                camera, step
            )
        except ValueError as exc:
            return _error(str(exc))
        if point is not None:
            if not _inside(point, displayed):
                return _error("point outside the image", **metadata)
            point = map_pixel(point, displayed, native_shape)
        world = self._depth_aligned(
            self._state.load(f"{camera}_world.npz", step=index), native_shape
        )
        if f"{camera}_raw.png" in self._state.get(index).artifacts:
            rgb = self._state.load_bytes(f"{camera}_raw.png", step=index)
        try:
            segmentation = self._sam3.segment(
                rgb,
                text_prompt=prompt,
                point=point,
                min_score=self._config.segment_min_score,
            )
        except Exception as exc:
            self._audit_sam(
                step=index,
                camera=camera,
                query={"prompt": prompt, "point": point},
                min_score=self._config.segment_min_score,
                error=exc,
            )
            return _error(f"segmentation_failed: {exc}", **metadata)
        self._audit_sam(
            step=index,
            camera=camera,
            query={"prompt": prompt, "point": point},
            min_score=self._config.segment_min_score,
            result=segmentation,
        )
        data = metadata
        if not segmentation.found:
            return ToolResult(
                data={
                    **metadata,
                    "code": "not_found",
                    "message": "Try a clearer object description, a point on the object, or another camera.",
                },
            )

        mask = segmentation.mask
        if mask is None or mask.shape != native_shape or not mask.any():
            return _error(
                "SAM3 mask dimensions differ from the selected image", **metadata
            )
        mask_id = f"m_{index}_{camera}_{next(self._mask_ids)}"
        self._masks[mask_id] = (index, camera, mask)
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
        image = self._state.load(f"{camera}.png", step=index)
        self._state.save(f"{mask_id}.png", _overlay(image, mask), step=index)
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
            step, camera, _, native_shape, displayed, metadata = self._query_view(
                camera, step
            )
        except ValueError as exc:
            return _error(str(exc))
        world = self._depth_aligned(
            self._state.load(f"{camera}_world.npz", step=step), native_shape
        )

        if pixel is not None:
            if z_range_m is not None:
                return _error("z_range_m applies to box and mask")
            if not _inside(pixel, displayed):
                return _error("pixel outside the image", **metadata)
            row, col = map_pixel(pixel, displayed, native_shape)
            samples = world[row, col]
        else:
            if box is not None:
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

    def _record(self, step: int | None) -> StepRecord | None:
        try:
            return self._state.get(-1 if step is None else step)
        except KeyError:
            return None


def _error(message: str, **metadata) -> ToolResult:
    return ToolResult(data={**metadata, "error": message})


def _inside(pixel: list[int], shape: tuple[int, ...]) -> bool:
    return 0 <= pixel[0] < shape[0] and 0 <= pixel[1] < shape[1]


def _overlay(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Tint the mask red for log inspection."""
    overlay = np.asarray(image)[..., :3].copy()
    overlay[mask] = (0.55 * overlay[mask] + [0.45 * 255, 0, 0]).astype(np.uint8)
    return overlay
