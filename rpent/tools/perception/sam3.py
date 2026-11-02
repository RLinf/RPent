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

"""SAM3 segmentation and world localization over recorded observations."""

from __future__ import annotations

import threading
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Annotated, Any, Literal

import numpy as np
from pydantic import Field

from rpent.tools.base import Tool, ToolResult, tool

if TYPE_CHECKING:
    from rpent.robots.components.sam3_client import Sam3Client
    from rpent.session import EnvState, StepRecord


class Sam3Tools:
    """SAM3 tools over a robot's recorded RGB and world-map artifacts.

    A robot opts in by building an instance while its toolkit initializes and
    registering it with ``iter_tools``. Keep one instance per toolkit so
    concurrent calls share a unique artifact index. World maps must be
    pixel-aligned with their RGB images and contain world XYZ in meters.
    """

    def __init__(
        self,
        client: Sam3Client,
        *,
        state: EnvState,
        cameras: Mapping[str, Sequence[tuple[str, str]]],
        fallback: str,
    ) -> None:
        """Bind the SAM3 client, artifact store, and robot camera configuration.

        Args:
            client: SAM3 inference client; the caller owns its service lifecycle.
            state: Run store used for both input and output artifacts.
            cameras: Camera names mapped to ordered (RGB, world-map) artifact
                pairs, preferred pair first. The first camera is the default.
            fallback: Robot-specific guidance returned when segmentation fails.
        """
        self._client = client
        self._state = state
        self._cameras = cameras
        self._fallback = fallback
        self._segment_index_lock = threading.Lock()
        self._segment_index = 0
        self.segment = self._declare_segment()

    def _declare_segment(self) -> Tool:
        # The camera enum comes from the robot, so each instance declares its own.
        default_camera = next(iter(self._cameras))
        Camera = Annotated[
            Literal[tuple(self._cameras)],
            Field(description=f"Artifact camera to use (default {default_camera})."),
        ]

        @tool(readonly=True)
        def segment(
            prompt: str = "",
            camera: Camera = default_camera,
            step: Annotated[int, Field(json_schema_extra={"default": -1})] = -1,
            point: Annotated[list[int], Field(min_length=2, max_length=2)]
            | None = None,
            min_score: float = 0.2,
        ) -> ToolResult:
            """SAM3 visual segmentation over an existing run artifact. It never renders a new camera view. Provide exactly one text prompt or single positive point. A successful top-ranked mask is projected through the matching world map to produce world_xyz.

            Args:
                prompt: Object/text prompt to segment.
                step: Step to segment; -1 = latest.
                point: Optional single positive point as [row, col]. Mutually exclusive with prompt.
                min_score: Minimum accepted mask score (default 0.2).
            """
            return self._segment(
                prompt=prompt,
                camera=camera,
                step=step,
                point=point,
                min_score=min_score,
            )

        return segment

    def _segment(
        self,
        *,
        prompt: str,
        camera: str,
        step: int,
        point: list[int] | None,
        min_score: float,
    ) -> ToolResult:
        state = self._state
        images: list[bytes] = []
        try:
            record = state.get(step)
        except Exception as exc:
            return ToolResult(data={"error": f"state step not available: {exc}"})
        nn = record.step_idx

        prompt = prompt.strip()
        has_prompt = bool(prompt)
        has_point = point is not None
        if has_prompt == has_point:
            return ToolResult(
                data={"error": "segment needs exactly one of prompt or point"}
            )
        artifact_pair = _select_artifacts(state, record, self._cameras[camera])
        if artifact_pair is None:
            return ToolResult(
                data={
                    "error": "complete segment artifacts not found",
                    "step": nn,
                    "camera": camera,
                    "checked_artifacts": [
                        name for pair in self._cameras[camera] for name in pair
                    ],
                }
            )
        image_name, world_name = artifact_pair

        try:
            data = self._client.segment(
                state.load_bytes(image_name, step=nn),
                text_prompt=prompt if has_prompt else None,
                point=point,
                min_score=min_score,
            )
        except ValueError as e:
            return ToolResult(
                data={
                    "error": str(e),
                    "step": nn,
                    "camera": camera,
                    "image_artifact": image_name,
                }
            )
        except Exception as e:
            return ToolResult(
                data={
                    "error": f"segmentation service call failed: {e}",
                    "step": nn,
                    "camera": camera,
                    "image_artifact": image_name,
                    "fallback": self._fallback,
                }
            )

        # Reserve an index before writing; failed saves leave gaps, never reuse.
        with self._segment_index_lock:
            segment_index = self._segment_index
            self._segment_index += 1
        segment_name = f"segment_{segment_index:02d}.json"
        overlay_name = f"segment_overlay_{segment_index:02d}.png"
        saved_overlay = None
        mask = data.mask
        if data.found and isinstance(mask, np.ndarray):
            try:
                world_map = state.load(world_name, step=nn)
            except Exception as exc:
                world_result = {
                    "world_xyz": None,
                    "world_error": f"world map artifact not available: {exc}",
                    "expected_world_artifact": world_name,
                }
            else:
                world_result = _mask_to_world(mask, world_map)
                world_result["world_artifact"] = world_name
            overlay = _make_segment_overlay(state.load(image_name, step=nn), mask)
            if overlay is not None and state.save(
                overlay_name,
                overlay,
                step=nn,
            ):
                saved_overlay = overlay_name
        else:
            world_result = {
                "world_xyz": None,
                "world_error": data.reason or "segmentation did not find a mask",
            }

        segment_blob = {
            "found": data.found,
            "mode": "text" if has_prompt else "point",
            "camera": camera,
            "source_step": nn,
            "segment_index": segment_index,
            "image_artifact": image_name,
            "min_score": min_score,
            "score": round(float(data.score), 3) if data.score is not None else None,
            "box": data.box,
            "mask_shape": list(data.mask_shape) if data.mask_shape else None,
        }
        if has_prompt:
            segment_blob["prompt"] = prompt
        else:
            segment_blob["point"] = point
        if not data.found:
            segment_blob["error"] = data.reason or "SAM3 found no mask"
        segment_blob.update(world_result)
        saved_segment = state.save(
            segment_name,
            segment_blob,
            step=nn,
        )

        result = {
            "found": data.found,
            "step": nn,
            "camera": camera,
            "image_artifact": image_name,
            "score": segment_blob["score"],
            "box": segment_blob["box"],
            "world_xyz": segment_blob["world_xyz"],
            "world_error": segment_blob.get("world_error"),
        }
        if saved_segment is None:
            result["error"] = f"failed to persist segment artifact {segment_name}"
            result["code"] = "segment_artifact_save_failed"
            result["attempted_segment_artifact"] = segment_name
            if "error" in segment_blob:
                result["segmentation_error"] = segment_blob["error"]
            result["fallback"] = self._fallback
        else:
            result["segment_artifact"] = saved_segment
        if saved_segment is not None and "error" in segment_blob:
            result["error"] = segment_blob["error"]
            result["fallback"] = self._fallback
        if saved_overlay is not None:
            result["overlay_artifact"] = saved_overlay
            images.append(state.load_bytes(saved_overlay, step=nn))
        return ToolResult(data=result, images=images)


def _select_artifacts(
    state: EnvState,
    record: StepRecord,
    pairs: Sequence[tuple[str, str]],
) -> tuple[str, str] | None:
    for image_name, world_name in pairs:
        if (
            image_name in record.artifacts
            and world_name in record.artifacts
            and state.exists(image_name, step=record.step_idx)
            and state.exists(world_name, step=record.step_idx)
        ):
            return image_name, world_name
    return None


def _mask_to_world(
    mask: np.ndarray, world_map: np.ndarray, min_valid: int = 10
) -> dict[str, Any]:
    if world_map.ndim != 3 or world_map.shape[2] < 3:
        return {
            "world_xyz": None,
            "world_error": f"invalid world map shape: {tuple(world_map.shape)}",
            "n_pixels": int(mask.sum()),
            "n_valid": 0,
            "mask_resized_to_world_shape": False,
        }

    if mask.shape != world_map.shape[:2]:
        return {
            "world_xyz": None,
            "world_error": (
                f"mask/world shape mismatch: mask={tuple(mask.shape)}, "
                f"world={tuple(world_map.shape[:2])}"
            ),
            "n_pixels": int(mask.sum()),
            "n_valid": 0,
            "mask_resized_to_world_shape": False,
        }

    ys, xs = np.where(mask)
    if ys.size == 0:
        return {"world_xyz": None, "world_error": "empty mask"}

    pts = world_map[ys, xs].astype(np.float64)
    valid = np.isfinite(pts).all(axis=1) & (np.abs(pts).sum(axis=1) > 1e-6)
    pts = pts[valid]
    result = {
        "centroid_pixel": [
            int(round(float(np.median(xs)))),
            int(round(float(np.median(ys)))),
        ],
        "n_pixels": int(mask.sum()),
        "n_valid": int(pts.shape[0]),
        "mask_resized_to_world_shape": False,
    }
    if pts.shape[0] < min_valid:
        result.update(
            {
                "world_xyz": None,
                "world_error": f"too few valid depth pixels ({int(pts.shape[0])})",
            }
        )
        return result

    result["world_xyz"] = [
        round(float(np.median(pts[:, 0])), 4),
        round(float(np.median(pts[:, 1])), 4),
        round(float(np.median(pts[:, 2])), 4),
    ]
    return result


def _make_segment_overlay(
    image: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray | None:
    if image.ndim != 3 or image.shape[:2] != mask.shape:
        return None
    overlay = image.copy()
    red = np.zeros_like(overlay)
    red[..., 0] = 255
    overlay[mask] = (
        0.55 * overlay[mask].astype(np.float32) + 0.45 * red[mask].astype(np.float32)
    ).astype(np.uint8)
    return overlay
