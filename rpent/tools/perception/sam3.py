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

from collections.abc import Sequence
from typing import TYPE_CHECKING, Annotated, Any, Protocol

import numpy as np
from pydantic import Field

from rpent.tools.base import ToolResult, tool
from rpent.tools.perception.perception_tool import CameraView, PerceptionToolBase

if TYPE_CHECKING:
    from rpent.session import EnvState


class Segmenter(Protocol):
    """Segmentation service used by :class:`Sam3Tools`, such as ``Sam3Client``."""

    def segment(
        self,
        image: bytes,
        *,
        text_prompt: str | None = None,
        point: list[int] | None = None,
        min_score: float = 0.2,
    ) -> Any:
        """Segment ``image`` with exactly one text prompt or ``[row, col]`` point.

        The result exposes ``found``, ``score``, ``box``, ``mask`` (a boolean
        array), ``mask_shape``, and ``reason``.
        """
        ...


class Sam3Tools(PerceptionToolBase):
    """SAM3 ``segment`` over a robot's recorded camera views.

    Results and overlays are saved as ``segment_NN`` artifacts at the segmented
    step, numbered through the shared :class:`EnvState`.
    """

    missing_artifacts_error = "complete segment artifacts not found"

    def __init__(
        self,
        segmenter: Segmenter,
        *,
        state: EnvState,
        cameras: dict[str, Sequence[CameraView]],
        fallback: str,
    ) -> None:
        """Bind the segmentation service and the robot's camera configuration.

        Args:
            segmenter: Segmentation service; the caller owns its lifecycle.
            state: Run store used for both input and output artifacts.
            cameras: Camera names mapped to their views in preference order.
                The first camera is the default.
            fallback: Robot-specific guidance returned when segmentation fails.
        """
        if segmenter is None:
            raise ValueError("Sam3Tools requires a segmenter")
        super().__init__(state=state, cameras=cameras, fallback=fallback)
        self._segmenter = segmenter

    @tool(readonly=True)
    def segment(
        self,
        prompt: str = "",
        camera: str = "",
        step: Annotated[int, Field(json_schema_extra={"default": -1})] = -1,
        point: Annotated[list[int], Field(min_length=2, max_length=2)] | None = None,
        min_score: float = 0.2,
    ) -> ToolResult:
        """SAM3 visual segmentation over an existing run artifact. It never renders a new camera view. Provide exactly one text prompt or single positive point. A successful top-ranked mask is projected through the matching world map to produce world_xyz.

        Args:
            prompt: Object/text prompt to segment.
            camera: Artifact camera to use.
            step: Step to segment; -1 = latest.
            point: Optional single positive point as [row, col]. Mutually exclusive with prompt.
            min_score: Minimum accepted mask score (default 0.2).
        """
        state = self._state
        camera = camera or next(iter(self._cameras))
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
        view = self._select_view(record, camera)
        if isinstance(view, ToolResult):
            return view
        image_name = view.image

        try:
            data = self._segmenter.segment(
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
        segment_index = state.reserve_artifact_index("segment")
        segment_name = f"segment_{segment_index:02d}.json"
        overlay_name = f"segment_overlay_{segment_index:02d}.png"
        saved_overlay = None
        mask = data.mask
        if data.found and isinstance(mask, np.ndarray):
            world_result = view.locate_mask(mask, state, nn)
            image = state.load(image_name, step=nn)
            if image.ndim == 3 and image.shape[:2] == mask.shape:
                # Tint the mask red so the planner can check the segmented object.
                red = np.zeros(image.shape[2], dtype=np.float32)
                red[0] = 255
                overlay = image.copy()
                overlay[mask] = (
                    0.55 * overlay[mask].astype(np.float32) + 0.45 * red
                ).astype(np.uint8)
            else:
                overlay = None
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
