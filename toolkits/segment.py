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

"""Segment a recorded camera view and retain its aligned evidence."""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Annotated, Protocol
from uuid import uuid4

import numpy as np
from pydantic import Field, ValidationError

from rpent.session import EnvState
from rpent.tools import ToolResult, tool
from toolkits.observation import (
    GeometryObservation,
    ImageSelection,
    Index,
    ObservationProvider,
    Size,
)

if TYPE_CHECKING:
    from rpent.robots.components.sam3_client import Sam3Result


class Segmenter(Protocol):
    """Produce masks using text or a positive row/column point prompt."""

    def segment(
        self,
        image: bytes,
        *,
        text_prompt: str | None = None,
        point: list[int] | None = None,
        min_score: float = 0.2,
    ) -> Sam3Result:
        """Return a mask at the supplied RGB resolution, or no candidate."""
        ...


class SegmentTool:
    """Segment pinned RGB-D snapshots using the caller's model, calibration and artifact store."""

    def __init__(
        self,
        *,
        observations: ObservationProvider,
        state: EnvState,
        segmenter: Segmenter | None,
    ) -> None:
        self._observations = observations
        self._state = state
        self._namespace = uuid4().hex[:12]
        self._segmenter = segmenter
        self._lock = threading.Lock()
        self._index = 0

    @tool(
        readonly=True,
        json_schema_extra={
            "additionalProperties": False,
            "if": {"required": ["point"], "properties": {"point": {"type": "array"}}},
            "then": {
                "required": ["image_shape"],
                "properties": {
                    "image_shape": {"type": "array"},
                    "prompt": {"pattern": r"^\s*$"},
                },
            },
            "else": {
                "required": ["prompt"],
                "properties": {"prompt": {"pattern": r"\S"}},
            },
        },
    )
    def segment(
        self,
        camera: Annotated[str, Field(min_length=1, max_length=128)],
        prompt: Annotated[str, Field(max_length=1024)] = "",
        step: Annotated[int, Field(ge=-1)] = -1,
        point: Annotated[list[Index], Field(min_length=2, max_length=2)] | None = None,
        image_shape: Annotated[list[Size], Field(min_length=2, max_length=2)]
        | None = None,
        min_score: Annotated[float, Field(ge=0, le=1)] = 0.2,
    ) -> ToolResult:
        """Get a candidate mask, overlay, detection score and optional world median from a recorded camera frame.

        Supply exactly one prompt. Use text in prompt or a positive point as
        [row, col]. For a point, give the image_shape [height, width] of the
        image used to select it. Confirm the object in the overlay before
        using the candidate's coordinates.
        """
        try:
            prompt = prompt.strip()
            if bool(prompt) == (point is not None):
                raise ValueError("Supply exactly one of prompt or point.")
            if point is not None and image_shape is None:
                raise ValueError(
                    "Give the image_shape of the image where the point was selected."
                )
            observation = self._observations.get_observation(camera=camera, step=step)
            if observation.camera != camera or (step >= 0 and observation.step != step):
                raise ValueError("Provider returned a different camera or step.")
            if point is not None:
                selection = ImageSelection(image_shape=image_shape, points_rc=[point])
                pixels, _ = selection.native(observation)
                point = list(pixels[0])
            return self._segment_observation(
                observation, prompt=prompt or None, point=point, min_score=min_score
            )
        except (ValueError, LookupError, OSError, RuntimeError) as exc:
            return ToolResult(
                data={
                    "status": "error",
                    "error": exc.errors(include_input=False)[0]["msg"]
                    if isinstance(exc, ValidationError)
                    else str(exc),
                    "camera": camera,
                    "step": step,
                }
            )

    def _segment_observation(
        self,
        observation: GeometryObservation,
        *,
        prompt: str | None,
        point: list[int] | None,
        min_score: float,
    ) -> ToolResult:
        """Segment the selected frame and save the result with its source step."""
        if self._segmenter is None:
            raise ValueError("segment requires an injected segmenter")
        result = self._segmenter.segment(
            observation.png_bytes(),
            text_prompt=prompt,
            point=point,
            min_score=min_score,
        )
        mask = np.asarray(result.mask) if result.found else None
        if mask is not None and (
            mask.dtype != np.bool_ or mask.shape != observation.rgb.shape[:2]
        ):
            raise ValueError("segmentation mask must be boolean and match RGB")
        with self._lock:
            index = self._index
            self._index += 1
        artifact_id = f"{self._namespace}_{index:03d}"
        step = observation.step
        state = self._state
        data = {
            "found": mask is not None,
            "camera": observation.camera,
            "source_step": step,
            "step": step,
            "world_frame_id": observation.world_frame_id,
            "mode": "text" if prompt else "point",
            "prompt": prompt,
            "point": point,
            "image_shape": list(observation.rgb.shape[:2]),
            "min_score": min_score,
            "score": None if result.score is None else float(result.score),
            "evidence_status": "identity_unchecked",
            "units": "m",
            "world_xyz": None,
            "mask_artifact": None,
        }
        images = []
        if mask is not None:
            if not mask.any():
                raise ValueError("Segmenter returned an empty foreground mask.")
            if point is not None and not mask[tuple(point)]:
                raise ValueError("Returned mask does not include the positive point.")
            valid = mask & observation.valid_mask
            n_valid = int(valid.sum())
            data.update(
                box=getattr(result, "box", None),
                n_valid=n_valid,
                definition="visible_mask_coordinate_median",
            )
            if n_valid >= 10:
                data["world_xyz"] = np.median(
                    observation.world_xyz[valid], axis=0
                ).tolist()
            else:
                data["world_error"] = "too few valid metric samples in mask"
            mask_name = f"segment_mask_{artifact_id}.npz"
            data["mask_artifact"] = state.save(mask_name, mask, step=step)
            if data["mask_artifact"] is None:
                raise RuntimeError("failed to persist segmentation mask")
            overlay_name = f"segment_overlay_{artifact_id}.png"
            data["overlay_artifact"] = state.save(
                overlay_name, observation.mask_overlay(mask), step=step
            )
            if data["overlay_artifact"] is None:
                raise RuntimeError("failed to persist segmentation overlay")
            images.append(state.load_bytes(overlay_name, step=step))
        else:
            data["reason"] = result.reason or "segmenter found no mask"
        name = f"segment_{artifact_id}.json"
        if state.save(name, data, step=step) is None:
            raise RuntimeError("failed to persist segmentation evidence")
        data["segment_artifact"] = name
        return ToolResult(data=data, images=images)
