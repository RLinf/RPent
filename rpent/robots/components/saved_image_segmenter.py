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

"""SAM3 segmentation of saved robot observations."""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np

from rpent.tools import ToolResult

if TYPE_CHECKING:
    from rpent.robots.components.sam3_client import Sam3Client
    from rpent.session import EnvState


def segment_prompt_error(prompt: str, point: list[int] | None) -> str | None:
    """Return the diagnostic shared by tools that accept one text or point prompt."""
    if bool(prompt.strip()) == (point is not None):
        return "segment needs exactly one of prompt or point"
    return None


class SavedImageSegmenter:
    """Run native SAM3 segmentation and save artifacts without advancing state.

    Each toolkit owns one instance. Calls may run concurrently; artifact indices
    are reserved after inference and are not reused when persistence fails.
    """

    def __init__(self, client: Sam3Client) -> None:
        self._client = client
        self._index_lock = threading.Lock()
        self._index = 0

    def segment(
        self,
        *,
        state: EnvState,
        step: int,
        camera: str,
        image_artifact: str,
        prompt: str = "",
        point: list[int] | None = None,
        min_score: float = 0.2,
        project_mask: Callable[[np.ndarray | None, str | None], dict[str, Any]],
        result_fields: tuple[str, ...] | None = None,
        fallback: str = "Use manual visual localization and back_project.",
    ) -> ToolResult:
        """Save the mask projection and overlay at their source observation step."""
        prompt = prompt.strip()
        source = {
            "step": step,
            "camera": camera,
            "image_artifact": image_artifact,
        }
        try:
            data = self._client.segment(
                state.load_bytes(image_artifact, step=step),
                text_prompt=prompt or None,
                point=point,
                min_score=min_score,
            )
        except ValueError as exc:
            return ToolResult(data={"error": str(exc), **source})
        except Exception as exc:
            return ToolResult(
                data={
                    "error": f"segmentation service call failed: {exc}",
                    **source,
                    "fallback": fallback,
                }
            )

        # Preserve the native reservation order; failed saves leave gaps.
        with self._index_lock:
            segment_index = self._index
            self._index += 1
        segment_name = f"segment_{segment_index:02d}.json"
        overlay_name = f"segment_overlay_{segment_index:02d}.png"
        mask = data.mask if data.found and isinstance(data.mask, np.ndarray) else None
        projection = project_mask(mask, data.reason)
        saved_overlay = None
        if mask is not None:
            overlay = _make_segment_overlay(state.load(image_artifact, step=step), mask)
            if overlay is not None and state.save(overlay_name, overlay, step=step):
                saved_overlay = overlay_name

        segment_blob = {
            "found": data.found,
            "mode": "text" if prompt else "point",
            "camera": camera,
            "source_step": step,
            "segment_index": segment_index,
            "image_artifact": image_artifact,
            "min_score": min_score,
            "score": round(float(data.score), 3) if data.score is not None else None,
            "box": data.box,
            "mask_shape": list(data.mask_shape) if data.mask_shape else None,
        }
        if prompt:
            segment_blob["prompt"] = prompt
        else:
            segment_blob["point"] = point
        if not data.found:
            segment_blob["error"] = data.reason or "SAM3 found no mask"
        segment_blob.update(projection)
        saved_segment = state.save(segment_name, segment_blob, step=step)

        result = {
            "found": data.found,
            **source,
            "score": segment_blob["score"],
            "box": data.box,
            **(
                projection
                if result_fields is None
                else {k: projection.get(k) for k in result_fields}
            ),
        }
        if saved_segment is None:
            result["error"] = f"failed to persist segment artifact {segment_name}"
            result["code"] = "segment_artifact_save_failed"
            result["attempted_segment_artifact"] = segment_name
            if "error" in segment_blob:
                result["segmentation_error"] = segment_blob["error"]
            result["fallback"] = fallback
        else:
            result["segment_artifact"] = saved_segment
            if "error" in segment_blob:
                result["error"] = segment_blob["error"]
                result["fallback"] = fallback
        images: list[bytes] = []
        if saved_overlay is not None:
            result["overlay_artifact"] = saved_overlay
            images.append(state.load_bytes(saved_overlay, step=step))
        return ToolResult(data=result, images=images)


def _make_segment_overlay(image: np.ndarray, mask: np.ndarray) -> np.ndarray | None:
    if image.ndim != 3 or image.shape[:2] != mask.shape:
        return None
    overlay = image.copy()
    red = np.zeros_like(overlay)
    red[..., 0] = 255
    overlay[mask] = (
        0.55 * overlay[mask].astype(np.float32) + 0.45 * red[mask].astype(np.float32)
    ).astype(np.uint8)
    return overlay
