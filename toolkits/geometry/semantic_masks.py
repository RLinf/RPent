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

"""Propose and select a whole-object mask in the source RGB image."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Protocol

import numpy as np

from toolkits.segment import Segmenter

if TYPE_CHECKING:
    from rpent.robots.components.molmo_client import MolmoResult
    from rpent.robots.components.sam3_client import Sam3Result
    from toolkits.geometry.local_point_agent import VisualGrounder
    from toolkits.observation import GeometryObservation

TEXT_MIN_SCORE = 0.4


class ObjectSegmenter(Segmenter, Protocol):
    """Provide ranked text candidates and positive-point segmentation on the same RGB."""

    def segment_candidates(
        self, image: bytes | np.ndarray, *, text_prompt: str, min_score: float
    ) -> list[Sam3Result]: ...


def _source_pixel(
    point: MolmoResult, shape: tuple[int, int]
) -> tuple[list[int] | None, str | None]:
    if not point.found or point.point_xy is None:
        return None, "no_single_molmo_point"
    h, w = shape
    col, row = point.point_xy
    if point.image_size != (w, h) or not (0 <= row < h and 0 <= col < w):
        return None, "molmo_point_does_not_match_source_image"
    return [min(h - 1, round(row)), min(w - 1, round(col))], None


def _valid_mask(result: Sam3Result, shape: tuple[int, int]) -> bool:
    mask = result.mask
    return (
        result.found
        and isinstance(mask, np.ndarray)
        and mask.dtype == np.bool_
        and mask.shape == shape
        and bool(mask.any())
    )


def select_object(
    obs: GeometryObservation,
    data: dict,
    segmenter: ObjectSegmenter,
    grounder: VisualGrounder,
    cancel: Callable[[], None],
    object_prompt: str,
) -> tuple[np.ndarray | None, str | None]:
    """Select a whole-object mask, consulting Molmo only for zero or multiple candidates."""
    image = obs.png_bytes()
    shape = obs.rgb.shape[:2]
    calls = data["calls"]
    calls["segment"] += 1
    candidates = segmenter.segment_candidates(
        image, text_prompt=object_prompt, min_score=TEXT_MIN_SCORE
    )
    if any(
        not _valid_mask(c, shape)
        or c.score is None
        or not TEXT_MIN_SCORE < c.score <= 1
        for c in candidates
    ):
        raise ValueError("SAM3 returned an invalid text candidate for the supplied RGB")
    attempt = {
        "stage": "object",
        "mode": "text",
        "prompt": object_prompt,
        "scores": [c.score for c in candidates],
        "candidate_count": len(candidates),
    }
    data["segment_attempts"].append(attempt)

    if len(candidates) == 1:
        attempt["selected_index"] = 0
        data["identity_evidence"]["object"] = {"method": "sam3_single_candidate"}
        return candidates[0].mask, None

    cancel()
    calls["grounding"] += 1
    point, warning = _source_pixel(grounder.ground(image, data["query"]), shape)
    if point is not None:
        attempt["point_rc"] = point

    if candidates:
        hits = []
        if point is not None:
            hits = [i for i, c in enumerate(candidates) if c.mask[tuple(point)]]
            attempt["point_hit_indices"] = hits
        if len(hits) == 1:
            index, method = hits[0], "molmo_point_in_sam3_mask"
        else:
            index = max(
                hits or range(len(candidates)), key=lambda i: candidates[i].score
            )
            method = "sam3_highest_score_at_point" if hits else "sam3_highest_score"
            warning = warning or (
                "molmo_point_hits_multiple_object_masks"
                if hits
                else "molmo_point_outside_object_masks"
            )
            attempt["reason"] = warning
            data.setdefault("warnings", []).append(
                {"stage": "object", "reason": warning, "source_camera": obs.camera}
            )
        attempt["selected_index"] = index
        mask = candidates[index].mask
    else:
        if point is None:
            attempt["reason"] = warning
            return None, warning
        cancel()
        calls["segment"] += 1
        result = segmenter.segment(image, point=point)
        attempt = {
            "stage": "object",
            "mode": "point",
            "point_rc": point,
            "score": result.score,
        }
        data["segment_attempts"].append(attempt)
        if not _valid_mask(result, shape):
            reason = result.reason or "invalid_sam3_point_mask"
            attempt["reason"] = reason
            return None, reason
        if not result.mask[tuple(point)]:
            attempt["reason"] = "sam3_point_mask_misses_point"
            return None, attempt["reason"]
        method, mask = "sam3_point_prompt", result.mask
    data["identity_evidence"]["object"] = {
        "method": method,
        "accepted": warning is None,
        "point_rc": point if warning is None else None,
    }
    return mask, None
