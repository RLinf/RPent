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

"""Transport-independent client for RPent's Grounding DINO + SAM 2 service."""

from __future__ import annotations

import base64
import io
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import imageio.v2 as imageio
import numpy as np

from rpent.utils.images import encode_rgb_png
from rpent.utils.rpc import RpcClient

DEFAULT_TIMEOUT_S = 120.0
#: Grounding DINO + SAM 2 accept at most this many prompts per request.
MAX_LABELS_PER_CALL = 16


def normalize_labels(
    labels: Sequence[str], *, max_labels: int = MAX_LABELS_PER_CALL
) -> list[str]:
    """Strip labels and reject empties, case-insensitive duplicates, and too many."""
    prompts = [str(label).strip() for label in labels]
    if not prompts or any(not label for label in prompts):
        raise ValueError("labels must be non-empty strings")
    folded = [label.casefold() for label in prompts]
    if len(folded) != len(set(folded)):
        raise ValueError("labels must be unique, ignoring case")
    if len(prompts) > max_labels:
        raise ValueError(f"at most {max_labels} labels per call")
    return prompts


@dataclass(frozen=True)
class Detection:
    """One open-vocabulary detection with its segmentation mask."""

    label: str
    score: float
    box: list[float]
    mask: np.ndarray


@dataclass(frozen=True)
class DetectSegmentResult:
    """All detections for one prompt against one frame."""

    found: bool
    detections: tuple[Detection, ...]
    image_shape: tuple[int, int]
    reason: str | None = None


class GdinoSam2Client:
    """Client wrapping the Grounding DINO + SAM 2 service over any RpcClient."""

    def __init__(
        self, client: RpcClient, *, timeout_s: float = DEFAULT_TIMEOUT_S
    ) -> None:
        self._client = client
        self._timeout_s = timeout_s

    def detect_segment(
        self,
        image: bytes | bytearray | memoryview | np.ndarray,
        *,
        labels: list[str],
        min_score: float = 0.3,
    ) -> DetectSegmentResult:
        """Detect and segment each label in one RGB frame.

        ``labels`` are open-vocabulary text prompts; every returned detection
        carries its box in [x1, y1, x2, y2] pixel coordinates and a boolean
        mask at the source image's resolution.
        """
        if isinstance(image, np.ndarray):
            image_bytes = encode_rgb_png(image)
        else:
            image_bytes = bytes(image)
        if not image_bytes:
            raise ValueError("image is empty")
        prompts = normalize_labels(labels)
        if not 0.0 <= float(min_score) <= 1.0:
            raise ValueError("min_score must be between 0 and 1")
        payload = self._client.call(
            "gdino_sam2.detect_segment",
            kwargs={
                "image_base64": base64.b64encode(image_bytes).decode("ascii"),
                "labels": prompts,
                "min_score": float(min_score),
            },
            timeout_s=self._timeout_s,
        )
        return self._decode_result(payload)

    @staticmethod
    def _decode_result(payload: Any) -> DetectSegmentResult:
        if not isinstance(payload, dict) or not isinstance(payload.get("found"), bool):
            raise RuntimeError(f"invalid detect_segment response: {payload!r}")
        shape = payload.get("image_shape")
        if (
            not isinstance(shape, list)
            or len(shape) != 2
            or not all(isinstance(value, int) and value > 0 for value in shape)
        ):
            raise RuntimeError(f"invalid image_shape in response: {shape!r}")
        image_shape = (shape[0], shape[1])
        if not payload["found"]:
            return DetectSegmentResult(
                found=False,
                detections=(),
                image_shape=image_shape,
                reason=str(payload.get("reason") or "no detection above min_score"),
            )
        rows = payload.get("detections")
        if not isinstance(rows, list) or not rows:
            raise RuntimeError("response marked found but had no detections")
        detections: list[Detection] = []
        for row in rows:
            if not isinstance(row, dict):
                raise RuntimeError(f"invalid detection entry: {row!r}")
            label = row.get("label")
            score = row.get("score")
            box = row.get("box")
            encoded_mask = row.get("mask_png_base64")
            if not isinstance(label, str) or not label:
                raise RuntimeError(f"invalid detection label: {row!r}")
            if not isinstance(score, (int, float)):
                raise RuntimeError(f"invalid detection score: {row!r}")
            if not isinstance(box, list) or len(box) != 4:
                raise RuntimeError(f"invalid detection box: {row!r}")
            try:
                raw = base64.b64decode(encoded_mask, validate=True)
                decoded = np.asarray(imageio.imread(io.BytesIO(raw)))
            except Exception as exc:
                raise RuntimeError(f"could not decode detection mask: {exc}") from exc
            if decoded.ndim == 3:
                decoded = decoded[..., 0]
            if decoded.shape != image_shape:
                raise RuntimeError(
                    f"mask shape {decoded.shape} does not match image {image_shape}"
                )
            detections.append(
                Detection(
                    label=label,
                    score=float(score),
                    box=[float(value) for value in box],
                    mask=decoded > 0,
                )
            )
        return DetectSegmentResult(
            found=True,
            detections=tuple(detections),
            image_shape=image_shape,
        )
