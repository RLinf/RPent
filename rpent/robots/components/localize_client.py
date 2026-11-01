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

"""Transport-independent client for RPent's map localization service."""

from __future__ import annotations

import base64
from typing import Any

import numpy as np

from rpent.tools.clearance import (
    DEFAULT_HALF_WIDTH,
    DEFAULT_LENGTH,
    DEFAULT_MAX_HEIGHT,
    DEFAULT_MIN_HEIGHT,
    DEFAULT_UNIT,
    ClearanceResult,
)
from rpent.tools.localize import (
    DEFAULT_MAX_RMS_PX,
    DEFAULT_MIN_INLIER_RATIO,
    DEFAULT_MIN_INLIERS,
    DEFAULT_REPROJECTION_ERROR,
    DEFAULT_TOP_K,
    LocalizationResult,
)
from rpent.utils.images import encode_rgb_png
from rpent.utils.rpc import RpcClient

#: Exhaustive localization of a large map can take tens of seconds on first use.
DEFAULT_TIMEOUT_S = 120.0


class LocalizeClient:
    """Client wrapping the map localization service over any :class:`RpcClient`."""

    def __init__(
        self, client: RpcClient, *, timeout_s: float = DEFAULT_TIMEOUT_S
    ) -> None:
        self._client = client
        self._timeout_s = timeout_s

    def localize(
        self,
        image: bytes | bytearray | memoryview | np.ndarray,
        *,
        prior_index: int | None = None,
        prior_pose: np.ndarray | None = None,
        exhaustive: bool = False,
        top_k: int = DEFAULT_TOP_K,
        exclude: list[int] | None = None,
        min_inliers: int = DEFAULT_MIN_INLIERS,
        min_inlier_ratio: float = DEFAULT_MIN_INLIER_RATIO,
        reprojection_error: float = DEFAULT_REPROJECTION_ERROR,
        max_rms_px: float = DEFAULT_MAX_RMS_PX,
    ) -> LocalizationResult:
        """Estimate one RGB frame's camera pose in the map frame.

        ``image`` is an RGB ``[H,W,3]`` array or encoded image bytes. The
        result's ``pose`` is a camera-to-world 4x4 matrix in the map's frame;
        ``status`` is ``"ok"``, ``"ambiguous"``, or ``"lost"``.
        """
        if isinstance(image, np.ndarray):
            image_bytes = encode_rgb_png(image)
        else:
            image_bytes = bytes(image)
        if not image_bytes:
            raise ValueError("image is empty")
        pose = None
        if prior_pose is not None:
            pose = np.asarray(prior_pose, dtype=np.float64)
            if pose.shape != (4, 4) or not np.isfinite(pose).all():
                raise ValueError("prior_pose must be a finite 4x4 matrix")
        payload = self._client.call(
            "localize.localize",
            kwargs={
                "image_base64": base64.b64encode(image_bytes).decode("ascii"),
                "prior_index": prior_index,
                "prior_pose": None if pose is None else pose.tolist(),
                "exhaustive": bool(exhaustive),
                "top_k": int(top_k),
                "exclude": None if exclude is None else [int(i) for i in exclude],
                "min_inliers": int(min_inliers),
                "min_inlier_ratio": float(min_inlier_ratio),
                "reprojection_error": float(reprojection_error),
                "max_rms_px": float(max_rms_px),
            },
            timeout_s=self._timeout_s,
        )
        try:
            return LocalizationResult.from_dict(payload)
        except ValueError as exc:
            raise RuntimeError(f"invalid localize response: {exc}") from exc

    def check_path(
        self,
        image: bytes | bytearray | memoryview | np.ndarray | None = None,
        *,
        pose: np.ndarray | None = None,
        prior_index: int | None = None,
        prior_pose: np.ndarray | None = None,
        exhaustive: bool = False,
        heading_deg: float = 0.0,
        length: float = DEFAULT_LENGTH,
        half_width: float = DEFAULT_HALF_WIDTH,
        min_height: float = DEFAULT_MIN_HEIGHT,
        max_height: float = DEFAULT_MAX_HEIGHT,
        unit: str = DEFAULT_UNIT,
    ) -> ClearanceResult:
        """Report whether a short straight drive from a camera pose is clear.

        Pass exactly one of ``image`` (an RGB ``[H,W,3]`` array or encoded
        image bytes to localize) or ``pose`` (a camera-to-world 4x4 matrix in
        the map frame). The corridor starts at the camera centre, follows the
        optical axis projected onto the floor, and turns by ``heading_deg``
        counter-clockwise. ``length``, ``half_width``, and the height band are
        measured in ``unit``: ``camera_heights``, ``map_units``, or ``m``.
        """
        if (image is None) == (pose is None):
            raise ValueError("pass exactly one of image or pose")
        camera_pose = None
        if pose is not None:
            if prior_index is not None or prior_pose is not None or exhaustive:
                raise ValueError(
                    "prior_index, prior_pose, and exhaustive apply only to image"
                )
            camera_pose = np.asarray(pose, dtype=np.float64)
            if camera_pose.shape != (4, 4) or not np.isfinite(camera_pose).all():
                raise ValueError("pose must be a finite 4x4 matrix")
        image_base64 = None
        if image is not None:
            image_bytes = (
                encode_rgb_png(image) if isinstance(image, np.ndarray) else bytes(image)
            )
            if not image_bytes:
                raise ValueError("image is empty")
            image_base64 = base64.b64encode(image_bytes).decode("ascii")
            if prior_pose is not None:
                prior_pose = np.asarray(prior_pose, dtype=np.float64)
                if prior_pose.shape != (4, 4) or not np.isfinite(prior_pose).all():
                    raise ValueError("prior_pose must be a finite 4x4 matrix")
        payload = self._client.call(
            "localize.check_path",
            kwargs={
                "image_base64": image_base64,
                "pose": None if camera_pose is None else camera_pose.tolist(),
                "prior_index": prior_index,
                "prior_pose": (
                    None if prior_pose is None else np.asarray(prior_pose).tolist()
                ),
                "exhaustive": bool(exhaustive),
                "heading_deg": float(heading_deg),
                "length": float(length),
                "half_width": float(half_width),
                "min_height": float(min_height),
                "max_height": float(max_height),
                "unit": unit,
            },
            timeout_s=self._timeout_s,
        )
        try:
            return ClearanceResult.from_dict(payload)
        except ValueError as exc:
            raise RuntimeError(f"invalid check_path response: {exc}") from exc

    def info(self) -> dict[str, Any]:
        """Return the map's frame, scale, and localization geometry."""
        payload = self._client.call("localize.info", timeout_s=self._timeout_s)
        if not isinstance(payload, dict):
            raise RuntimeError(f"invalid localize.info response: {payload!r}")
        return payload
