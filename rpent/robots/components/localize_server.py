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

"""RPC server owning a localizer for one built map.

Run manually with::

    python -m rpent.robots.components.localize_server \
        --map-path /path/to/maps/workspace \
        --transport http --host 127.0.0.1 --port 8117

The service exposes three RPC methods. ``localize.localize`` estimates one RGB
frame's camera pose in the map frame; ``localize.check_path`` estimates a pose
or takes one directly and reports whether a short straight drive is clear on
the map; ``localize.info`` reports the map's frame, scale, localization, and
clearance geometry. Feature extraction and matching run on the server's GPU,
and the DISK + LightGlue weights download on first use.
"""

from __future__ import annotations

import argparse
import base64
import io
import logging
import os
from typing import Any

import numpy as np
from PIL import Image

from rpent.tools.clearance import (
    DEFAULT_HALF_WIDTH,
    DEFAULT_LENGTH,
    DEFAULT_MAX_HEIGHT,
    DEFAULT_MIN_HEIGHT,
    DEFAULT_UNIT,
    ClearanceRequest,
    MapClearance,
)
from rpent.tools.localize import (
    DEFAULT_MAX_RMS_PX,
    DEFAULT_MIN_INLIER_RATIO,
    DEFAULT_MIN_INLIERS,
    DEFAULT_REPROJECTION_ERROR,
    DEFAULT_TOP_K,
    MapLocalizer,
)
from rpent.utils.logging import get_logger
from rpent.utils.rpc import RpcFacade

logger = get_logger("localize_server")


def _decode_image(image_base64: str) -> np.ndarray:
    """Decode one base64 PNG/JPEG frame as an RGB array."""
    if not isinstance(image_base64, str) or not image_base64:
        raise ValueError("image_base64 must be a non-empty string")
    try:
        raw = base64.b64decode(image_base64, validate=True)
        with Image.open(io.BytesIO(raw)) as image:
            rgb = np.asarray(image.convert("RGB"))
    except Exception as exc:
        raise ValueError(f"invalid image data: {exc}") from exc
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"image must be RGB [H,W,3], got {rgb.shape}")
    return rgb


def _parse_pose(name: str, value: Any) -> np.ndarray | None:
    if value is None:
        return None
    try:
        pose = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a 4x4 numeric matrix") from exc
    if pose.shape != (4, 4):
        raise ValueError(f"{name} must be 4x4, got {pose.shape}")
    if not np.isfinite(pose).all():
        raise ValueError(f"{name} must contain only finite values")
    return pose


def _positive_int(name: str, value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer, got {value!r}")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer, got {value!r}") from exc
    if result < 1:
        raise ValueError(f"{name} must be positive, got {result}")
    return result


def _frame_index(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer, got {value!r}")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must not be negative, got {result}")
    return result


def _positive_float(name: str, value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a number, got {value!r}") from exc
    if not np.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be positive, got {result}")
    return result


def _inlier_ratio(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"min_inlier_ratio must be a number, got {value!r}") from exc
    if not 0.0 <= result <= 1.0:
        raise ValueError("min_inlier_ratio must be between 0 and 1")
    return result


def _index_list(value: Any) -> list[int]:
    if value is None:
        return []
    if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)):
        raise ValueError(f"exclude must be a list of frame indices, got {value!r}")
    indices: list[int] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, np.integer)):
            raise ValueError(f"exclude entries must be integers, got {item!r}")
        indices.append(int(item))
    return indices


class LocalizeFacade(RpcFacade):
    """RPC server wrapping one :class:`~rpent.tools.localize.MapLocalizer`."""

    def __init__(
        self,
        map_path: str,
        *,
        device: str = "auto",
        localizer: MapLocalizer | None = None,
        camera_height_m: float | None = None,
    ) -> None:
        super().__init__()
        self._localizer = (
            localizer
            if localizer is not None
            else MapLocalizer(map_path, device=device)
        )
        self._camera_height_m = camera_height_m
        self._clearance: MapClearance | None = None
        self._register_rpc()

    def _register_rpc(self) -> None:
        self._rpc["localize.localize"] = self.localize
        self._rpc["localize.check_path"] = self.check_path
        self._rpc["localize.info"] = self.info
        self._readonly_methods.add("localize.info")

    @property
    def _clearance_checker(self) -> MapClearance:
        """Return the lazily built corridor checker for this map."""
        if self._clearance is None:
            self._clearance = MapClearance(
                self._localizer.map_path,
                metadata=getattr(self._localizer.source, "metadata", None),
                camera_height_m=self._camera_height_m,
            )
        return self._clearance

    def localize(
        self,
        *,
        image_base64: str,
        prior_index: int | None = None,
        prior_pose: Any = None,
        exhaustive: bool = False,
        top_k: int = DEFAULT_TOP_K,
        exclude: Any = None,
        min_inliers: int = DEFAULT_MIN_INLIERS,
        min_inlier_ratio: float = DEFAULT_MIN_INLIER_RATIO,
        reprojection_error: float = DEFAULT_REPROJECTION_ERROR,
        max_rms_px: float = DEFAULT_MAX_RMS_PX,
    ) -> dict[str, Any]:
        """Estimate one RGB frame's pose in the map frame.

        ``prior_index`` restricts candidates to a temporal window around the
        nearest model keyframe; ``prior_pose`` ranks keyframes by pose
        distance; ``exhaustive`` checks every model keyframe. Without any of
        them the coarse global descriptor shortlists ``top_k`` keyframes.
        ``exclude`` hides frame indices from candidate selection.
        """
        image = _decode_image(image_base64)
        if prior_index is not None:
            prior_index = _frame_index("prior_index", prior_index)
        pose = _parse_pose("prior_pose", prior_pose)
        if not isinstance(exhaustive, bool):
            raise ValueError(f"exhaustive must be a boolean, got {exhaustive!r}")
        excluded = _index_list(exclude)
        result = self._localizer.localize(
            image,
            prior_index=prior_index,
            prior_pose=pose,
            exhaustive=exhaustive,
            top_k=_positive_int("top_k", top_k),
            exclude=excluded,
            min_inliers=_positive_int("min_inliers", min_inliers),
            min_inlier_ratio=_inlier_ratio(min_inlier_ratio),
            reprojection_error=_positive_float(
                "reprojection_error", reprojection_error
            ),
            max_rms_px=_positive_float("max_rms_px", max_rms_px),
        )
        return result.to_dict()

    def check_path(
        self,
        *,
        image_base64: str | None = None,
        pose: Any = None,
        prior_index: int | None = None,
        prior_pose: Any = None,
        exhaustive: bool = False,
        heading_deg: float = 0.0,
        length: float = DEFAULT_LENGTH,
        half_width: float = DEFAULT_HALF_WIDTH,
        min_height: float = DEFAULT_MIN_HEIGHT,
        max_height: float = DEFAULT_MAX_HEIGHT,
        unit: str = DEFAULT_UNIT,
    ) -> dict[str, Any]:
        """Report whether a short straight drive from a camera pose is clear.

        Pass exactly one of ``image_base64`` (a live frame, localized with the
        same options as :meth:`localize`) or ``pose`` (a 4x4 model-frame
        camera-to-world matrix). The corridor starts at the camera centre,
        follows the optical axis projected onto the floor, and is rotated by
        ``heading_deg`` counter-clockwise. Distances in the request and result
        use ``unit``: ``camera_heights`` (default), ``map_units``, or ``m``
        when the map records a metric scale.
        """
        if (image_base64 is None) == (pose is None):
            raise ValueError("pass exactly one of image_base64 or pose")
        request = ClearanceRequest(
            heading_deg=heading_deg,
            length=length,
            half_width=half_width,
            min_height=min_height,
            max_height=max_height,
            unit=unit,
        )
        keyframe: int | None = None
        localization: dict[str, Any] | None = None
        camera_pose: np.ndarray | None
        if image_base64 is None:
            if prior_index is not None or prior_pose is not None or exhaustive:
                raise ValueError(
                    "prior_index, prior_pose, and exhaustive apply only to image_base64"
                )
            camera_pose = _parse_pose("pose", pose)
        else:
            localized = self.localize(
                image_base64=image_base64,
                prior_index=prior_index,
                prior_pose=prior_pose,
                exhaustive=exhaustive,
            )
            localization = {
                name: localized[name]
                for name in (
                    "status",
                    "keyframe",
                    "inliers",
                    "inlier_ratio",
                    "reprojection_rms_px",
                    "reason",
                )
            }
            keyframe = localized["keyframe"]
            # A failed localization leaves no pose; the checker reports that as
            # unlocalized instead of guessing where the camera is.
            raw_pose = localized["pose"]
            camera_pose = (
                None if raw_pose is None else np.asarray(raw_pose, dtype=np.float64)
            )
        result = self._clearance_checker.check(
            camera_pose,
            request,
            keyframe=keyframe,
            localization=localization,
        )
        return result.to_dict()

    def info(self) -> dict[str, Any]:
        """Return the map's frame, scale, localization, and clearance geometry."""
        localizer = self._localizer
        try:
            clearance = self._clearance_checker.info()
        except Exception as exc:
            # Localization still works when the map lacks the artifacts a
            # clearance check needs; report why instead of failing the call.
            logger.warning(
                "clearance is unavailable for %s: %s", localizer.map_path, exc
            )
            clearance = {
                "available": False,
                "status": "unreadable",
                "reason": str(exc),
            }
        return {
            "map_path": localizer.map_path,
            "frame": localizer.frame,
            "scale": dict(localizer.scale),
            "frames": localizer.source.frame_count,
            "model_keyframes": len(localizer.model_indices),
            "intrinsic": localizer.intrinsic.tolist(),
            "clearance": clearance,
        }


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RPent map localization server")
    parser.add_argument("--map-path", required=True, help="built map directory")
    parser.add_argument("--transport", choices=["socket", "http"], default="http")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8117)
    parser.add_argument(
        "--device",
        default="auto",
        help="torch device for feature extraction ('auto', 'cpu', 'cuda', ...)",
    )
    parser.add_argument(
        "--cuda-device",
        type=int,
        default=None,
        help="GPU device exposed through CUDA_VISIBLE_DEVICES.",
    )
    parser.add_argument(
        "--camera-height-m",
        type=float,
        default=None,
        help=(
            "metric camera height; anchors clearance distances on a map "
            "without a metric scale"
        ),
    )
    parser.add_argument(
        "--parent-watch",
        action="store_true",
        help="watch parent process via stdin pipe and exit when it dies",
    )
    return parser


def main() -> None:
    """Load the map and serve localize requests until terminated."""
    args = _build_argparser().parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s")
    if args.cuda_device is not None:
        target = str(args.cuda_device)
        prev = os.environ.get("CUDA_VISIBLE_DEVICES")
        if prev is not None and prev != target:
            logging.warning(
                "CUDA_VISIBLE_DEVICES=%s is already set; overriding with --cuda-device=%s",
                prev,
                args.cuda_device,
            )
        os.environ["CUDA_VISIBLE_DEVICES"] = target
    facade = LocalizeFacade(
        args.map_path,
        device=args.device,
        camera_height_m=args.camera_height_m,
    )
    facade.serve(
        transport=args.transport,
        host=args.host,
        port=args.port,
        parent_watch=args.parent_watch,
    )


if __name__ == "__main__":
    main()
