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

"""Localize a query frame against a built LingBot-Map map.

LingBot-Map reconstructs a map from a frame sequence but does not relocalize a
later run. This module closes that gap with a separate component:

1. preprocess the query exactly as the mapping server preprocessed map frames;
2. extract local features and match them against archived keyframes (DISK +
   LightGlue by default);
3. lift matched keyframe pixels to 3D with that keyframe's depth, intrinsics,
   and camera-to-world pose;
4. solve PnP RANSAC for the query camera pose in the map frame;
5. report the pose, inlier support, and enough diagnostics to reject a false
   match.

Candidate keyframes can be searched exhaustively for a cold start, restricted
to the neighborhood of a prior pose while tracking, or retrieved by a coarse
global descriptor. Exhaustive search is the reliable mode; retrieval is a
fast hint whose result is still verified by PnP.

The offline evaluator replays archived frames as held-out queries and reports
success, translation/rotation error, and latency::

    python -m rpent.tools.localize \\
        --map-path /path/to/maps/workspace \\
        --evaluate --query-stride 10

Localize a single archived or live frame::

    python -m rpent.tools.localize \\
        --map-path /path/to/maps/workspace \\
        --image /path/to/live_frame.png

Maps store camera poses in the reconstruction's model frame, whose scale is
arbitrary. A returned pose is correct in that frame; its translation, the
keyframe depths, and every reported distance are model units. When the map
metadata records a metric anchor, it reports how many metres one unit is; the
poses here stay in the model frame so they can be compared with the map's
depth and point cloud without a second convention.
"""

from __future__ import annotations

import argparse
import io
import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from PIL import Image

from rpent.tools.ground_map import LocalMapSource, PreprocessSpec
from rpent.utils.logging import get_logger

logger = get_logger("localize")

#: Candidate keyframes verified when retrieval or a prior pose is used.
DEFAULT_TOP_K = 8
#: Keyframes on each side of a prior frame considered while tracking.
DEFAULT_PRIOR_WINDOW = 6
#: RANSAC inliers required before a pose is reported as ``ok``.
DEFAULT_MIN_INLIERS = 25
#: Minimum inlier fraction of the lifted correspondences.
DEFAULT_MIN_INLIER_RATIO = 0.10
#: PnP RANSAC reprojection threshold in model-space pixels.
DEFAULT_REPROJECTION_ERROR = 12.0
#: Maximum inlier RMS after refinement; larger values mean a bad pose.
DEFAULT_MAX_RMS_PX = 10.0
#: LightGlue match score below which a correspondence is discarded.
DEFAULT_MIN_MATCH_SCORE = 0.2
#: Minimum reconstruction confidence for a keyframe depth pixel.
DEFAULT_MIN_CONFIDENCE = 1.0
#: Optional temporal exclusion window for replay, in archived frame indices.
DEFAULT_EXCLUSION = 0
#: Hide archived frames whose camera is this close to a query (map units).
DEFAULT_EXCLUSION_RADIUS = 0.1
#: Hide archived frames whose camera is this close in rotation (degrees).
DEFAULT_EXCLUSION_ROTATION_DEG = 10.0
#: Two solutions farther apart than this are treated as distinct places.
DEFAULT_AMBIGUITY_TRANSLATION = 1.0
DEFAULT_AMBIGUITY_ROTATION_DEG = 20.0
#: A distinct pose is a real competitor only with this share of the best
#: pose's inliers.
DEFAULT_AMBIGUITY_SUPPORT_RATIO = 0.5
#: Every this many model keyframes is a query in the default replay.
DEFAULT_QUERY_STRIDE = 10
#: Pose error thresholds used to label a localized query "accurate".
DEFAULT_ACCURATE_TRANSLATION = 0.5
DEFAULT_ACCURATE_ROTATION_DEG = 15.0

_BACKEND_HINT = (
    "the DISK + LightGlue backend requires torch and kornia; install the "
    "optional dependencies with 'uv pip install -e \".[localize]\"'"
)
_CV2_HINT = (
    "localization requires OpenCV for PnP; install it with "
    "'uv pip install -e \".[localize]\"'"
)


@dataclass(frozen=True)
class FeatureSet:
    """Local features in model-input pixel coordinates.

    ``image_size`` is the padded ``(width, height)`` passed to LightGlue for
    keypoint normalization. ``active_size`` is the unpadded model input and
    bounds the pixels that carry depth. Keypoints outside the model input are
    removed at extraction.
    """

    keypoints: np.ndarray
    descriptors: np.ndarray
    image_size: tuple[int, int]
    active_size: tuple[int, int]

    @property
    def count(self) -> int:
        """Number of usable features."""
        return int(self.keypoints.shape[0])


@dataclass(frozen=True)
class FeatureMatches:
    """Index-aligned matches between a query and a keyframe."""

    query_indices: np.ndarray
    keyframe_indices: np.ndarray
    scores: np.ndarray

    @property
    def count(self) -> int:
        """Number of matched feature pairs."""
        return int(self.query_indices.shape[0])


@dataclass(frozen=True)
class PoseSolution:
    """A verified camera pose in the map frame."""

    pose: np.ndarray
    inliers: int
    inlier_ratio: float
    reprojection_rms_px: float


@dataclass(frozen=True)
class LocalizationResult:
    """Outcome of one localization attempt."""

    status: str
    frame: str
    pose: np.ndarray | None
    keyframe: int | None
    matches: int
    correspondences: int
    inliers: int
    inlier_ratio: float
    reprojection_rms_px: float | None
    query_keypoints: int
    candidates_checked: int
    selection: str
    ambiguous: bool
    latency_ms: float
    scale: dict[str, Any]
    reason: str | None = None
    second_keyframe: int | None = None
    second_inliers: int = 0

    @property
    def ok(self) -> bool:
        """Whether a pose passed the inlier and reprojection checks."""
        return self.status == "ok"

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable record."""
        return {
            "status": self.status,
            "frame": self.frame,
            "pose": None if self.pose is None else self.pose.tolist(),
            "keyframe": self.keyframe,
            "matches": self.matches,
            "correspondences": self.correspondences,
            "inliers": self.inliers,
            "inlier_ratio": self.inlier_ratio,
            "reprojection_rms_px": self.reprojection_rms_px,
            "query_keypoints": self.query_keypoints,
            "candidates_checked": self.candidates_checked,
            "selection": self.selection,
            "ambiguous": self.ambiguous,
            "latency_ms": self.latency_ms,
            "scale": self.scale,
            "reason": self.reason,
            "second_keyframe": self.second_keyframe,
            "second_inliers": self.second_inliers,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> LocalizationResult:
        """Rebuild a result from :meth:`to_dict` output.

        Raises:
            ValueError: If required fields are missing or malformed.
        """
        if not isinstance(payload, Mapping):
            raise ValueError(f"localization result must be a mapping, got {payload!r}")
        try:
            pose = payload.get("pose")
            scale = payload.get("scale")
            if not isinstance(scale, Mapping):
                raise ValueError("scale must be a mapping")
            array = None if pose is None else np.asarray(pose, dtype=np.float64)
            if array is not None and array.shape != (4, 4):
                raise ValueError(f"pose must be 4x4, got {array.shape}")
            if array is not None and not np.isfinite(array).all():
                raise ValueError("pose must be finite")
            return cls(
                status=str(payload["status"]),
                frame=str(payload.get("frame", "model")),
                pose=array,
                keyframe=_optional_int(payload.get("keyframe")),
                matches=int(payload.get("matches", 0)),
                correspondences=int(payload.get("correspondences", 0)),
                inliers=int(payload.get("inliers", 0)),
                inlier_ratio=float(payload.get("inlier_ratio", 0.0)),
                reprojection_rms_px=_optional_float(payload.get("reprojection_rms_px")),
                query_keypoints=int(payload.get("query_keypoints", 0)),
                candidates_checked=int(payload.get("candidates_checked", 0)),
                selection=str(payload.get("selection", "unknown")),
                ambiguous=bool(payload.get("ambiguous", False)),
                latency_ms=float(payload.get("latency_ms", 0.0)),
                scale=dict(scale),
                reason=None
                if payload.get("reason") is None
                else str(payload["reason"]),
                second_keyframe=_optional_int(payload.get("second_keyframe")),
                second_inliers=int(payload.get("second_inliers", 0)),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid localization result: {exc}") from exc


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


class FeatureBackend(Protocol):
    """Local feature extraction and matching used by :class:`MapLocalizer`."""

    def extract(self, image: np.ndarray) -> FeatureSet:
        """Extract features from an RGB model-input image."""

    def match(self, query: FeatureSet, keyframe: FeatureSet) -> FeatureMatches:
        """Match query features to keyframe features."""


class DiskLightGlueBackend:
    """DISK keypoints with LightGlue matching, implemented with kornia.

    DISK requires 3-channel input whose height and width are divisible by 16.
    The model input is padded with replicate borders for extraction, but
    keypoints in the padding are discarded so matching uses model coordinates.
    """

    name = "disk-lightglue"

    def __init__(
        self,
        *,
        device: str = "auto",
        min_match_score: float = DEFAULT_MIN_MATCH_SCORE,
    ) -> None:
        self.device = device
        self.min_match_score = float(min_match_score)
        self._loaded = False
        self._torch: Any = None
        self._functional: Any = None
        self._disk: Any = None
        self._matcher: Any = None
        self._device_name = "cpu"

    def _load(self) -> None:
        if self._loaded:
            return
        try:
            import torch
            import torch.nn.functional as functional
            from kornia.feature import DISK, LightGlue
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(_BACKEND_HINT) from exc
        device = self.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._torch = torch
        self._functional = functional
        self._device_name = device
        self._disk = DISK.from_pretrained("depth").to(device).eval()
        self._matcher = (
            LightGlue(features="disk", depth_confidence=-1, width_confidence=-1)
            .to(device)
            .eval()
        )
        self._loaded = True
        logger.info("loaded DISK + LightGlue on %s", device)

    def extract(self, image: np.ndarray) -> FeatureSet:
        """Extract DISK features from an RGB uint8 model-input image."""
        self._load()
        array = np.asarray(image, dtype=np.uint8)
        if array.ndim != 3 or array.shape[2] != 3:
            raise ValueError(f"expected an RGB image, got shape {tuple(array.shape)}")
        height, width = array.shape[:2]
        pad_height = (-height) % 16
        pad_width = (-width) % 16
        tensor = (
            self._torch.from_numpy(array.astype(np.float32) / 255.0)
            .permute(2, 0, 1)[None]
            .to(self._device_name)
        )
        if pad_height or pad_width:
            tensor = self._functional.pad(
                tensor,
                (0, pad_width, 0, pad_height),
                mode="replicate",
            )
        with self._torch.inference_mode():
            features = self._disk(tensor)[0]
        keypoints = features.keypoints.detach().cpu().numpy().astype(np.float32)
        descriptors = features.descriptors.detach().cpu().numpy().astype(np.float32)
        keep = (
            (keypoints[:, 0] >= 0)
            & (keypoints[:, 0] < width)
            & (keypoints[:, 1] >= 0)
            & (keypoints[:, 1] < height)
        )
        return FeatureSet(
            keypoints=keypoints[keep],
            descriptors=descriptors[keep],
            image_size=(int(width + pad_width), int(height + pad_height)),
            active_size=(int(width), int(height)),
        )

    def match(self, query: FeatureSet, keyframe: FeatureSet) -> FeatureMatches:
        """Match query features to one keyframe with LightGlue."""
        if query.count < 2 or keyframe.count < 2:
            empty = np.zeros(0, dtype=np.int64)
            return FeatureMatches(empty, empty, np.zeros(0, dtype=np.float32))
        self._load()
        torch = self._torch

        def _data(features: FeatureSet) -> dict[str, Any]:
            return {
                "keypoints": torch.from_numpy(features.keypoints)[None].to(
                    self._device_name
                ),
                "descriptors": torch.from_numpy(features.descriptors)[None].to(
                    self._device_name
                ),
                "image_size": torch.tensor(
                    [features.image_size], device=self._device_name
                ),
            }

        with torch.inference_mode():
            output = self._matcher({"image0": _data(query), "image1": _data(keyframe)})
        matches = output["matches0"][0].detach().cpu().numpy()
        scores = output["matching_scores0"][0].detach().cpu().numpy()
        valid = (matches >= 0) & (scores >= self.min_match_score)
        query_indices = np.nonzero(valid)[0].astype(np.int64)
        keyframe_indices = matches[valid].astype(np.int64)
        return FeatureMatches(
            query_indices=query_indices,
            keyframe_indices=keyframe_indices,
            scores=scores[valid].astype(np.float32),
        )


def _cv2() -> Any:
    try:
        import cv2
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(_CV2_HINT) from exc
    return cv2


def preprocess_image(image: np.ndarray, spec: PreprocessSpec) -> np.ndarray:
    """Apply the mapping server's resize and crop to a source RGB image.

    The server resizes to ``spec.resize_width`` and a patch-aligned height,
    then center-crops the height to ``spec.input_size[0]`` only when the
    resized image is taller. Landscape sources therefore keep a model input
    shorter than ``spec.input_size[0]``, matching how their depth was stored.
    """
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"expected an RGB image, got shape {tuple(array.shape)}")
    if array.dtype != np.uint8:
        array = array.astype(np.uint8)
    if spec.crop != "center":
        raise ValueError(f"unsupported crop mode: {spec.crop!r}")
    height, width = int(array.shape[0]), int(array.shape[1])
    new_height, new_width = spec.resized_size((height, width))
    pil_image = Image.fromarray(array, mode="RGB")
    resized = pil_image.resize((new_width, new_height), Image.Resampling.BICUBIC)
    input_height = int(spec.input_size[0])
    if new_height > input_height:
        top = spec.crop_top((height, width))
        resized = resized.crop((0, top, new_width, top + input_height))
    return np.asarray(resized, dtype=np.uint8)


def _global_descriptor(image: np.ndarray) -> np.ndarray:
    """Return a small mean-centered RGB descriptor for coarse retrieval."""
    with Image.fromarray(np.asarray(image, dtype=np.uint8), mode="RGB") as pil:
        small = pil.resize((32, 32), Image.Resampling.BILINEAR)
        values = np.asarray(small, dtype=np.float32).ravel()
    values = values - float(values.mean())
    norm = float(np.linalg.norm(values))
    if norm > 0.0:
        values = values / norm
    return values


def rotation_angle_deg(first: np.ndarray, second: np.ndarray) -> float:
    """Return the rotation angle between two camera-to-world poses."""
    rotation = first[:3, :3].T @ second[:3, :3]
    cosine = (float(np.trace(rotation)) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def pose_error(estimated: np.ndarray, ground_truth: np.ndarray) -> tuple[float, float]:
    """Return ``(translation, rotation_deg)`` between two 4x4 poses."""
    translation = float(np.linalg.norm(estimated[:3, 3] - ground_truth[:3, 3]))
    return translation, rotation_angle_deg(estimated, ground_truth)


def _pose_distance(first: np.ndarray, second: np.ndarray) -> float:
    translation, rotation = pose_error(first, second)
    return translation + 0.02 * rotation


def _frame_array(values: np.ndarray) -> np.ndarray:
    """Drop a serialized frame's trailing singleton channel."""
    array = np.asarray(values)
    while array.ndim > 2 and array.shape[-1] == 1:
        array = array[..., 0]
    return array


class MapLocalizer:
    """Match query frames to a map and solve their pose in the map frame."""

    def __init__(
        self,
        map_path: str | Path,
        *,
        backend: FeatureBackend | None = None,
        device: str = "auto",
        min_match_score: float = DEFAULT_MIN_MATCH_SCORE,
        min_confidence: float = DEFAULT_MIN_CONFIDENCE,
        prior_window: int = DEFAULT_PRIOR_WINDOW,
    ) -> None:
        self.map_path = str(Path(map_path))
        self.source = LocalMapSource(map_path)
        if not self.source.has_depth:
            raise ValueError(
                f"{self.map_path} has no depth, intrinsics, and poses; "
                "localization needs a reconstructed map"
            )
        if self.source.preprocess is None:
            raise ValueError(f"{self.map_path} records no preprocessing geometry")
        self.preprocess = self.source.preprocess
        self.min_confidence = float(min_confidence)
        self.prior_window = max(1, int(prior_window))
        if backend is None:
            backend = DiskLightGlueBackend(
                device=device, min_match_score=min_match_score
            )
        self.backend = backend
        self._model_indices = self._load_model_indices()
        if not self._model_indices:
            raise ValueError(f"{self.map_path} has no model keyframes")
        self._features: dict[int, FeatureSet] = {}
        self._descriptors: dict[int, np.ndarray] = {}
        self._intrinsic = self._calibrated_intrinsic()
        self._model_size = self._load_model_size(self._model_indices[0])
        self.frame = str(self.source.metadata.get("frame", "model"))
        self.scale = self._scale_record()
        logger.info(
            "localizing against %s: %d model keyframes, input %dx%d",
            self.map_path,
            len(self._model_indices),
            self._model_size[1],
            self._model_size[0],
        )

    @property
    def model_indices(self) -> list[int]:
        """Frame indices that carry model poses and depth."""
        return list(self._model_indices)

    @property
    def all_indices(self) -> list[int]:
        """Every archived frame index."""
        return list(range(self.source.frame_count))

    @property
    def intrinsic(self) -> np.ndarray:
        """Query intrinsics at the model input resolution."""
        return self._intrinsic.copy()

    def _load_model_indices(self) -> list[int]:
        index_path = Path(self.map_path) / "keyframe_index.json"
        indices: list[int] = []
        if index_path.is_file():
            entries = json.loads(index_path.read_text())
            if isinstance(entries, list):
                for position, entry in enumerate(entries):
                    if not isinstance(entry, dict):
                        continue
                    if not entry.get("model_keyframe"):
                        continue
                    index = int(entry.get("index", position))
                    if 0 <= index < self.source.frame_count:
                        indices.append(index)
        if not indices:
            logger.info(
                "%s has no model-keyframe flags; using every frame",
                self.map_path,
            )
            indices = list(range(self.source.frame_count))
        return sorted(indices)

    def _calibrated_intrinsic(self) -> np.ndarray:
        intrinsics = self.source.intrinsics
        if intrinsics is None:
            raise ValueError(f"{self.map_path} has no intrinsics")
        values = np.asarray(intrinsics, dtype=np.float64)
        selected = values[self._model_indices]
        selected = selected[np.isfinite(selected).all(axis=(1, 2))]
        if selected.size == 0:
            raise ValueError(f"{self.map_path} has no finite intrinsics")
        return np.median(selected, axis=0)

    def _load_model_size(self, index: int) -> tuple[int, int]:
        frame = self.source.load(index)
        if frame.depth is None:
            raise ValueError(f"frame {index} has no depth")
        depth = _frame_array(frame.depth)
        return int(depth.shape[0]), int(depth.shape[1])

    def _scale_record(self) -> dict[str, Any]:
        scale = self.source.metadata.get("scale")
        if not isinstance(scale, dict):
            return {"metric": False, "meters_per_unit": None}
        return {
            "metric": bool(scale.get("metric", False)),
            "meters_per_unit": scale.get("meters_per_unit"),
        }

    def load_image(self, index: int) -> np.ndarray:
        """Decode one archived frame as RGB."""
        frame = self.source.load(index)
        with Image.open(io.BytesIO(frame.image)) as image:
            return np.asarray(image.convert("RGB"))

    def _prepare(self, index: int) -> np.ndarray:
        prepared = preprocess_image(self.load_image(index), self.preprocess)
        self._check_model_size(prepared.shape[:2], f"frame {index}")
        return prepared

    def _check_model_size(self, size: tuple[int, int], description: str) -> None:
        expected = self._model_size
        if (int(size[0]), int(size[1])) != expected:
            raise ValueError(
                f"{description} preprocesses to {int(size[0])}x{int(size[1])}, "
                f"but the map model input is {expected[0]}x{expected[1]}; "
                "use a frame with the same resolution and orientation as the "
                "mapping run"
            )

    def _features_for(self, index: int) -> FeatureSet:
        if index not in self._features:
            self._features[index] = self.backend.extract(self._prepare(index))
        return self._features[index]

    def _descriptor_for(self, index: int) -> np.ndarray:
        if index not in self._descriptors:
            self._descriptors[index] = _global_descriptor(self._prepare(index))
        return self._descriptors[index]

    def _select_candidates(
        self,
        prepared: np.ndarray,
        *,
        candidates: Sequence[int] | None,
        prior_index: int | None,
        prior_pose: np.ndarray | None,
        exhaustive: bool,
        top_k: int,
        exclude: Sequence[int],
    ) -> tuple[list[int], str]:
        excluded = {int(index) for index in exclude}
        available = [index for index in self._model_indices if index not in excluded]
        if not available:
            raise ValueError("no model keyframes remain after exclusion")
        if candidates is not None:
            selected = [
                int(index)
                for index in candidates
                if int(index) in self._model_indices and int(index) not in excluded
            ]
            if not selected:
                raise ValueError("no explicit candidates remain after exclusion")
            return selected, "explicit"
        if exhaustive:
            return available, "exhaustive"
        if prior_index is not None:
            nearest = min(available, key=lambda index: abs(index - prior_index))
            selected = [
                index
                for index in available
                if abs(index - nearest) <= self.prior_window
            ]
            return selected, "prior_index"
        if prior_pose is not None:
            poses = self.source.camera_to_world
            if poses is None:
                raise ValueError(f"{self.map_path} has no camera poses")
            ranked = sorted(
                available,
                key=lambda index: _pose_distance(prior_pose, poses[index]),
            )
            return ranked[: max(1, top_k)], "prior_pose"
        descriptor = _global_descriptor(prepared)
        ranked = sorted(
            available,
            key=lambda index: float(
                np.linalg.norm(descriptor - self._descriptor_for(index))
            ),
        )
        return ranked[: max(1, top_k)], "retrieval"

    def _lift_matches(
        self,
        index: int,
        keyframe: FeatureSet,
        query: FeatureSet,
        matches: FeatureMatches,
    ) -> tuple[np.ndarray, np.ndarray]:
        if matches.count == 0:
            empty_points = np.zeros((0, 3), dtype=np.float64)
            empty_pixels = np.zeros((0, 2), dtype=np.float64)
            return empty_points, empty_pixels
        frame = self.source.load(index)
        if (
            frame.depth is None
            or frame.intrinsic is None
            or frame.camera_to_world is None
        ):
            empty_points = np.zeros((0, 3), dtype=np.float64)
            empty_pixels = np.zeros((0, 2), dtype=np.float64)
            return empty_points, empty_pixels
        depth = _frame_array(frame.depth)
        if (depth.shape[1], depth.shape[0]) != keyframe.active_size:
            raise ValueError(
                f"frame {index} depth is {depth.shape[0]}x{depth.shape[1]}, "
                f"but its features are {keyframe.active_size[1]}x"
                f"{keyframe.active_size[0]}; the map is inconsistent"
            )
        confidence = (
            None if frame.confidence is None else _frame_array(frame.confidence)
        )
        rows = np.rint(keyframe.keypoints[matches.keyframe_indices, 1]).astype(np.int64)
        cols = np.rint(keyframe.keypoints[matches.keyframe_indices, 0]).astype(np.int64)
        height, width = depth.shape[:2]
        valid = (rows >= 0) & (rows < height) & (cols >= 0) & (cols < width)
        rows = np.clip(rows, 0, height - 1)
        cols = np.clip(cols, 0, width - 1)
        z = np.asarray(depth[rows, cols], dtype=np.float64)
        valid &= np.isfinite(z) & (z > 1e-6)
        if confidence is not None:
            support = np.asarray(confidence[rows, cols], dtype=np.float64)
            valid &= np.isfinite(support) & (support >= self.min_confidence)
        if not np.any(valid):
            empty_points = np.zeros((0, 3), dtype=np.float64)
            empty_pixels = np.zeros((0, 2), dtype=np.float64)
            return empty_points, empty_pixels
        rows = rows[valid]
        cols = cols[valid]
        z = z[valid]
        intrinsic = np.asarray(frame.intrinsic, dtype=np.float64)
        fx, fy = intrinsic[0, 0], intrinsic[1, 1]
        cx, cy = intrinsic[0, 2], intrinsic[1, 2]
        camera_points = np.stack(
            [
                (cols.astype(np.float64) - cx) * z / fx,
                (rows.astype(np.float64) - cy) * z / fy,
                z,
            ],
            axis=1,
        )
        pose = np.asarray(frame.camera_to_world, dtype=np.float64)
        world_points = camera_points @ pose[:3, :3].T + pose[:3, 3]
        image_points = np.asarray(
            query.keypoints[matches.query_indices[valid]], dtype=np.float64
        )
        return world_points, image_points

    def _solve(
        self,
        object_points: np.ndarray,
        image_points: np.ndarray,
        *,
        reprojection_error: float,
    ) -> PoseSolution | None:
        if object_points.shape[0] < 6:
            return None
        cv2 = _cv2()
        try:
            ok, rvec, tvec, inliers = cv2.solvePnPRansac(
                object_points,
                image_points,
                self._intrinsic,
                None,
                reprojectionError=float(reprojection_error),
                iterationsCount=5000,
                confidence=0.999,
                flags=cv2.SOLVEPNP_EPNP,
            )
            if not ok or inliers is None:
                return None
            inlier_indices = np.asarray(inliers).reshape(-1)
            if inlier_indices.size < 6:
                return None
            criteria = (
                cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT,
                50,
                1e-8,
            )
            try:
                rvec, tvec = cv2.solvePnPRefineLM(
                    object_points[inlier_indices],
                    image_points[inlier_indices],
                    self._intrinsic,
                    None,
                    rvec,
                    tvec,
                    criteria,
                )
            except cv2.error:
                logger.debug("PnP refinement failed; keeping the RANSAC pose")
            projected, _ = cv2.projectPoints(
                object_points[inlier_indices],
                rvec,
                tvec,
                self._intrinsic,
                None,
            )
        except cv2.error as exc:
            logger.debug(
                "PnP failed on %d correspondences: %s", object_points.shape[0], exc
            )
            return None
        residuals = projected.reshape(-1, 2) - image_points[inlier_indices]
        rms = float(np.sqrt(np.mean(np.sum(residuals * residuals, axis=1))))
        rotation, _ = cv2.Rodrigues(rvec)
        world_to_camera = np.eye(4, dtype=np.float64)
        world_to_camera[:3, :3] = rotation
        world_to_camera[:3, 3] = np.asarray(tvec).reshape(3)
        pose = np.linalg.inv(world_to_camera)
        return PoseSolution(
            pose=pose,
            inliers=int(inlier_indices.size),
            inlier_ratio=float(inlier_indices.size / max(1, object_points.shape[0])),
            reprojection_rms_px=rms,
        )

    def localize(
        self,
        image: np.ndarray,
        *,
        candidates: Sequence[int] | None = None,
        prior_index: int | None = None,
        prior_pose: np.ndarray | None = None,
        exhaustive: bool = False,
        top_k: int = DEFAULT_TOP_K,
        exclude: Sequence[int] = (),
        min_inliers: int = DEFAULT_MIN_INLIERS,
        min_inlier_ratio: float = DEFAULT_MIN_INLIER_RATIO,
        reprojection_error: float = DEFAULT_REPROJECTION_ERROR,
        max_rms_px: float = DEFAULT_MAX_RMS_PX,
    ) -> LocalizationResult:
        """Estimate the camera pose of one RGB frame in the map frame.

        Candidate selection is explicit ``candidates``, else ``exhaustive``,
        else a window around ``prior_index``, else the nearest ``prior_pose``
        keyframes, else a global-descriptor shortlist. Every candidate is
        verified with PnP; retrieval alone never produces a pose.
        """
        start = time.perf_counter()
        prepared = preprocess_image(image, self.preprocess)
        self._check_model_size(prepared.shape[:2], "the query image")
        query_features = self.backend.extract(prepared)
        selected, selection = self._select_candidates(
            prepared,
            candidates=candidates,
            prior_index=prior_index,
            prior_pose=prior_pose,
            exhaustive=exhaustive,
            top_k=top_k,
            exclude=exclude,
        )
        attempts: list[tuple[int, int, int, PoseSolution | None]] = []
        for index in selected:
            keyframe_features = self._features_for(index)
            matches = self.backend.match(query_features, keyframe_features)
            object_points, image_points = self._lift_matches(
                index, keyframe_features, query_features, matches
            )
            solution = self._solve(
                object_points,
                image_points,
                reprojection_error=reprojection_error,
            )
            attempts.append((index, matches.count, object_points.shape[0], solution))
        solutions = [attempt for attempt in attempts if attempt[3] is not None]
        solutions.sort(
            key=lambda attempt: (
                attempt[3].inliers,
                -attempt[3].reprojection_rms_px,
            ),
            reverse=True,
        )
        best = solutions[0] if solutions else None
        second = solutions[1] if len(solutions) > 1 else None
        latency_ms = (time.perf_counter() - start) * 1000.0
        if best is None:
            return LocalizationResult(
                status="lost",
                frame=self.frame,
                pose=None,
                keyframe=None,
                matches=max((attempt[1] for attempt in attempts), default=0),
                correspondences=max((attempt[2] for attempt in attempts), default=0),
                inliers=0,
                inlier_ratio=0.0,
                reprojection_rms_px=None,
                query_keypoints=query_features.count,
                candidates_checked=len(selected),
                selection=selection,
                ambiguous=False,
                latency_ms=latency_ms,
                scale=self.scale,
                reason="no candidate produced a PnP pose",
            )
        best_index, match_count, correspondence_count, solution = best
        assert solution is not None
        second_keyframe = None
        second_inliers = 0
        ambiguous = False
        if second is not None and second[3] is not None:
            other = second[3]
            second_keyframe = second[0]
            second_inliers = other.inliers
            plausible = (
                other.inliers >= min_inliers
                and other.inliers >= DEFAULT_AMBIGUITY_SUPPORT_RATIO * solution.inliers
                and other.inlier_ratio >= min_inlier_ratio
                and other.reprojection_rms_px <= max_rms_px
            )
            if plausible:
                translation, rotation = pose_error(solution.pose, other.pose)
                ambiguous = (
                    translation > DEFAULT_AMBIGUITY_TRANSLATION
                    or rotation > DEFAULT_AMBIGUITY_ROTATION_DEG
                )
        reason = None
        status = "ok"
        if solution.inliers < min_inliers:
            status = "lost"
            reason = (
                f"best pose has {solution.inliers} inliers, below "
                f"the required {min_inliers}"
            )
        elif solution.inlier_ratio < min_inlier_ratio:
            status = "lost"
            reason = (
                f"inlier ratio {solution.inlier_ratio:.2f} is below "
                f"the required {min_inlier_ratio:.2f}"
            )
        elif solution.reprojection_rms_px > max_rms_px:
            status = "lost"
            reason = (
                f"inlier RMS {solution.reprojection_rms_px:.1f} px exceeds "
                f"the allowed {max_rms_px:.1f} px"
            )
        elif ambiguous:
            status = "ambiguous"
            reason = (
                f"a distinct pose has {second_inliers} inliers versus "
                f"{solution.inliers}; verify with an independent sensor or a "
                "pose prior"
            )
        return LocalizationResult(
            status=status,
            frame=self.frame,
            pose=solution.pose,
            keyframe=best_index,
            matches=match_count,
            correspondences=correspondence_count,
            inliers=solution.inliers,
            inlier_ratio=solution.inlier_ratio,
            reprojection_rms_px=solution.reprojection_rms_px,
            query_keypoints=query_features.count,
            candidates_checked=len(selected),
            selection=selection,
            ambiguous=ambiguous,
            latency_ms=latency_ms,
            scale=self.scale,
            reason=reason,
            second_keyframe=second_keyframe,
            second_inliers=second_inliers,
        )


def _distribution(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "median": None, "p75": None, "max": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "median": float(np.median(array)),
        "p75": float(np.percentile(array, 75)),
        "max": float(array.max()),
    }


def _scene_summary(localizer: MapLocalizer) -> dict[str, Any]:
    poses = localizer.source.camera_to_world
    summary: dict[str, Any] = {
        "frames": localizer.source.frame_count,
        "model_keyframes": len(localizer.model_indices),
        "metric": bool(localizer.scale.get("metric", False)),
        "meters_per_unit": localizer.scale.get("meters_per_unit"),
    }
    if poses is not None and len(poses) > 1:
        centers = np.asarray(poses[:, :3, 3], dtype=np.float64)
        summary["camera_span"] = float(np.linalg.norm(np.ptp(centers, axis=0)))
        summary["trajectory_length"] = float(
            np.sum(np.linalg.norm(np.diff(centers, axis=0), axis=1))
        )
    return summary


def evaluate(
    map_path: str | Path,
    *,
    queries: Sequence[int] | None = None,
    stride: int = DEFAULT_QUERY_STRIDE,
    exclusion: int = DEFAULT_EXCLUSION,
    exclusion_radius: float = DEFAULT_EXCLUSION_RADIUS,
    exclusion_rotation: float = DEFAULT_EXCLUSION_ROTATION_DEG,
    mode: str = "exhaustive",
    top_k: int = DEFAULT_TOP_K,
    min_inliers: int = DEFAULT_MIN_INLIERS,
    min_inlier_ratio: float = DEFAULT_MIN_INLIER_RATIO,
    reprojection_error: float = DEFAULT_REPROJECTION_ERROR,
    max_rms_px: float = DEFAULT_MAX_RMS_PX,
    accurate_translation: float = DEFAULT_ACCURATE_TRANSLATION,
    accurate_rotation: float = DEFAULT_ACCURATE_ROTATION_DEG,
    localizer: MapLocalizer | None = None,
    output: str | Path | None = None,
) -> dict[str, Any]:
    """Replay held-out frames and write localization metrics.

    The query frame and archived frames whose camera is within
    ``exclusion_radius`` map units and ``exclusion_rotation`` degrees of it are
    hidden, so the query cannot match itself or a near-identical duplicate.
    ``exclusion`` additionally hides archived frames within that many frame
    indices on each side. Spatial hiding is disabled when either spatial
    threshold is non-positive. ``mode`` is ``"exhaustive"`` or ``"retrieval"``.
    The returned payload is also written to ``localization_eval.json`` in the
    map directory unless ``output`` names another path.
    """
    if stride <= 0:
        raise ValueError("stride must be positive")
    if exclusion < 0:
        raise ValueError("exclusion must not be negative")
    if exclusion_radius < 0 or exclusion_rotation < 0:
        raise ValueError("exclusion thresholds must not be negative")
    if mode not in {"exhaustive", "retrieval"}:
        raise ValueError(f"unknown mode: {mode}")
    if localizer is None:
        localizer = MapLocalizer(map_path)
    if Path(localizer.map_path) != Path(map_path):
        raise ValueError("localizer and map_path refer to different maps")
    if queries is None:
        selected_queries = localizer.model_indices[::stride]
    else:
        selected_queries = [int(index) for index in queries]
    if not selected_queries:
        raise ValueError("no query frames were selected")
    for query in selected_queries:
        if not 0 <= query < localizer.source.frame_count:
            raise ValueError(
                f"query frame {query} is outside the map's "
                f"{localizer.source.frame_count} frames"
            )
    records: list[dict[str, Any]] = []
    poses = localizer.source.camera_to_world
    spatial_exclusion = exclusion_radius > 0 and exclusion_rotation > 0
    for query in selected_queries:
        ground_truth_frame = localizer.source.load(query)
        ground_truth = ground_truth_frame.camera_to_world
        if ground_truth is None:
            logger.warning("frame %d has no pose; skipping", query)
            continue
        hidden = {query}
        for index in localizer.all_indices:
            if index == query:
                continue
            if abs(index - query) <= exclusion:
                hidden.add(index)
                continue
            if not spatial_exclusion or poses is None:
                continue
            translation, rotation = pose_error(
                np.asarray(poses[index], dtype=np.float64),
                np.asarray(ground_truth, dtype=np.float64),
            )
            if translation < exclusion_radius and rotation < exclusion_rotation:
                hidden.add(index)
        try:
            image = localizer.load_image(query)
            result = localizer.localize(
                image,
                exhaustive=mode == "exhaustive",
                top_k=top_k,
                exclude=sorted(hidden),
                min_inliers=min_inliers,
                min_inlier_ratio=min_inlier_ratio,
                reprojection_error=reprojection_error,
                max_rms_px=max_rms_px,
            )
        except (ValueError, RuntimeError) as exc:
            logger.warning("frame %d could not be localized: %s", query, exc)
            records.append(
                {
                    "frame": query,
                    "status": "error",
                    "keyframe": None,
                    "matches": 0,
                    "correspondences": 0,
                    "inliers": 0,
                    "inlier_ratio": 0.0,
                    "reprojection_rms_px": None,
                    "query_keypoints": 0,
                    "candidates_checked": 0,
                    "selection": mode,
                    "ambiguous": False,
                    "latency_ms": 0.0,
                    "translation_error": None,
                    "rotation_error": None,
                    "accurate": False,
                    "reason": str(exc),
                }
            )
            continue
        translation_error = None
        rotation_error = None
        if result.pose is not None:
            translation_error, rotation_error = pose_error(
                result.pose, np.asarray(ground_truth, dtype=np.float64)
            )
        accurate = bool(
            result.ok
            and translation_error is not None
            and rotation_error is not None
            and translation_error <= accurate_translation
            and rotation_error <= accurate_rotation
        )
        record = result.to_dict()
        record.update(
            {
                "frame": query,
                "translation_error": translation_error,
                "rotation_error": rotation_error,
                "accurate": accurate,
            }
        )
        records.append(record)
    if not records:
        raise ValueError("none of the selected query frames had a ground-truth pose")
    localized = [record for record in records if record["status"] == "ok"]
    ambiguous = [record for record in records if record["status"] == "ambiguous"]
    accurate = [record for record in localized if record["accurate"]]
    translation_values = [
        record["translation_error"]
        for record in localized
        if record["translation_error"] is not None
    ]
    rotation_values = [
        record["rotation_error"]
        for record in localized
        if record["rotation_error"] is not None
    ]
    rich = [record for record in records if record.get("query_keypoints", 0) >= 200]
    rich_success = [record for record in rich if record["status"] == "ok"]
    meters_per_unit = localizer.scale.get("meters_per_unit")
    summary = {
        "attempted": len(records),
        "localized": len(localized),
        "accurate": len(accurate),
        "ambiguous": len(ambiguous),
        "lost": len(records) - len(localized) - len(ambiguous),
        "success_rate": len(localized) / len(records),
        "accurate_rate": len(accurate) / len(records),
        "queries_with_200_keypoints": len(rich),
        "success_rate_with_200_keypoints": (
            len(rich_success) / len(rich) if rich else None
        ),
        "translation_error": _distribution(translation_values),
        "rotation_error_deg": _distribution(rotation_values),
        "latency_ms": _distribution([record["latency_ms"] for record in records]),
        "accurate_translation_threshold": accurate_translation,
        "accurate_rotation_threshold_deg": accurate_rotation,
        "translation_unit": "map units",
        "scene": _scene_summary(localizer),
    }
    if meters_per_unit is not None:
        summary["meters_per_unit"] = float(meters_per_unit)
        summary["translation_error_m"] = _distribution(
            [value * float(meters_per_unit) for value in translation_values]
        )
    payload = {
        "schema_version": 1,
        "map_path": localizer.map_path,
        "mode": mode,
        "exclusion": {
            "index": exclusion,
            "radius": exclusion_radius,
            "rotation_deg": exclusion_rotation,
        },
        "summary": summary,
        "results": records,
    }
    output_path = (
        Path(output)
        if output is not None
        else Path(localizer.map_path) / "localization_eval.json"
    )
    output_path.write_text(json.dumps(payload, indent=2))
    logger.info(
        "localized %d/%d queries (%d accurate, %d ambiguous); wrote %s",
        len(localized),
        len(records),
        len(accurate),
        len(ambiguous),
        output_path,
    )
    return payload


def _parse_frames(value: str) -> list[int]:
    try:
        return [int(part) for part in value.split(",") if part.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"expected comma-separated frame indices, got {value!r}"
        ) from exc


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Localize a frame against a built LingBot-Map map."
    )
    parser.add_argument("--map-path", required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image", help="query image on disk")
    source.add_argument(
        "--frame-index", type=int, help="archived map frame to use as query"
    )
    source.add_argument(
        "--evaluate",
        action="store_true",
        help="replay held-out map frames and write metrics",
    )
    parser.add_argument(
        "--mode",
        choices=("exhaustive", "retrieval"),
        default="exhaustive",
        help="evaluation candidate mode (default: exhaustive)",
    )
    parser.add_argument(
        "--prior-frame",
        type=int,
        help="search only keyframes near this frame while tracking",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help="keyframes verified in retrieval or prior-pose mode",
    )
    parser.add_argument(
        "--query-stride",
        type=int,
        default=DEFAULT_QUERY_STRIDE,
        help="evaluate every N-th model keyframe",
    )
    parser.add_argument(
        "--query-frames",
        type=_parse_frames,
        help="comma-separated query frame indices for evaluation",
    )
    parser.add_argument(
        "--exclusion",
        type=int,
        default=DEFAULT_EXCLUSION,
        help="also hide frames within this many indices on each side of a query",
    )
    parser.add_argument(
        "--exclusion-radius",
        type=float,
        default=DEFAULT_EXCLUSION_RADIUS,
        help="hide map frames this close to a query (map units)",
    )
    parser.add_argument(
        "--exclusion-rotation",
        type=float,
        default=DEFAULT_EXCLUSION_ROTATION_DEG,
        help="hide map frames this close in rotation (degrees)",
    )
    parser.add_argument("--min-inliers", type=int, default=DEFAULT_MIN_INLIERS)
    parser.add_argument(
        "--min-inlier-ratio", type=float, default=DEFAULT_MIN_INLIER_RATIO
    )
    parser.add_argument(
        "--reprojection-error",
        type=float,
        default=DEFAULT_REPROJECTION_ERROR,
    )
    parser.add_argument("--max-rms", type=float, default=DEFAULT_MAX_RMS_PX)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument(
        "--output", help="evaluation output path (default: in the map dir)"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Run the localization CLI."""
    args = _build_argparser().parse_args(argv)
    localizer = MapLocalizer(
        args.map_path,
        device=args.device,
        min_match_score=DEFAULT_MIN_MATCH_SCORE,
    )
    if args.evaluate:
        payload = evaluate(
            args.map_path,
            queries=args.query_frames,
            stride=args.query_stride,
            exclusion=args.exclusion,
            exclusion_radius=args.exclusion_radius,
            exclusion_rotation=args.exclusion_rotation,
            mode=args.mode,
            top_k=args.top_k,
            min_inliers=args.min_inliers,
            min_inlier_ratio=args.min_inlier_ratio,
            reprojection_error=args.reprojection_error,
            max_rms_px=args.max_rms,
            localizer=localizer,
            output=args.output,
        )
        print(json.dumps(payload["summary"], indent=2))
        return
    if args.frame_index is not None:
        image = localizer.load_image(args.frame_index)
    else:
        with Image.open(args.image) as source:
            image = np.asarray(source.convert("RGB"))
    result = localizer.localize(
        image,
        prior_index=args.prior_frame,
        exhaustive=args.prior_frame is None and args.mode == "exhaustive",
        top_k=args.top_k,
        min_inliers=args.min_inliers,
        min_inlier_ratio=args.min_inlier_ratio,
        reprojection_error=args.reprojection_error,
        max_rms_px=args.max_rms,
    )
    print(json.dumps(result.to_dict(), indent=2))


if __name__ == "__main__":
    main()
