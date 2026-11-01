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

"""Ground open-vocabulary labels on a reconstructed LingBot-Map map.

The mapping server stores, for every archived frame, the source RGB image, the
model-resolution depth, the camera intrinsics, and the camera-to-world pose.
This module turns Grounding DINO + SAM 2 detections on those frames into object
instances in the map's 3D frame:

1. run detection and segmentation on archived source frames;
2. map each source-resolution mask through the server's resize and center crop
   onto the model depth grid;
3. unproject the masked depth pixels with the frame's intrinsics and camera
   pose, and reject depth outliers;
4. associate same-label observations across frames by 3D proximity;
5. write ``objects.json``, ``semantic_points.npz``, ``semantic_points.ply``,
   and per-frame mask/overlay PNGs.

Run it offline on a built map directory::

    python -m rpent.tools.ground_map \\
        --map-path /path/to/maps/workspace \\
        --labels chair desk monitor \\
        --perception-endpoint http://127.0.0.1:8115

The same pipeline runs online inside ``build_map(labels=...)``: a
:class:`ClientMapSource` fetches frames and depth over RPC, and the resulting
annotations are written back into the map directory.
"""

from __future__ import annotations

import argparse
import io
import json
import math
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from PIL import Image, ImageDraw

from rpent.utils.logging import get_logger
from rpent.utils.rpc.http_rpc import HttpRpcClient
from toolkits.grounding.client import (
    MAX_LABELS_PER_CALL,
    Detection,
    GdinoSam2Client,
    normalize_labels,
)
from toolkits.mapping.client import LingBotMapClient
from toolkits.mapping.server import (
    ANNOTATIONS_DIR,
    MAP_SCHEMA_VERSION,
    OBJECTS_FILE,
    SEMANTIC_PLY_FILE,
    SEMANTIC_POINTS_FILE,
    camera_to_world,
)

logger = get_logger("ground_map")

#: Object semantic-map schema version.
SEMANTIC_SCHEMA_VERSION = 1
#: Default number of archived frames to ground when the caller names none.
DEFAULT_MAX_KEYFRAMES = 12

#: Indexed by label position; also written into the semantic PLY and overlays.
LABEL_PALETTE = np.asarray(
    [
        [230, 25, 75],
        [60, 180, 75],
        [255, 225, 25],
        [0, 130, 200],
        [245, 130, 48],
        [145, 30, 180],
        [70, 240, 240],
        [240, 50, 230],
        [210, 245, 60],
        [250, 190, 190],
        [0, 128, 128],
        [170, 110, 40],
        [255, 250, 200],
        [128, 0, 0],
        [170, 255, 195],
        [128, 128, 0],
    ],
    dtype=np.uint8,
)


@dataclass(frozen=True)
class PreprocessSpec:
    """How the mapping server turned a source frame into its model input.

    ``input_size`` is ``(height, width)``. The server resizes every frame to
    ``resize_width`` and a patch-aligned height, then center-crops the height
    to ``input_size[0]`` when it is taller.
    """

    resize_width: int = 518
    patch_size: int = 14
    input_size: tuple[int, int] = (518, 518)
    crop: str = "center"

    @classmethod
    def from_metadata(
        cls,
        metadata: dict[str, Any],
        *,
        model_size: tuple[int, int] | None = None,
    ) -> PreprocessSpec | None:
        """Read the preprocessing block, falling back for version-1 maps.

        Version-1 maps recorded no block, so ``model_size`` (the depth grid
        read from the map) supplies the input size. Without it the square
        518x518 grid is assumed, which is wrong for landscape source frames.
        """
        block = metadata.get("preprocess")
        if isinstance(block, dict):
            try:
                input_size = block.get("input_size", [518, 518])
                return cls(
                    resize_width=int(block.get("resize_width", 518)),
                    patch_size=int(block.get("patch_size", 14)),
                    input_size=(int(input_size[0]), int(input_size[1])),
                    crop=str(block.get("crop", "center")),
                )
            except (TypeError, ValueError, IndexError) as exc:
                raise ValueError(f"invalid preprocess block: {block!r}") from exc
        if metadata.get("backend") == "gct_stream":
            # Version-1 maps predate the explicit block; the streaming backend
            # used these values with a height that depends on the source aspect.
            if model_size is None:
                return cls()
            return cls(input_size=(int(model_size[0]), int(model_size[1])))
        return None

    def resized_size(self, source_size: tuple[int, int]) -> tuple[int, int]:
        """Return ``(height, width)`` after the server's resize."""
        height, width = int(source_size[0]), int(source_size[1])
        if height <= 0 or width <= 0:
            raise ValueError(f"invalid source size: {source_size!r}")
        new_width = self.resize_width
        new_height = int(round(height * (new_width / width) / self.patch_size))
        return max(self.patch_size, new_height * self.patch_size), new_width

    def crop_top(self, source_size: tuple[int, int]) -> int:
        """Return the first resized row kept by the center crop."""
        new_height, _ = self.resized_size(source_size)
        return max(0, (new_height - self.input_size[0]) // 2)

    def source_mask_to_model(
        self, mask: np.ndarray, source_size: tuple[int, int]
    ) -> np.ndarray:
        """Warp a source-resolution mask onto the model depth grid."""
        source_mask = np.asarray(mask, dtype=bool)
        height, width = int(source_size[0]), int(source_size[1])
        if source_mask.shape != (height, width):
            raise ValueError(
                f"mask shape {source_mask.shape} does not match source {source_size}"
            )
        if self.crop != "center":
            raise ValueError(f"unsupported crop mode: {self.crop!r}")
        new_height, new_width = self.resized_size(source_size)
        if new_width != self.input_size[1]:
            raise ValueError(
                "the server does not crop width; resize_width must match input width"
            )
        crop_top = self.crop_top(source_size)
        rows = np.floor(
            (np.arange(self.input_size[0]) + crop_top) * (height / new_height)
        ).astype(np.int64)
        cols = np.floor(np.arange(self.input_size[1]) * (width / new_width)).astype(
            np.int64
        )
        row_valid = (
            (rows >= 0)
            & (rows < height)
            & ((np.arange(self.input_size[0]) + crop_top) < new_height)
        )
        col_valid = (cols >= 0) & (cols < width)
        clamped_rows = np.clip(rows, 0, height - 1)
        clamped_cols = np.clip(cols, 0, width - 1)
        warped = source_mask[np.ix_(clamped_rows, clamped_cols)]
        warped[~row_valid, :] = False
        warped[:, ~col_valid] = False
        return warped

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON form recorded in ``objects.json``."""
        return {
            "resize_width": self.resize_width,
            "patch_size": self.patch_size,
            "input_size": list(self.input_size),
            "crop": self.crop,
        }


@dataclass(frozen=True)
class MapFrameData:
    """One archived frame with the geometry needed to lift detections."""

    index: int
    image: bytes
    source_size: tuple[int, int]
    timestamp_s: float | None
    pushed_base_pose: list[float] | None
    depth: np.ndarray | None
    confidence: np.ndarray | None
    intrinsic: np.ndarray | None
    camera_to_world: np.ndarray | None


class MapSource(Protocol):
    """Frames and geometry of one built map."""

    map_path: str
    metadata: dict[str, Any]
    frame_count: int
    has_depth: bool
    preprocess: PreprocessSpec | None

    def load(self, index: int) -> MapFrameData:
        """Load one archived frame and its geometry."""


@dataclass
class Observation:
    """One detection on one keyframe, optionally lifted into 3D."""

    label: str
    keyframe: int
    timestamp_s: float | None
    pushed_base_pose: list[float] | None
    score: float
    box: list[float]
    mask_pixels: int
    valid_depth_pixels: int
    mask_name: str
    lifted: bool
    lift_reason: str | None = None
    points: np.ndarray | None = None
    summary: dict[str, Any] | None = None
    object_id: str | None = None
    object_index: int = -1

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON record stored under ``detections``."""
        record: dict[str, Any] = {
            "label": self.label,
            "keyframe": self.keyframe,
            "timestamp_s": self.timestamp_s,
            "pushed_base_pose": self.pushed_base_pose,
            "score": round(float(self.score), 4),
            "box": [round(float(value), 2) for value in self.box],
            "mask_pixels": self.mask_pixels,
            "valid_depth_pixels": self.valid_depth_pixels,
            "mask": f"{ANNOTATIONS_DIR}/{self.mask_name}",
            "lifted": self.lifted,
            "lift_reason": self.lift_reason,
            "object_id": self.object_id,
        }
        if self.summary is not None:
            record.update(self.summary)
        return record


@dataclass
class ObjectInstance:
    """Same-label observations associated across keyframes."""

    object_id: str
    label: str
    members: list[int]
    points: np.ndarray
    summary: dict[str, Any]
    robot_summary: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON record stored under ``objects``."""
        record: dict[str, Any] = {
            "id": self.object_id,
            "label": self.label,
            "frame": "model",
            "observation_count": len(self.members),
            **self.summary,
        }
        if self.robot_summary is not None:
            record["robot_frame"] = self.robot_summary
        return record


@dataclass
class GroundMapResult:
    """Objects, semantic points, masks, and evaluation for one map."""

    map_path: str
    metadata: dict[str, Any]
    labels: list[str]
    keyframes: list[int]
    observations: list[Observation]
    objects: list[ObjectInstance]
    masks: dict[str, bytes]
    semantic_points: np.ndarray
    semantic_labels: np.ndarray
    semantic_scores: np.ndarray
    semantic_object_ids: np.ndarray
    semantic_frame_indices: np.ndarray
    summary: dict[str, Any]
    preprocess: PreprocessSpec | None

    def objects_payload(self) -> dict[str, Any]:
        """Return the full ``objects.json`` document."""
        return {
            "schema_version": SEMANTIC_SCHEMA_VERSION,
            "map_schema_version": int(
                self.metadata.get("schema_version", MAP_SCHEMA_VERSION)
            ),
            "map_path": self.map_path,
            "map_frame": self.metadata.get("frame", "model"),
            "scale": self.metadata.get("scale"),
            "gravity": self.metadata.get("gravity"),
            "floor": self.metadata.get("floor"),
            "alignment": self.metadata.get("alignment"),
            "preprocess": (
                None if self.preprocess is None else self.preprocess.to_dict()
            ),
            "labels": self.labels,
            "keyframes": self.keyframes,
            "detections": [observation.to_dict() for observation in self.observations],
            "objects": [instance.to_dict() for instance in self.objects],
            "evaluation": self.summary,
        }

    def semantic_npz_bytes(self) -> bytes:
        """Serialize the semantic point cloud as an ``.npz`` payload."""
        buffer = io.BytesIO()
        np.savez_compressed(
            buffer,
            points=self.semantic_points.astype(np.float32),
            labels=self.semantic_labels.astype(np.int32),
            scores=self.semantic_scores.astype(np.float32),
            object_ids=self.semantic_object_ids.astype(np.int32),
            frame_indices=self.semantic_frame_indices.astype(np.int32),
            label_names=np.asarray(self.labels),
        )
        return buffer.getvalue()

    def semantic_ply_bytes(self) -> bytes:
        """Serialize the semantic point cloud as a labeled binary PLY."""
        return _semantic_ply_bytes(
            self.semantic_points, self.semantic_labels, len(self.labels)
        )


class LocalMapSource:
    """Read a map directory directly from disk."""

    def __init__(self, map_dir: str | Path) -> None:
        self.map_path = str(Path(map_dir))
        self._map_dir = Path(map_dir)
        metadata_path = self._map_dir / "map.json"
        if not metadata_path.is_file():
            raise FileNotFoundError(f"map metadata not found: {metadata_path}")
        self.metadata = json.loads(metadata_path.read_text())
        self.frame_count = int(self.metadata.get("frames", 0))
        depth_path = self._map_dir / "depth.npy"
        self._depth = (
            np.load(depth_path, mmap_mode="r") if depth_path.is_file() else None
        )
        self.preprocess = PreprocessSpec.from_metadata(
            self.metadata,
            model_size=None if self._depth is None else _model_grid(self._depth),
        )
        self._extrinsic = _camera_array(self.metadata, "extrinsic", (3, 4))
        self._intrinsic = _camera_array(self.metadata, "intrinsic", (3, 3))
        self._c2w = self._load_camera_to_world()
        confidence_path = self._map_dir / "world_points_conf.npy"
        self._confidence = (
            np.load(confidence_path, mmap_mode="r")
            if confidence_path.is_file()
            else None
        )
        self.has_depth = (
            self._depth is not None
            and self._intrinsic is not None
            and self._c2w is not None
            and bool(self.metadata.get("has_depth", True))
        )
        if self.metadata.get("has_depth") and not self.has_depth:
            logger.warning(
                "%s declares depth but is missing depth, intrinsics, or poses",
                self.map_path,
            )
        self._entries = self._load_entries()

    @property
    def camera_to_world(self) -> np.ndarray | None:
        """Per-frame camera-to-world poses, or ``None`` without camera data."""
        return self._c2w

    @property
    def intrinsics(self) -> np.ndarray | None:
        """Per-frame model-resolution intrinsics, or ``None`` without cameras."""
        return self._intrinsic

    def _load_camera_to_world(self) -> np.ndarray | None:
        block = self.metadata.get("camera_to_world")
        path = None
        if isinstance(block, dict):
            path = block.get("path")
        candidates = [path, "cameras_c2w.npy"]
        for candidate in candidates:
            if not candidate:
                continue
            candidate_path = self._map_dir / candidate
            if candidate_path.is_file():
                values = np.asarray(np.load(candidate_path), dtype=np.float64)
                while values.ndim > 3 and values.shape[0] == 1:
                    values = values[0]
                if values.shape[1:] == (4, 4):
                    return values
        if self._extrinsic is not None:
            return camera_to_world(self._extrinsic)
        return None

    def _load_entries(self) -> dict[int, dict[str, Any]]:
        index_path = self._map_dir / "keyframe_index.json"
        if index_path.is_file():
            entries = json.loads(index_path.read_text())
            if isinstance(entries, list):
                return {
                    int(entry.get("index", position)): entry
                    for position, entry in enumerate(entries)
                    if isinstance(entry, dict)
                }
        poses_path = self._map_dir / "poses.json"
        if poses_path.is_file():
            poses = json.loads(poses_path.read_text())
            if isinstance(poses, list):
                return {
                    index: {
                        "index": index,
                        "timestamp_s": entry.get("timestamp_s"),
                        "pushed_base_pose": entry.get("pose"),
                    }
                    for index, entry in enumerate(poses)
                    if isinstance(entry, dict)
                }
        return {}

    def load(self, index: int) -> MapFrameData:
        """Load one archived frame from the map directory."""
        if not 0 <= int(index) < self.frame_count:
            raise ValueError(
                f"frame index {index} is outside the map's {self.frame_count} frames"
            )
        frame_index = int(index)
        image_path = self._map_dir / "frames" / f"{frame_index:06d}.png"
        if not image_path.is_file():
            raise FileNotFoundError(f"archived frame not found: {image_path}")
        image_bytes = image_path.read_bytes()
        with Image.open(io.BytesIO(image_bytes)) as image:
            width, height = image.size
        entry = self._entries.get(frame_index, {})
        depth = None
        confidence = None
        intrinsic = None
        pose = None
        if self.has_depth:
            depth = select_serialized_frame(
                self._depth, frame_index, self.frame_count, "depth"
            )
            if self._confidence is not None:
                confidence = select_serialized_frame(
                    self._confidence, frame_index, self.frame_count, "confidence"
                )
            intrinsic = self._intrinsic[frame_index]
            pose = self._c2w[frame_index]
        return MapFrameData(
            index=frame_index,
            image=image_bytes,
            source_size=(int(height), int(width)),
            timestamp_s=entry.get("timestamp_s"),
            pushed_base_pose=entry.get("pushed_base_pose"),
            depth=depth,
            confidence=confidence,
            intrinsic=intrinsic,
            camera_to_world=pose,
        )


class ClientMapSource:
    """Read frames and depth from a running mapping service over RPC."""

    def __init__(
        self,
        client: LingBotMapClient,
        map_path: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self._client = client
        self.map_path = str(map_path)
        self.metadata = (
            metadata if metadata is not None else client.load_map_metadata(map_path)
        )
        self.frame_count = int(self.metadata.get("frames", 0))
        self.preprocess = PreprocessSpec.from_metadata(self.metadata)
        self._extrinsic = _camera_array(self.metadata, "extrinsic", (3, 4))
        self._intrinsic = _camera_array(self.metadata, "intrinsic", (3, 3))
        if self._extrinsic is not None:
            self._c2w = camera_to_world(self._extrinsic)
        else:
            self._c2w = None
        self.has_depth = (
            bool(self.metadata.get("has_depth"))
            and self._intrinsic is not None
            and self._c2w is not None
        )

    def load(self, index: int) -> MapFrameData:
        """Fetch one archived frame, its depth, and its camera pose over RPC."""
        frame = self._client.load_frame(self.map_path, index)
        depth = None
        confidence = None
        intrinsic = None
        pose = None
        if self.has_depth:
            depth, confidence = self._client.load_depth(self.map_path, index)
            intrinsic = self._intrinsic[frame.index]
            pose = self._c2w[frame.index]
        return MapFrameData(
            index=frame.index,
            image=frame.image,
            source_size=frame.source_size,
            timestamp_s=frame.timestamp_s,
            pushed_base_pose=frame.pushed_base_pose,
            depth=depth,
            confidence=confidence,
            intrinsic=intrinsic,
            camera_to_world=pose,
        )


def select_keyframes(
    frame_count: int,
    *,
    max_keyframes: int = DEFAULT_MAX_KEYFRAMES,
    keyframes: Sequence[int] | None = None,
) -> list[int]:
    """Choose evenly spaced frames unless the caller names explicit indices."""
    if frame_count <= 0:
        raise ValueError("the map has no archived frames")
    if keyframes is not None:
        selected = sorted({int(index) for index in keyframes})
        if not selected:
            raise ValueError("keyframes must not be empty")
        if any(index < 0 or index >= frame_count for index in selected):
            raise ValueError(
                f"keyframes must be within [0, {frame_count - 1}], got {selected}"
            )
        return selected
    if max_keyframes < 1:
        raise ValueError("max_keyframes must be positive")
    if max_keyframes >= frame_count:
        return list(range(frame_count))
    values = np.linspace(0, frame_count - 1, max_keyframes)
    return sorted({int(round(float(value))) for value in values})


def lift_mask(
    mask: np.ndarray,
    depth: np.ndarray,
    confidence: np.ndarray | None,
    intrinsic: np.ndarray,
    camera_to_world_pose: np.ndarray,
    *,
    min_confidence: float = 1.0,
) -> np.ndarray:
    """Unproject the masked depth pixels into the model world frame."""
    mask_array = np.asarray(mask, dtype=bool)
    depth_array = np.asarray(depth, dtype=np.float64)
    if depth_array.ndim == 3 and depth_array.shape[-1] == 1:
        depth_array = depth_array[..., 0]
    if mask_array.shape != depth_array.shape:
        raise ValueError(
            f"mask shape {mask_array.shape} does not match depth {depth_array.shape}"
        )
    valid = mask_array & np.isfinite(depth_array) & (depth_array > 1e-6)
    if confidence is not None:
        confidence_array = np.asarray(confidence, dtype=np.float64)
        if confidence_array.ndim == 3 and confidence_array.shape[-1] == 1:
            confidence_array = confidence_array[..., 0]
        if confidence_array.shape != depth_array.shape:
            raise ValueError(
                f"confidence shape {confidence_array.shape} does not match depth "
                f"{depth_array.shape}"
            )
        valid &= np.isfinite(confidence_array) & (
            confidence_array >= float(min_confidence)
        )
    rows, cols = np.nonzero(valid)
    if rows.size == 0:
        return np.empty((0, 3), dtype=np.float64)
    k = np.asarray(intrinsic, dtype=np.float64)
    fx, fy = float(k[0, 0]), float(k[1, 1])
    cx, cy = float(k[0, 2]), float(k[1, 2])
    if fx <= 0.0 or fy <= 0.0:
        raise ValueError(f"intrinsic has non-positive focal lengths: {k.tolist()}")
    z = depth_array[rows, cols]
    camera_points = np.stack([(cols - cx) * z / fx, (rows - cy) * z / fy, z], axis=1)
    transform = np.asarray(camera_to_world_pose, dtype=np.float64)
    return camera_points @ transform[:3, :3].T + transform[:3, 3]


def _robust_filter(points: np.ndarray) -> np.ndarray:
    """Reject depth outliers that a segmentation mask dragged in."""
    if len(points) < 10:
        return points
    centre = np.median(points, axis=0)
    distances = np.linalg.norm(points - centre, axis=1)
    median = float(np.median(distances))
    mad = float(np.median(np.abs(distances - median)))
    limit = median + 4.0 * 1.4826 * mad
    if limit <= median:
        limit = median * 1.5 + 1e-9
    return points[distances <= limit]


def _summarize_points(points: np.ndarray) -> dict[str, Any]:
    """Summarize a 3D point set for association and JSON output."""
    centroid = np.median(points, axis=0)
    distances = np.linalg.norm(points - centroid, axis=1)
    low, high = np.percentile(points, [5.0, 95.0], axis=0)
    return {
        "point_count": int(points.shape[0]),
        "centroid": [float(value) for value in centroid],
        "aabb_min": [float(value) for value in points.min(axis=0)],
        "aabb_max": [float(value) for value in points.max(axis=0)],
        "extent": [float(value) for value in (high - low)],
        "radius": float(np.percentile(distances, 75.0)),
    }


def _robot_frame_summary(
    points: np.ndarray, metadata: dict[str, Any]
) -> dict[str, Any] | None:
    """Transform a model-frame summary into the robot frame when aligned."""
    alignment = metadata.get("alignment")
    if not isinstance(alignment, dict) or alignment.get("status") != "aligned":
        return None
    scale = float(alignment["scale"])
    rotation = np.asarray(alignment["rotation"], dtype=np.float64)
    translation = np.asarray(alignment["translation"], dtype=np.float64)
    transformed = scale * points @ rotation.T + translation
    summary = _summarize_points(transformed)
    summary["frame"] = "robot_base"
    summary["meters_per_unit"] = scale
    return summary


def _slug(label: str) -> str:
    """Make a label safe for an artifact file name."""
    cleaned = "".join(
        character if character.isalnum() or character in "-_" else "_"
        for character in label.strip().lower()
    )
    return cleaned.strip("_") or "object"


def _encode_png(array: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(np.asarray(array, dtype=np.uint8)).save(buffer, format="PNG")
    return buffer.getvalue()


def _mask_png(mask: np.ndarray) -> bytes:
    return _encode_png(np.asarray(mask, dtype=bool) * 255)


def _overlay_png(
    image_bytes: bytes,
    detections: Sequence[Detection],
    label_colors: dict[str, tuple[int, int, int]],
) -> bytes:
    """Render masks and boxes for one keyframe."""
    fallback = (255, 255, 255)
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    frame = np.asarray(image, dtype=np.uint8).copy()
    for detection in detections:
        color = np.asarray(
            label_colors.get(detection.label, fallback), dtype=np.float32
        )
        mask = np.asarray(detection.mask, dtype=bool)
        if mask.shape != frame.shape[:2]:
            continue
        frame[mask] = (0.55 * frame[mask].astype(np.float32) + 0.45 * color).astype(
            np.uint8
        )
    rendered = Image.fromarray(frame)
    draw = ImageDraw.Draw(rendered)
    for detection in detections:
        color = label_colors.get(detection.label, fallback)
        x1, y1, x2, y2 = (int(round(float(value))) for value in detection.box)
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        draw.text(
            (max(0, x1 + 3), max(0, y1 + 2)),
            f"{detection.label} {detection.score:.2f}",
            fill=color,
        )
    buffer = io.BytesIO()
    rendered.save(buffer, format="PNG")
    return buffer.getvalue()


def _fuse_instances(
    observations: Sequence[Observation], merge_factor: float
) -> list[ObjectInstance]:
    """Associate same-label lifted observations into instances."""
    clusters: list[dict[str, Any]] = []
    order = sorted(
        (index for index, observation in enumerate(observations) if observation.lifted),
        key=lambda index: -observations[index].score,
    )
    for observation_index in order:
        observation = observations[observation_index]
        assert observation.summary is not None
        centroid = np.asarray(observation.summary["centroid"])
        radius = float(observation.summary["radius"])
        best: dict[str, Any] | None = None
        best_distance = math.inf
        for cluster in clusters:
            if cluster["label"] != observation.label:
                continue
            distance = float(np.linalg.norm(cluster["centroid"] - centroid))
            threshold = merge_factor * max(cluster["radius"], radius, 1e-6)
            if distance <= threshold and distance < best_distance:
                best = cluster
                best_distance = distance
        if best is None:
            clusters.append(
                {
                    "label": observation.label,
                    "members": [observation_index],
                    "centroid": centroid,
                    "radius": radius,
                }
            )
        else:
            previous_count = len(best["members"])
            best["members"].append(observation_index)
            best["centroid"] = (best["centroid"] * previous_count + centroid) / (
                previous_count + 1
            )
            best["radius"] = max(best["radius"], radius)

    instances: list[ObjectInstance] = []
    ordered = sorted(
        clusters,
        key=lambda cluster: (
            cluster["label"],
            float(cluster["centroid"][0]),
            float(cluster["centroid"][1]),
            float(cluster["centroid"][2]),
        ),
    )
    counts: defaultdict[str, int] = defaultdict(int)
    for cluster in ordered:
        members = sorted(
            cluster["members"], key=lambda index: observations[index].keyframe
        )
        member_points: list[np.ndarray] = []
        for index in members:
            assert observations[index].points is not None
            member_points.append(observations[index].points)
        points = np.concatenate(member_points, axis=0)
        label = cluster["label"]
        object_id = f"{_slug(label)}-{counts[label]}"
        counts[label] += 1
        summary = _summarize_points(points)
        summary["best_score"] = max(observations[index].score for index in members)
        summary["mean_score"] = float(
            np.mean([observations[index].score for index in members])
        )
        summary["keyframes"] = sorted(
            {observations[index].keyframe for index in members}
        )
        instance = ObjectInstance(
            object_id=object_id,
            label=label,
            members=members,
            points=points,
            summary=summary,
        )
        for index in members:
            observations[index].object_id = object_id
        instances.append(instance)
    return instances


def _nearest_centroid_distance(observations: Sequence[Observation]) -> float | None:
    """Median nearest-neighbour distance among same-label lifted frames."""
    values: list[float] = []
    for first_index, first in enumerate(observations):
        if not first.lifted or first.summary is None:
            continue
        best: float | None = None
        for second_index, second in enumerate(observations):
            if first_index == second_index or second.summary is None:
                continue
            if second.keyframe == first.keyframe:
                continue
            distance = float(
                np.linalg.norm(
                    np.asarray(first.summary["centroid"])
                    - np.asarray(second.summary["centroid"])
                )
            )
            best = distance if best is None else min(best, distance)
        if best is not None:
            values.append(best)
    if not values:
        return None
    return float(np.median(values))


def _evaluation(
    *,
    map_path: str,
    labels: Sequence[str],
    keyframes: Sequence[int],
    observations: Sequence[Observation],
    objects: Sequence[ObjectInstance],
) -> dict[str, Any]:
    """Summarize per-label detection and cross-frame consistency."""
    per_label: dict[str, Any] = {}
    for label in labels:
        rows = [
            observation for observation in observations if observation.label == label
        ]
        lifted = [observation for observation in rows if observation.lifted]
        instances = [instance for instance in objects if instance.label == label]
        scores = [observation.score for observation in rows]
        agreement = _nearest_centroid_distance(rows)
        radii = [
            float(observation.summary["radius"])
            for observation in lifted
            if observation.summary is not None
        ]
        typical_radius = float(np.median(radii)) if radii else None
        per_label[label] = {
            "detections": len(rows),
            "lifted_detections": len(lifted),
            "keyframes": sorted({observation.keyframe for observation in rows}),
            "objects": len(instances),
            "best_score": max(scores) if scores else None,
            "mean_score": float(np.mean(scores)) if scores else None,
            "median_nearest_centroid_distance": agreement,
            "median_nearest_centroid_over_radius": (
                None
                if agreement is None or typical_radius is None or typical_radius <= 0.0
                else agreement / typical_radius
            ),
        }
    return {
        "map_path": map_path,
        "keyframes": list(keyframes),
        "labels": per_label,
        "notes": (
            "Precision and recall need independent ground-truth labels. "
            "median_nearest_centroid_distance measures cross-frame consistency "
            "and is reported per label as a proxy, not as accuracy."
        ),
    }


def ground_map(
    source: MapSource,
    perception: GdinoSam2Client,
    labels: Sequence[str],
    *,
    keyframes: Sequence[int] | None = None,
    max_keyframes: int = DEFAULT_MAX_KEYFRAMES,
    min_score: float = 0.35,
    min_confidence: float = 1.0,
    min_mask_pixels: int = 32,
    max_points_per_detection: int = 4000,
    merge_factor: float = 0.75,
    allow_2d: bool = False,
) -> GroundMapResult:
    """Detect and lift object labels on a built map.

    Args:
        source: Map frames and geometry.
        perception: Grounding DINO + SAM 2 client.
        labels: Open-vocabulary prompts. More than 16 are split across calls.
        keyframes: Explicit archived frame indices; defaults to evenly spaced
            frames.
        max_keyframes: Keyframes to use when ``keyframes`` is omitted.
        min_score: Minimum detection score passed to the perception service.
        min_confidence: Minimum depth confidence for lifted points.
        min_mask_pixels: Minimum pixels for a detection or lifted point set.
        max_points_per_detection: Per-observation cap on semantic points.
        merge_factor: Association radius as a multiple of the observation
            radius.
        allow_2d: Keep 2D observations when the map has no depth (the archive
            backend). Without it, a map with no depth raises.

    Returns:
        The grounded observations, instances, semantic points, masks, and
        evaluation summary.
    """
    cleaned = normalize_labels(labels, max_labels=4 * MAX_LABELS_PER_CALL)
    slugs = [_slug(label) for label in cleaned]
    duplicates = sorted(
        {label for label, slug in zip(cleaned, slugs) if slugs.count(slug) > 1}
    )
    if duplicates:
        raise ValueError(
            f"labels {duplicates} map to the same artifact name; use distinct "
            "words instead of punctuation or case variants"
        )
    if not 0.0 <= float(min_score) <= 1.0:
        raise ValueError("min_score must be between 0 and 1")
    if not math.isfinite(float(min_confidence)) or float(min_confidence) < 0.0:
        raise ValueError("min_confidence must be finite and non-negative")
    if int(min_mask_pixels) < 1:
        raise ValueError("min_mask_pixels must be positive")
    if int(max_points_per_detection) < 1:
        raise ValueError("max_points_per_detection must be positive")
    if not math.isfinite(float(merge_factor)) or float(merge_factor) <= 0.0:
        raise ValueError("merge_factor must be finite and positive")
    if not source.has_depth and not allow_2d:
        raise RuntimeError(
            "this map has no depth; pass allow_2d to keep 2D-only annotations"
        )
    selected = select_keyframes(
        source.frame_count, max_keyframes=max_keyframes, keyframes=keyframes
    )
    label_colors = {
        label: tuple(int(value) for value in LABEL_PALETTE[index % len(LABEL_PALETTE)])
        for index, label in enumerate(cleaned)
    }
    label_index = {label: index for index, label in enumerate(cleaned)}
    canonical_label = {label.casefold(): label for label in cleaned}
    observations: list[Observation] = []
    masks: dict[str, bytes] = {}
    used_masks: set[str] = set()
    preprocess = source.preprocess

    for frame_index in selected:
        frame = source.load(frame_index)
        detections: list[Detection] = []
        for start in range(0, len(cleaned), MAX_LABELS_PER_CALL):
            chunk = cleaned[start : start + MAX_LABELS_PER_CALL]
            result = perception.detect_segment(
                frame.image, labels=chunk, min_score=min_score
            )
            detections.extend(result.detections)
        if not detections:
            continue
        masks[f"overlay_{frame_index:06d}.png"] = _overlay_png(
            frame.image, detections, label_colors
        )
        if preprocess is None and frame.depth is not None:
            # A version-1 map recorded no preprocessing block; the depth grid
            # it stored is the geometry the masks must be warped onto.
            preprocess = PreprocessSpec.from_metadata(
                source.metadata,
                model_size=(int(frame.depth.shape[-2]), int(frame.depth.shape[-1])),
            )
        for detection in detections:
            label = canonical_label.get(detection.label.casefold())
            if label is None:
                continue
            mask = np.asarray(detection.mask, dtype=bool)
            if mask.shape != frame.source_size:
                raise RuntimeError(
                    f"detection mask {mask.shape} does not match frame "
                    f"{frame.source_size} at {frame_index}"
                )
            mask_pixels = int(mask.sum())
            if mask_pixels < min_mask_pixels:
                continue
            base_name = f"{frame_index:06d}_{_slug(label)}"
            mask_name = f"{base_name}.png"
            suffix = 1
            while mask_name in used_masks:
                mask_name = f"{base_name}_{suffix}.png"
                suffix += 1
            used_masks.add(mask_name)
            masks[mask_name] = _mask_png(mask)

            points: np.ndarray | None = None
            summary: dict[str, Any] | None = None
            valid_depth_pixels = 0
            lift_reason: str | None = None
            if source.has_depth and frame.depth is not None:
                if frame.intrinsic is None or frame.camera_to_world is None:
                    lift_reason = "the frame has no intrinsics or camera pose"
                elif preprocess is None:
                    lift_reason = "the map records no preprocessing geometry"
                else:
                    model_mask = preprocess.source_mask_to_model(
                        mask, frame.source_size
                    )
                    if int(model_mask.sum()) < min_mask_pixels:
                        lift_reason = "the mask falls outside the model crop"
                    else:
                        lifted = lift_mask(
                            model_mask,
                            frame.depth,
                            frame.confidence,
                            frame.intrinsic,
                            frame.camera_to_world,
                            min_confidence=min_confidence,
                        )
                        valid_depth_pixels = int(lifted.shape[0])
                        if valid_depth_pixels < min_mask_pixels:
                            lift_reason = "too few valid depth pixels"
                        else:
                            filtered = _robust_filter(lifted)
                            if filtered.shape[0] < min_mask_pixels:
                                lift_reason = (
                                    "too few points survived outlier filtering"
                                )
                            else:
                                points = filtered
                                summary = _summarize_points(filtered)
            else:
                lift_reason = "the map has no depth"

            observations.append(
                Observation(
                    label=label,
                    keyframe=frame_index,
                    timestamp_s=frame.timestamp_s,
                    pushed_base_pose=frame.pushed_base_pose,
                    score=float(detection.score),
                    box=[float(value) for value in detection.box],
                    mask_pixels=mask_pixels,
                    valid_depth_pixels=valid_depth_pixels,
                    mask_name=mask_name,
                    lifted=points is not None,
                    lift_reason=lift_reason,
                    points=points,
                    summary=summary,
                )
            )

    objects = _fuse_instances(observations, merge_factor)
    object_index = {instance.object_id: index for index, instance in enumerate(objects)}
    for instance in objects:
        instance.robot_summary = _robot_frame_summary(instance.points, source.metadata)
    for observation in observations:
        observation.object_index = (
            -1 if observation.object_id is None else object_index[observation.object_id]
        )

    semantic_points: list[np.ndarray] = []
    semantic_labels: list[np.ndarray] = []
    semantic_scores: list[np.ndarray] = []
    semantic_object_ids: list[np.ndarray] = []
    semantic_frame_indices: list[np.ndarray] = []
    for observation in observations:
        if observation.points is None or observation.summary is None:
            continue
        points = observation.points
        if points.shape[0] > max_points_per_detection:
            choice = (
                np.linspace(0, points.shape[0] - 1, max_points_per_detection)
                .round()
                .astype(np.int64)
            )
            points = points[choice]
        count = int(points.shape[0])
        semantic_points.append(points.astype(np.float32))
        semantic_labels.append(
            np.full(count, label_index[observation.label], dtype=np.int32)
        )
        semantic_scores.append(np.full(count, observation.score, dtype=np.float32))
        semantic_object_ids.append(
            np.full(count, observation.object_index, dtype=np.int32)
        )
        semantic_frame_indices.append(
            np.full(count, observation.keyframe, dtype=np.int32)
        )
    if semantic_points:
        points_array = np.concatenate(semantic_points, axis=0)
        labels_array = np.concatenate(semantic_labels, axis=0)
        scores_array = np.concatenate(semantic_scores, axis=0)
        object_ids_array = np.concatenate(semantic_object_ids, axis=0)
        frame_indices_array = np.concatenate(semantic_frame_indices, axis=0)
    else:
        points_array = np.empty((0, 3), dtype=np.float32)
        labels_array = np.empty((0,), dtype=np.int32)
        scores_array = np.empty((0,), dtype=np.float32)
        object_ids_array = np.empty((0,), dtype=np.int32)
        frame_indices_array = np.empty((0,), dtype=np.int32)

    summary = _evaluation(
        map_path=source.map_path,
        labels=cleaned,
        keyframes=selected,
        observations=observations,
        objects=objects,
    )
    return GroundMapResult(
        map_path=source.map_path,
        metadata=source.metadata,
        labels=cleaned,
        keyframes=selected,
        observations=observations,
        objects=objects,
        masks=masks,
        semantic_points=points_array,
        semantic_labels=labels_array,
        semantic_scores=scores_array,
        semantic_object_ids=object_ids_array,
        semantic_frame_indices=frame_indices_array,
        summary=summary,
        preprocess=preprocess,
    )


def save_grounded_map(
    result: GroundMapResult,
    output_dir: str | Path | None = None,
    *,
    write_ply: bool = True,
) -> dict[str, str]:
    """Write the grounded artifacts to a map directory (or another directory)."""
    out = Path(output_dir) if output_dir is not None else Path(result.map_path)
    out.mkdir(parents=True, exist_ok=True)
    annotations = out / ANNOTATIONS_DIR
    annotations.mkdir(parents=True, exist_ok=True)
    for name, data in result.masks.items():
        (annotations / name).write_bytes(data)
    objects_path = out / OBJECTS_FILE
    objects_path.write_text(json.dumps(result.objects_payload(), indent=2))
    semantic_path = out / SEMANTIC_POINTS_FILE
    semantic_path.write_bytes(result.semantic_npz_bytes())
    written = {
        "objects": str(objects_path),
        "semantic_points": str(semantic_path),
        "annotations": str(annotations),
        "evaluation": str(out / "evaluation.json"),
    }
    (out / "evaluation.json").write_text(json.dumps(result.summary, indent=2))
    if write_ply:
        ply_path = out / SEMANTIC_PLY_FILE
        ply_path.write_bytes(result.semantic_ply_bytes())
        written["semantic_ply"] = str(ply_path)
    return written


def save_grounded_map_via_client(
    result: GroundMapResult,
    client: LingBotMapClient,
) -> dict[str, Any]:
    """Write grounded artifacts into a map that lives on the mapping server.

    Offline callers that read a map directory on the local filesystem use
    :func:`save_grounded_map`; this variant sends the same artifacts over RPC
    so a client can label a map whose files it cannot reach directly.
    """
    has_points = result.semantic_points.shape[0] > 0
    return client.save_annotations(
        result.map_path,
        objects=result.objects_payload(),
        masks=result.masks,
        semantic_points_npz=result.semantic_npz_bytes() if has_points else None,
        semantic_points_ply=result.semantic_ply_bytes() if has_points else None,
    )


def _camera_array(
    metadata: dict[str, Any], name: str, trailing: tuple[int, ...]
) -> np.ndarray | None:
    cameras = metadata.get("cameras")
    if not isinstance(cameras, dict):
        return None
    values = cameras.get(name)
    if values is None:
        return None
    array = np.asarray(values, dtype=np.float64)
    while array.ndim > len(trailing) + 1 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != len(trailing) + 1 or array.shape[1:] != trailing:
        raise ValueError(
            f"cameras.{name} must be [N,{','.join(str(v) for v in trailing)}], "
            f"got {array.shape}"
        )
    return array


def _model_grid(serialized: np.ndarray) -> tuple[int, int] | None:
    """Return a serialized depth array's ``(height, width)`` grid."""
    shape = tuple(int(value) for value in serialized.shape)
    while shape and shape[-1] == 1:
        shape = shape[:-1]
    if len(shape) < 2:
        return None
    return shape[-2], shape[-1]


def select_serialized_frame(
    array: np.ndarray, index: int, frames: int, name: str
) -> np.ndarray:
    """Select one frame from a serialized depth or confidence array."""
    if array.ndim == 5:
        if array.shape[0] != 1 or array.shape[1] != frames:
            raise ValueError(f"unexpected {name} shape: {array.shape}")
        selected = array[0, index]
    elif array.ndim == 4:
        if array.shape[0] == 1 and array.shape[1] == frames:
            selected = array[0, index]
        elif array.shape[0] == frames:
            selected = array[index]
        else:
            raise ValueError(f"unexpected {name} shape: {array.shape}")
    elif array.ndim == 3 and array.shape[0] == frames:
        selected = array[index]
    else:
        raise ValueError(f"unexpected {name} shape: {array.shape}")
    selected = np.asarray(selected, dtype=np.float32)
    if selected.ndim == 3 and selected.shape[-1] == 1:
        selected = selected[..., 0]
    return selected


def _semantic_ply_bytes(
    points: np.ndarray, labels: np.ndarray, label_count: int
) -> bytes:
    """Serialize labeled points as a binary little-endian PLY."""
    colors = LABEL_PALETTE[np.clip(labels, 0, len(LABEL_PALETTE) - 1)].astype(np.uint8)
    dtype = np.dtype(
        [
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
            ("label", "u1"),
        ]
    )
    vertices = np.empty(points.shape[0], dtype=dtype)
    vertices["x"] = points[:, 0]
    vertices["y"] = points[:, 1]
    vertices["z"] = points[:, 2]
    vertices["red"] = colors[:, 0]
    vertices["green"] = colors[:, 1]
    vertices["blue"] = colors[:, 2]
    vertices["label"] = np.clip(labels, 0, min(label_count, 255)).astype(np.uint8)
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        "comment RPent semantic map: red/green/blue encode the label color\n"
        f"element vertex {points.shape[0]}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "property uchar label\n"
        "end_header\n"
    ).encode("ascii")
    return header + vertices.tobytes()


def _parse_keyframes(value: str | None) -> list[int] | None:
    if value is None:
        return None
    parts = [part for part in value.replace(",", " ").split() if part]
    if not parts:
        raise ValueError("keyframes must not be empty")
    try:
        return [int(part) for part in parts]
    except ValueError as exc:
        raise ValueError(f"invalid keyframes: {value!r}") from exc


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Ground object labels on a built LingBot-Map map"
    )
    parser.add_argument("--map-path", required=True, help="built map directory")
    parser.add_argument("--labels", nargs="+", required=True, help="object prompts")
    parser.add_argument(
        "--perception-endpoint",
        default="http://127.0.0.1:8115",
        help="Grounding DINO + SAM 2 server URL",
    )
    parser.add_argument(
        "--keyframes",
        default=None,
        help="comma- or space-separated frame indices (default: evenly spaced)",
    )
    parser.add_argument("--max-keyframes", type=int, default=DEFAULT_MAX_KEYFRAMES)
    parser.add_argument("--min-score", type=float, default=0.35)
    parser.add_argument("--min-confidence", type=float, default=1.0)
    parser.add_argument("--min-mask-pixels", type=int, default=32)
    parser.add_argument("--max-points-per-detection", type=int, default=4000)
    parser.add_argument("--merge-factor", type=float, default=0.75)
    parser.add_argument(
        "--output-dir",
        default=None,
        help="artifact directory (default: the map directory)",
    )
    parser.add_argument("--no-ply", action="store_true")
    parser.add_argument(
        "--allow-2d",
        action="store_true",
        help="keep 2D annotations when the map has no depth",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Run the offline grounder and write artifacts into the map directory."""
    args = _build_argparser().parse_args(argv)
    try:
        keyframes = _parse_keyframes(args.keyframes)
        source = LocalMapSource(args.map_path)
        perception = GdinoSam2Client(
            HttpRpcClient(args.perception_endpoint), timeout_s=300.0
        )
        result = ground_map(
            source,
            perception,
            args.labels,
            keyframes=keyframes,
            max_keyframes=args.max_keyframes,
            min_score=args.min_score,
            min_confidence=args.min_confidence,
            min_mask_pixels=args.min_mask_pixels,
            max_points_per_detection=args.max_points_per_detection,
            merge_factor=args.merge_factor,
            allow_2d=args.allow_2d,
        )
        written = save_grounded_map(result, args.output_dir, write_ply=not args.no_ply)
    except Exception as exc:
        logger.error("grounding failed: %s", exc)
        raise SystemExit(1) from exc
    logger.info(
        "grounded %d detections into %d objects on %d keyframes",
        len(result.observations),
        len(result.objects),
        len(result.keyframes),
    )
    for name, path in written.items():
        logger.info("wrote %s: %s", name, path)


if __name__ == "__main__":
    main()
