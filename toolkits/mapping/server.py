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

"""RPC server owning a LingBot-Map workspace mapping session.

Run manually with::

    LINGBOTMAP_OUTPUT_DIR=/path/to/maps \\
    LINGBOTMAP_CHECKPOINT_PATH=/path/to/lingbot-map.pt \\
        python -m toolkits.mapping.server \\
        --transport http --host 127.0.0.1 --port 8116

RPent can spawn this process as a dashboard runtime component, or attach to a
server started by hand through a ``--map-endpoint`` URL. The service exposes
``map.start_session`` / ``map.push_frame`` / ``map.build`` / ``map.stop_session``
``/ map.load_metadata`` RPC methods over either HTTP or socket transport.

The LingBot-Map builder is wrapped behind the small :class:`MapBackend`
protocol so the concrete integration stays in one adapter function. With the
``lingbot-map`` package and a model checkpoint
(:class:`GctStreamBackend`), pushed frames are reconstructed into camera
poses, per-frame depth, and a world point cloud. Without them the reference
:class:`FrameArchiveBackend` records frames and poses and serializes them as
the map artifact, which keeps the RPC contract and the tool flow verifiable
offline.

Reconstructions are defined only up to a similarity: the model fixes the world
frame to its first frames and its scale is arbitrary. When every pushed frame
carries a robot base pose, :class:`GctStreamBackend` fits that similarity
(Umeyama) from the model camera trajectory to the pushed positions and writes
the map in the robot's metric frame. Maps built from frames without poses stay
in the model frame with arbitrary scale, which is enough to inspect the
reconstruction but not to localize with.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import logging
import os
import re
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from PIL import Image

from rpent.utils.logging import get_logger
from rpent.utils.rpc import RpcFacade

logger = get_logger("lingbotmap_server")

#: Version of the map artifact contract written by this server.
MAP_SCHEMA_VERSION = 2
#: Model input geometry used by the GCT streaming backend.
MODEL_INPUT_SIZE = 518
MODEL_PATCH_SIZE = 14
#: Map artifact names shared by the server, client, and offline grounder.
KEYFRAME_INDEX_FILE = "keyframe_index.json"
CAMERAS_C2W_FILE = "cameras_c2w.npy"
CAMERAS_C2W_ROBOT_FRAME_FILE = "cameras_c2w_robot_frame.npy"
OBJECTS_FILE = "objects.json"
SEMANTIC_POINTS_FILE = "semantic_points.npz"
SEMANTIC_PLY_FILE = "semantic_points.ply"
ANNOTATIONS_DIR = "annotations"

#: Annotation file names are written below ``annotations/``. Restricting them
#: to a flat, PNG-only name keeps a remote client from writing elsewhere in the
#: map directory.
_ANNOTATION_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,120}\.png$")
#: Map names become one directory below the output root, so they are restricted
#: to a single flat path segment.
_MAP_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,120}$")
_MAX_ANNOTATION_BYTES = 32 * 1024 * 1024
_MAX_SEMANTIC_ARTIFACT_BYTES = 96 * 1024 * 1024

#: Frames retained in one mapping session. The GCT backend stacks every
#: preprocessed frame on the GPU in ``build``, and a 24 GB device runs out of
#: memory above roughly 300 frames at the 392x518 model grid. Raise it with
#: ``--max-frames`` on hosts with more memory.
DEFAULT_MAX_SESSION_FRAMES = 300

#: Minimum spread of the pushed base positions, in meters, for the scale fit.
#: Below it the similarity is dominated by pose noise and reconstruction error.
MIN_ALIGNMENT_SPREAD_M = 0.05
#: Minimum ratio between the second and first singular values of the pushed
#: positions. A straight-line sweep fixes the direction it travelled but leaves
#: roll about that line unconstrained, so it cannot anchor the map orientation.
MIN_ALIGNMENT_AREA_RATIO = 0.02
#: Maximum residual of the fitted camera trajectory against the pushed
#: positions, as a fraction of the trajectory spread. A camera offset from the
#: base axis stays within it; a reconstruction that the poses do not explain
#: (for example one mirrored relative to them) exceeds it.
MAX_ALIGNMENT_RESIDUAL_RATIO = 0.25


@dataclass(frozen=True)
class BaseAlignment:
    """Similarity from the model world frame to the pushed base-pose frame.

    The transform is ``points_robot = scale * rotation @ points_model +
    translation``, so ``scale`` is robot-map units (meters) per model unit and
    a model depth map multiplied by it becomes metric. ``rms_error_m`` is the
    root-mean-square residual of the camera trajectory against the pushed base
    positions. The fit uses camera centres as the model side and base
    positions as the robot side, so a rigidly mounted camera offsets them by a
    vector that rotates with the base yaw; that offset appears in the residual
    and can bias the scale slightly, which is why a large residual relative to
    the trajectory spread rejects the fit. Mount the camera close to the base
    axis when metric scale matters.
    """

    scale: float
    rotation: np.ndarray
    translation: np.ndarray
    rms_error_m: float
    frames: int
    unposed_frames: int = 0


class MapBackend(Protocol):
    """The slice of LingBot-Map the mapping tool depends on."""

    #: Root directory holding one subdirectory per built map.
    output_dir: Path

    def start(self, map_name: str) -> None:
        """Open a new mapping run named ``map_name``."""

    def push_frame(
        self, image: np.ndarray, pose: np.ndarray | None, timestamp_s: float
    ) -> None:
        """Record one synchronized RGB frame and base pose."""

    def build(self) -> dict[str, Any]:
        """Finalize the map; return artifact metadata including ``map_path``."""

    def reset(self) -> None:
        """Release every buffer retained for the current mapping session."""


class FrameArchiveBackend:
    """Reference backend that archives frames and poses as the map artifact.

    The archive is a directory with ``frames/*.png``, ``poses.json``, and a
    ``map.json`` metadata file. It exists so the mapping RPC contract and the
    ``build_map`` tool flow run without LingBot-Map installed; a real
    LingBot-Map adapter implements :class:`MapBackend` instead.
    """

    def __init__(self, output_dir: str) -> None:
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._map_name = ""
        self._frames: list[np.ndarray] = []
        self._poses: list[list[float] | None] = []
        self._timestamps: list[float] = []

    @property
    def output_dir(self) -> Path:
        """Root directory holding one subdirectory per built map."""
        return self._output_dir

    def start(self, map_name: str) -> None:
        self.reset()
        self._map_name = map_name

    def reset(self) -> None:
        """Drop the archived frames and poses of the current session."""
        self._map_name = ""
        self._frames = []
        self._poses = []
        self._timestamps = []

    def push_frame(
        self, image: np.ndarray, pose: np.ndarray | None, timestamp_s: float
    ) -> None:
        frame = np.asarray(image)
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(f"pushed frame must be RGB [H,W,3], got {frame.shape}")
        self._frames.append(frame)
        self._poses.append(
            np.asarray(pose, dtype=np.float64).tolist() if pose is not None else None
        )
        self._timestamps.append(float(timestamp_s))

    def build(self) -> dict[str, Any]:
        if not self._map_name:
            raise RuntimeError("no mapping run started")
        map_dir = self._output_dir / self._map_name
        frames_dir = map_dir / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        for index, frame in enumerate(self._frames):
            Image.fromarray(np.asarray(frame, dtype=np.uint8)).save(
                frames_dir / f"{index:06d}.png"
            )
        poses = [
            {"pose": pose, "timestamp_s": timestamp}
            for pose, timestamp in zip(self._poses, self._timestamps, strict=True)
        ]
        (map_dir / "poses.json").write_text(json.dumps(poses))
        source_sizes = [frame.shape[:2] for frame in self._frames]
        metadata = _build_archive_metadata(
            map_name=self._map_name,
            timestamps=self._timestamps,
            poses=self._poses,
            source_sizes=source_sizes,
        )
        _write_keyframe_index(
            map_dir,
            _frame_entries(
                timestamps=self._timestamps,
                poses=self._poses,
                source_sizes=source_sizes,
                model_keyframe_interval=None,
            ),
        )
        (map_dir / "map.json").write_text(json.dumps(metadata))
        logger.info("built map %s with %d frames", self._map_name, len(self._frames))
        return {
            "map_path": str(map_dir),
            "frames": len(self._frames),
            "metadata": metadata,
        }


def build_lingbot_backend(output_dir: str) -> MapBackend:
    """Resolve the LingBot-Map backend for this server run.

    The GCT streaming model is used when the ``lingbot-map`` package is
    importable and ``LINGBOTMAP_CHECKPOINT_PATH`` points at a checkpoint;
    otherwise the reference archive backend keeps the service running.
    """
    checkpoint = os.environ.get("LINGBOTMAP_CHECKPOINT_PATH")
    if checkpoint:
        try:
            return GctStreamBackend(output_dir, checkpoint)
        except ImportError as exc:
            logger.warning(
                "LINGBOTMAP_CHECKPOINT_PATH is set but lingbot-map is not "
                "installed (%s); falling back to the frame-archive backend",
                exc,
            )
    else:
        logger.info(
            "LINGBOTMAP_CHECKPOINT_PATH is not set; using the frame-archive backend"
        )
    return FrameArchiveBackend(output_dir)


def _estimate_similarity(
    source: np.ndarray, target: np.ndarray
) -> tuple[float, np.ndarray, np.ndarray]:
    """Least-squares similarity mapping ``source`` onto ``target``.

    Implements Umeyama's closed-form solution for
    ``target ~= scale * rotation @ source + translation``.
    """
    count = source.shape[0]
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    source_centred = source - source_mean
    variance = float((source_centred**2).sum() / count)
    covariance = (target - target_mean).T @ source_centred / count
    left, singular, right = np.linalg.svd(covariance)
    correction = np.ones(3)
    if np.linalg.det(left) * np.linalg.det(right) < 0:
        correction[2] = -1.0
    rotation = left @ np.diag(correction) @ right
    scale = float(np.dot(singular, correction) / variance)
    translation = target_mean - scale * rotation @ source_mean
    return scale, rotation, translation


def _estimate_base_alignment(
    extrinsic: np.ndarray, poses: Sequence[Sequence[float] | None]
) -> tuple[BaseAlignment | None, str]:
    """Fit the model world frame to the pushed robot base poses.

    ``extrinsic`` holds the model's world-to-camera extrinsics as ``[S,3,4]``
    and ``poses`` the per-frame base poses pushed by the robot (``[x, y, yaw,
    roll, pitch, yaw_head]``, or ``None``). The fit uses the camera centres as
    sources and the pushed ``(x, y)`` positions as targets, which puts the
    reconstructed trajectory on the floor plane with ``+z`` up and recovers the
    metric scale. Frames pushed without a pose are left out of the fit, so
    intermittent odometry does not discard the whole alignment.

    Returns ``(alignment, reason)``. ``alignment`` is ``None`` and ``reason``
    explains why when the poses cannot anchor a trustworthy fit; callers keep
    the map in the model frame in that case.
    """
    extrinsics = np.asarray(extrinsic, dtype=np.float64)
    if extrinsics.ndim != 3 or extrinsics.shape[1:] != (3, 4):
        raise ValueError(f"extrinsic must be [S,3,4], got {extrinsics.shape}")
    frames = extrinsics.shape[0]
    if len(poses) != frames:
        return None, f"{len(poses)} poses were pushed for {frames} frames"

    posed = [index for index, pose in enumerate(poses) if pose is not None]
    missing = frames - len(posed)
    if len(posed) < 3:
        return None, (
            f"need at least 3 posed frames to fit a similarity, got {len(posed)} "
            f"of {frames}"
        )

    base = np.asarray(
        [[float(poses[index][0]), float(poses[index][1]), 0.0] for index in posed],
        dtype=np.float64,
    )

    centred = base - base.mean(axis=0)
    spread = float(np.linalg.norm(centred, axis=1).max())
    if spread < MIN_ALIGNMENT_SPREAD_M:
        return None, (
            f"the robot moved {spread:.3f} m, below the {MIN_ALIGNMENT_SPREAD_M:.2f} m "
            "needed to fix the scale"
        )
    singular = np.linalg.svd(centred, compute_uv=False)
    if singular[0] <= 0.0 or singular[1] < MIN_ALIGNMENT_AREA_RATIO * singular[0]:
        return (
            None,
            "the robot drove in a straight line, which leaves the map roll free",
        )

    rotations = extrinsics[posed, :3, :3]
    centres = camera_to_world(extrinsics)[posed, :3, 3]
    if not np.isfinite(centres).all():
        return None, "the reconstructed camera poses contain non-finite values"
    if float(np.linalg.norm(centres - centres.mean(axis=0), axis=1).max()) <= 0.0:
        return None, "the reconstructed camera trajectory is static"

    scale, rotation, translation = _estimate_similarity(centres, base)
    if not (np.isfinite(scale) and scale > 0.0):
        return None, "the similarity fit produced a degenerate scale"
    residual = scale * centres @ rotation.T + translation - base
    rms = float(np.sqrt((residual**2).sum(axis=1).mean()))
    if not np.isfinite(rms):
        return None, "the similarity fit is not finite"
    if rms > MAX_ALIGNMENT_RESIDUAL_RATIO * spread:
        return None, (
            f"the fitted similarity leaves a {rms:.3f} m residual on a {spread:.3f} m "
            "trajectory, so the base poses do not explain the reconstruction"
        )
    # A planar trajectory fixes the in-plane orientation but not which side of
    # its plane is up. Upright cameras settle it: camera y (OpenCV convention)
    # points down, so it must end up pointing down in the robot frame.
    mapped_down = rotations[:, 1, :] @ rotation.T
    if float(mapped_down[:, 2].mean()) >= 0.0:
        return None, (
            "the fitted similarity turns the reconstruction upside down, so the "
            "camera orientations do not match the pushed poses"
        )
    return (
        BaseAlignment(
            scale=scale,
            rotation=rotation,
            translation=translation,
            rms_error_m=rms,
            frames=len(posed),
            unposed_frames=missing,
        ),
        "",
    )


def _apply_alignment(points: np.ndarray, alignment: BaseAlignment) -> np.ndarray:
    """Map ``[...,3]`` points from the model frame into the robot frame."""
    return alignment.scale * points @ alignment.rotation.T + alignment.translation


def camera_to_world(extrinsic: np.ndarray) -> np.ndarray:
    """Convert ``[S,3,4]`` world-to-camera extrinsics to 4x4 camera-to-world."""
    rotations = np.swapaxes(extrinsic[..., :3, :3], -1, -2)
    camera_to_world = np.tile(np.eye(4), (*extrinsic.shape[:-2], 1, 1))
    camera_to_world[..., :3, :3] = rotations
    camera_to_world[..., :3, 3] = -np.einsum(
        "...ij,...j->...i", rotations, extrinsic[..., :3, 3]
    )
    return camera_to_world


def _model_keyframe_interval(frames: int) -> int:
    """Return the GCT keyframe spacing used for ``frames`` pushed frames."""
    return 1 if frames <= 320 else (frames + 319) // 320


def _select_serialized_frame(
    array: np.ndarray, index: int, frames: int, name: str
) -> np.ndarray:
    """Select one frame from a serialized depth or confidence array.

    Reconstruction arrays are saved as ``(1,N,H,W,1)`` for depth and
    ``(N,H,W)`` for confidence. Reading through a memory map keeps the
    per-frame RPC payload to one frame.
    """
    if array.ndim == 5:
        if array.shape[0] != 1 or array.shape[1] != frames:
            raise ValueError(f"unexpected {name} shape: {array.shape}")
        return array[0, index]
    if array.ndim == 4:
        if array.shape[0] == 1 and array.shape[1] == frames:
            return array[0, index]
        if array.shape[0] == frames:
            return array[index]
        raise ValueError(f"unexpected {name} shape: {array.shape}")
    if array.ndim == 3 and array.shape[0] == frames:
        return array[index]
    raise ValueError(f"unexpected {name} shape: {array.shape}")


def _frame_entries(
    *,
    timestamps: Sequence[float],
    poses: Sequence[Sequence[float] | None],
    source_sizes: Sequence[Sequence[int]],
    model_keyframe_interval: int | None,
) -> list[dict[str, Any]]:
    """Describe every archived frame for retrieval and offline grounding."""
    if not (len(timestamps) == len(poses) == len(source_sizes)):
        raise ValueError("timestamps, poses, and source sizes must have equal length")
    entries: list[dict[str, Any]] = []
    for index, (timestamp, pose, size) in enumerate(
        zip(timestamps, poses, source_sizes, strict=True)
    ):
        height, width = int(size[0]), int(size[1])
        entries.append(
            {
                "index": index,
                "timestamp_s": float(timestamp),
                "image": f"frames/{index:06d}.png",
                "source_size": [height, width],
                "pushed_base_pose": None if pose is None else [float(v) for v in pose],
                "model_keyframe": (
                    None
                    if model_keyframe_interval is None
                    else index % model_keyframe_interval == 0
                ),
            }
        )
    return entries


def _write_keyframe_index(map_dir: Path, entries: list[dict[str, Any]]) -> None:
    """Persist the per-frame index next to the archived frames."""
    (map_dir / KEYFRAME_INDEX_FILE).write_text(json.dumps(entries))


def _pushed_poses_summary(
    poses: Sequence[Sequence[float] | None],
) -> dict[str, Any]:
    """Describe ``poses.json`` without duplicating its entries in ``map.json``."""
    with_pose = sum(pose is not None for pose in poses)
    return {
        "path": "poses.json",
        "count": len(poses),
        "with_pose": with_pose,
        "without_pose": len(poses) - with_pose,
        "status": "present" if with_pose else "absent",
    }


def _estimate_gravity(extrinsic: np.ndarray) -> dict[str, Any]:
    """Estimate the scene's up axis from the reconstructed camera orientations.

    Camera ``+y`` points down in the OpenCV convention, so the mean negated
    camera y axis is the scene up direction. The trajectory-plane normal is
    reported alongside it: a forward-facing sweep should have a high
    ``trajectory_plane_agreement``.
    """
    c2w = camera_to_world(extrinsic)
    rotations = c2w[:, :3, :3]
    centres = c2w[:, :3, 3]
    down = rotations[:, :, 1].mean(axis=0)
    norm = float(np.linalg.norm(down))
    if not np.isfinite(down).all() or norm < 1e-6:
        return {"status": "unavailable", "reason": "camera orientations are degenerate"}
    up = -down / norm
    gravity: dict[str, Any] = {
        "status": "estimated",
        "up": up.tolist(),
        "method": "mean_camera_y_axis",
    }
    if centres.shape[0] >= 3:
        centred = centres - centres.mean(axis=0)
        _, singular, right = np.linalg.svd(centred, full_matrices=False)
        if singular.shape[0] >= 3 and singular[1] > 1e-9:
            normal = right[2]
            if float(normal @ up) < 0.0:
                normal = -normal
            gravity["trajectory_plane_normal"] = normal.tolist()
            gravity["trajectory_plane_agreement"] = float(abs(normal @ up))
    return gravity


def _estimate_floor(
    points: np.ndarray | None, up: np.ndarray, camera_centres: np.ndarray
) -> dict[str, Any]:
    """Estimate the dominant low surface along ``up`` in the point cloud.

    Only a subsample is examined, so the estimate is cheap even for a full
    multi-gigabyte point cloud. The returned ``offset`` is the signed distance
    of the floor plane from the model origin along ``normal``; a point ``p``
    lies on the floor when ``dot(p, normal) == offset``.
    """
    if points is None:
        return {"status": "unavailable", "reason": "the map has no point cloud"}
    if points.ndim != 4 or points.shape[-1] != 3:
        return {
            "status": "unavailable",
            "reason": "the point cloud has an unexpected shape",
        }
    frame_step = max(1, points.shape[0] // 32)
    pixel_step = max(1, points.shape[1] // 64)
    sample = np.asarray(points[::frame_step, ::pixel_step, ::pixel_step])
    sample = sample.reshape(-1, 3)
    sample = sample[np.isfinite(sample).all(axis=1)]
    if sample.shape[0] < 100:
        return {"status": "unavailable", "reason": "too few finite points"}

    heights = sample @ up
    camera_z = float(np.median(camera_centres @ up))
    if not np.isfinite(camera_z):
        return {"status": "unavailable", "reason": "camera heights are not finite"}
    low = float(np.percentile(heights, 0.5))
    high = min(camera_z, float(np.percentile(heights, 99.5)))
    if not high > low:
        return {
            "status": "unavailable",
            "reason": "no points lie below the camera plane",
        }
    candidates = heights[(heights >= low) & (heights <= high)]
    histogram, edges = np.histogram(candidates, bins=160)
    best = int(np.argmax(histogram))
    width = float(edges[1] - edges[0])
    floor_offset = float((edges[best] + edges[best + 1]) / 2.0)
    in_bin = candidates[np.abs(candidates - floor_offset) <= width]
    if in_bin.size:
        floor_offset = float(np.median(in_bin))
    camera_height = camera_z - floor_offset
    support = float(histogram[best] / max(1, candidates.size))
    if camera_height <= 0.0 or support < 0.005:
        return {
            "status": "unavailable",
            "reason": "no dominant floor surface was found below the camera",
        }
    return {
        "status": "estimated",
        "normal": up.tolist(),
        "offset": floor_offset,
        "camera_height": camera_height,
        "support": support,
        "method": "gravity_histogram_low_mode",
    }


def _scale_metadata(
    alignment: BaseAlignment | None, floor: dict[str, Any]
) -> dict[str, Any]:
    """Describe whether the map's distances are metric and where scale came from."""
    if alignment is not None:
        block: dict[str, Any] = {
            "status": "metric",
            "metric": True,
            "meters_per_unit": float(alignment.scale),
            "source": "base_pose_alignment",
        }
        if floor.get("status") == "estimated":
            block["camera_height_m"] = float(floor["camera_height"] * alignment.scale)
        return block
    return {
        "status": "arbitrary",
        "metric": False,
        "meters_per_unit": None,
        "source": None,
        "note": (
            "no base poses anchored the reconstruction; distances are in arbitrary "
            "model units"
        ),
    }


def _build_gct_metadata(
    *,
    map_name: str,
    extrinsic: np.ndarray,
    intrinsic: np.ndarray,
    source_sizes: Sequence[Sequence[int]],
    timestamps: Sequence[float],
    poses: Sequence[Sequence[float] | None],
    keyframe_interval: int,
    points: np.ndarray | None,
    input_size: tuple[int, int],
    alignment: BaseAlignment | None,
    alignment_reason: str,
) -> dict[str, Any]:
    """Build the version-2 metadata contract for a reconstructed map.

    ``input_size`` is the actual model grid ``(height, width)`` of the
    reconstruction. It equals the nominal square input only for sources whose
    resized height reaches that height; landscape sources stay shorter, and
    consumers use this value to map source masks onto the depth grid.

    ``extrinsic`` is ``[N,3,4]`` world-to-camera in the model frame and
    ``intrinsic`` is ``[N,3,3]`` in pixels. The camera-to-world poses,
    preprocessing geometry, gravity direction, floor plane, and scale
    provenance are recorded so later localization and grounding do not have to
    rediscover them from the reconstruction.
    """
    extrinsics = np.asarray(extrinsic, dtype=np.float64)
    intrinsics = np.asarray(intrinsic, dtype=np.float64)
    if extrinsics.ndim != 3 or extrinsics.shape[1:] != (3, 4):
        raise ValueError(f"extrinsic must be [N,3,4], got {extrinsics.shape}")
    if intrinsics.shape != (extrinsics.shape[0], 3, 3):
        raise ValueError(
            f"intrinsic must be [N,3,3] matching extrinsic, got {intrinsics.shape}"
        )
    if len(source_sizes) != extrinsics.shape[0]:
        raise ValueError("one source size is required per frame")

    c2w = camera_to_world(extrinsics)
    gravity = _estimate_gravity(extrinsics)
    # ``alignment`` transforms cameras and points into the robot frame; the
    # gravity and floor estimates stay in the model frame so consumers know
    # which coordinates their normal, offset, and camera height describe.
    gravity["frame"] = "model"
    if gravity.get("status") == "estimated":
        up = np.asarray(gravity["up"], dtype=np.float64)
        floor = _estimate_floor(points, up, c2w[:, :3, 3])
    else:
        floor = {
            "status": "unavailable",
            "reason": "gravity could not be estimated from the camera orientations",
        }
    floor["frame"] = "model"
    entries = _frame_entries(
        timestamps=timestamps,
        poses=poses,
        source_sizes=source_sizes,
        model_keyframe_interval=keyframe_interval,
    )
    return {
        "schema_version": MAP_SCHEMA_VERSION,
        "backend": "gct_stream",
        "map_name": map_name,
        "frame": "model",
        "frames": int(extrinsics.shape[0]),
        "has_depth": points is not None,
        "has_world_points": points is not None,
        "preprocess": {
            "resize_width": MODEL_INPUT_SIZE,
            "patch_size": MODEL_PATCH_SIZE,
            "crop": "center",
            "input_size": [int(input_size[0]), int(input_size[1])],
        },
        "cameras": {
            "frame": "model",
            "extrinsic_convention": "world_to_camera",
            "intrinsic_convention": "pixel",
            "shape": [int(extrinsics.shape[0]), 3, 4],
            "extrinsic": extrinsics.tolist(),
            "intrinsic_shape": [int(intrinsics.shape[0]), 3, 3],
            "intrinsic": intrinsics.tolist(),
        },
        "camera_to_world": {
            "path": CAMERAS_C2W_FILE,
            "convention": "camera_to_world",
            "shape": list(c2w.shape),
        },
        "gravity": gravity,
        "floor": floor,
        "scale": _scale_metadata(alignment, floor),
        "keyframes": {
            "path": KEYFRAME_INDEX_FILE,
            "count": len(entries),
            "model_keyframe_interval": int(keyframe_interval),
        },
        "pushed_poses": _pushed_poses_summary(poses),
        "alignment": (
            {
                "status": "aligned",
                "frame": "robot_base",
                "scale": alignment.scale,
                "rotation": alignment.rotation.tolist(),
                "translation": alignment.translation.tolist(),
                "rms_error_m": alignment.rms_error_m,
                "frames": alignment.frames,
                "unposed_frames": alignment.unposed_frames,
            }
            if alignment is not None
            else {"status": "unavailable", "reason": alignment_reason}
        ),
    }


def _build_archive_metadata(
    *,
    map_name: str,
    timestamps: Sequence[float],
    poses: Sequence[Sequence[float] | None],
    source_sizes: Sequence[Sequence[int]],
) -> dict[str, Any]:
    """Build the version-2 metadata contract for the frame-archive backend."""
    entries = _frame_entries(
        timestamps=timestamps,
        poses=poses,
        source_sizes=source_sizes,
        model_keyframe_interval=None,
    )
    return {
        "schema_version": MAP_SCHEMA_VERSION,
        "backend": "frame_archive",
        "map_name": map_name,
        "frame": "model",
        "frames": len(entries),
        "has_depth": False,
        "has_world_points": False,
        "preprocess": None,
        "cameras": None,
        "camera_to_world": None,
        "gravity": None,
        "floor": None,
        "scale": {
            "status": "arbitrary",
            "metric": False,
            "meters_per_unit": None,
            "source": None,
            "note": "the archive backend records no reconstruction geometry",
        },
        "keyframes": {
            "path": KEYFRAME_INDEX_FILE,
            "count": len(entries),
            "model_keyframe_interval": None,
        },
        "pushed_poses": _pushed_poses_summary(poses),
        "alignment": {
            "status": "unavailable",
            "reason": "the archive backend does not reconstruct camera poses",
        },
    }


class GctStreamBackend:
    """Map backend running the real LingBot-Map GCT streaming model.

    Frames pushed during a session are preprocessed on arrival (resize to the
    model's input size, patch-aligned, cropped) and stacked; :meth:`build`
    runs :meth:`GCTStream.inference_streaming` over them and serializes the
    predictions — camera poses, per-frame depth, and a world point cloud
    unprojected from that depth — plus the archived source frames as the map
    artifact. When every pushed frame carries a base pose, the artifact also
    includes the same poses and points transformed into the robot's metric
    frame together with the fitted similarity.
    """

    def __init__(self, output_dir: str, checkpoint: str) -> None:
        resolved = Path(checkpoint).expanduser().resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"LingBot-Map checkpoint not found: {resolved}")
        import torch
        from lingbot_map.models.gct_stream import GCTStream

        self._torch = torch
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._map_name = ""
        self._preprocessed: list[Any] = []
        self._source_frames: list[np.ndarray] = []
        self._poses: list[list[float] | None] = []
        self._timestamps: list[float] = []

        logger.info("loading LingBot-Map GCTStream: %s", resolved)
        self._model = GCTStream(
            img_size=MODEL_INPUT_SIZE,
            patch_size=MODEL_PATCH_SIZE,
            enable_3d_rope=True,
            max_frame_num=1024,
            kv_cache_sliding_window=64,
            kv_cache_scale_frames=8,
            kv_cache_cross_frame_special=True,
            kv_cache_include_scale_frames=True,
            use_sdpa=True,
            camera_num_iterations=4,
        )
        ckpt = torch.load(resolved, map_location=self._device, weights_only=False)
        state_dict = ckpt.get("model", ckpt)
        missing, unexpected = self._model.load_state_dict(state_dict, strict=False)
        if missing:
            logger.warning("LingBot-Map checkpoint missing %d keys", len(missing))
        if unexpected:
            logger.warning(
                "LingBot-Map checkpoint has %d unexpected keys", len(unexpected)
            )
        self._model = self._model.to(self._device).eval()
        logger.info("LingBot-Map model loaded on %s", self._device)

    @property
    def output_dir(self) -> Path:
        """Root directory holding one subdirectory per built map."""
        return self._output_dir

    def start(self, map_name: str) -> None:
        self.reset()
        self._map_name = map_name

    def reset(self) -> None:
        """Drop the preprocessed frames, source frames, poses, and timestamps."""
        self._map_name = ""
        self._preprocessed = []
        self._source_frames = []
        self._poses = []
        self._timestamps = []

    def push_frame(
        self, image: np.ndarray, pose: np.ndarray | None, timestamp_s: float
    ) -> None:
        frame = np.asarray(image)
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(f"pushed frame must be RGB [H,W,3], got {frame.shape}")
        self._source_frames.append(frame)
        self._poses.append(
            np.asarray(pose, dtype=np.float64).tolist() if pose is not None else None
        )
        self._timestamps.append(float(timestamp_s))
        self._preprocessed.append(self._preprocess(frame))

    def build(self) -> dict[str, Any]:
        if not self._map_name:
            raise RuntimeError("no mapping run started")
        torch = self._torch
        images = torch.stack(self._preprocessed, dim=0).to(self._device)
        num_frames = int(images.shape[0])
        keyframe_interval = _model_keyframe_interval(num_frames)
        dtype = (
            torch.bfloat16
            if self._device == "cuda" and torch.cuda.get_device_capability()[0] >= 8
            else torch.float32
        )
        logger.info(
            "reconstructing %s: %d frames, keyframe_interval=%d, dtype=%s",
            self._map_name,
            num_frames,
            keyframe_interval,
            str(dtype),
        )
        with (
            torch.no_grad(),
            torch.amp.autocast(
                self._device, dtype=dtype, enabled=self._device == "cuda"
            ),
        ):
            predictions = self._model.inference_streaming(
                images,
                num_scale_frames=8,
                keyframe_interval=keyframe_interval,
                output_device=torch.device("cpu"),
            )
        return self._serialize(self._map_name, predictions, keyframe_interval)

    def _preprocess(self, frame: np.ndarray) -> Any:
        """Resize one RGB frame to the model input (patch-aligned, cropped)."""
        torch = self._torch
        image = Image.fromarray(np.asarray(frame, dtype=np.uint8))
        width, height = image.size
        new_width = MODEL_INPUT_SIZE
        new_height = (
            round(height * (new_width / width) / MODEL_PATCH_SIZE) * MODEL_PATCH_SIZE
        )
        image = image.resize((new_width, new_height), Image.Resampling.BICUBIC)
        tensor = torch.from_numpy(np.asarray(image)).permute(2, 0, 1).float() / 255.0
        if new_height > MODEL_INPUT_SIZE:
            start_y = (new_height - MODEL_INPUT_SIZE) // 2
            tensor = tensor[:, start_y : start_y + MODEL_INPUT_SIZE, :]
        return tensor

    def _serialize(
        self,
        map_name: str,
        predictions: dict[str, Any],
        keyframe_interval: int,
    ) -> dict[str, Any]:
        torch = self._torch
        map_dir = self._output_dir / map_name
        frames_dir = map_dir / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        for index, frame in enumerate(self._source_frames):
            Image.fromarray(np.asarray(frame, dtype=np.uint8)).save(
                frames_dir / f"{index:06d}.png"
            )

        from lingbot_map.utils.pose_enc import pose_encoding_to_extri_intri

        pose_enc = predictions["pose_enc"]
        image_shape = tuple(predictions["depth"].shape[-3:-1])
        extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, image_shape)
        # pose_enc yields camera-to-world; invert to world-to-camera extrinsics.
        extrinsic_4x4 = torch.zeros(
            (*extrinsic.shape[:-2], 4, 4),
            device=extrinsic.device,
            dtype=extrinsic.dtype,
        )
        extrinsic_4x4[..., :3, :4] = extrinsic
        extrinsic_4x4[..., 3, 3] = 1.0
        from lingbot_map.utils.geometry import closed_form_inverse_se3_general

        extrinsic_4x4 = closed_form_inverse_se3_general(extrinsic_4x4)
        extrinsic_np = extrinsic_4x4[..., :3, :4].detach().cpu().numpy()
        intrinsic_np = intrinsic.detach().cpu().numpy()

        points_np: np.ndarray | None = None
        depth = predictions.get("depth")
        if depth is not None:
            depth_np = depth.detach().cpu().numpy()
            np.save(map_dir / "depth.npy", depth_np)
            # The published checkpoints contain no point-head weights
            # (``enable_point`` heads stay random), so derive the world point
            # cloud from the depth head via upstream unprojection, as the
            # LingBot-Map viewer does by default.
            from lingbot_map.utils.geometry import (
                unproject_depth_map_to_point_map,
            )

            extrinsic_squeezed = extrinsic_np[0]
            intrinsic_squeezed = intrinsic_np[0]
            points_np = unproject_depth_map_to_point_map(
                depth_np[0], extrinsic_squeezed, intrinsic_squeezed
            )
            np.save(map_dir / "world_points.npy", points_np)
            points_conf = predictions.get("depth_conf")
            conf_np = (
                points_conf.detach().cpu().numpy()
                if points_conf is not None
                else np.ones(points_np.shape[:-1], dtype=np.float32)
            )
            np.save(map_dir / "world_points_conf.npy", conf_np[0])

        c2w = camera_to_world(extrinsic_np[0])
        np.save(map_dir / CAMERAS_C2W_FILE, c2w)

        alignment, alignment_reason = _estimate_base_alignment(
            extrinsic_np[0], self._poses
        )
        cameras_robot_frame: dict[str, Any] | None = None
        if alignment is not None:
            robot_camera_to_world = np.tile(np.eye(4), (c2w.shape[0], 1, 1))
            robot_camera_to_world[:, :3, :3] = alignment.rotation @ c2w[:, :3, :3]
            robot_camera_to_world[:, :3, 3] = _apply_alignment(c2w[:, :3, 3], alignment)
            np.save(map_dir / CAMERAS_C2W_ROBOT_FRAME_FILE, robot_camera_to_world)
            cameras_robot_frame = {
                "path": CAMERAS_C2W_ROBOT_FRAME_FILE,
                "convention": "camera_to_world",
                "shape": list(robot_camera_to_world.shape),
                "intrinsic": intrinsic_np[0].tolist(),
            }
            if points_np is not None:
                np.save(
                    map_dir / "world_points_robot_frame.npy",
                    _apply_alignment(points_np, alignment),
                )
            logger.info(
                "aligned %s to the base poses: scale=%.4f, rms=%.3f m",
                map_name,
                alignment.scale,
                alignment.rms_error_m,
            )
        else:
            logger.info("%s stays in the model frame: %s", map_name, alignment_reason)

        poses = [
            {"pose": pose, "timestamp_s": timestamp}
            for pose, timestamp in zip(self._poses, self._timestamps, strict=True)
        ]
        (map_dir / "poses.json").write_text(json.dumps(poses))
        source_sizes = [frame.shape[:2] for frame in self._source_frames]
        metadata = _build_gct_metadata(
            map_name=map_name,
            extrinsic=extrinsic_np[0],
            intrinsic=intrinsic_np[0],
            source_sizes=source_sizes,
            timestamps=self._timestamps,
            poses=self._poses,
            keyframe_interval=keyframe_interval,
            points=points_np,
            input_size=(int(image_shape[-2]), int(image_shape[-1])),
            alignment=alignment,
            alignment_reason=alignment_reason,
        )
        _write_keyframe_index(
            map_dir,
            _frame_entries(
                timestamps=self._timestamps,
                poses=self._poses,
                source_sizes=source_sizes,
                model_keyframe_interval=keyframe_interval,
            ),
        )
        if cameras_robot_frame is not None:
            metadata["cameras_robot_frame"] = cameras_robot_frame
        (map_dir / "map.json").write_text(json.dumps(metadata))
        logger.info("built map %s with %d frames", map_name, len(self._source_frames))
        return {
            "map_path": str(map_dir),
            "frames": len(self._source_frames),
            "metadata": metadata,
        }


class LingBotMapFacade(RpcFacade):
    """RPC server owning one LingBot-Map mapping session at a time."""

    def __init__(
        self,
        backend: MapBackend,
        *,
        max_frames: int = DEFAULT_MAX_SESSION_FRAMES,
    ) -> None:
        super().__init__()
        self._backend = backend
        self._session_id: str | None = None
        self._session_frames = 0
        self._max_frames = int(max_frames)
        if self._max_frames < 1:
            raise ValueError("max_frames must be positive")
        self._register_rpc()

    def _register_rpc(self) -> None:
        self._rpc["map.start_session"] = self.start_session
        self._rpc["map.push_frame"] = self.push_frame
        self._rpc["map.build"] = self.build
        self._rpc["map.stop_session"] = self.stop_session
        self._rpc["map.load_metadata"] = self.load_metadata
        self._rpc["map.load_frame"] = self.load_frame
        self._rpc["map.load_depth"] = self.load_depth
        self._rpc["map.save_annotations"] = self.save_annotations
        self._readonly_methods.add("map.load_metadata")
        self._readonly_methods.add("map.load_frame")
        self._readonly_methods.add("map.load_depth")

    # ---- RPC methods -----------------------------------------------------

    def start_session(self, *, map_name: str) -> dict[str, Any]:
        name = str(map_name).strip()
        if not name:
            raise ValueError("map_name must be a non-empty string")
        if not _MAP_NAME.fullmatch(name):
            raise ValueError(
                "map_name must be one directory name of 1-121 letters, digits, "
                "dots, underscores, or hyphens, starting with a letter or digit"
            )
        if self._session_id is not None:
            raise RuntimeError(
                "a mapping session is already active; build or stop it first"
            )
        target = Path(self._backend.output_dir) / name
        if target.exists():
            raise ValueError(
                f"map {name!r} already exists at {target}; choose another name "
                "or remove the existing map directory"
            )
        self._backend.start(name)
        self._session_id = uuid.uuid4().hex[:12]
        self._session_frames = 0
        return self._status()

    def push_frame(
        self,
        *,
        session_id: str,
        image_base64: str,
        pose: list[float] | None,
        timestamp_s: float,
    ) -> dict[str, Any]:
        self._require_session(session_id)
        if self._session_frames >= self._max_frames:
            raise RuntimeError(
                f"mapping session reached its {self._max_frames}-frame limit; "
                "build the map from a shorter sweep or start the service with a "
                "larger --max-frames"
            )
        if not isinstance(image_base64, str) or not image_base64:
            raise ValueError("image_base64 must be a non-empty string")
        try:
            raw = base64.b64decode(image_base64, validate=True)
            image = np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"))
        except Exception as exc:
            raise ValueError(f"invalid image data: {exc}") from exc
        if pose is not None:
            pose_array = np.asarray(pose, dtype=np.float64).reshape(-1)
            if pose_array.shape != (6,) or not np.isfinite(pose_array).all():
                raise ValueError("pose must be a finite flat 6-vector")
        else:
            pose_array = None
        if not np.isfinite(float(timestamp_s)):
            raise ValueError("timestamp_s must be finite")
        self._backend.push_frame(image, pose_array, float(timestamp_s))
        self._session_frames += 1
        return self._status()

    def build(self, *, session_id: str) -> dict[str, Any]:
        self._require_session(session_id)
        if self._session_frames == 0:
            self._session_id = None
            self._backend.reset()
            raise RuntimeError("no frames were pushed; refusing to build an empty map")
        try:
            result = self._backend.build()
        except BaseException:
            self._session_id = None
            self._backend.reset()
            raise
        self._session_id = None
        self._backend.reset()
        return result

    def stop_session(self, *, session_id: str) -> dict[str, Any]:
        self._require_session(session_id)
        self._session_id = None
        self._session_frames = 0
        self._backend.reset()
        return self._status()

    def load_metadata(self, *, map_path: str) -> dict[str, Any]:
        map_dir = self._require_map_dir(map_path)
        return {"metadata": json.loads((map_dir / "map.json").read_text())}

    def load_frame(self, *, map_path: str, index: int) -> dict[str, Any]:
        """Return one archived source frame and its recorded pose metadata."""
        map_dir = self._require_map_dir(map_path)
        metadata = json.loads((map_dir / "map.json").read_text())
        frame_index = self._check_frame_index(metadata, index)
        image_path = map_dir / "frames" / f"{frame_index:06d}.png"
        if not image_path.is_file():
            raise FileNotFoundError(f"archived frame not found: {image_path}")
        with Image.open(image_path) as image:
            width, height = image.size
        # Archived frames are already RGB PNGs, so forward the bytes as-is
        # instead of decoding and re-encoding them on every RPC.
        image_bytes = image_path.read_bytes()
        entry = self._keyframe_entry(map_dir, frame_index)
        return {
            "index": frame_index,
            "image_base64": base64.b64encode(image_bytes).decode("ascii"),
            "source_size": [int(height), int(width)],
            "timestamp_s": entry.get("timestamp_s"),
            "pushed_base_pose": entry.get("pushed_base_pose"),
        }

    def load_depth(self, *, map_path: str, index: int) -> dict[str, Any]:
        """Return one frame's model-resolution depth and confidence arrays."""
        map_dir = self._require_map_dir(map_path)
        metadata = json.loads((map_dir / "map.json").read_text())
        frame_index = self._check_frame_index(metadata, index)
        depth_path = map_dir / "depth.npy"
        if not depth_path.is_file():
            raise FileNotFoundError(
                f"map has no depth map (the frame-archive backend records none): {map_dir}"
            )
        frames = int(metadata.get("frames", 0))
        depth = _select_serialized_frame(
            np.load(depth_path, mmap_mode="r"), frame_index, frames, "depth"
        )
        depth_array = np.ascontiguousarray(depth, dtype=np.float32)
        confidence_encoded: str | None = None
        confidence_shape: list[int] | None = None
        confidence_path = map_dir / "world_points_conf.npy"
        if confidence_path.is_file():
            confidence = _select_serialized_frame(
                np.load(confidence_path, mmap_mode="r"),
                frame_index,
                frames,
                "confidence",
            )
            confidence_array = np.ascontiguousarray(confidence, dtype=np.float32)
            confidence_encoded = base64.b64encode(confidence_array.tobytes()).decode(
                "ascii"
            )
            confidence_shape = list(confidence_array.shape)
        return {
            "index": frame_index,
            "shape": list(depth_array.shape),
            "dtype": "float32",
            "depth_base64": base64.b64encode(depth_array.tobytes()).decode("ascii"),
            "confidence_shape": confidence_shape,
            "confidence_base64": confidence_encoded,
        }

    def save_annotations(
        self,
        *,
        map_path: str,
        objects: dict[str, Any],
        masks: dict[str, str] | None = None,
        semantic_points_base64: str | None = None,
        semantic_points_ply_base64: str | None = None,
    ) -> dict[str, Any]:
        """Persist grounded objects, masks, and optional semantic point files.

        ``masks`` maps a flat PNG file name to base64 bytes and is written
        below ``annotations/``. ``objects`` is written to ``objects.json``.
        The optional semantic arrays are written as ``semantic_points.npz``
        and ``semantic_points.ply``. Only these known artifact names can be
        written, so a client cannot rewrite ``map.json`` or read outside the
        map directory.
        """
        map_dir = self._require_map_dir(map_path)
        if not isinstance(objects, dict):
            raise ValueError("objects must be a JSON object")
        try:
            encoded_objects = json.dumps(objects, indent=2)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"objects must be JSON serializable: {exc}") from exc

        annotations_dir = map_dir / ANNOTATIONS_DIR
        annotations_dir.mkdir(parents=True, exist_ok=True)
        masks_written = 0
        for name, encoded in (masks or {}).items():
            if not isinstance(name, str) or _ANNOTATION_NAME.match(name) is None:
                raise ValueError(f"invalid annotation file name: {name!r}")
            if not isinstance(encoded, str) or not encoded:
                raise ValueError(f"annotation {name!r} must be base64 text")
            raw = base64.b64decode(encoded, validate=True)
            if len(raw) > _MAX_ANNOTATION_BYTES:
                raise ValueError(f"annotation {name!r} is too large")
            if not raw.startswith(b"\x89PNG\r\n\x1a\n"):
                raise ValueError(f"annotation {name!r} is not a PNG image")
            (annotations_dir / name).write_bytes(raw)
            masks_written += 1

        semantic_written: list[str] = []
        for name, encoded in (
            (SEMANTIC_POINTS_FILE, semantic_points_base64),
            (SEMANTIC_PLY_FILE, semantic_points_ply_base64),
        ):
            if encoded is None:
                continue
            if not isinstance(encoded, str) or not encoded:
                raise ValueError(f"{name} must be non-empty base64 text")
            raw = base64.b64decode(encoded, validate=True)
            if len(raw) > _MAX_SEMANTIC_ARTIFACT_BYTES:
                raise ValueError(f"{name} is too large")
            (map_dir / name).write_bytes(raw)
            semantic_written.append(name)

        objects_path = map_dir / OBJECTS_FILE
        objects_path.write_text(encoded_objects)
        return {
            "map_path": str(map_dir),
            "objects_path": str(objects_path),
            "masks_written": masks_written,
            "semantic_artifacts": semantic_written,
        }

    # ---- helpers ---------------------------------------------------------

    def _check_frame_index(self, metadata: dict[str, Any], index: int) -> int:
        frames = int(metadata.get("frames", 0))
        try:
            frame_index = int(index)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"frame index must be an integer, got {index!r}") from exc
        if not 0 <= frame_index < frames:
            raise ValueError(
                f"frame index {frame_index} is outside the map's {frames} frames"
            )
        return frame_index

    def _keyframe_entry(self, map_dir: Path, index: int) -> dict[str, Any]:
        index_path = map_dir / KEYFRAME_INDEX_FILE
        if index_path.is_file():
            entries = json.loads(index_path.read_text())
            if isinstance(entries, list) and index < len(entries):
                entry = entries[index]
                if isinstance(entry, dict):
                    return entry
        poses_path = map_dir / "poses.json"
        if poses_path.is_file():
            poses = json.loads(poses_path.read_text())
            if isinstance(poses, list) and index < len(poses):
                return poses[index]
        return {}

    def _require_map_dir(self, map_path: str) -> Path:
        root = Path(self._backend.output_dir).expanduser().resolve()
        candidate = Path(str(map_path)).expanduser().resolve()
        if candidate.parent != root:
            raise ValueError(
                f"map_path must be a direct child of the map output directory: {root}"
            )
        if not candidate.is_dir():
            raise FileNotFoundError(f"map directory not found: {candidate}")
        if not (candidate / "map.json").is_file():
            raise ValueError(f"map directory has no map.json: {candidate}")
        return candidate

    def _require_session(self, session_id: str) -> None:
        if not isinstance(session_id, str) or session_id != self._session_id:
            raise RuntimeError("unknown or expired mapping session")

    def _status(self) -> dict[str, Any]:
        return {
            "session_id": self._session_id or "",
            "running": self._session_id is not None,
            "frames": self._session_frames,
        }


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RPent LingBot-Map server")
    parser.add_argument("--transport", choices=["socket", "http"], default="http")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8116)
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for built map artifacts (default: LINGBOTMAP_OUTPUT_DIR)",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="LingBot-Map GCT checkpoint path (default: LINGBOTMAP_CHECKPOINT_PATH); "
        "unset falls back to the frame-archive backend",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=DEFAULT_MAX_SESSION_FRAMES,
        help=(
            "maximum frames retained in one mapping session "
            f"(default: {DEFAULT_MAX_SESSION_FRAMES}); raise it on hosts with "
            "more GPU memory"
        ),
    )
    parser.add_argument(
        "--parent-watch",
        action="store_true",
        help="watch parent process via stdin pipe and exit when it dies",
    )
    return parser


def main() -> None:
    """Start the mapping facade and serve until terminated."""
    args = _build_argparser().parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s")
    output_dir = args.output_dir or os.environ.get("LINGBOTMAP_OUTPUT_DIR")
    if not output_dir:
        raise RuntimeError(
            "map output directory is not set; pass --output-dir or export "
            "LINGBOTMAP_OUTPUT_DIR before starting RPent"
        )
    if args.checkpoint:
        os.environ["LINGBOTMAP_CHECKPOINT_PATH"] = args.checkpoint
    facade = LingBotMapFacade(
        build_lingbot_backend(output_dir), max_frames=args.max_frames
    )
    facade.serve(
        transport=args.transport,
        host=args.host,
        port=args.port,
        parent_watch=args.parent_watch,
    )


if __name__ == "__main__":
    main()
