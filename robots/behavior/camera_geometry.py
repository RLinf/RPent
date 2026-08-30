"""Small camera-geometry helpers for BEHAVIOR RGB-D tools.

The full simulator geometry lives behind the environment RPC.  This module
keeps only import-safe validation and math helpers used by lightweight clients
and tests; live calibration should be supplied by the env server.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

CANONICAL_CAMERAS = ("head", "left_wrist", "right_wrist")
HAND_GEOMETRY_TRANSLATION_TOLERANCE_M = 0.025
HAND_GEOMETRY_ROTATION_TOLERANCE_DEG = 8.0
HAND_GEOMETRY_FINGER_JOINT_TOLERANCE_M = 0.015
HAND_GEOMETRY_SYNC_RENDER_ITERATIONS = 6


class CameraGeometryError(ValueError):
    """Raised when camera metadata or RGB-D geometry is invalid."""


class FrameTtlExpired(CameraGeometryError):
    """Raised when a frame-bound claim is too old for action use."""


@dataclass(frozen=True)
class CameraIntrinsics:
    """Pinhole camera intrinsics in pixel coordinates."""

    fx: float
    fy: float
    cx: float
    cy: float

    def matrix(self) -> np.ndarray:
        return np.asarray(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )


def canonical_camera(value: Any) -> str:
    if not isinstance(value, str) or value not in CANONICAL_CAMERAS:
        raise CameraGeometryError(
            f"camera must be one of {', '.join(CANONICAL_CAMERAS)}"
        )
    return value


def validated_rigid_transform(value: Any, *, name: str = "transform") -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (4, 4) or not np.isfinite(array).all():
        raise CameraGeometryError(f"{name} must be a finite 4x4 transform")
    if not np.allclose(array[3], np.asarray([0.0, 0.0, 0.0, 1.0]), atol=1e-6):
        raise CameraGeometryError(f"{name} has invalid homogeneous row")
    rotation = array[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-3):
        raise CameraGeometryError(f"{name} rotation is not orthonormal")
    return array


def camera_point_from_pixel(
    *,
    u: int,
    v: int,
    depth_m: float,
    intrinsics: CameraIntrinsics,
) -> np.ndarray:
    if isinstance(u, bool) or isinstance(v, bool):
        raise CameraGeometryError("pixel coordinates must be integers")
    if not np.isfinite(depth_m) or depth_m <= 0.0:
        raise CameraGeometryError("depth_m must be positive and finite")
    return np.asarray(
        [
            (int(u) - intrinsics.cx) * float(depth_m) / intrinsics.fx,
            (int(v) - intrinsics.cy) * float(depth_m) / intrinsics.fy,
            float(depth_m),
        ],
        dtype=np.float64,
    )


def backproject_pixel_to_world(
    *,
    u: int,
    v: int,
    depth_m: float,
    intrinsics: CameraIntrinsics,
    camera_to_world: Any,
) -> np.ndarray:
    point = camera_point_from_pixel(
        u=u,
        v=v,
        depth_m=depth_m,
        intrinsics=intrinsics,
    )
    transform = validated_rigid_transform(camera_to_world, name="camera_to_world")
    return (transform @ np.asarray([*point, 1.0], dtype=np.float64))[:3]


def robust_depth_sample(
    depth: Any,
    *,
    u: int,
    v: int,
    window_px: int = 7,
) -> float:
    image = np.asarray(depth, dtype=np.float64)
    if image.ndim != 2:
        raise CameraGeometryError(f"depth image must be [H,W], got {image.shape}")
    if isinstance(window_px, bool) or int(window_px) <= 0:
        raise CameraGeometryError("window_px must be positive")
    h, w = image.shape
    if not (0 <= int(u) < w and 0 <= int(v) < h):
        raise CameraGeometryError("pixel is outside depth image")
    radius = int(window_px) // 2
    crop = image[
        max(0, int(v) - radius) : min(h, int(v) + radius + 1),
        max(0, int(u) - radius) : min(w, int(u) + radius + 1),
    ]
    values = crop[np.isfinite(crop) & (crop > 0.0)]
    if values.size == 0:
        raise CameraGeometryError("depth window contains no positive finite samples")
    return float(np.median(values))


class FrameCache:
    """Minimal in-process frame cache keyed by public frame id."""

    def __init__(self) -> None:
        self._frames: dict[str, dict[str, Any]] = {}

    def put(self, frame_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(frame_id, str) or not frame_id:
            raise CameraGeometryError("frame_id must be non-empty")
        self._frames[frame_id] = dict(payload)
        return dict(self._frames[frame_id])

    def get(self, frame_id: str) -> dict[str, Any]:
        try:
            return dict(self._frames[frame_id])
        except KeyError as exc:
            raise CameraGeometryError(f"unknown frame_id: {frame_id}") from exc


def load_camera_correction_profiles(_path: str | None = None) -> dict[str, Any]:
    """Return an explicit empty correction set for minimal upstream builds."""

    return {"schema_version": 1, "profiles": {}, "source": "not_configured"}


def r1pro_wrist_camera_reference_transforms() -> dict[str, np.ndarray]:
    """Return identity placeholders only for import-safe static validation."""

    return {"left_wrist": np.eye(4), "right_wrist": np.eye(4)}


def rigid_transform_residual(a: Any, b: Any) -> dict[str, float]:
    left = validated_rigid_transform(a, name="a")
    right = validated_rigid_transform(b, name="b")
    delta = np.linalg.inv(left) @ right
    translation_m = float(np.linalg.norm(delta[:3, 3]))
    rotation_trace = float(np.clip((np.trace(delta[:3, :3]) - 1.0) / 2.0, -1.0, 1.0))
    rotation_deg = float(np.degrees(np.arccos(rotation_trace)))
    return {"translation_m": translation_m, "rotation_deg": rotation_deg}


def hand_geometry_sync_certificate_is_valid(value: Any) -> bool:
    return isinstance(value, dict) and value.get("valid") is True


def frame_bound_hand_distance_report(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
    raise CameraGeometryError("live hand geometry is available only through env RPC")


__all__ = [
    "CANONICAL_CAMERAS",
    "HAND_GEOMETRY_FINGER_JOINT_TOLERANCE_M",
    "HAND_GEOMETRY_ROTATION_TOLERANCE_DEG",
    "HAND_GEOMETRY_SYNC_RENDER_ITERATIONS",
    "HAND_GEOMETRY_TRANSLATION_TOLERANCE_M",
    "CameraGeometryError",
    "CameraIntrinsics",
    "FrameCache",
    "FrameTtlExpired",
    "backproject_pixel_to_world",
    "camera_point_from_pixel",
    "canonical_camera",
    "frame_bound_hand_distance_report",
    "hand_geometry_sync_certificate_is_valid",
    "load_camera_correction_profiles",
    "r1pro_wrist_camera_reference_transforms",
    "rigid_transform_residual",
    "robust_depth_sample",
    "validated_rigid_transform",
]
