# Copyright 2026 The RPent Authors.
# Licensed under the Apache License, Version 2.0.
"""Measured mounting frames; native control and depth remain in simulator world."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation


def grasp_axes(closing: np.ndarray, approach: np.ndarray) -> np.ndarray:
    """Build canonical tool axes from measured jaw and wrist-to-finger directions.

    Inputs and output use the native TCP orientation. The sign of a parallel
    jaw's closing line is immaterial; use its positive dominant native axis so
    opening/closing cannot flip the reported orientation by pi.
    """
    x, z = (np.asarray(value, dtype=np.float64).copy() for value in (closing, approach))
    if any(
        value.shape != (3,)
        or not np.isfinite(value).all()
        or np.linalg.norm(value) < 1e-6
        for value in (x, z)
    ):
        raise ValueError("general requires measurable jaw and approach axes")
    x /= np.linalg.norm(x)
    if x[np.argmax(np.abs(x))] < 0:
        x *= -1
    z /= np.linalg.norm(z)
    if abs(x @ z) > 0.1:
        raise ValueError("measured jaw and approach axes are not perpendicular")
    z -= x * (x @ z)
    z /= np.linalg.norm(z)
    return np.column_stack((x, np.cross(z, x), z))


@dataclass(frozen=True)
class Coordinates:
    """Map simulator-world positions and rotations into the measured robot mounting frame."""

    origin: np.ndarray
    rotation: np.ndarray
    table_height_m: float | None = None
    floor_height_m: float | None = None
    version: int = 0
    tcp: str = "native_grasp_tcp"

    @classmethod
    def from_control(cls, control: dict[str, Any]) -> Coordinates:
        """Validate native calibration once, before any general-frame action."""
        try:
            calibration = control["coordinate_calibration"]
            origin = np.asarray(calibration["base_xyz"], dtype=np.float64)
            base_rotation = np.asarray(calibration["base_rotation"], dtype=np.float64)
            kind = "floor" if "floor_top_xyz" in calibration else "table"
            surface = np.asarray(calibration[f"{kind}_top_xyz"], dtype=np.float64)
        except KeyError as exc:
            raise ValueError(
                "general requires native mounting-base and support-surface calibration"
            ) from exc
        if any(v.shape != (3,) or not np.isfinite(v).all() for v in (origin, surface)):
            raise ValueError("general calibration positions must be finite xyz vectors")
        if (
            base_rotation.shape != (3, 3)
            or not np.isfinite(base_rotation).all()
            or not np.allclose(base_rotation.T @ base_rotation, np.eye(3), atol=1e-6)
            or not np.isclose(np.linalg.det(base_rotation), 1, atol=1e-6)
        ):
            raise ValueError("general calibration needs a proper base rotation")
        forward = base_rotation[:, 0].copy()
        forward[2] = 0
        if np.linalg.norm(forward) < 1e-6:
            raise ValueError(
                "robot base forward axis must have a horizontal projection"
            )
        forward /= np.linalg.norm(forward)
        up = np.array([0.0, 0.0, 1.0])
        rotation = np.column_stack((forward, np.cross(up, forward), up))
        return cls(
            origin.copy(),
            rotation,
            **{f"{kind}_height_m": float(surface[2] - origin[2])},
            tcp=str(calibration.get("tcp", "native_grasp_tcp")),
        )

    def after(self, previous: Coordinates) -> Coordinates:
        """Advance the version only when the measured mount frame changes."""
        changed = not (
            np.allclose(self.origin, previous.origin, rtol=0, atol=1e-6)
            and np.allclose(self.rotation, previous.rotation, rtol=0, atol=1e-6)
        )
        return replace(self, version=previous.version + int(changed))

    def points(self, world: np.ndarray) -> np.ndarray:
        """Convert already-valid native points; filter invalid depth before calling."""
        return (np.asarray(world, dtype=np.float64) - self.origin) @ self.rotation

    def world_position(self, position: np.ndarray) -> np.ndarray:
        """Convert an absolute model position to simulator world."""
        return self.origin + self.rotation @ position

    def orientation(self, world_rotation: np.ndarray) -> np.ndarray:
        """Convert a native orientation by matrix composition."""
        return self.rotation.T @ world_rotation

    def world_orientation(self, rotation: np.ndarray) -> np.ndarray:
        """Convert a model orientation to simulator world."""
        return self.rotation @ rotation

    def metadata(self) -> dict[str, Any]:
        """Describe model coordinates and raw point-map storage without ambiguity."""
        return {
            "tcp": self.tcp,
            "units": {"position": "metres", "orientation": "radians"},
            "axes": {"x": "forward", "y": "robot_left", "z": "up"},
            "origin_in_sim_world": self.origin.tolist(),
            "rotation_to_sim_world": self.rotation.tolist(),
            "raw_point_map_frame": "sim_world",
            "raw_point_map_unit": "m",
            "contract_version": "episode_coordinates_v2",
            "frame_version": self.version,
            "table_height_above_base_m": self.table_height_m,
            **(
                {
                    "support_surface": "floor",
                    "floor_height_above_base_m": self.floor_height_m,
                }
                if self.floor_height_m is not None
                else {}
            ),
        }

    def description(self) -> dict[str, Any]:
        """Give the model the frame convention, without the simulator-world origin."""
        metadata = self.metadata()
        return {
            "tcp": metadata["tcp"],
            "units": metadata["units"],
            "axes": metadata["axes"],
            "contract_version": metadata["contract_version"],
            "frame_version": self.version,
            "orientation": "fixed-axis XYZ RPY radians; tool +x closes the jaws, +z approaches",
            "table_height_above_base_m": None
            if self.table_height_m is None
            else round(self.table_height_m, 6),
            **(
                {
                    "support_surface": "floor",
                    "floor_height_above_base_m": round(self.floor_height_m, 6),
                }
                if self.floor_height_m is not None
                else {}
            ),
        }


def pose_matrix(pose: Any) -> np.ndarray:
    """Convert a world xyz/wxyz pose to a rigid transform."""
    value = np.asarray(pose, dtype=np.float64)
    if value.shape != (7,) or not np.isfinite(value).all():
        raise ValueError("Expected seven finite xyz/wxyz values")
    if not np.isclose(np.linalg.norm(value[3:]), 1, atol=1e-4):
        raise ValueError("Expected a unit wxyz quaternion")
    result = np.eye(4)
    result[:3, :3] = Rotation.from_quat(value[[4, 5, 6, 3]]).as_matrix()
    result[:3, 3] = value[:3]
    return result


def mounting_calibration(
    left: Any, right: Any, surface: Any, *, support_surface: str = "table"
) -> dict[str, Any]:
    """Construct the shared arm frame and record the measured support surface."""
    if support_surface not in ("table", "floor"):
        raise ValueError("Support surface must be a table or a floor")
    left, right, surface = (
        np.asarray(v, dtype=np.float64) for v in (left, right, surface)
    )
    if any(v.shape != (3,) or not np.isfinite(v).all() for v in (left, right, surface)):
        raise ValueError("Mount and support positions must be finite xyz vectors")
    lateral = left - right
    lateral[2] = 0
    if np.linalg.norm(lateral) < 1e-6:
        raise ValueError("Two distinguishable arm mounting centres are required")
    lateral /= np.linalg.norm(lateral)
    up = np.array([0.0, 0.0, 1.0])
    rotation = np.column_stack((np.cross(lateral, up), lateral, up))
    return {
        "base_xyz": ((left + right) / 2).tolist(),
        "base_rotation": rotation.tolist(),
        f"{support_surface}_top_xyz": surface.tolist(),
        "left_mount_xyz": left.tolist(),
        "right_mount_xyz": right.tolist(),
    }


def finger_contact_centres(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Use the opposed inner faces in native axes where the jaws slide along Y."""
    first_below = (first[:, 1].min() + first[:, 1].max()) < (
        second[:, 1].min() + second[:, 1].max()
    )
    inward_limits = (np.max, np.min) if first_below else (np.min, np.max)
    centres = []
    for vertices, inward in zip((first, second), inward_limits, strict=True):
        lateral = vertices[:, 1]
        face = vertices[np.isclose(lateral, inward(lateral), rtol=0, atol=1e-5)]
        if len(face) < 3:
            raise ValueError("Missing flat inner contact face on a finger")
        centres.append((face.min(axis=0) + face.max(axis=0)) / 2)
    return np.asarray(centres)
