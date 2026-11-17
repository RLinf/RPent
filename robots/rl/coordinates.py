# Copyright 2026 The RPent Authors.
# Licensed under the Apache License, Version 2.0.
"""Episode-fixed LIBERO frames; native control and depth remain in simulator world."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Coordinates:
    """Map simulator-world positions and rotations into the selected world frame."""

    mode: str
    origin: np.ndarray
    rotation: np.ndarray
    table_height_m: float | None = None

    @classmethod
    def from_control(cls, mode: str, control: dict[str, Any]) -> Coordinates:
        """Validate native calibration once, before any general-frame action."""
        if mode == "base":
            calibration = control.get("coordinate_calibration")
            height = (
                None
                if calibration is None
                else float(
                    np.asarray(calibration["table_top_xyz"])[2]
                    - np.asarray(calibration["base_xyz"])[2]
                )
            )
            return cls(mode, np.zeros(3), np.eye(3), height)
        if mode != "general":
            raise ValueError("world_cord must be base or general")
        try:
            calibration = control["coordinate_calibration"]
            origin = np.asarray(calibration["base_xyz"], dtype=np.float64)
            base_rotation = np.asarray(calibration["base_rotation"], dtype=np.float64)
            table = np.asarray(calibration["table_top_xyz"], dtype=np.float64)
            tcp = np.asarray(control["pad_tcp_xyz"], dtype=np.float64)
        except KeyError as exc:
            raise ValueError(
                "general requires native base, table and finger-pad calibration"
            ) from exc
        if any(
            v.shape != (3,) or not np.isfinite(v).all() for v in (origin, table, tcp)
        ):
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
        return cls(mode, origin.copy(), rotation, float(table[2] - origin[2]))

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
            "world_cord": self.mode,
            "tcp": "finger_pad_midpoint" if self.mode == "general" else "grip_site",
            "units": {"position": "metres", "orientation": "radians"},
            "axes": {"x": "forward", "y": "robot_left", "z": "up"}
            if self.mode == "general"
            else {"x": "sim_world_x", "y": "sim_world_y", "z": "sim_world_z"},
            "origin_in_sim_world": self.origin.tolist(),
            "rotation_to_sim_world": self.rotation.tolist(),
            "raw_point_map_frame": "sim_world",
            "table_height_above_base_m": self.table_height_m,
        }

    def description(self) -> dict[str, Any]:
        """Give the model the frame convention, without the simulator-world origin."""
        metadata = self.metadata()
        return {
            "world_cord": self.mode,
            "tcp": metadata["tcp"],
            "units": metadata["units"],
            "axes": metadata["axes"],
            "table_height_above_base_m": None
            if self.table_height_m is None
            else round(self.table_height_m, 6),
        }
