# Copyright 2026 The RPent Authors.
# Licensed under the Apache License, Version 2.0.
"""Shared mounting coordinates and measured Aloha finger-pad references."""

from dataclasses import dataclass
from itertools import product
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from robots.rl.coordinates import Coordinates


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


def mounting_calibration(left: Any, right: Any, table: Any) -> dict[str, Any]:
    """Construct forward/left/up axes at the midpoint of both mounting bases."""
    left, right, table = (np.asarray(v, dtype=np.float64) for v in (left, right, table))
    if any(v.shape != (3,) or not np.isfinite(v).all() for v in (left, right, table)):
        raise ValueError("Mount and table positions must be finite xyz vectors")
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
        "table_top_xyz": table.tolist(),
        "left_mount_xyz": left.tolist(),
        "right_mount_xyz": right.tolist(),
    }


@dataclass(frozen=True)
class RobotwinFrame:
    """Episode-fixed shared mounting frame for the two finger-pad TCPs."""

    coordinates: Coordinates
    calibration: dict[str, Any]

    @classmethod
    def from_calibration(cls, calibration: dict[str, Any]) -> "RobotwinFrame":
        coordinates = Coordinates.from_control("general", calibration)
        return cls(coordinates, calibration)

    def model_pose(self, native_tcp: Any, pad_world: Any) -> dict[str, list[float]]:
        """Describe the measured pad centre and native orientation in the shared frame."""
        pose = pose_matrix(native_tcp)
        rotation = self.coordinates.orientation(pose[:3, :3])
        return {
            "xyz": self.coordinates.points(pad_world).tolist(),
            "orientation_rpy": Rotation.from_matrix(rotation).as_euler("xyz").tolist(),
        }

    def world_target(
        self, xyz: Any, rpy: Any, current: Any, pad_world: Any
    ) -> np.ndarray:
        """Map an absolute pad target back to the native world TCP."""
        native = pose_matrix(current)
        offset = native[:3, :3].T @ (np.asarray(pad_world) - native[:3, 3])
        position = np.asarray(
            self.coordinates.points(pad_world) if xyz is None else xyz, dtype=np.float64
        )
        angles = None if rpy is None else np.asarray(rpy, dtype=np.float64)
        if (
            position.shape != (3,)
            or not np.isfinite(position).all()
            or (
                angles is not None
                and (angles.shape != (3,) or not np.isfinite(angles).all())
            )
        ):
            raise ValueError(
                "xyz and orientation_rpy must contain three finite numbers"
            )
        rotation = (
            native[:3, :3]
            if angles is None
            else self.coordinates.world_orientation(
                Rotation.from_euler("xyz", angles).as_matrix()
            )
        )
        native_position = self.coordinates.world_position(position) - rotation @ offset
        quaternion = Rotation.from_matrix(rotation).as_quat()[[3, 0, 1, 2]]
        return np.r_[native_position, quaternion]

    def metadata(self) -> dict[str, Any]:
        return {
            **self.coordinates.metadata(),
            "calibration": self.calibration,
        }


def finger_contact_centres(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Use the opposed inner faces in axes aligned with the Aloha TCP."""
    first_below = (first[:, 1].min() + first[:, 1].max()) < (
        second[:, 1].min() + second[:, 1].max()
    )
    inward_limits = (np.max, np.min) if first_below else (np.min, np.max)
    centres = []
    for vertices, inward in zip((first, second), inward_limits, strict=True):
        lateral = vertices[:, 1]
        face = vertices[np.isclose(lateral, inward(lateral), rtol=0, atol=1e-5)]
        if len(face) < 3:
            raise ValueError("Missing flat inner contact face on an Aloha finger")
        centres.append((face.min(axis=0) + face.max(axis=0)) / 2)
    return np.asarray(centres)


def _finger_vertices(link: Any) -> np.ndarray:
    """Read collision vertices in the measured finger-link frame."""
    points = []
    for shape in link.get_collision_shapes():
        if hasattr(shape, "vertices"):
            vertices = np.asarray(shape.vertices, dtype=np.float64)
        elif hasattr(shape, "half_size"):
            vertices = np.asarray(list(product((-1, 1), repeat=3))) * shape.half_size
        else:
            continue
        local = shape.get_local_pose().to_transformation_matrix()
        points.append(vertices @ local[:3, :3].T + local[:3, 3])
    if not points:
        raise ValueError(f"Missing collision geometry for finger {link.get_name()}")
    return np.concatenate(points)


def measured_pad_position(
    robot: Any, arm: str, sites: list[list[float]]
) -> list[float]:
    """Read the midpoint of the two contact sites from the live finger links."""
    pads = []
    for (joint, _, _), site in zip(
        getattr(robot, f"{arm}_gripper"), sites, strict=True
    ):
        world = joint.child_link.get_pose().to_transformation_matrix()
        pads.append(world[:3, 3] + world[:3, :3] @ site)
    return np.mean(pads, axis=0).tolist()


def robotwin_calibration(task: Any) -> dict[str, Any]:
    """Measure the mounting frame and identify the two jaw contact sites per arm."""
    robot = task.robot
    mounts, offsets, pad_positions, pad_sites = {}, {}, {}, {}
    for arm in ("left", "right"):
        root = getattr(robot, f"{arm}_arm_joints")[0].parent_link
        mounts[arm] = root.get_pose().p
        fingers = [joint.child_link for joint, _, _ in getattr(robot, f"{arm}_gripper")]
        tcp = pose_matrix(getattr(robot, f"get_{arm}_tcp_pose")())
        tcp_rotation, tcp_position = tcp[:3, :3], tcp[:3, 3]
        finger_poses = [link.get_pose().to_transformation_matrix() for link in fingers]
        points = [
            (_finger_vertices(link) @ pose[:3, :3].T + pose[:3, 3] - tcp_position)
            @ tcp_rotation
            for link, pose in zip(fingers, finger_poses, strict=True)
        ]
        centres = finger_contact_centres(*points)
        pad_sites[arm] = [
            (
                (tcp_position + tcp_rotation @ centre - pose[:3, 3]) @ pose[:3, :3]
            ).tolist()
            for centre, pose in zip(centres, finger_poses, strict=True)
        ]
        offset = centres.mean(axis=0)
        offsets[arm] = offset.tolist()
        pad_positions[arm] = (tcp_position + tcp_rotation @ offset).tolist()
    return {
        "coordinate_calibration": mounting_calibration(
            mounts["left"], mounts["right"], task.table.get_pose().p
        ),
        "pad_tcp_xyz": pad_positions["left"],
        "native_tcp_to_pad": offsets,
        "pad_measurement": "midpoint_of_opposed_finger_contact_faces",
        "finger_pad_sites": pad_sites,
    }
