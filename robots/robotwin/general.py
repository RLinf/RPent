# Copyright 2026 The RPent Authors.
# Licensed under the Apache License, Version 2.0.
"""Shared mounting frame and measured axes of the native Aloha grasp TCP."""

from dataclasses import dataclass
from itertools import product
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from robots.rl.geometry import (
    Coordinates,
    finger_contact_centres,
    grasp_axes,
    mounting_calibration,
    pose_matrix,
)


@dataclass(frozen=True)
class RobotwinFrame:
    """Episode-fixed shared mounting frame for both native grasp TCPs."""

    coordinates: Coordinates
    calibration: dict[str, Any]

    @classmethod
    def from_calibration(cls, calibration: dict[str, Any]) -> "RobotwinFrame":
        coordinates = Coordinates.from_control(calibration)
        return cls(coordinates, calibration)

    def tool_rotation(self, arm: str) -> np.ndarray:
        rotation = np.asarray(
            self.calibration["tool_rotation_to_native"][arm], dtype=float
        )
        if (
            rotation.shape != (3, 3)
            or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6)
            or not np.isclose(np.linalg.det(rotation), 1)
        ):
            raise ValueError("invalid measured native grasp orientation")
        return rotation

    def model_pose(self, native_tcp: Any, arm: str) -> dict[str, list[float]]:
        """Keep the native grasp point and express calibrated axes in the shared frame."""
        pose = pose_matrix(native_tcp)
        rotation = self.coordinates.orientation(pose[:3, :3] @ self.tool_rotation(arm))
        return {
            "xyz": self.coordinates.points(pose[:3, 3]).tolist(),
            "orientation_rpy": Rotation.from_matrix(rotation).as_euler("xyz").tolist(),
        }

    def world_target(self, xyz: Any, rpy: Any, current: Any, arm: str) -> np.ndarray:
        """Map a common-axis absolute pose back to the native grasp TCP."""
        native = pose_matrix(current)
        position = (
            self.coordinates.points(native[:3, 3])
            if xyz is None
            else np.asarray(xyz, dtype=float)
        )
        angles = None if rpy is None else np.asarray(rpy, dtype=float)
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
            else (
                self.coordinates.world_orientation(
                    Rotation.from_euler("xyz", angles).as_matrix()
                )
                @ self.tool_rotation(arm).T
            )
        )
        quaternion = Rotation.from_matrix(rotation).as_quat()[[3, 0, 1, 2]]
        return np.r_[self.coordinates.world_position(position), quaternion]

    def metadata(self) -> dict[str, Any]:
        return {
            **self.coordinates.metadata(),
            "calibration": self.calibration,
        }


def _collision_vertices(body: Any) -> np.ndarray:
    """Read collision vertices in a SAPIEN link or static entity's frame."""
    if not hasattr(body, "get_collision_shapes"):
        from sapien.physx import PhysxRigidBaseComponent

        rigid = body.find_component_by_type(PhysxRigidBaseComponent)
        if rigid is None:
            raise ValueError(f"Missing rigid component for {body.get_name()}")
    else:
        rigid = body
    points = []
    for shape in rigid.get_collision_shapes():
        if hasattr(shape, "vertices"):
            vertices = np.asarray(shape.vertices, dtype=np.float64) * shape.scale
        elif hasattr(shape, "half_size"):
            vertices = np.asarray(list(product((-1, 1), repeat=3))) * shape.half_size
        else:
            continue
        local = shape.get_local_pose().to_transformation_matrix()
        points.append(vertices @ local[:3, :3].T + local[:3, 3])
    if not points:
        raise ValueError(f"Missing collision geometry for {body.get_name()}")
    return np.concatenate(points)


def robotwin_calibration(task: Any) -> dict[str, Any]:
    """Read installation mounts, collision-table top and each native tool's axes."""
    robot = task.robot
    mounts, axes = {}, {}
    for arm in ("left", "right"):
        root = getattr(robot, f"{arm}_arm_joints")[0].parent_link
        mounts[arm] = root.get_pose().p
        tcp = pose_matrix(getattr(robot, f"get_{arm}_tcp_pose")())
        wrist = pose_matrix(getattr(robot, f"get_{arm}_ee_pose")())
        fingers = [joint.child_link for joint, _, _ in getattr(robot, f"{arm}_gripper")]
        finger_poses = [link.get_pose().to_transformation_matrix() for link in fingers]
        points = [
            (_collision_vertices(link) @ pose[:3, :3].T + pose[:3, 3] - tcp[:3, 3])
            @ tcp[:3, :3]
            for link, pose in zip(fingers, finger_poses, strict=True)
        ]
        centres = finger_contact_centres(*points)
        approach = tcp[:3, :3].T @ (tcp[:3, 3] - wrist[:3, 3])
        axes[arm] = grasp_axes(centres[1] - centres[0], approach).tolist()
    table_pose = task.table.get_pose().to_transformation_matrix()
    table_points = (
        _collision_vertices(task.table) @ table_pose[:3, :3].T + table_pose[:3, 3]
    )
    top = table_points.mean(axis=0)
    top[2] = table_points[:, 2].max()
    calibration = mounting_calibration(mounts["left"], mounts["right"], top)
    calibration["tcp"] = "native_grasp_tcp"
    return {"coordinate_calibration": calibration, "tool_rotation_to_native": axes}
