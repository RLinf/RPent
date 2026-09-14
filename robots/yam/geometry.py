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

"""Geometry helpers for the RPent YAM facade.

The motion runtime and IK implementation remain owned by RLinf/i2rt. This
module only normalizes qpos14, calibration transforms, and planning inputs for
the RPent agent boundary.
"""

from __future__ import annotations

import json
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from rpent.utils.config import get_rlinf_repo_path

ARM_SLICES = {"left": slice(0, 7), "right": slice(7, 14)}
ARM_JOINT_INDICES = np.array([0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12])
GRIPPER_INDICES = np.array([6, 13])
DEFAULT_JOINT_LIMIT_MIN = np.array(
    [
        [-2.61799, 0.0, 0.0, -1.69297, -1.5708, -2.0944],
        [-2.61799, 0.0, 0.0, -1.69297, -1.5708, -2.0944],
    ],
    dtype=np.float64,
)
DEFAULT_JOINT_LIMIT_MAX = np.array(
    [
        [3.14159, 3.66519, 3.14159, 1.5708, 1.5708, 2.0944],
        [3.14159, 3.66519, 3.14159, 1.5708, 1.5708, 2.0944],
    ],
    dtype=np.float64,
)


def as_qpos14(value: Any, *, name: str = "qpos") -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (14,):
        raise ValueError(f"{name} must have shape (14,), got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    if np.any((array[GRIPPER_INDICES] < 0.0) | (array[GRIPPER_INDICES] > 1.0)):
        raise ValueError(f"{name} grippers must be in [0,1], 0=closed, 1=open")
    return array.copy()


def split_qpos14(qpos: Any) -> tuple[np.ndarray, np.ndarray]:
    vector = as_qpos14(qpos)
    return vector[ARM_SLICES["left"]].copy(), vector[ARM_SLICES["right"]].copy()


def joint_limits_from_config(
    config: dict[str, Any] | None,
) -> tuple[np.ndarray, np.ndarray]:
    config = config or {}
    lower = np.asarray(
        config.get("joint_limit_min", DEFAULT_JOINT_LIMIT_MIN), dtype=np.float64
    )
    upper = np.asarray(
        config.get("joint_limit_max", DEFAULT_JOINT_LIMIT_MAX), dtype=np.float64
    )
    if lower.shape != (2, 6) or upper.shape != (2, 6):
        raise ValueError(
            f"YAM joint limits must have shape (2,6), got {lower.shape}/{upper.shape}"
        )
    if not np.isfinite(lower).all() or not np.isfinite(upper).all():
        raise ValueError("YAM joint limits must be finite")
    if not np.all(lower < upper):
        raise ValueError("each YAM joint lower limit must be below its upper limit")
    return lower.copy(), upper.copy()


def enforce_hard_limits(
    qpos: Any,
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    name: str = "qpos",
) -> np.ndarray:
    vector = as_qpos14(qpos, name=name)
    left, right = split_qpos14(vector)
    for arm_name, arm, lo, hi in (
        ("left", left, lower[0], upper[0]),
        ("right", right, lower[1], upper[1]),
    ):
        below = arm[:6] < lo
        above = arm[:6] > hi
        if np.any(below) or np.any(above):
            indexes = np.flatnonzero(below | above).tolist()
            raise ValueError(
                f"{name} {arm_name} arm joints outside hard limits at {indexes}"
            )
    return vector


def apply_previous_command_slew(
    target: Any,
    previous_command: Any,
    max_joint_delta: float | None,
) -> tuple[np.ndarray, bool]:
    """Clamp arm joints relative to the previous command, not measured qpos."""
    accepted = as_qpos14(target, name="target")
    if max_joint_delta is None:
        return accepted, False
    delta = float(max_joint_delta)
    if not np.isfinite(delta) or delta <= 0:
        raise ValueError("max_joint_delta_per_step must be finite and positive")
    previous = as_qpos14(previous_command, name="previous_command")
    before = accepted.copy()
    accepted[ARM_JOINT_INDICES] = np.clip(
        accepted[ARM_JOINT_INDICES],
        previous[ARM_JOINT_INDICES] - delta,
        previous[ARM_JOINT_INDICES] + delta,
    )
    return accepted, not np.array_equal(before, accepted)


def matrix4(value: Any, *, name: str = "transform") -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError(f"{name} must be a finite 4x4 matrix")
    if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0]):
        raise ValueError(f"{name} bottom row must be [0,0,0,1]")
    return matrix.copy()


def pose_to_matrix(value: Any) -> np.ndarray:
    """Accept either 4x4 or xyz+wxyz pose and return a 4x4 matrix."""
    array = np.asarray(value, dtype=np.float64)
    if array.shape == (4, 4):
        return matrix4(array, name="target_pose")
    if array.shape != (7,):
        raise ValueError("target_pose must be 4x4 or xyz+wxyz shape (7,)")
    xyz = array[:3]
    quat = array[3:]
    if not np.isfinite(array).all():
        raise ValueError("target_pose must contain only finite values")
    norm = float(np.linalg.norm(quat))
    if norm <= 0:
        raise ValueError("target_pose quaternion must be non-zero")
    w, x, y, z = quat / norm
    rot = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = rot
    result[:3, 3] = xyz
    return result


def matrix_to_xyz_wxyz(matrix: Any) -> np.ndarray:
    transform = matrix4(matrix)
    rot = transform[:3, :3]
    trace = float(np.trace(rot))
    if trace > 0:
        s = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (rot[2, 1] - rot[1, 2]) / s
        y = (rot[0, 2] - rot[2, 0]) / s
        z = (rot[1, 0] - rot[0, 1]) / s
    else:
        index = int(np.argmax(np.diag(rot)))
        if index == 0:
            s = np.sqrt(1.0 + rot[0, 0] - rot[1, 1] - rot[2, 2]) * 2.0
            w = (rot[2, 1] - rot[1, 2]) / s
            x = 0.25 * s
            y = (rot[0, 1] + rot[1, 0]) / s
            z = (rot[0, 2] + rot[2, 0]) / s
        elif index == 1:
            s = np.sqrt(1.0 + rot[1, 1] - rot[0, 0] - rot[2, 2]) * 2.0
            w = (rot[0, 2] - rot[2, 0]) / s
            x = (rot[0, 1] + rot[1, 0]) / s
            y = 0.25 * s
            z = (rot[1, 2] + rot[2, 1]) / s
        else:
            s = np.sqrt(1.0 + rot[2, 2] - rot[0, 0] - rot[1, 1]) * 2.0
            w = (rot[1, 0] - rot[0, 1]) / s
            x = (rot[0, 2] + rot[2, 0]) / s
            y = (rot[1, 2] + rot[2, 1]) / s
            z = 0.25 * s
    quat = np.asarray([w, x, y, z], dtype=np.float64)
    quat /= np.linalg.norm(quat)
    return np.concatenate([transform[:3, 3], quat])


def _load_json_path(value: str | Path | None) -> dict[str, Any]:
    if value is None:
        return {}
    return json.loads(Path(value).expanduser().read_text())


@dataclass(frozen=True)
class YamCalibration:
    """Transforms into the RPent world frame, defined as the left robot base."""

    camera_to_world: dict[str, np.ndarray]
    world_from_right_base: np.ndarray | None
    wrist_grasp_from_camera: dict[str, np.ndarray]
    world_frame: str = "left_base"

    def cam2world_cv(self, camera_name: str) -> np.ndarray:
        if camera_name not in self.camera_to_world:
            raise ValueError(f"missing cam2world_cv for YAM camera {camera_name!r}")
        return self.camera_to_world[camera_name].copy()

    def arm_target_from_world(
        self, arm: Literal["left", "right"], target_world: Any
    ) -> np.ndarray:
        target = pose_to_matrix(target_world)
        if arm == "left":
            return target
        if self.world_from_right_base is None:
            raise ValueError(
                "right-arm planning requires extrinsics_path with "
                "top_camera.T_base_to_cam via both arms"
            )
        return np.linalg.inv(self.world_from_right_base) @ target

    def pose_to_world(
        self, arm: Literal["left", "right"], pose_in_arm_base: Any
    ) -> np.ndarray:
        pose = matrix4(pose_in_arm_base, name=f"{arm}_eef_pose")
        if arm == "left":
            return pose
        if self.world_from_right_base is None:
            raise ValueError("right-arm pose requires T_world_from_right_base")
        return self.world_from_right_base @ pose


def load_calibration(config: dict[str, Any] | None) -> YamCalibration:
    """Load the RLinf solve_handeye extrinsics.json producer format."""
    config = config or {}
    calibration = _load_json_path(config.get("extrinsics_path"))
    camera_to_world: dict[str, np.ndarray] = {}
    wrist_grasp_from_camera: dict[str, np.ndarray] = {}
    right_base_from_top: np.ndarray | None = None

    top_transforms = calibration.get("top_camera", {}).get("T_base_to_cam", {})
    if "via_left_arm" in top_transforms:
        camera_to_world["top"] = matrix4(
            top_transforms["via_left_arm"],
            name="top_camera.T_base_to_cam.via_left_arm",
        )
    if "via_right_arm" in top_transforms:
        right_base_from_top = matrix4(
            top_transforms["via_right_arm"],
            name="top_camera.T_base_to_cam.via_right_arm",
        )

    for key, arm in (("left_wrist", "left"), ("right_wrist", "right")):
        values = calibration.get(key)
        if isinstance(values, dict) and values.get("T_grasp_to_cam") is not None:
            # RLinf solve_handeye.py verifies FK @ t_tcp_cam @ board_in_cam
            # and writes t_tcp_cam under this historical key. Do not invert.
            wrist_grasp_from_camera[arm] = matrix4(
                values["T_grasp_to_cam"], name=f"{key}.T_grasp_to_cam"
            )

    world_from_right = None
    if "top" in camera_to_world and right_base_from_top is not None:
        world_from_right = camera_to_world["top"] @ np.linalg.inv(right_base_from_top)

    return YamCalibration(
        camera_to_world=camera_to_world,
        world_from_right_base=world_from_right,
        wrist_grasp_from_camera=wrist_grasp_from_camera,
    )


@dataclass(frozen=True)
class _TableSurface:
    """Site-supplied convex tabletop, extruded downward in world Z.

    This is a keep-out volume, not a certification of the physical tabletop.
    The caller must verify the footprint, plane, depth and uncertainty on site.
    """

    plane: np.ndarray
    footprint: np.ndarray
    depth_m: float
    uncertainty_m: float

    @classmethod
    def from_config(cls, config: Any) -> _TableSurface:
        required = {"plane_z_equals_ax_by_c", "footprint_xy", "depth_m"}
        if (
            not isinstance(config, dict)
            or not required.issubset(config)
            or set(config) - required - {"uncertainty_m"}
        ):
            raise ValueError(
                "table_surface requires plane_z_equals_ax_by_c, footprint_xy, depth_m; optional uncertainty_m"
            )
        plane = np.asarray(config["plane_z_equals_ax_by_c"], dtype=np.float64)
        points = np.asarray(config["footprint_xy"], dtype=np.float64)
        depth = float(config["depth_m"])
        uncertainty = float(config.get("uncertainty_m", 0.0))
        if plane.shape != (3,) or not np.isfinite(plane).all():
            raise ValueError(
                "table_surface plane must contain three finite coefficients"
            )
        if (
            points.ndim != 2
            or points.shape[1] != 2
            or len(points) < 3
            or not np.isfinite(points).all()
        ):
            raise ValueError(
                "table_surface footprint_xy must contain at least three finite XY vertices"
            )
        if (
            not np.isfinite(depth)
            or depth <= 0
            or not np.isfinite(uncertainty)
            or uncertainty < 0
        ):
            raise ValueError(
                "table_surface depth_m must be positive and uncertainty_m non-negative, both finite"
            )
        # Require cyclic convex vertices: never silently take a convex hull or
        # reorder an invalid footprint and change the site's approved region.
        edges = np.roll(points, -1, axis=0) - points
        next_edges = np.roll(edges, -1, axis=0)
        turns = edges[:, 0] * next_edges[:, 1] - edges[:, 1] * next_edges[:, 0]
        if np.all(turns < -1e-10):
            points = points[::-1].copy()
            edges = np.roll(points, -1, axis=0) - points
        elif not np.all(turns > 1e-10):
            raise ValueError(
                "table_surface footprint must be strictly convex and cyclic"
            )
        offsets = points[None, :, :] - points[:, None, :]
        sides = (
            edges[:, None, 0] * offsets[:, :, 1] - edges[:, None, 1] * offsets[:, :, 0]
        )
        if np.any(sides < -1e-10):
            raise ValueError("table_surface footprint must not self-intersect")
        return cls(plane.copy(), points.copy(), depth, uncertainty)

    def vertices(self) -> np.ndarray:
        z = self.footprint @ self.plane[:2] + self.plane[2]
        top = np.column_stack((self.footprint, z))
        bottom = top - [0, 0, self.depth_m]
        return np.concatenate((top, bottom))

    def tcp_clearance(self, point: np.ndarray) -> float:
        edges = np.roll(self.footprint, -1, axis=0) - self.footprint
        offsets = point[:2] - self.footprint
        sides = edges[:, 0] * offsets[:, 1] - edges[:, 1] * offsets[:, 0]
        # Expanded footprint keeps the TCP check conservative near uncertain
        # table edges. Mesh distance separately covers the entire robot shape.
        if np.any(sides < -self.uncertainty_m * np.linalg.norm(edges, axis=1) - 1e-10):
            return float("inf")
        a, b, c = self.plane
        return float(
            (point[2] - a * point[0] - b * point[1] - c) / np.sqrt(1 + a * a + b * b)
            - self.uncertainty_m
        )


def _link3_convex_parts(vertices: np.ndarray, faces: np.ndarray) -> list[np.ndarray]:
    """Conservative local hulls without discarding or clipping any triangle.

    Each slab includes whole triangles intersecting it. Their hull encloses
    the original solid's slab intersection (including every boundary point).
    Overlap across slabs is intentional; this is not mesh decimation.
    """
    axis = int(np.argmax(np.ptp(vertices, axis=0)))
    bounds = np.linspace(vertices[:, axis].min(), vertices[:, axis].max(), 5)
    coordinates = vertices[faces, axis]
    lower, upper = coordinates.min(axis=1), coordinates.max(axis=1)
    parts = []
    for low, high in zip(bounds[:-1], bounds[1:]):
        selected = faces[(lower <= high) & (upper >= low)]
        if selected.size:
            parts.append(vertices[np.unique(selected)])
    return parts


class _ModelCollisionGuard:
    """Check convex MuJoCo meshes from the same models used for FK/IK.

    Adjacent/welded bodies and the model's explicit exclusions are mechanical
    contacts, not tested self-collision pairs. All cross-arm pairs are tested,
    including static bases. Camera mounts, cables, held objects, leader arms,
    and fixtures absent from these models are not covered. Path sampling is
    not a continuous collision or tracking-error guarantee.
    """

    def __init__(self, geometry: YamGeometry, clearance_m: float) -> None:
        import mujoco

        self._mj = mujoco
        self.clearance_m = clearance_m
        self.base_link2_clearance_m = geometry.base_link2_clearance_m
        self.left_link3_link5_clearance_m = geometry.left_link3_link5_clearance_m
        right_base = geometry.calibration.world_from_right_base
        if right_base is None:
            raise ValueError("collision_guard requires calibrated right base transform")
        if not np.allclose(
            right_base[:3, :3].T @ right_base[:3, :3], np.eye(3)
        ) or not np.isclose(np.linalg.det(right_base[:3, :3]), 1.0):
            raise ValueError("collision_guard requires a rigid right base transform")
        spec = mujoco.MjSpec()
        for arm, transform in (("left", np.eye(4)), ("right", right_base)):
            kin = geometry._kinematics_for(arm)
            if not hasattr(kin, "model"):
                raise ValueError(
                    "collision_guard requires the actual MuJoCo kinematics model"
                )
            self._validate_model(kin.model)
            # Serialize each already-loaded model instead of reconstructing a
            # different gripper/arm model or modifying RLinf's mutable IK data.
            with tempfile.TemporaryDirectory(
                prefix="rpent-yam-collision-"
            ) as directory:
                xml_path = str(Path(directory) / "arm.xml")
                mujoco.mj_saveLastXML(xml_path, kin.model)
                child = mujoco.MjSpec.from_file(xml_path)
                if arm == "left":
                    # Only refine the observed link3/link5 concavity. Other
                    # pairs retain the original whole-link convex geometry.
                    model = kin.model
                    geom = int(model.body_geomadr[model.body("link3").id])
                    mesh = int(model.geom_dataid[geom])
                    start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
                    vertices = model.mesh_vert[start : start + count]
                    start, count = model.mesh_faceadr[mesh], model.mesh_facenum[mesh]
                    faces = model.mesh_face[start : start + count]
                    rotation = np.empty(9)
                    mujoco.mju_quat2Mat(rotation, model.geom_quat[geom])
                    vertices = (
                        vertices @ rotation.reshape(3, 3).T + model.geom_pos[geom]
                    )
                    for index, part in enumerate(_link3_convex_parts(vertices, faces)):
                        name = f"guard_link3_part{index}"
                        child.add_mesh(name=name, uservert=part.reshape(-1))
                        child.body("link3").add_geom(
                            name=name,
                            type=mujoco.mjtGeom.mjGEOM_MESH,
                            meshname=name,
                            density=0,
                            contype=0,
                            conaffinity=0,
                            rgba=[0, 0, 0, 0],
                        )
                pose = matrix_to_xyz_wxyz(transform)
                frame = spec.worldbody.add_frame(pos=pose[:3], quat=pose[3:])
                spec.attach(child, prefix=f"{arm}_", frame=frame)
        self.table_z = geometry.table_z
        self.table_surface = geometry.table_surface
        self.table_mesh_configured = geometry.table_guard_configured
        if self.table_surface is not None:
            spec.add_mesh(
                name="guard_table_mesh",
                uservert=self.table_surface.vertices().reshape(-1),
            )
            spec.worldbody.add_geom(
                name="guard_table",
                type=mujoco.mjtGeom.mjGEOM_MESH,
                meshname="guard_table_mesh",
            )
        elif self.table_z is not None:
            spec.worldbody.add_geom(
                name="guard_table",
                type=mujoco.mjtGeom.mjGEOM_PLANE,
                pos=[0, 0, self.table_z],
                size=[0, 0, 0.01],
            )
        self.model = spec.compile()
        self.data = mujoco.MjData(self.model)
        self._link3_parts = tuple(
            i
            for i in range(self.model.ngeom)
            if self.model.geom(i).name.startswith("left_guard_link3_part")
        )
        self._q_addresses = {
            arm: np.array(
                [self.model.joint(f"{arm}_joint{i}").qposadr[0] for i in range(1, 9)]
            )
            for arm in ("left", "right")
        }
        self._finger_ranges = {
            arm: np.array([self.model.joint(f"{arm}_joint{i}").range for i in (7, 8)])
            for arm in ("left", "right")
        }
        self._pairs = []
        excluded = {int(value) for value in self.model.exclude_signature}
        table_id = (
            self.model.geom("guard_table").id if self.table_mesh_configured else None
        )
        self._table_pairs = []
        for first in range(self.model.ngeom):
            for second in range(first + 1, self.model.ngeom):
                if first in self._link3_parts or second in self._link3_parts:
                    continue
                a = int(self.model.geom_bodyid[first])
                b = int(self.model.geom_bodyid[second])
                if table_id in (first, second):
                    other = second if first == table_id else first
                    # Only fixed installation geometry may touch its support.
                    if self.model.body_weldid[self.model.geom_bodyid[other]] != 0:
                        self._table_pairs.append((first, second))
                    continue
                same_arm = (
                    self.model.body(a).name.split("_", 1)[0]
                    == self.model.body(b).name.split("_", 1)[0]
                )
                if same_arm:
                    wa, wb = self.model.body_weldid[[a, b]]
                    adjacent = (
                        self.model.body_weldid[self.model.body_parentid[wa]] == wb
                        or self.model.body_weldid[self.model.body_parentid[wb]] == wa
                    )
                    if (
                        wa == wb
                        or adjacent
                        or (min(a, b) << 16 | max(a, b)) in excluded
                    ):
                        continue
                self._pairs.append((first, second))
        if not any(
            self.model.body(int(self.model.geom_bodyid[a])).name.startswith("left_")
            and self.model.body(int(self.model.geom_bodyid[b])).name.startswith(
                "right_"
            )
            for a, b in self._pairs
        ):
            raise ValueError("collision_guard requires cross-arm collision pairs")

    def _validate_model(self, model: Any) -> None:
        """Reject incomplete/unsupported models instead of reporting a safe path.

        These names and mesh counts are the installed YAM + flexible_4310
        model contract, also used by RLinf's kinematics adapter. They are not
        tunable safety parameters or a substitute for physical model validation.
        """
        bodies = [
            "base",
            "link1",
            "link2",
            "link3",
            "link4",
            "link5",
            "gripper",
            "linear_module",
            "linear_module_2",
        ]
        if (
            model.njnt != 8
            or model.nq != 8
            or model.nv != 8
            or {model.joint(i).name for i in range(model.njnt)}
            != {f"joint{i}" for i in range(1, 9)}
        ):
            raise ValueError(
                "collision_guard requires the supported eight-joint YAM model"
            )
        if model.nbody != 10 or {
            model.body(i).name for i in range(1, model.nbody)
        } != set(bodies):
            raise ValueError("collision_guard requires the supported YAM body layout")
        for index, name in enumerate(bodies):
            body = model.body(name)
            expected_parent = (
                "world" if index == 0 else bodies[index - 1] if index < 8 else "gripper"
            )
            if model.body(int(body.parentid[0])).name != expected_parent:
                raise ValueError(f"collision_guard unsupported parent for {name}")
            count = 2 if name in ("linear_module", "linear_module_2") else 1
            geoms = np.flatnonzero(model.geom_bodyid == body.id)
            if len(geoms) != count or np.any(
                model.geom_type[geoms] != self._mj.mjtGeom.mjGEOM_MESH
            ):
                raise ValueError(
                    f"collision_guard missing or unsupported mesh geometry for {name}"
                )
            if not np.isfinite(model.geom_rbound[geoms]).all() or np.any(
                model.geom_rbound[geoms] <= 0
            ):
                raise ValueError(f"collision_guard invalid mesh bounds for {name}")
        expected_exclusions = {
            frozenset(("gripper", "linear_module")),
            frozenset(("gripper", "linear_module_2")),
            frozenset(("linear_module", "linear_module_2")),
        }
        exclusions = {
            frozenset(
                (model.body(int(value) >> 16).name, model.body(int(value) & 65535).name)
            )
            for value in model.exclude_signature
        }
        if exclusions != expected_exclusions:
            raise ValueError("collision_guard unsupported model collision exclusions")
        for index in range(1, 9):
            joint = model.joint(f"joint{index}")
            expected_type = (
                self._mj.mjtJoint.mjJNT_HINGE
                if index <= 6
                else self._mj.mjtJoint.mjJNT_SLIDE
            )
            if (
                joint.type[0] != expected_type
                or model.body(int(joint.bodyid[0])).name != bodies[index]
            ):
                raise ValueError(
                    f"collision_guard unsupported joint{index} type or body"
                )
            if (
                not joint.limited[0]
                or not np.isfinite(joint.range).all()
                or joint.range[0] >= joint.range[1]
            ):
                raise ValueError(f"collision_guard invalid joint{index} range")

    def check(self, qpos: np.ndarray, *, table_escape=None) -> dict[str, Any]:
        for arm in ("left", "right"):
            values = qpos[ARM_SLICES[arm]]
            ranges = self._finger_ranges[arm]
            self.data.qpos[self._q_addresses[arm]] = np.concatenate(
                [values[:6], ranges[:, 0] + values[6] * (ranges[:, 1] - ranges[:, 0])]
            )
        self._mj.mj_kinematics(self.model, self.data)
        closest_points = np.zeros(6)
        for first, second in self._pairs + self._table_pairs:
            table_pair = (first, second) in self._table_pairs
            margin = self.clearance_m + (
                self.table_surface.uncertainty_m
                if table_pair and self.table_surface is not None
                else 0.0
            )
            if not table_pair:
                names = [
                    self.model.body(int(self.model.geom_bodyid[i])).name
                    for i in (first, second)
                ]
                arm_a, body_a = names[0].split("_", 1)
                arm_b, body_b = names[1].split("_", 1)
                if arm_a == arm_b and {body_a, body_b} == {"base", "link2"}:
                    margin = self.base_link2_clearance_m
                elif set(names) == {"left_link3", "left_link5"}:
                    margin = self.left_link3_link5_clearance_m
            # Bounding spheres only reject distant pairs; near pairs always
            # use mesh distance rather than an invented link radius.
            separation = np.linalg.norm(
                self.data.geom_xpos[first] - self.data.geom_xpos[second]
            )
            if (
                not table_pair
                and separation
                > self.model.geom_rbound[first]
                + self.model.geom_rbound[second]
                + self.clearance_m
            ):
                continue
            distance = self._mj.mj_geomDistance(
                self.model, self.data, first, second, margin + 1e-6, closest_points
            )
            distance_kind = "model_convex_surface"
            if (
                not table_pair
                and set(names) == {"left_link3", "left_link5"}
                and distance <= margin
            ):
                link5 = first if names[0] == "left_link5" else second
                candidates = []
                for part in self._link3_parts:
                    points = np.zeros(6)
                    gap = self._mj.mj_geomDistance(
                        self.model, self.data, part, link5, margin + 1e-6, points
                    )
                    candidates.append((gap, points))
                distance, closest_points = min(candidates, key=lambda item: item[0])
                if names[0] == "left_link5":
                    closest_points = closest_points.reshape(2, 3)[::-1].reshape(6)
                distance_kind = "model_convex_parts"
            # Operator retreat may leave one positive table clearance below
            # its normal margin. All other pairs retain their normal checks.
            if table_pair and table_escape is not None:
                bodies, floor = table_escape
                names = [
                    self.model.body(int(self.model.geom_bodyid[i])).name
                    for i in (first, second)
                ]
                if set(names) == set(bodies) and distance >= floor > 0:
                    continue
            if distance <= margin:
                return {
                    "ok": False,
                    "checked": True,
                    "reason": "link_table_guard" if table_pair else "collision_guard",
                    "bodies": [
                        self.model.body(int(self.model.geom_bodyid[i])).name
                        for i in (first, second)
                    ],
                    "distance_m": float(distance),
                    "distance_kind": distance_kind,
                    "closest_points_world": closest_points.reshape(2, 3).tolist(),
                    "world_frame": "left_base",
                    "required_clearance_m": margin,
                    "limitation": "model convex geometry distance, not motor-center distance or measured physical clearance; convex hulls fill mesh concavities and may underestimate the original mesh surface gap",
                }
        return {
            "ok": True,
            "checked": True,
            "reason": None,
            "required_clearance_m": self.clearance_m,
            "table_mesh_checked": self.table_mesh_configured,
            "limitation": "sampled arm mesh convex hulls only; excludes adjacent/model-excluded pairs, cameras, cables, held objects, leader arms and fixtures; table geometry checked only when table_z or table_surface is configured",
        }


class YamGeometry:
    """Planner and pose adapter backed by RLinf's YAM kinematics."""

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        *,
        kinematics: dict[str, Any] | Any | None = None,
    ) -> None:
        self.config = config or {}
        self.calibration = load_calibration(self.config)
        self.lower, self.upper = joint_limits_from_config(self.config)
        self.table_z = (
            None
            if self.config.get("table_z") is None
            else float(self.config["table_z"])
        )
        self.table_surface = (
            None
            if self.config.get("table_surface") is None
            else _TableSurface.from_config(self.config["table_surface"])
        )
        if self.table_z is not None and self.table_surface is not None:
            raise ValueError("configure only one of table_z and table_surface")
        self.table_clearance_m = float(self.config.get("table_clearance_m", 0.03))
        if self.table_z is not None and not np.isfinite(self.table_z):
            raise ValueError("table_z must be finite")
        if not np.isfinite(self.table_clearance_m) or self.table_clearance_m < 0:
            raise ValueError("table_clearance_m must be finite and non-negative")
        self.path_joint_delta = float(self.config.get("path_joint_delta", 0.02))
        if not np.isfinite(self.path_joint_delta) or self.path_joint_delta <= 0:
            raise ValueError("path_joint_delta must be finite and positive")
        self._kinematics = kinematics
        guard = self.config.get("collision_guard", {})
        if not isinstance(guard, dict) or not isinstance(
            guard.get("enabled", False), bool
        ):
            raise ValueError("collision_guard must be a mapping with boolean enabled")
        self.collision_guard_enabled = guard.get("enabled", False)
        self.collision_clearance_m = float(guard.get("clearance_m", 0.01))
        if (
            not np.isfinite(self.collision_clearance_m)
            or self.collision_clearance_m <= 0
        ):
            raise ValueError("collision_guard.clearance_m must be finite and positive")
        self.base_link2_clearance_m = float(
            guard.get("base_link2_clearance_m", self.collision_clearance_m)
        )
        if (
            not np.isfinite(self.base_link2_clearance_m)
            or not 0 < self.base_link2_clearance_m <= self.collision_clearance_m
        ):
            raise ValueError(
                "collision_guard.base_link2_clearance_m must be finite, positive and <= clearance_m"
            )
        self.left_link3_link5_clearance_m = float(
            guard.get("left_link3_link5_clearance_m", self.collision_clearance_m)
        )
        if (
            not np.isfinite(self.left_link3_link5_clearance_m)
            or not 0 < self.left_link3_link5_clearance_m <= self.collision_clearance_m
        ):
            raise ValueError(
                "collision_guard.left_link3_link5_clearance_m must be finite, positive and <= clearance_m"
            )
        if self.table_surface is not None and not self.collision_guard_enabled:
            raise ValueError(
                "table_surface requires collision_guard.enabled for mesh protection"
            )
        self._collision_guard: _ModelCollisionGuard | None = None

    @property
    def table_guard_configured(self) -> bool:
        """Whether a validated table representation is configured, not certified."""
        return self.table_z is not None or self.table_surface is not None

    def _tcp_table_clearance(self, point: np.ndarray) -> float:
        if self.table_surface is not None:
            return self.table_surface.tcp_clearance(point)
        return float(point[2] - self.table_z)

    def _kinematics_for(self, arm: Literal["left", "right"]) -> Any:
        if isinstance(self._kinematics, dict):
            if arm in self._kinematics:
                return self._kinematics[arm]
        elif self._kinematics is not None:
            return self._kinematics
        rlinf_root = get_rlinf_repo_path()
        if rlinf_root is not None and str(rlinf_root) not in sys.path:
            sys.path.insert(0, str(rlinf_root))
        try:
            from rlinf.envs.realworld.yam.kinematics import YamKinematicsAdapter
        except ModuleNotFoundError as error:
            raise RuntimeError(
                "missing dependency while importing RLinf YAM kinematics. "
                "Set RPENT_RLINF_ROOT or RLINF_REPO_PATH to the RLinf checkout "
                "and install the pinned YAM/i2rt dependencies on the robot host."
            ) from error
        adapter = YamKinematicsAdapter(
            joint_lower=self.lower[0 if arm == "left" else 1],
            joint_upper=self.upper[0 if arm == "left" else 1],
        )
        if self._kinematics is None:
            self._kinematics = {}
        self._kinematics[arm] = adapter
        return adapter

    def eef_pose(self, arm: Literal["left", "right"], qpos14: Any) -> np.ndarray:
        qpos = enforce_hard_limits(qpos14, self.lower, self.upper)
        arm_qpos = qpos[ARM_SLICES[arm]]
        fk = self._kinematics_for(arm).fk(arm_qpos[:6], float(arm_qpos[6]))
        return matrix_to_xyz_wxyz(self.calibration.pose_to_world(arm, fk))

    def robot_state(self, qpos14: Any) -> dict[str, Any]:
        qpos = enforce_hard_limits(qpos14, self.lower, self.upper)
        return {
            "qpos": qpos.astype(np.float64),
            "left_eef_pose": self.eef_pose("left", qpos),
            "right_eef_pose": self.eef_pose("right", qpos),
            "world_frame": self.calibration.world_frame,
        }

    def camera_meta_for_qpos(
        self, camera_name: str, base_meta: dict[str, Any], qpos14: Any
    ) -> dict[str, Any]:
        """Return camera metadata with wrist cam poses tied to measured qpos."""
        name = str(camera_name)
        meta = dict(base_meta)
        if name == "top":
            meta["cam2world_cv"] = self.calibration.cam2world_cv("top")
            meta["cam2world_source"] = "static_top_extrinsics"
            return meta
        if name not in {"left", "right"}:
            raise ValueError(f"unknown YAM camera {camera_name!r}")
        qpos = enforce_hard_limits(qpos14, self.lower, self.upper)
        arm_qpos = qpos[ARM_SLICES[name]]
        grasp_from_camera = self.calibration.wrist_grasp_from_camera.get(name)
        if grasp_from_camera is None:
            raise ValueError(
                f"{name} wrist camera requires dynamic hand-eye calibration: "
                f"set extrinsics_path to an RLinf solve_handeye output with "
                f"{name}_wrist.T_grasp_to_cam"
            )
        base_from_grasp = self._kinematics_for(name).fk(
            arm_qpos[:6], float(arm_qpos[6])
        )
        world_from_camera = self.calibration.pose_to_world(
            name, base_from_grasp
        ) @ matrix4(grasp_from_camera, name=f"{name}_wrist.T_grasp_to_cam")
        meta["cam2world_cv"] = world_from_camera
        meta["cam2world_source"] = "dynamic_fk_current_qpos_approx_cached_frame"
        return meta

    def plan_arm_path(
        self,
        arm: Literal["left", "right"],
        target_pose: Any,
        *,
        current_qpos14: Any,
    ) -> dict[str, Any]:
        if arm not in ("left", "right"):
            raise ValueError("arm must be 'left' or 'right'")
        qpos = enforce_hard_limits(current_qpos14, self.lower, self.upper)
        arm_index = 0 if arm == "left" else 1
        seed = qpos[ARM_SLICES[arm]][:6]
        gripper = float(qpos[ARM_SLICES[arm]][6])
        target_in_base = self.calibration.arm_target_from_world(arm, target_pose)
        kin = self._kinematics_for(arm)
        result = kin.solve(target_in_base, seed, gripper)
        if not bool(result.success):
            return {
                "status": "Failure",
                "position": None,
                "reason": getattr(result, "reason", "ik_failed"),
                "position_error": float(getattr(result, "position_error", np.inf)),
                "rotation_error": float(getattr(result, "rotation_error", np.inf)),
            }

        q_target = np.asarray(result.q_target, dtype=np.float64).reshape(6)
        if (
            not np.isfinite(q_target).all()
            or np.any(q_target < self.lower[arm_index])
            or np.any(q_target > self.upper[arm_index])
        ):
            return {
                "status": "Failure",
                "position": None,
                "reason": "joint_limits",
            }
        max_delta = float(np.max(np.abs(q_target - seed)))
        steps = max(2, int(np.ceil(max_delta / self.path_joint_delta)) + 1)
        position = np.linspace(seed, q_target, steps, dtype=np.float64)
        table_check = self._check_table_guard(arm, position, gripper, kin)
        if table_check["checked"] and not table_check["ok"]:
            return {
                "status": "Failure",
                "position": None,
                "reason": "table_guard",
                **table_check,
            }
        collision_check = self._check_collision_path(
            [
                np.concatenate([q, qpos[6:]])
                if arm == "left"
                else np.concatenate([qpos[:7], q, qpos[13:]])
                for q in position
            ]
        )
        if not collision_check["ok"]:
            return {"status": "Failure", "position": None, **collision_check}
        return {
            "status": "Success",
            "position": position,
            "collision_guard": collision_check,
            "reason": None,
            "position_error": float(result.position_error),
            "rotation_error": float(result.rotation_error),
            "table_guard": table_check,
        }

    def check_qpos_transition(
        self, previous_qpos14: Any, target_qpos14: Any, *, samples: int = 8
    ) -> dict[str, Any]:
        """Check sampled TCP height and optional two-arm model clearance."""
        previous = enforce_hard_limits(previous_qpos14, self.lower, self.upper)
        target = enforce_hard_limits(target_qpos14, self.lower, self.upper)
        steps = max(
            2,
            int(samples),
            int(np.ceil(np.max(np.abs(target - previous)) / self.path_joint_delta)) + 1,
        )
        path = np.linspace(previous, target, steps)
        collision_check = self._check_collision_path(path)
        if not collision_check["ok"]:
            return collision_check
        if not self.table_guard_configured:
            if collision_check["checked"]:
                return collision_check
            return {
                "ok": True,
                "checked": False,
                "reason": None,
                "limitation": "no table_z or table_surface configured; table guard disabled",
            }
        min_clearance = float("inf")
        for qpos in path:
            for arm in ("left", "right"):
                arm_qpos = qpos[ARM_SLICES[arm]]
                pose_world = self.calibration.pose_to_world(
                    arm,
                    self._kinematics_for(arm).fk(arm_qpos[:6], float(arm_qpos[6])),
                )
                clearance = self._tcp_table_clearance(pose_world[:3, 3])
                min_clearance = min(min_clearance, clearance)
        return {
            "ok": min_clearance >= self.table_clearance_m,
            "checked": True,
            "reason": None
            if min_clearance >= self.table_clearance_m
            else "table_guard",
            "min_tcp_clearance_m": min_clearance,
            "required_clearance_m": self.table_clearance_m,
            "collision_guard": collision_check,
            "limitation": "TCP table surface clearance within footprint only; additional arm mesh checks are reported separately",
        }

    def _check_collision_path(self, path: Any) -> dict[str, Any]:
        if not self.collision_guard_enabled:
            return {
                "ok": True,
                "checked": False,
                "reason": None,
                "limitation": "collision_guard.enabled is false; no arm collision check",
            }
        if self._collision_guard is None:
            self._collision_guard = _ModelCollisionGuard(
                self, self.collision_clearance_m
            )
        result = {}
        for index, qpos in enumerate(path):
            result = self._collision_guard.check(qpos)
            if not result["ok"]:
                return {**result, "path_sample": index}
        return result

    def _check_table_guard(
        self,
        arm: Literal["left", "right"],
        position: np.ndarray,
        gripper: float,
        kin: Any,
    ) -> dict[str, Any]:
        if not self.table_guard_configured:
            return {
                "ok": True,
                "checked": False,
                "reason": None,
                "limitation": "no table_z or table_surface configured; table guard disabled",
            }
        min_clearance = float("inf")
        for q in position:
            pose_world = self.calibration.pose_to_world(arm, kin.fk(q, gripper))
            clearance = self._tcp_table_clearance(pose_world[:3, 3])
            min_clearance = min(min_clearance, clearance)
        return {
            "ok": min_clearance >= self.table_clearance_m,
            "checked": True,
            "reason": None
            if min_clearance >= self.table_clearance_m
            else "table_guard",
            "min_tcp_clearance_m": min_clearance,
            "required_clearance_m": self.table_clearance_m,
            "limitation": "checks TCP height only; no self, object, or fixture collision guarantee",
        }
