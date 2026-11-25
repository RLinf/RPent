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

"""Pose conversion and bounded motion execution helpers for VLABench."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np

from rpent.tools import ToolResult

if TYPE_CHECKING:
    from robots.vlabench.adapter import VLABenchEnvClient


def quat_from_matrix(matrix: Any) -> np.ndarray:
    """Return the unit xyzw quaternion of a rotation matrix."""
    m = np.asarray(matrix, dtype=float)
    trace = float(np.trace(m))
    if trace > 0:
        s = 2.0 * np.sqrt(trace + 1.0)
        q = [
            (m[2, 1] - m[1, 2]) / s,
            (m[0, 2] - m[2, 0]) / s,
            (m[1, 0] - m[0, 1]) / s,
            0.25 * s,
        ]
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        q = [
            0.25 * s,
            (m[0, 1] + m[1, 0]) / s,
            (m[0, 2] + m[2, 0]) / s,
            (m[2, 1] - m[1, 2]) / s,
        ]
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        q = [
            (m[0, 1] + m[1, 0]) / s,
            0.25 * s,
            (m[1, 2] + m[2, 1]) / s,
            (m[0, 2] - m[2, 0]) / s,
        ]
    else:
        s = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        q = [
            (m[0, 2] + m[2, 0]) / s,
            (m[1, 2] + m[2, 1]) / s,
            0.25 * s,
            (m[1, 0] - m[0, 1]) / s,
        ]
    result = np.asarray(q, dtype=float)
    return result / np.linalg.norm(result)


def rotation_from_vector(vector: Any) -> np.ndarray:
    """Return the rotation matrix of a rotation vector in radians (Rodrigues)."""
    v = np.asarray(vector, dtype=float)
    angle = float(np.linalg.norm(v))
    if angle < 1e-12:
        return np.eye(3)
    x, y, z = v / angle
    k = np.asarray([[0, -z, y], [z, 0, -x], [-y, x, 0]], dtype=float)
    return np.eye(3) + np.sin(angle) * k + (1 - np.cos(angle)) * (k @ k)


def validate_unit_quat(quaternion_xyzw: Any) -> np.ndarray:
    """Validate and return a finite unit xyzw quaternion."""
    quat = np.asarray(quaternion_xyzw, dtype=float)
    if quat.shape != (4,) or not np.isfinite(quat).all():
        raise ValueError("quaternion_xyzw must be a finite unit 4-vector")
    norm = float(np.linalg.norm(quat))
    if norm < 1e-8 or abs(norm - 1.0) > 0.01:
        raise ValueError("quaternion_xyzw must be a finite unit 4-vector")
    return quat / norm


class VLABenchMotion:
    """Bounded pose, delta, gripper, and terminal motion logic for VLABench."""

    def __init__(
        self, *, env: VLABenchEnvClient, check_cancelled: Callable[[], None]
    ) -> None:
        self.env = env
        self._check_cancelled = check_cancelled
        self.native_actions = 0

    def status(self) -> dict[str, Any]:
        """Return the native task status, including success and the step budget."""
        return self.env.task_status()

    def eef_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the current world EEF position and rotation matrix."""
        pose = np.asarray(self.env.robot_model_state()["world_from_eef"], dtype=float)
        return pose[:3, 3], pose[:3, :3]

    def _receipt(self, receipt: dict[str, Any]) -> ToolResult:
        self.native_actions += int(receipt["steps_used"])
        return ToolResult(data={**receipt, "success": bool(receipt["ok"])})

    def _servo(
        self,
        xyz: np.ndarray,
        quat_xyzw: np.ndarray,
        *,
        gripper: Any,
        max_steps: int,
        tol: float,
        step_clip: float,
        ori_tol: float,
    ) -> ToolResult:
        self._check_cancelled()
        receipt = self.env.execute_pose(
            xyz=xyz.tolist(),
            quaternion_xyzw=quat_xyzw.tolist(),
            gripper=gripper,
            max_steps=max_steps,
            tol=tol,
            step_clip=step_clip,
            ori_tol=ori_tol,
            check_cancelled=self._check_cancelled,
        )
        return self._receipt(receipt)

    def finish(self, *, status: str, summary: str) -> ToolResult:
        """Stop the run after checking authoritative native task success."""
        requested_success = status.lower() == "success"
        try:
            native = self.status()
        except Exception as error:  # The terminal tool must still stop the planner.
            return ToolResult(
                data={
                    "_finish": True,
                    "status": "error",
                    "summary": summary,
                    "requested_status": status,
                    "requested_success": requested_success,
                    "runtime_error": f"{type(error).__name__}: {error}",
                }
            )
        verified = native.get("task_success") is True
        return ToolResult(
            data={
                "_finish": True,
                "status": (
                    "success"
                    if verified
                    else ("failure" if requested_success else status)
                ),
                "summary": summary,
                "requested_success": requested_success,
                "success": verified,
                "episode_status": native,
            }
        )

    def move_pose(
        self,
        *,
        xyz: list[float] | None = None,
        quaternion_xyzw: list[float] | None = None,
        gripper: Any = "hold",
        max_steps: int = 150,
        tol: float = 0.012,
        step_clip: float = 0.02,
        ori_tol: float = 0.05,
    ) -> ToolResult:
        """Move the end effector toward an absolute world pose."""
        if xyz is None or quaternion_xyzw is None:
            current_xyz, current_rot = self.eef_pose()
        position = np.asarray(xyz, dtype=float) if xyz is not None else current_xyz
        if position.shape != (3,) or not np.isfinite(position).all():
            raise ValueError("xyz must be a finite 3-vector")
        quat = (
            validate_unit_quat(quaternion_xyzw)
            if quaternion_xyzw is not None
            else quat_from_matrix(current_rot)
        )
        return self._servo(
            position,
            quat,
            gripper=gripper,
            max_steps=max_steps,
            tol=tol,
            step_clip=step_clip,
            ori_tol=ori_tol,
        )

    def move_delta(
        self,
        *,
        dxyz: list[float] | None = None,
        rotation_vector_world: list[float] | None = None,
        gripper: Any = "hold",
        max_steps: int = 120,
        tol: float = 0.012,
        step_clip: float = 0.02,
        ori_tol: float = 0.05,
    ) -> ToolResult:
        """Move relative to the measured world EEF pose."""
        position, rotation = self.eef_pose()
        if dxyz is not None:
            delta = np.asarray(dxyz, dtype=float)
            if delta.shape != (3,) or not np.isfinite(delta).all():
                raise ValueError("dxyz must be a finite 3-vector")
            position = position + delta
        if rotation_vector_world is not None:
            rotvec = np.asarray(rotation_vector_world, dtype=float)
            if rotvec.shape != (3,) or not np.isfinite(rotvec).all():
                raise ValueError("rotation_vector_world must be a finite 3-vector")
            rotation = rotation_from_vector(rotvec) @ rotation
        return self._servo(
            position,
            quat_from_matrix(rotation),
            gripper=gripper,
            max_steps=max_steps,
            tol=tol,
            step_clip=step_clip,
            ori_tol=ori_tol,
        )

    def set_gripper(self, *, gripper: float, steps: int = 5) -> ToolResult:
        """Hold the arm and drive the fingers for a bounded number of steps."""
        self._check_cancelled()
        receipt = self.env.execute_gripper(
            gripper=gripper, steps=steps, check_cancelled=self._check_cancelled
        )
        return self._receipt(receipt)
