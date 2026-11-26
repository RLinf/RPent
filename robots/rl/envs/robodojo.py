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

"""RoboDojo dual-arm ARX-X5 general-frame RL adapter."""

from __future__ import annotations

from typing import Annotated, Any, Literal

import numpy as np
from pydantic import Field
from scipy.spatial.transform import Rotation

from robots.rl.geometry import Coordinates, pose_matrix
from robots.rl.perception import RecordedPerception
from robots.robodojo.access import CAMERAS
from robots.robodojo.primitives import RoboDojoPrimitives as NativePrimitives
from robots.robodojo.tools import _world_from_depth
from rpent.tools import ToolResult, tool
from rpent.tools.toolkit import ToolCancelled
from rpent.utils.transforms import quaternion_wxyz

Vector3 = Annotated[list[float], Field(min_length=3, max_length=3)]


class RoboDojoRLPrimitives(RecordedPerception):
    """Episode adapter using RoboDojo's native planner, acknowledged actions, and RGB-D."""

    cameras = CAMERAS

    def __init__(
        self,
        *,
        env: Any,
        sam3_client: Any = None,
        state: Any = None,
        check_cancelled,
        max_episode_steps: int = 10000,
        record_artifacts: bool = False,
    ) -> None:
        self._init_perception(
            env=env,
            sam3_client=sam3_client,
            state=state,
            check_cancelled=check_cancelled,
            max_episode_steps=max_episode_steps,
            record_artifacts=record_artifacts,
        )
        self.native = NativePrimitives(
            env=self.env, check_cancelled=self._check_cancelled
        )
        self.calibration = self.env.coordinate_calibration()
        self.coordinates = Coordinates.from_control(self.calibration)
        self._save_artifact(
            "coordinates.json",
            {**self.coordinates.metadata(), "calibration": self.calibration},
            None,
        )
        self._captured: dict[str, Any] | None = None
        self._last_robot_state: dict[str, Any] | None = None

    def _check_execution(self, result: ToolResult) -> bool:
        status = result.data.get("episode_status")
        if status is not None:
            self.env_steps = int(status["step"])
            self._success |= bool(status.get("success", False))
        if result.data.get("stop_reason") in {
            "execution_unknown",
            "synchronization_lost",
            "runtime_failure",
        } or result.data.get("feedback_error"):
            self.infrastructure_error = dict(result.data)
            self._save_artifact(
                "infrastructure_error.json", self.infrastructure_error, None
            )
            result.data["status"] = "error"
            return False
        return True

    def _pose(self, arm: str, motion: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
        native = pose_matrix(motion[f"{arm}_eef_pose"])
        cal = self.calibration["tools"][arm]
        offset = np.asarray(cal["offset_in_native_m"])
        rotation = np.asarray(cal["rotation_to_native"])
        xyz = native[:3, 3] + native[:3, :3] @ offset
        axes = native[:3, :3] @ rotation
        return self.coordinates.points(xyz), self.coordinates.orientation(axes)

    def _native_call(self, operation, *args, **kwargs):
        if self.infrastructure_error is not None:
            raise RuntimeError("native environment unavailable; attempt has ended")
        try:
            return operation(*args, **kwargs)
        except ToolCancelled:
            raise
        except Exception as error:
            self.infrastructure_error = {
                "stop_reason": "runtime_failure",
                "error": f"{type(error).__name__}: {error}",
                "execution_uncertain": True,
            }
            self._save_artifact(
                "infrastructure_error.json", self.infrastructure_error, None
            )
            raise

    def native_success(self) -> bool:
        if self.infrastructure_error is not None:
            return self._success
        status = self._native_call(self.env.get_status)
        self.env_steps = int(status["step"])
        self._success |= bool(status["success"])
        return self._success

    @property
    def ended(self) -> bool:
        return (
            self.native_success()
            or self.env.episode_ended
            or self.env_steps >= self._max_episode_steps
        )

    def robot_state(self) -> dict[str, Any]:
        if self.infrastructure_error is not None:
            return {**(self._last_robot_state or {}), "state_stale": True}
        status = self._native_call(self.env.get_status)
        self.env_steps = int(status["step"])
        motion = self._native_call(self.env.get_motion_state)
        result: dict[str, Any] = {"env_steps": self.env_steps}
        for arm in ("left", "right"):
            xyz, rotation = self._pose(arm, motion)
            result[arm] = {
                "pose": {
                    "xyz": xyz.tolist(),
                    "orientation_rpy": Rotation.from_matrix(rotation)
                    .as_euler("xyz")
                    .tolist(),
                },
                "opening": motion[f"{arm}_gripper"],
            }
        self._last_robot_state = result
        return result

    def record_observation(self, **kwargs):
        self._captured = self._native_call(self.env.get_obs)
        try:
            return super().record_observation(**kwargs)
        finally:
            self._captured = None

    def capture(self, camera: str) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        frame = self._captured["vision"][camera]
        depth = frame.get("distance_to_image_plane")
        if depth is None:
            depth = frame.get("depth")
        world = _world_from_depth(
            depth, frame["intrinsic_matrix"], frame["extrinsic_matrix"]
        )
        return (
            np.asarray(frame["color"]),
            world,
            {
                "intrinsic_matrix": np.asarray(frame["intrinsic_matrix"]).tolist(),
                "camera_to_world": np.asarray(frame["extrinsic_matrix"]).tolist(),
                "camera_axes": "USD +X right, +Y up, -Z forward",
                "raw_point_map_frame": "sim_world",
                "raw_point_map_unit": "m",
                "coordinates": self.coordinates.metadata(),
                "env_steps": self.env_steps,
            },
        )

    @tool
    def move_to(
        self,
        arm: Literal["left", "right"],
        xyz: Vector3 | None = None,
        orientation_rpy: Vector3 | None = None,
    ) -> ToolResult:
        """Move one grasp TCP to absolute xyz in metres and fixed-axis XYZ RPY in radians."""
        if xyz is None and orientation_rpy is None:
            raise ValueError("move_to needs xyz, orientation_rpy, or both")
        motion = self._native_call(self.env.get_motion_state)
        current_xyz, current_rotation = self._pose(arm, motion)
        target = current_xyz if xyz is None else np.asarray(xyz)
        rotation = (
            current_rotation
            if orientation_rpy is None
            else Rotation.from_euler("xyz", orientation_rpy).as_matrix()
        )
        native_rotation = self.coordinates.world_orientation(rotation)
        native_xyz = self.coordinates.world_position(target)
        cal = self.calibration["tools"][arm]
        native_rotation = native_rotation @ np.asarray(cal["rotation_to_native"]).T
        native_xyz -= native_rotation @ np.asarray(cal["offset_in_native_m"])
        result = self._native_call(
            self.native.move_to,
            arm=arm,
            xyz=native_xyz.tolist(),
            quat=quaternion_wxyz(Rotation.from_matrix(native_rotation)).tolist(),
        )
        stop_reason = result.data.get("stop_reason")
        if stop_reason == "cancelled":
            raise ToolCancelled(result.data.get("error", "cancelled"))
        if not self._check_execution(result):
            return result
        if stop_reason == "plan_failed":
            result.data.update(status="partial", reason=stop_reason)
            return result
        result.data.pop("final_eef_xyz", None)
        result.data.pop("final_dist_m", None)
        measured_xyz, measured_rotation = self._pose(
            arm, self._native_call(self.env.get_motion_state)
        )
        distance = float(np.linalg.norm(measured_xyz - target))
        angle = float(Rotation.from_matrix(rotation @ measured_rotation.T).magnitude())
        result.data.update(
            position_error_m=distance,
            orientation_error_rad=angle,
            status="error"
            if result.data.get("error")
            else "reached"
            if distance <= 0.005 and angle <= 0.05
            else "partial",
        )
        return result

    @tool
    def set_gripper(
        self,
        arm: Literal["left", "right"],
        opening: Annotated[float, Field(ge=0, le=1)],
    ) -> ToolResult:
        """Set the selected gripper opening, 0 closed to 1 open, retaining both arm targets."""
        result = self._native_call(self.native.set_gripper, arm=arm, val=opening)
        if not self._check_execution(result):
            return result
        measured = self._native_call(self.env.get_motion_state)[f"{arm}_gripper"]
        result.data.update(
            opening=measured,
            status="error"
            if result.data.get("error")
            else "reached"
            if abs(measured - opening) <= 0.05
            else "partial",
        )
        return result
