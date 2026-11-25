# Copyright 2026 The RPent Authors.
# Licensed under the Apache License, Version 2.0.
"""Episode adapter using RoboDojo's native planner, acknowledged actions and RGB-D."""

from typing import Annotated, Literal

import numpy as np
from pydantic import Field
from scipy.spatial.transform import Rotation

from robots.rl.actions import Vector3
from robots.rl.backends import ScenePrimitives
from robots.rl.coordinates import Coordinates, pose_matrix
from robots.rl.tools import LiberoRLPrimitives
from robots.robodojo.access import CAMERAS
from robots.robodojo.primitives import RoboDojoPrimitives as NativePrimitives
from robots.robodojo.tools import _world_from_depth
from rpent.tools import ToolResult, tool
from rpent.utils.transforms import quaternion_wxyz


class RoboDojoPrimitives(ScenePrimitives):
    cameras = CAMERAS
    _model_points = LiberoRLPrimitives._model_points

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.native = NativePrimitives(
            env=self.env, check_cancelled=self._check_cancelled
        )
        self.calibration = self.env.coordinate_calibration()
        self.coordinates = Coordinates.from_control(self.calibration)
        self._state.save(
            "coordinates.json",
            {**self.coordinates.metadata(), "calibration": self.calibration},
            step=None,
        )
        self._captured = None
        self.infrastructure_error = None
        self._last_robot_state = None

    def _check_execution(self, result):
        """An unacknowledged motion ends this attempt; never send it again."""
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
            self._state.save(
                "infrastructure_error.json", self.infrastructure_error, step=None
            )
            result.data["status"] = "error"
            return False
        return True

    def _pose(self, arm, motion):
        native = pose_matrix(motion[f"{arm}_eef_pose"])
        calibration = self.calibration["tools"][arm]
        offset = np.asarray(calibration["offset_in_native_m"])
        rotation = np.asarray(calibration["rotation_to_native"])
        xyz = native[:3, 3] + native[:3, :3] @ offset
        axes = native[:3, :3] @ rotation
        return self.coordinates.points(xyz), self.coordinates.orientation(axes)

    def _native_call(self, operation, *args, **kwargs):
        """Latch native RPC/planner failures before another action can run."""
        from rpent.tools.toolkit import ToolCancelled

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
            self._state.save(
                "infrastructure_error.json", self.infrastructure_error, step=None
            )
            raise

    def native_success(self):
        if self.infrastructure_error is not None:
            return self._success
        status = self._native_call(self.env.get_status)
        self.env_steps = int(status["step"])
        self._success |= bool(status["success"])
        return self._success

    @property
    def ended(self):
        return (
            self.native_success()
            or self.env.episode_ended
            or self.env_steps >= self._max_episode_steps
        )

    def robot_state(self):
        if self.infrastructure_error is not None:
            return {**(self._last_robot_state or {}), "state_stale": True}
        status = self._native_call(self.env.get_status)
        self.env_steps = int(status["step"])
        motion = self._native_call(self.env.get_motion_state)
        result = {"env_steps": self.env_steps}
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
        # Capture RGB, optical-axis depth and extrinsics together. No physics
        # action is issued while the scene's exclusive tool lock is held.
        self._captured = self._native_call(self.env.get_obs)
        try:
            return super().record_observation(**kwargs)
        finally:
            self._captured = None

    def capture(self, camera):
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
        """Move one grasp TCP to absolute xyz in metres and fixed-axis XYZ RPY in radians. Omitted components keep their measured values. The other arm and both grippers retain their targets; reached/partial describes measured arrival, not grasp success."""
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
        calibration = self.calibration["tools"][arm]
        native_rotation = (
            native_rotation @ np.asarray(calibration["rotation_to_native"]).T
        )
        native_xyz -= native_rotation @ np.asarray(calibration["offset_in_native_m"])
        result = self._native_call(
            self.native.move_to,
            arm=arm,
            xyz=native_xyz.tolist(),
            quat=quaternion_wxyz(Rotation.from_matrix(native_rotation)).tolist(),
        )
        if not self._check_execution(result):
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
        """Set the selected gripper opening, 0 closed to 1 open, retaining both arm targets. Contact or a changed opening alone does not prove a successful grasp."""
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
