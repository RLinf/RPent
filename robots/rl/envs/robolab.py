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

"""Measured RoboLab contact-pad TCP control and RGB-D episode observations."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Annotated, Any

import numpy as np
from pydantic import Field
from scipy.spatial.transform import Rotation

from robots.rl.geometry import Coordinates, grasp_axes, rpy_of, world_from_depth
from robots.rl.perception import RecordedPerception
from robots.robolab.access import CAMERAS
from rpent.tools import ToolResult, tool
from rpent.tools.toolkit import ToolCancelled
from rpent.utils.transforms import quaternion_wxyz, rotation_from_wxyz

Vector3 = Annotated[list[float], Field(min_length=3, max_length=3)]


@dataclass(frozen=True)
class _ControlConfig:
    position_step_m: float = 0.03
    rotation_step_rad: float = 0.12
    position_tolerance_m: float = 0.004
    rotation_tolerance_rad: float = 0.02
    settle_speed_m: float = 0.001
    settle_speed_rad: float = 0.005
    max_motion_steps: int = 300
    stall_steps: int = 15
    stall_progress_m: float = 0.002
    stall_progress_rad: float = 0.01
    opening_tolerance: float = 0.08
    max_gripper_steps: int = 40
    settle_steps: int = 3
    settle_opening: float = 0.01
    segment_min_score: float = 0.2


@dataclass(frozen=True)
class _Measurement:
    xyz: np.ndarray
    rotation: np.ndarray
    opening: float

    def state(self) -> dict[str, Any]:
        return {
            "pose": {
                "xyz": [round(float(value), 4) for value in self.xyz],
                "orientation_rpy": rpy_of(self.rotation),
            },
            "opening": round(self.opening, 3),
        }


def _clip_norm(vector: np.ndarray, limit: float) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    return vector if norm <= limit else vector * (limit / norm)


class RoboLabRLPrimitives(RecordedPerception):
    """Servo the measured pad midpoint with the native DROID flange controller."""

    cameras = CAMERAS
    defer_unrecorded_capture = True

    def __init__(
        self,
        *,
        env: Any,
        sam3_client: Any = None,
        state: Any = None,
        check_cancelled: Callable[[], None],
        max_episode_steps: int = 10000,
        record_artifacts: bool = False,
    ) -> None:
        self._config = _ControlConfig()
        metadata = env.get_env_meta()
        self._native_max_episode_steps: int | None = metadata.get(
            "native_max_episode_steps"
        )
        limits = (
            max_episode_steps,
            metadata.get("max_episode_steps"),
            self._native_max_episode_steps,
        )
        self._init_perception(
            env=env,
            sam3_client=sam3_client,
            state=state,
            check_cancelled=check_cancelled,
            max_episode_steps=min(limit for limit in limits if limit is not None),
            record_artifacts=record_artifacts,
            segment_min_score=self._config.segment_min_score,
        )
        calibration = self.env.coordinate_calibration()
        self.coordinates = Coordinates.from_control(calibration)
        self.tool_rotation = grasp_axes(
            calibration["left_to_right_in_flange"],
            calibration["flange_to_tcp_in_flange"],
        )
        self._accept_motion(self.env.get_motion_state())
        self._opening_target = float(self._motion["opening"] >= 0.5)
        self._save_artifact("environment.json", metadata, None)
        self._save_artifact(
            "coordinates.json",
            {
                **self.coordinates.metadata(),
                "calibration": calibration,
                "tool_rotation_to_native_flange": self.tool_rotation.tolist(),
            },
            None,
        )

    def _accept_motion(self, motion: dict[str, Any]) -> None:
        self._motion = motion
        self._status = motion["status"]
        self.env_steps = int(self._status["step"])
        self._success |= bool(self._status["success"])

    def native_success(self) -> bool:
        return self._success

    @property
    def ended(self) -> bool:
        return (
            self.infrastructure_error is not None
            or self._status["terminated"]
            or self._status["truncated"]
            or self.env_steps >= self._max_episode_steps
        )

    def _measure_native(self) -> _Measurement:
        return _Measurement(
            np.asarray(self._motion["flange_world_xyz"], dtype=np.float64),
            rotation_from_wxyz(self._motion["flange_quaternion_wxyz"]).as_matrix(),
            self._motion["opening"],
        )

    def _tcp_world(self) -> np.ndarray:
        return np.asarray(self._motion["tcp_world_xyz"], dtype=np.float64)

    def _site_target(
        self, target: np.ndarray, rotation: np.ndarray, current: _Measurement
    ) -> np.ndarray:
        offset = current.rotation.T @ (self._tcp_world() - current.xyz)
        return target - rotation @ offset

    def measure(self) -> _Measurement:
        """Read the pad midpoint and grasp orientation in general coordinates."""
        current = self._measure_native()
        return _Measurement(
            self.coordinates.points(self._tcp_world()),
            self.coordinates.orientation(current.rotation @ self.tool_rotation),
            current.opening,
        )

    def terminal_flags(self, state: dict[str, Any]) -> dict[str, bool]:
        """Combine the simulator verdict and the toolkit step budget."""
        terminated = self.native_success()
        truncated = not terminated and (
            bool(state.get("truncated")) or self.env_steps >= self._max_episode_steps
        )
        return {"terminated": terminated, "truncated": truncated}

    def robot_state(self) -> dict[str, Any]:
        pads = np.asarray(self._motion["pad_centres_world_m"])
        state = {
            **self.measure().state(),
            "finger_gap_m": round(float(np.linalg.norm(pads[0] - pads[1])), 5),
            "env_steps": self.env_steps,
            "max_episode_steps": self._max_episode_steps,
            "native_max_episode_steps": self._native_max_episode_steps,
            "truncated": self._status["truncated"],
            **({"state_stale": True} if self.infrastructure_error else {}),
        }
        return {**state, **self.terminal_flags(state)}

    def _fail(self, error: Exception, *, execution_uncertain: bool) -> None:
        self.infrastructure_error = {
            "stop_reason": "runtime_failure",
            "error": f"{type(error).__name__}: {error}",
            "execution_uncertain": execution_uncertain,
        }
        self._save_artifact(
            "infrastructure_error.json", self.infrastructure_error, step=None
        )

    def _step(self, action: np.ndarray) -> None:
        self._check_cancelled()
        try:
            feedback = self.env.step_to(
                action[:3],
                action[3:],
                self._opening_target,
                expected_step=self.env_steps,
            )
            self._accept_motion(feedback)
        except ToolCancelled:
            raise
        except Exception as error:
            self._fail(error, execution_uncertain=True)
            raise

    def _pose_action(
        self, current: _Measurement, xyz: np.ndarray, rotation: np.ndarray
    ) -> tuple[np.ndarray, float]:
        offset = xyz - current.xyz
        rotvec = Rotation.from_matrix(rotation @ current.rotation.T).as_rotvec()
        xyz = current.xyz + _clip_norm(offset, self._config.position_step_m)
        axes = (
            Rotation.from_rotvec(
                _clip_norm(rotvec, self._config.rotation_step_rad)
            ).as_matrix()
            @ current.rotation
        )
        return (
            np.r_[xyz, quaternion_wxyz(Rotation.from_matrix(axes))],
            float(np.linalg.norm(rotvec)),
        )

    def _move_world_pose(self, target: np.ndarray, goal: np.ndarray) -> ToolResult:
        """Servo the pad midpoint by recomputing the flange target each step."""
        config = self._config
        errors: deque[tuple[float, float]] = deque(maxlen=config.stall_steps + 1)
        reason = None
        previous = self._measure_native()
        for steps in range(config.max_motion_steps + 1):
            current = self._measure_native()
            site_target = self._site_target(target, goal, current)
            action, angle = self._pose_action(current, site_target, goal)
            distance = float(np.linalg.norm(target - self._tcp_world()))
            errors.append((distance, angle))
            settled = (
                np.linalg.norm(current.xyz - previous.xyz) < config.settle_speed_m
                and Rotation.from_matrix(
                    current.rotation @ previous.rotation.T
                ).magnitude()
                < config.settle_speed_rad
            )
            previous = current
            within = (
                distance < config.position_tolerance_m
                and angle < config.rotation_tolerance_rad
            )
            stalled = (
                len(errors) > config.stall_steps
                and errors[0][0] - distance < config.stall_progress_m
                and errors[0][1] - angle < config.stall_progress_rad
            )
            if self.ended:
                reason = self._ended_reason()
            elif within and (settled or stalled or steps == config.max_motion_steps):
                break
            elif steps == config.max_motion_steps:
                reason = "max_steps"
            elif stalled:
                reason = "stalled"
            if reason is not None:
                break
            self._step(action)
        return ToolResult(
            data={
                "status": "reached" if reason is None else "partial",
                **({} if reason is None else {"reason": reason}),
                "env_steps": steps,
                "position_error_m": round(distance, 4),
                "rotation_error_rad": round(angle, 3),
            }
        )

    @tool
    def move_to(
        self,
        xyz: Vector3 | None = None,
        orientation_rpy: Vector3 | None = None,
    ) -> ToolResult:
        """Move the grasp TCP to absolute xyz in metres and fixed-axis XYZ RPY in radians. Omitted components keep their measured values. The previous gripper target stays active."""
        if self.ended:
            return ToolResult(data={"error": "episode_ended"})
        if xyz is None and orientation_rpy is None:
            raise ValueError("move_to needs xyz, orientation_rpy, or both")
        current = self._measure_native()
        target = (
            self._tcp_world().copy()
            if xyz is None
            else self.coordinates.world_position(np.asarray(xyz))
        )
        goal = (
            current.rotation
            if orientation_rpy is None
            else self.coordinates.world_orientation(
                Rotation.from_euler("xyz", orientation_rpy).as_matrix()
            )
            @ self.tool_rotation.T
        )
        return self._move_world_pose(target, goal)

    @tool
    def set_gripper(self, opening: Annotated[float, Field(ge=0, le=1)]) -> ToolResult:
        """Hold the TCP pose and drive the binary gripper. A blocked close reports partial and stops the batch. Use a fresh view and a short lift to verify the hold."""
        if self.ended:
            return ToolResult(data={"error": "episode_ended"})
        config = self._config
        opening_target = float(opening >= 0.5)
        start = self._measure_native()
        already_reached = (
            self._opening_target == opening_target
            and abs(start.opening - opening_target) <= config.opening_tolerance
            and abs(start.opening - opening) <= config.opening_tolerance
        )
        self._check_cancelled()
        self._opening_target = opening_target
        target = self._tcp_world().copy()
        previous = start.opening
        start_step = self.env_steps
        still = 0
        reason = None if already_reached else "max_steps"
        for _ in range(0 if already_reached else config.max_gripper_steps):
            if self.ended:
                reason = self._ended_reason()
                break
            current = self._measure_native()
            action, _ = self._pose_action(
                current,
                self._site_target(target, start.rotation, current),
                start.rotation,
            )
            self._step(action)
            if self.ended:
                reason = self._ended_reason()
                break
            measured = self.measure()
            still = (
                still + 1
                if abs(measured.opening - previous) < config.settle_opening
                else 0
            )
            previous = measured.opening
            if still >= config.settle_steps:
                at_opening = (
                    abs(measured.opening - self._opening_target)
                    <= config.opening_tolerance
                )
                if self._opening_target == 0 and not at_opening:
                    reason = "blocked"
                    break
                tcp_distance = np.linalg.norm(self._tcp_world() - target)
                if at_opening and tcp_distance <= config.position_tolerance_m:
                    reason = (
                        None
                        if abs(measured.opening - opening) <= config.opening_tolerance
                        else "binary_gripper"
                    )
                    break
        measured = self.measure().opening
        return ToolResult(
            data={
                "status": "reached" if reason is None else "partial",
                **({} if reason is None else {"reason": reason}),
                "opening": round(measured, 3),
                "opening_error": round(opening - measured, 3),
                "env_steps": self.env_steps - start_step,
            }
        )

    def _display_shape(self, native_shape: tuple[int, int]) -> tuple[int, int]:
        if self.display_size is None or max(native_shape) <= self.display_size:
            return native_shape
        height, width = native_shape
        scale = self.display_size / max(native_shape)
        return max(1, round(height * scale)), max(1, round(width * scale))

    def capture(self, camera: str) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        """Read a named RoboLab camera at the acknowledged native step."""
        if camera not in self.cameras:
            raise ValueError(f"available cameras: {self.cameras}")
        return self._capture((camera,))[camera]

    def capture_all(self) -> dict[str, tuple[np.ndarray, np.ndarray, dict[str, Any]]]:
        """Acquire the two RGB-D frames together at the acknowledged native step."""
        return self._capture(self.cameras)

    def _capture(
        self, cameras: Sequence[str]
    ) -> dict[str, tuple[np.ndarray, np.ndarray, dict[str, Any]]]:
        if self.infrastructure_error is not None:
            raise RuntimeError("RoboLab observation unavailable after native failure")
        try:
            observation = self.env.get_obs()
            if observation["status"]["step"] != self.env_steps:
                raise RuntimeError("RoboLab advanced outside the acknowledged action")
            return {
                camera: self._camera_capture(observation["vision"][camera])
                for camera in cameras
            }
        except ToolCancelled:
            raise
        except Exception as error:
            self._fail(error, execution_uncertain=False)
            raise

    def _camera_capture(
        self, frame: dict[str, Any]
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        depth = np.asarray(frame["distance_to_image_plane"], dtype=np.float32)
        depth = np.where(np.isfinite(depth) & (depth > 0), depth, np.nan)
        intrinsics = np.asarray(frame["intrinsic_matrix"])
        # Convert the simulator's OpenGL camera axes to the pinhole convention
        # used by the shared projector: +X right, +Y down, +Z forward.
        transform = np.asarray(frame["camera_to_world"]) @ np.diag([1, -1, -1, 1])
        calibration = {
            "intrinsic_K": intrinsics.tolist(),
            "extrinsic_cam2world": transform.tolist(),
            "camera_axes": "+X right, +Y down, +Z forward",
            "raw_point_map_frame": "sim_world",
            "raw_point_map_unit": "m",
            "env_steps": self.env_steps,
        }
        world = world_from_depth(depth, calibration)
        return np.asarray(frame["color"]).copy(), world, calibration
