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

"""RoboCasa PandaOmron mobile-base + lift + arm general-frame RL adapter."""

from __future__ import annotations

import itertools
from typing import Annotated, Any

import numpy as np
from pydantic import Field
from scipy.spatial.transform import Rotation

from robots.rl.envs.libero import ControlConfig, Measurement, _clip_norm, _stalled
from robots.rl.geometry import Coordinates, grasp_axes
from robots.rl.perception import RecordedPerception
from rpent.tools import ToolResult, tool


class RoboCasaRLPrimitives(RecordedPerception):
    """Dynamic mounting coordinates with native grip_site, mobile base, and lift."""

    cameras = ("agentview", "agentview_right", "wrist")
    BASE_LIMITS = np.array([0.35, 0.35, 0.5], dtype=float)

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
        self._config = ControlConfig()
        self._meta = self.env._client.call("env.control_meta")
        self._open_gap = max(float(self._meta["gripper_max_gap"]), 1e-3)
        self._hold_opening = self._opening()
        self._refresh_coordinates()
        self._save_artifact("coordinates.json", self._coordinates_metadata(), None)

    def native_success(self) -> bool:
        self._success |= bool(self.env.check_success())
        return self._success

    def _finger_gap(self) -> float:
        return float(np.abs(self.env.gripper_qpos).sum())

    def _opening(self) -> float:
        return min(1.0, self._finger_gap() / self._open_gap)

    def _measure_native(self) -> Measurement:
        self._control = self.env._client.call("env.coordinate_calibration")
        return Measurement(
            np.asarray(self._control["site_xyz"], dtype=np.float64),
            np.asarray(self._control["site_rotation"], dtype=np.float64),
            self._opening(),
        )

    def _refresh_coordinates(self) -> Measurement:
        native = self._measure_native()
        coordinates = Coordinates.from_control(self._control)
        if previous := getattr(self, "coordinates", None):
            coordinates = coordinates.after(previous)
        self.coordinates = coordinates
        self.tool_rotation = grasp_axes(
            self._control["jaw_slide_axes_site"][0],
            self._control["wrist_to_site_in_site"],
        )
        return Measurement(
            self.coordinates.points(native.xyz),
            self.coordinates.orientation(native.rotation @ self.tool_rotation),
            native.opening,
        )

    def measure(self) -> Measurement:
        native = self._measure_native()
        return Measurement(
            self.coordinates.points(native.xyz),
            self.coordinates.orientation(native.rotation @ self.tool_rotation),
            native.opening,
        )

    def _coordinates_metadata(self) -> dict[str, Any]:
        metadata = {
            **self.coordinates.metadata(),
            "support_surface_kind": self._control.get("support_surface_kind"),
            "support_surface_source": self._control.get("support_surface_source"),
            "lift_m": round(float(self._control["lift_m"]), 6)
            if "lift_m" in self._control
            else None,
            "site_world_m": np.asarray(self._control["site_xyz"], dtype=float).tolist(),
        }
        if "jaw_slide_axes_site" in self._control:
            metadata["jaw_slide_axes_site"] = np.asarray(
                self._control["jaw_slide_axes_site"], dtype=float
            ).tolist()
        return metadata

    def robot_state(self) -> dict[str, Any]:
        measured = self._refresh_coordinates()
        return {
            **measured.state(),
            "finger_gap_m": round(self._finger_gap(), 5),
            "lift_m": round(float(self._control["lift_m"]), 4),
            "coordinates": self.coordinates.description(),
            "env_steps": self.env_steps,
        }

    def capture(self, camera: str) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        rgb, depth = self.env.render_camera(camera, depth=True, mask_invalid_depth=True)
        meta = self.env.get_camera_meta(camera)
        transform = self.env._client.call(
            "env.get_camera_transform",
            kwargs={
                "camera_name": self.env._resolve_cam(camera),
                "height": self.env.camera_h,
                "width": self.env.camera_w,
            },
        )
        depth_arr = np.asarray(depth, dtype=np.float64)
        near_m, far_m = float(meta["depth_near"]), float(meta["depth_far"])
        valid = (
            np.isfinite(depth_arr)
            & (depth_arr > near_m * (1.0 + 1e-4))
            & (depth_arr < far_m * (1.0 - 1e-5))
        )
        safe = np.where(valid, depth_arr, 0.0)
        rows, cols = np.mgrid[: depth_arr.shape[0], : depth_arr.shape[1]]
        homogeneous = np.stack(
            (cols * safe, rows * safe, safe, np.ones_like(safe)), axis=-1
        )
        world = (homogeneous @ np.asarray(transform, dtype=np.float64).T)[..., :3]
        world[~valid] = np.nan
        return (
            np.asarray(rgb),
            world,
            {
                "pixel_to_world": np.asarray(transform).tolist(),
                "height": depth_arr.shape[0],
                "width": depth_arr.shape[1],
                "intrinsic_K": np.asarray(meta["intrinsic"]).tolist(),
                "depth_near": near_m,
                "depth_far": far_m,
            },
        )

    def _native(self, part: str, physical: np.ndarray) -> np.ndarray:
        cfg = self._meta["parts"][part]
        low, high = np.asarray(cfg["output_min"]), np.asarray(cfg["output_max"])
        return np.asarray(cfg["input_min"]) + (physical - low) / (high - low) * (
            np.asarray(cfg["input_max"]) - np.asarray(cfg["input_min"])
        )

    def _base_velocity(self, physical: np.ndarray) -> np.ndarray:
        cfg = self._meta["parts"]["base"]
        theta = float(self.env._client.call("env.control_frame")["base_theta"])
        rotation = Rotation.from_euler("z", -theta).as_matrix()
        low, high = np.asarray(cfg["actuator_min"]), np.asarray(cfg["actuator_max"])
        return rotation.T @ (
            (rotation @ physical - (high + low) / 2) / ((high - low) / 2)
        )

    def _step(self, action: np.ndarray) -> None:
        self._check_cancelled()
        if self.env_steps >= self._max_episode_steps:
            return
        native = np.zeros(12)
        native[-1] = -1
        local = np.asarray(action[:6]).copy()
        if self._meta["parts"]["right"]["input_ref_frame"] != "world":
            base = np.asarray(
                self.env._client.call("env.control_frame")["arm_origin_rotation"]
            )
            local[:3] = base.T @ local[:3]
            local[3:] = base.T @ local[3:]
        start, stop = self._meta["slices"]["right"]
        native[start:stop] = self._native("right", local)
        native[6] = float(np.clip(4 * (self._opening() - self._hold_opening), -1, 1))
        self.env.step(native)
        self.env_steps += 1

    def _pose_action(
        self, current: Measurement, xyz: np.ndarray, rotation: np.ndarray
    ) -> tuple[np.ndarray, float, float]:
        position = xyz - current.xyz
        angle = Rotation.from_matrix(rotation @ current.rotation.T).as_rotvec()
        action = np.zeros(7)
        action[:3] = _clip_norm(position, self._config.position_step_m)
        action[3:6] = _clip_norm(angle, self._config.rotation_step_rad)
        return action, float(np.linalg.norm(position)), float(np.linalg.norm(angle))

    @tool
    def move_to(
        self,
        xyz: Annotated[list[float], Field(min_length=3, max_length=3)] | None = None,
        orientation_rpy: Annotated[list[float], Field(min_length=3, max_length=3)]
        | None = None,
    ) -> ToolResult:
        """Servo the grip_site toward an absolute world pose, without collision avoidance."""
        if self.ended:
            return ToolResult(data={"error": "episode_ended"})
        if xyz is None and orientation_rpy is None:
            return ToolResult(data={"error": "Supply xyz, orientation_rpy, or both."})
        start = self.measure()
        target = self.coordinates.world_position(
            start.xyz if xyz is None else np.asarray(xyz, dtype=np.float64)
        )
        goal = (
            self.coordinates.world_orientation(start.rotation)
            if orientation_rpy is None
            else self.coordinates.world_orientation(
                Rotation.from_euler("xyz", orientation_rpy).as_matrix()
            )
        ) @ self.tool_rotation.T
        config = self._config
        errors: list[tuple[float, float]] = []
        reason = None
        previous = self._measure_native()
        for steps in itertools.count():
            current = self._measure_native()
            action, distance, angle = self._pose_action(current, target, goal)
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
            stalled = _stalled(errors, config)
            if within and (settled or stalled or steps >= config.max_motion_steps):
                break
            if self.ended:
                reason = self._ended_reason()
            elif steps >= config.max_motion_steps:
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
    def set_gripper(
        self, opening: Annotated[float, Field(ge=0.0, le=1.0)]
    ) -> ToolResult:
        """Drive the gripper to a normalized opening while holding the current pose."""
        if self.ended:
            return ToolResult(data={"error": "episode_ended"})
        self._hold_opening = float(opening)
        config = self._config
        hold = self._measure_native()
        previous, still, reason = hold.opening, 0, None
        for steps in itertools.count():
            current = self._measure_native()
            error = opening - current.opening
            still = (
                still + 1
                if abs(current.opening - previous) < config.settle_opening
                else 0
            )
            previous = current.opening
            settled = still >= config.settle_steps
            within = abs(error) <= config.opening_tolerance
            if within and settled:
                break
            if self.ended:
                reason = self._ended_reason()
            elif steps >= config.max_gripper_steps:
                reason = "max_steps"
            elif settled and error < 0:
                reason = "blocked"
            if reason is not None:
                break
            action, _, _ = self._pose_action(current, hold.xyz, hold.rotation)
            self._step(action)
        return ToolResult(
            data={
                "status": "reached" if reason is None else "partial",
                **({} if reason is None else {"reason": reason}),
                "env_steps": steps,
                "opening_error": round(error, 3),
                "opening": round(self._opening(), 3),
            }
        )

    def _mobility_state(self):
        self._refresh_coordinates()
        return (
            np.asarray(self._control["mobile_base_xyz"], dtype=float).copy(),
            Rotation.from_matrix(self._control["mobile_base_rotation"]),
            self.coordinates.origin.copy(),
        )

    def _mobile_step(
        self, velocity=None, lift: float = 0.0, *, check_cancelled: bool = True
    ) -> None:
        if check_cancelled:
            self._check_cancelled()
        if self.env_steps >= self._max_episode_steps:
            return
        native = np.zeros(12)
        native[-1] = 1
        start, stop = self._meta["slices"]["base"]
        native[start:stop] = self._base_velocity(
            np.zeros(3) if velocity is None else velocity
        )
        start, stop = self._meta["slices"]["torso"]
        native[start:stop] = self._native("torso", np.array([lift]))
        native[6] = float(np.clip(4 * (self._opening() - self._hold_opening), -1, 1))
        self.env.step(native)
        self.env_steps += 1

    def _mobility_delta(self, before):
        after = self._mobility_state()
        initial_frame = before[1].inv()
        displacement = initial_frame.apply(after[0] - before[0])
        forward = (initial_frame * after[1]).as_matrix()[:, 0]
        yaw = float(np.arctan2(forward[1], forward[0]))
        return displacement, yaw, float(after[2][2] - before[2][2])

    def _mobility_feedback(self, motion, start_steps, request, reason):
        displacement, yaw, height_change = motion
        return {
            "status": "reached" if reason == "completed" else "partial",
            "reason": reason,
            "requested": request,
            "displacement_initial_base_m": np.round(displacement, 4).tolist(),
            "yaw_change_rad": round(yaw, 4),
            "mount_height_change_m": round(height_change, 4),
            "executed_steps": self.env_steps - start_steps,
            "executed_duration_s": round(
                (self.env_steps - start_steps) / self._meta["control_freq"], 4
            ),
            "env_steps": self.env_steps,
        }

    @tool
    def move_base(
        self,
        vx: Annotated[float, Field(ge=-0.35, le=0.35)] = 0,
        vy: Annotated[float, Field(ge=-0.35, le=0.35)] = 0,
        yaw_rate: Annotated[float, Field(ge=-0.5, le=0.5)] = 0,
        duration_s: Annotated[float, Field(ge=0.1, le=2)] = 1,
    ) -> ToolResult:
        """Move the mobile base at bounded velocity in its current local frame, then stop."""
        physical = np.asarray([vx, vy, yaw_rate], dtype=float)
        cfg = self._meta["parts"]["base"]
        if (
            not np.isfinite(physical).all()
            or not np.isfinite(duration_s)
            or not 0.1 <= duration_s <= 2
            or np.any(np.abs(physical) > self.BASE_LIMITS)
            or np.any(physical < cfg["actuator_min"])
            or np.any(physical > cfg["actuator_max"])
        ):
            return ToolResult(
                data={"error": "Invalid mobile-base velocity or duration"}
            )
        if self.ended:
            return ToolResult(data={"error": "episode_ended"})
        before, start_steps = self._mobility_state(), self.env_steps
        requested_steps = max(2, round(duration_s * self._meta["control_freq"]))
        count = min(requested_steps, self._max_episode_steps - self.env_steps)
        reason, drive_steps = "completed", 0
        try:
            for _ in range(max(0, count - 1)):
                if self.ended:
                    reason = "episode_ended"
                    break
                self._mobile_step(physical)
                drive_steps += 1
        finally:
            if not self.ended and self.env_steps < self._max_episode_steps:
                self._mobile_step(check_cancelled=False)
        if self.ended:
            reason = "episode_ended" if self.native_success() else "budget_exhausted"
        elif count < requested_steps:
            reason = "budget_exhausted"
        motion = self._mobility_delta(before)
        displacement, yaw, _ = motion
        result = self._mobility_feedback(
            motion,
            start_steps,
            {"vx": vx, "vy": vy, "yaw_rate": yaw_rate, "duration_s": duration_s},
            reason,
        )
        duration = drive_steps / self._meta["control_freq"]
        expected_yaw = yaw_rate * duration
        half_yaw = expected_yaw / 2
        velocity = Rotation.from_euler("z", half_yaw).apply([vx, vy, 0])[:2]
        expected_xy = duration * np.sinc(half_yaw / np.pi) * velocity
        residual_xy = expected_xy - displacement[:2]
        expected_distance = float(np.linalg.norm(expected_xy))
        translation_error = float(np.linalg.norm(residual_xy))
        yaw_error = expected_yaw - yaw
        result["translation_error_m"] = round(translation_error, 4)
        result["translation_shortfall_m"] = round(
            float(residual_xy @ expected_xy) / expected_distance
            if expected_distance
            else 0.0,
            4,
        )
        result["yaw_shortfall_rad"] = round(float(np.sign(expected_yaw) * yaw_error), 4)
        if reason == "completed":
            result["status"] = "commanded"
            if translation_error > max(0.05, expected_distance * 0.25) or abs(
                yaw_error
            ) > max(0.0524, abs(expected_yaw) * 0.25):
                result.update(status="partial", reason="limited_motion")
        return ToolResult(data=result)

    @tool
    def move_lift(
        self,
        delta_m: Annotated[float, Field(ge=-0.25, le=0.25)],
        duration_s: Annotated[float, Field(ge=0.1, le=5)] = 3,
    ) -> ToolResult:
        """Raise or lower the arm's mounting support relative to its current measured height."""
        if (
            not np.isfinite([delta_m, duration_s]).all()
            or not -0.25 <= delta_m <= 0.25
            or not 0.1 <= duration_s <= 5
        ):
            return ToolResult(data={"error": "Invalid lift displacement or duration"})
        if self.ended:
            return ToolResult(data={"error": "episode_ended"})
        before, start_steps = self._mobility_state(), self.env_steps
        target = before[2][2] + delta_m
        count = min(
            max(2, round(duration_s * self._meta["control_freq"])),
            self._max_episode_steps - self.env_steps,
        )
        reason, previous, still = "motion_budget", before[2][2], 0
        cfg = self._meta["parts"]["torso"]
        try:
            for _ in range(max(0, count - 1)):
                if self.ended:
                    reason = "episode_ended"
                    break
                height = self._mobility_state()[2][2]
                error = target - height
                if abs(error) <= 0.004:
                    reason = "completed"
                    break
                still = still + 1 if abs(height - previous) < 0.0002 else 0
                previous = height
                if still >= 12:
                    reason = "blocked_or_joint_limit"
                    break
                low, high = float(cfg["output_min"][0]), float(cfg["output_max"][0])
                self._mobile_step(
                    lift=float(np.clip(error, max(low, -0.025), min(high, 0.025)))
                )
        finally:
            if not self.ended and self.env_steps < self._max_episode_steps:
                self._mobile_step(check_cancelled=False)
        motion = self._mobility_delta(before)
        height_error = delta_m - motion[2]
        if self.ended:
            reason = "episode_ended" if self.native_success() else "budget_exhausted"
        elif abs(height_error) <= 0.004:
            reason = "completed"
        elif reason == "completed":
            reason = "outside_tolerance_after_stop"
        result = self._mobility_feedback(
            motion, start_steps, {"delta_m": delta_m, "duration_s": duration_s}, reason
        )
        result["height_error_m"] = round(height_error, 4)
        return ToolResult(data=result)
