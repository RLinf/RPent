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

"""Thin motion and RGB-D adapters for the shared clean workflow."""

from __future__ import annotations

import itertools
from typing import Annotated, Any, Literal

import numpy as np
from PIL import Image, ImageDraw
from pydantic import Field
from scipy.spatial.transform import Rotation

from robots.rl import geometry
from robots.rl.tools import ControlConfig, LiberoRLPrimitives, Measurement, _clip_norm
from rpent.tools import ToolResult, tool
from rpent.utils.transforms import quaternion_wxyz, rotation_from_wxyz


class ScenePrimitives(LiberoRLPrimitives):
    """Reuse the recorded-image tools while adapting native scene state."""

    cameras: tuple[str, ...]

    @staticmethod
    def _model_points(native: np.ndarray) -> np.ndarray:
        return native

    def __init__(
        self,
        *,
        env,
        sam3_client,
        state,
        check_cancelled,
        max_episode_steps: int = 10000,
    ) -> None:
        self.env, self._sam3, self._state = env, sam3_client, state
        self._check_cancelled = check_cancelled
        self._config = ControlConfig()
        self._max_episode_steps = max_episode_steps
        self.env_steps = 0
        self.frames = []
        self._masks = {}
        self._mask_ids = itertools.count(1)
        self._sam_calls = itertools.count(1)
        self._success = False

    @property
    def ended(self) -> bool:
        return self.native_success() or self.env_steps >= self._max_episode_steps

    def native_success(self) -> bool:
        """Read and latch the simulator's verdict, independently of tool status."""
        raise NotImplementedError

    def _ended_reason(self) -> str:
        return "terminated" if self.native_success() else "truncated"

    def robot_state(self) -> dict[str, Any]:
        raise NotImplementedError

    def record_observation(self, *, command=None, result=None, elapsed_s=None):
        state = self.robot_state()
        with self._state.record_step(
            state=state,
            terminated=self.native_success(),
            truncated=self.env_steps >= self._max_episode_steps,
            command=command,
            result=result,
            elapsed_s=elapsed_s,
            extras={"env_steps": self.env_steps},
        ) as step:
            if hasattr(self, "coordinates"):
                self._save("step_coordinates.json", self.coordinates.metadata(), step)
            for camera in self.cameras:
                rgb, world, calibration = self.capture(camera)
                self._save(f"{camera}_raw.png", rgb, step)
                rgb = self.annotate(rgb, calibration, state)
                self._save(f"{camera}.png", rgb, step)
                self._save(f"{camera}_world.npz", world.astype(np.float32), step)
                self._save(f"{camera}_calibration.json", calibration, step)
                if camera == self.cameras[0]:
                    self.frames.append(rgb)
        return self._state.get(step)

    def annotate(
        self, image: np.ndarray, calibration: dict, state: dict | None = None
    ) -> np.ndarray:
        """Return the agent view; single-arm embodiments need no arm markers."""
        return image

    def capture(self, camera: str) -> tuple[np.ndarray, np.ndarray, dict]:
        raise NotImplementedError


class RoboCasaPrimitives(ScenePrimitives):
    """World-pose servo over PandaOmron's native composite controller."""

    cameras = ("agentview", "agentview_right", "wrist")

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._meta = self.env._client.call("env.control_meta")
        self._open_gap = max(float(self._meta["gripper_max_gap"]), 1e-3)
        self._hold_opening = self._opening()

    def native_success(self) -> bool:
        self._success |= bool(self.env.check_success())
        return self._success

    def _finger_gap(self) -> float:
        return float(np.abs(self.env.gripper_qpos).sum())

    def _opening(self) -> float:
        return min(1.0, self._finger_gap() / self._open_gap)

    def measure(self) -> Measurement:
        return Measurement(
            self.env.eef_pos,
            Rotation.from_quat(self.env.eef_quat).as_matrix(),
            self._opening(),
        )

    def robot_state(self) -> dict[str, Any]:
        raw = self.env.current_raw_obs
        measurement = self.measure()
        return {
            "pose": {
                "xyz": np.round(measurement.xyz, 4).tolist(),
                "quaternion_wxyz": quaternion_wxyz(
                    Rotation.from_matrix(measurement.rotation)
                ).tolist(),
            },
            "opening": round(measurement.opening, 3),
            "finger_gap_m": round(self._finger_gap(), 5),
            "base_xyz": np.asarray(raw["robot0_base_pos"]).tolist(),
            "base_quaternion_xyzw": np.asarray(raw["robot0_base_quat"]).tolist(),
            "env_steps": self.env_steps,
        }

    def capture(self, camera: str):
        rgb, depth = self.env.render_camera(camera, depth=True)
        meta = self.env.get_camera_meta(camera)
        # Use the same native pixel-to-world transform as the existing adapter.
        transform = self.env._client.call(
            "env.get_camera_transform",
            kwargs={
                "camera_name": self.env._resolve_cam(camera),
                "height": self.env.camera_h,
                "width": self.env.camera_w,
            },
        )
        rows, cols = np.mgrid[: depth.shape[0], : depth.shape[1]]
        homogeneous = np.stack(
            (cols * depth, rows * depth, depth, np.ones_like(depth)), axis=-1
        )
        world = (homogeneous @ np.asarray(transform).T)[..., :3]
        return (
            np.asarray(rgb),
            world,
            {
                "pixel_to_world": np.asarray(transform).tolist(),
                "height": depth.shape[0],
                "width": depth.shape[1],
                "intrinsic_K": np.asarray(meta["intrinsic"]).tolist(),
            },
        )

    def _native(self, part: str, physical: np.ndarray) -> np.ndarray:
        cfg = self._meta["parts"][part]
        low, high = np.asarray(cfg["output_min"]), np.asarray(cfg["output_max"])
        return np.asarray(cfg["input_min"]) + (physical - low) / (high - low) * (
            np.asarray(cfg["input_max"]) - np.asarray(cfg["input_min"])
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

    def _pose_action(self, current, xyz, rotation):
        position = xyz - current.xyz
        angle = Rotation.from_matrix(rotation @ current.rotation.T).as_rotvec()
        action = np.zeros(7)
        action[:3] = _clip_norm(position, self._config.position_step_m)
        action[3:6] = _clip_norm(angle, self._config.rotation_step_rad)
        return action, float(np.linalg.norm(position)), float(np.linalg.norm(angle))

    @tool
    def set_gripper(self, opening: Annotated[float, Field(ge=0, le=1)]) -> ToolResult:
        """Set the normalized jaw opening while holding the current world pose. 0 closes and 1 opens. A blocked closing jaw is not proof of a grasp; the measured gap is held during later motions.

        Args:
            opening: Desired normalized opening, 0 closed to 1 open.
        """
        self._hold_opening = opening
        previous, still = self.measure().opening, 0
        for _ in range(self._config.max_gripper_steps):
            current = self.measure().opening
            still = still + 1 if abs(current - previous) < 0.005 else 0
            previous = current
            if self.ended or abs(current - opening) < 0.05 and still >= 3:
                break
            if still >= 4 and opening < current:
                break
            self._step(np.zeros(7))
        measured = self.measure().opening
        self._hold_opening = measured
        return ToolResult(
            data={
                "status": "reached" if abs(measured - opening) < 0.08 else "partial",
                "opening": round(measured, 3),
            }
        )

    @tool
    def move_base(
        self,
        vx: float = 0,
        vy: float = 0,
        yaw_rate: float = 0,
        duration_s: Annotated[float, Field(gt=0, le=5)] = 1,
    ) -> ToolResult:
        """Move the mobile base at constant local-frame velocity, then stop. x/y use the base frame; positive yaw rotates counterclockwise. Keep the arm and measured gripper opening held. Velocities outside native controller limits are refused. Native friction and contact can reduce achieved motion; inspect the returned measured displacement.

        Args:
            vx: Local x velocity in metres/second.
            vy: Local y velocity in metres/second.
            yaw_rate: Counterclockwise yaw rate in radians/second.
            duration_s: Duration in seconds, at most 5.
        """
        physical = np.array([vx, vy, yaw_rate], dtype=float)
        cfg = self._meta["parts"]["base"]
        if self.ended:
            return ToolResult(data={"error": "episode_ended"})
        if np.any(physical < cfg["actuator_min"]) or np.any(
            physical > cfg["actuator_max"]
        ):
            return ToolResult(
                data={
                    "error": "base velocity exceeds native limits",
                    "low": cfg["actuator_min"],
                    "high": cfg["actuator_max"],
                }
            )
        before = self.robot_state()
        native = np.zeros(12)
        native[-1] = 1
        start, stop = self._meta["slices"]["base"]
        count = min(
            round(duration_s * self._meta["control_freq"]),
            self._max_episode_steps - self.env_steps,
        )
        try:
            for _ in range(count):
                self._check_cancelled()
                if self.native_success():
                    break
                native[start:stop] = self._base_velocity(physical)
                native[6] = float(
                    np.clip(4 * (self.measure().opening - self._hold_opening), -1, 1)
                )
                self.env.step(native)
                self.env_steps += 1
        finally:
            native[start:stop] = self._base_velocity(np.zeros(3))
            if self.env_steps < self._max_episode_steps:
                self.env.step(native)
                self.env_steps += 1
        after = self.robot_state()
        before_rotation = Rotation.from_quat(before["base_quaternion_xyzw"])
        after_rotation = Rotation.from_quat(after["base_quaternion_xyzw"])
        return ToolResult(
            data={
                "status": "commanded",
                "commanded_local_velocity": physical.tolist(),
                "duration_s": count / self._meta["control_freq"],
                "displacement_world_m": (
                    np.asarray(after["base_xyz"]) - np.asarray(before["base_xyz"])
                ).tolist(),
                "yaw_change_rad": float(
                    (after_rotation * before_rotation.inv()).as_rotvec()[2]
                ),
                "env_steps": self.env_steps,
            }
        )

    def _base_velocity(self, physical: np.ndarray) -> np.ndarray:
        """Invert the native velocity controller's rotation and actuator scaling."""
        cfg = self._meta["parts"]["base"]
        theta = float(self.env._client.call("env.control_frame")["base_theta"])
        rotation = Rotation.from_euler("z", -theta).as_matrix()
        low, high = np.asarray(cfg["actuator_min"]), np.asarray(cfg["actuator_max"])
        return rotation.T @ (
            (rotation @ physical - (high + low) / 2) / ((high - low) / 2)
        )


class RoboCasaGeneralPrimitives(RoboCasaPrimitives):
    """Dynamic mounting coordinates with the native grip site, base and lift control."""

    BASE_LIMITS = np.array([0.35, 0.35, 0.5], dtype=float)
    move_to = LiberoRLPrimitives.move_to
    measure = LiberoRLPrimitives.measure
    _model_points = LiberoRLPrimitives._model_points

    def _surface_center(self, native: np.ndarray, points: np.ndarray) -> np.ndarray:
        return np.median(points, axis=0)

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._refresh_coordinates()
        self._state.save("coordinates.json", self._coordinates_metadata(), step=None)

    def _measure_native(self) -> Measurement:
        self._control = self.env._client.call("env.coordinate_calibration")
        return Measurement(
            np.asarray(self._control["site_xyz"], dtype=np.float64),
            np.asarray(self._control["site_rotation"], dtype=np.float64),
            self._opening(),
        )

    def _tcp_world(self, current: Measurement) -> np.ndarray:
        return current.xyz

    def _refresh_coordinates(self) -> Measurement:
        from robots.rl.coordinates import Coordinates, grasp_axes

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
            self.coordinates.points(self._tcp_world(native)),
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

    def capture(self, camera: str):
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
        near_m = float(meta["depth_near"])
        far_m = float(meta["depth_far"])
        valid = (
            np.isfinite(depth_arr)
            & (depth_arr > near_m * (1.0 + 1e-4))
            & (depth_arr < far_m * (1.0 - 1e-5))
        )
        safe_depth = np.where(valid, depth_arr, 0.0)
        rows, cols = np.mgrid[: depth_arr.shape[0], : depth_arr.shape[1]]
        homogeneous = np.stack(
            (
                cols * safe_depth,
                rows * safe_depth,
                safe_depth,
                np.ones_like(safe_depth),
            ),
            axis=-1,
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

    def record_observation(self, *, command=None, result=None, elapsed_s=None):
        state = self.robot_state()
        with self._state.record_step(
            state=state,
            terminated=self.native_success(),
            truncated=self.env_steps >= self._max_episode_steps,
            command=command,
            result=result,
            elapsed_s=elapsed_s,
            extras={"env_steps": self.env_steps},
        ) as step:
            self._save("step_coordinates.json", self._coordinates_metadata(), step)
            for camera in self.cameras:
                rgb, world, calibration = self.capture(camera)
                self._save(f"{camera}_raw.png", rgb, step)
                rgb = self.annotate(rgb, calibration)
                self._save(f"{camera}.png", rgb, step)
                self._save(f"{camera}_world.npz", world.astype(np.float32), step)
                self._save(f"{camera}_calibration.json", calibration, step)
                if camera == self.cameras[0]:
                    self.frames.append(rgb)
        return self._state.get(step)

    @tool
    def set_gripper(
        self, opening: Annotated[float, Field(ge=0.0, le=1.0)]
    ) -> ToolResult:
        """Drive the gripper to a normalized opening while holding the current pose. Closing on an object stops early and reports the measured opening; contact does not imply a stable grasp. The target stays in effect during later moves. Returns reached/partial/error, measured pose/opening, residual opening error and environment steps. Example: {"opening":1}.

        Args:
            opening: Target opening from 0 (closed) to 1 (fully open).
        """
        if self.ended:
            return ToolResult(data={"error": "episode_ended"})
        self._hold_opening = float(opening)
        result = self._drive_gripper(opening)
        result.data["opening"] = round(self._opening(), 3)
        return result

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
        # Track the installed arm goal while the base or lifting support moves.
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
        """Measure translation and signed heading change in the starting chassis frame."""
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
        """Move the mobile base at bounded velocity in its current local frame, then stop. Hold the arm's installed pose and existing gripper target. Contacts can prevent motion. Read achieved displacement and the measured frame in the action feedback.

        Args:
            vx: Forward speed in metres/second, within the schema bounds.
            vy: Leftward speed in metres/second, within the schema bounds.
            yaw_rate: Counterclockwise speed in rad/s, -0.5 to 0.5.
            duration_s: Requested duration, 0.1 to 2 seconds, including a final stop step.
        """
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
        reason = "completed"
        drive_steps = 0
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
        # Integrate constant body-frame velocity along its arc in the initial frame.
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
        """Raise or lower the arm's mounting support relative to its current measured height. Hold the installed arm pose and existing gripper target. Inspect achieved height change, updated support surface height, and call view_env before choosing the next arm motion; joint limits or contacts may cause a partial result.

        Args:
            delta_m: Signed vertical change in mounting-support height, metres; positive raises, negative lowers, within the schema bounds.
            duration_s: Execution budget from 0.1 to 5 seconds, including a final stop step.
        """
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
        reason = "motion_budget"
        previous, still = before[2][2], 0
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


class RoboTwinPrimitives(ScenePrimitives):
    """Native planned single-arm waypoint execution."""

    cameras = ("head", "left_wrist", "right_wrist")

    def marker_position(self, arm: str, state: dict) -> np.ndarray:
        """Return the measured TCP position in the camera calibration frame."""
        return np.asarray(state[arm]["tcp_xyz"])

    def annotate(
        self, image: np.ndarray, calibration: dict, state: dict | None = None
    ) -> np.ndarray:
        """Project measured TCPs into each view and label anatomical arm identity."""
        state = self.robot_state() if state is None else state
        view = Image.fromarray(image).convert("RGB")
        draw = ImageDraw.Draw(view)
        extrinsic = np.linalg.inv(np.asarray(calibration["extrinsic_cam2world"]))
        intrinsic = np.asarray(calibration["intrinsic_K"])
        for arm, color in (("left", "cyan"), ("right", "yellow")):
            point = np.r_[self.marker_position(arm, state), 1.0]
            camera = extrinsic @ point
            if camera[2] <= 0:
                continue
            pixel = intrinsic @ camera[:3]
            col, row = np.rint(pixel[:2] / pixel[2]).astype(int)
            if 0 <= col < view.width and 0 <= row < view.height:
                draw.ellipse(
                    (col - 5, row - 5, col + 5, row + 5), outline=color, width=2
                )
                draw.text((col + 6, row), arm.upper(), fill=color)
        return np.asarray(view)

    def native_success(self) -> bool:
        self._success |= bool(self.env.last_info["episode_status"]["eval_success"])
        return self._success

    def robot_state(self) -> dict[str, Any]:
        state = self.env.last_info["robot_state"]
        grippers = self.env._client.call("env.measured_grippers")
        return {
            arm: {
                "tcp_xyz": np.asarray(state[f"{arm}_tcp_pose"])[:3].tolist(),
                "tcp_quaternion_wxyz": np.asarray(state[f"{arm}_tcp_pose"])[
                    3:
                ].tolist(),
                **grippers[arm],
            }
            for arm in ("left", "right")
        }

    def capture(self, camera):
        rgb, depth = self.env.render_camera(camera, depth=True)
        meta = self.env.get_camera_meta(camera)
        extrinsic = np.asarray(meta["extrinsic_cv"], dtype=float)
        if extrinsic.shape == (3, 4):
            extrinsic = np.vstack((extrinsic, [0, 0, 0, 1]))
        elif extrinsic.shape != (4, 4):
            raise ValueError("camera extrinsic must be a 3x4 or 4x4 transform")
        camera_to_world = np.linalg.inv(extrinsic)
        calibration = {
            "intrinsic_K": np.asarray(meta["intrinsic_K"]).tolist(),
            "extrinsic_cam2world": camera_to_world.tolist(),
            "height": depth.shape[0],
            "width": depth.shape[1],
        }
        depth = np.asarray(depth, dtype=float)
        world = geometry.world_from_depth(depth, calibration)
        world[~np.isfinite(depth) | (depth <= 0)] = np.nan
        return np.asarray(rgb), world, calibration

    def _execute(self, waypoints):
        remaining = self._max_episode_steps - self.env_steps
        if self.ended or remaining <= 0:
            return ToolResult(data={"error": "episode_ended"})
        result = self.env.chunk_step(
            np.asarray(waypoints)[:remaining], action_type="qpos"
        )
        self.env_steps += int(result[4]["executed_actions"])
        return ToolResult(data={"status": "reached", "env_steps": self.env_steps})

    @staticmethod
    def _matrix(pose):
        pose = np.asarray(pose, dtype=float)
        result = np.eye(4)
        result[:3, :3] = rotation_from_wxyz(pose[3:]).as_matrix()
        result[:3, 3] = pose[:3]
        return result

    def _move_arm(self, arm: str, target: np.ndarray) -> ToolResult:
        state = self.env.last_info["robot_state"]
        tcp = self._matrix(state[f"{arm}_tcp_pose"])
        eef = self._matrix(state[f"{arm}_eef_pose"])
        desired = target @ np.linalg.inv(np.linalg.inv(eef) @ tcp)
        pose = [
            *desired[:3, 3],
            *quaternion_wxyz(Rotation.from_matrix(desired[:3, :3])),
        ]
        self._check_cancelled()
        plan = self.env.plan_arm_path(arm, pose)
        if (
            plan["status"] != "Success"
            or plan.get("position") is None
            or not len(plan["position"])
        ):
            return ToolResult(
                data={"status": "partial", "reason": "plan_failed", "arm": arm}
            )
        initial = np.asarray(state["qpos_target14"], dtype=float)
        actions = np.repeat(initial[None], len(plan["position"]), axis=0)
        joint_slice = slice(0, 6) if arm == "left" else slice(7, 13)
        actions[:, joint_slice] = plan["position"]
        goals = {arm: target}
        result = self._execute(actions)
        if result.is_error:
            return result
        # Native joint drives can still lag the final path waypoint. Hold that
        # same target for a bounded settling interval, checking measured TCPs.
        # This cannot turn an inaccurate plan into a reported success.
        settling_steps = 0
        errors, reached = self._pose_errors(goals)
        while not reached and not self.ended and settling_steps < 30:
            count = min(10, self._max_episode_steps - self.env_steps)
            if count <= 0:
                break
            result = self._execute(
                np.repeat(np.asarray(actions[-1])[None], count, axis=0)
            )
            if result.is_error:
                return result
            settling_steps += count
            errors, reached = self._pose_errors(goals)
        result.data.update(
            status="reached" if reached else "partial",
            errors=errors,
            settling_steps=settling_steps,
        )
        if not reached:
            result.data["reason"] = "not_at_target"
        return result

    def _pose_errors(self, goals):
        """Compare requested poses with native measured TCPs."""
        errors = {}
        for name, desired in goals.items():
            actual = self._matrix(self.env.last_info["robot_state"][f"{name}_tcp_pose"])
            errors[name] = {
                "position_error_m": float(
                    np.linalg.norm(desired[:3, 3] - actual[:3, 3])
                ),
                "rotation_error_rad": float(
                    Rotation.from_matrix(desired[:3, :3] @ actual[:3, :3].T).magnitude()
                ),
            }
        reached = all(
            value["position_error_m"] < self._config.position_tolerance_m
            and value["rotation_error_rad"] < self._config.rotation_tolerance_rad
            for value in errors.values()
        )
        return errors, reached

    @tool
    def set_gripper(
        self,
        arm: Literal["left", "right"],
        opening: Annotated[float, Field(ge=0, le=1)],
    ) -> ToolResult:
        """Set one normalized gripper opening while holding both arm poses. 0 closes, 1 opens; an opening target or jaw contact is not proof that an object is held.

        Args:
            arm: Anatomical left or right arm.
            opening: Normalized opening, from 0 closed to 1 open.
        """
        initial = np.asarray(
            self.env.last_info["robot_state"]["qpos_target14"], dtype=float
        )
        target = initial.copy()
        target[6 if arm == "left" else 13] = opening
        result = self._execute(
            [
                initial + (target - initial) * fraction
                for fraction in np.linspace(0.1, 1, 10)
            ]
        )
        if not result.is_error:
            measured = self.env._client.call("env.measured_grippers")
            reached = (
                abs(measured[arm]["opening"] - opening) < self._config.opening_tolerance
            )
            result.data.update(
                status="reached" if reached else "partial", grippers=measured
            )
            if not reached:
                result.data["reason"] = "opening_differs"
        return result


class RoboTwinGeneralPrimitives(RoboTwinPrimitives):
    """Expose the shared mounting frame while retaining native planning and RGB-D."""

    _model_points = LiberoRLPrimitives._model_points

    def __init__(self, **kwargs) -> None:
        from robots.robotwin.general import RobotwinFrame

        super().__init__(**kwargs)
        calibration = self.env._client.call("env.coordinate_calibration")
        self.frame = RobotwinFrame.from_calibration(calibration)
        self.coordinates = self.frame.coordinates
        self._state.save("coordinates.json", self.frame.metadata(), step=None)

    def robot_state(self) -> dict[str, Any]:
        native = self.env.last_info["robot_state"]
        state = self.env._client.call("env.measured_grippers")
        for arm, gripper in state.items():
            gripper["pose"] = self.frame.model_pose(native[f"{arm}_tcp_pose"], arm)
        return state

    def marker_position(self, arm: str, state: dict) -> np.ndarray:
        return self.coordinates.world_position(np.asarray(state[arm]["pose"]["xyz"]))

    def surface_points(self, native_points: np.ndarray) -> np.ndarray:
        """Convert the valid depth points before computing any surface geometry."""
        return self.coordinates.points(geometry.valid_points(native_points))

    @tool
    def move_to(
        self,
        arm: Literal["left", "right"],
        xyz: Annotated[list[float], Field(min_length=3, max_length=3)] | None = None,
        orientation_rpy: Annotated[list[float], Field(min_length=3, max_length=3)]
        | None = None,
    ) -> ToolResult:
        """Move an anatomical arm's native grasp TCP to an absolute general-frame target.

        Positions use metres. orientation_rpy uses radians with Rz(yaw)Ry(pitch)Rx(roll).
        Supply xyz, orientation_rpy, or both. Omitted parts hold their measured values.
        The other arm and both grippers keep their previous targets.
        Read the returned measured pose and reached/partial
        status to determine what actually happened.
        """
        if xyz is None and orientation_rpy is None:
            return ToolResult(data={"error": "Supply xyz, orientation_rpy, or both"})
        native = self.env.last_info["robot_state"]
        target = self.frame.world_target(
            xyz,
            orientation_rpy,
            native[f"{arm}_tcp_pose"],
            arm,
        )
        return self._move_arm(arm, self._matrix(target))
