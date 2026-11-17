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


class ScenePrimitives(LiberoRLPrimitives):
    """Reuse the recorded-image tools while adapting native scene state."""

    cameras: tuple[str, ...]

    surface_points = staticmethod(geometry.valid_points)

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
        with self._state.record_step(
            state=self.robot_state(),
            terminated=self.native_success(),
            truncated=self.env_steps >= self._max_episode_steps,
            command=command,
            result=result,
            elapsed_s=elapsed_s,
            extras={"env_steps": self.env_steps},
        ) as step:
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

    def annotate(self, image: np.ndarray, calibration: dict) -> np.ndarray:
        """Return the agent view; single-arm embodiments need no arm markers."""
        return image

    def capture(self, camera: str) -> tuple[np.ndarray, np.ndarray, dict]:
        raise NotImplementedError

    @tool(readonly=True)
    def view_env(self, camera: str = "all") -> ToolResult:
        """Return current recorded RGB views, without advancing physics. Pixels are [row,col] in the returned view and must use that view's observation step.

        Args:
            camera: A camera named in the robot description, or all.
        """
        names = self.cameras if camera == "all" else (camera,)
        if any(name not in self.cameras for name in names):
            return ToolResult(data={"error": f"available cameras: {self.cameras}"})
        step = self._state.latest_step
        shapes = [
            list(self._state.load(f"{name}.png", step=step).shape[:2]) for name in names
        ]
        return ToolResult(
            data={
                "step": step,
                "images": [
                    {"camera": name, "size": shape}
                    for name, shape in zip(names, shapes)
                ],
            },
            images=self.images(step, list(names)),
        )

    @tool(readonly=True)
    def segment(
        self,
        prompt: str | None = None,
        point: list[int] | None = None,
        camera: str | None = None,
        step: int | None = None,
    ) -> ToolResult:
        """Segment a named object or one [row,col] point in a recorded camera frame. Returns its mask identifier, exclusive row/col box, and visible-surface world_xyz, top_z_m, size_xyz_m and axis_yaw; these are measurements, not grasp poses.

        Args:
            prompt: Object description; supply prompt or point, not both.
            point: One [row,col] image pixel; requires step.
            camera: A named robot camera; omitted selects the first overview.
            step: Recorded observation step; text queries default to latest.
        """
        camera = camera or self.cameras[0]
        if camera not in self.cameras:
            return ToolResult(data={"error": f"available cameras: {self.cameras}"})
        return super().segment(prompt=prompt, point=point, camera=camera, step=step)

    @tool(readonly=True)
    def back_project(
        self,
        pixel: list[int] | None = None,
        box: list[int] | None = None,
        mask_id: str | None = None,
        camera: str | None = None,
        step: int | None = None,
        z_range_m: list[float] | None = None,
    ) -> ToolResult:
        """Measure a world-space visible surface from one pixel, box or saved segment mask. Boxes include background. Pixel/box queries require the exact camera and observation step; a mask identifier fixes both.

        Args:
            pixel: [row,col], requires step.
            box: [row_min,col_min,row_max,col_max], max exclusive; requires step.
            mask_id: Identifier returned by segment.
            camera: Named camera for pixel/box queries; omitted selects the overview.
            step: Recorded observation step for pixel/box queries.
            z_range_m: Optional [low,high] world-z filter for a box or mask.
        """
        if mask_id is None:
            camera = camera or self.cameras[0]
            if camera not in self.cameras:
                return ToolResult(data={"error": f"available cameras: {self.cameras}"})
        if pixel is not None and len(pixel) != 2 or box is not None and len(box) != 4:
            return ToolResult(data={"error": "pixel needs 2 values; box needs 4"})
        return super().back_project(
            pixel=pixel,
            box=box,
            mask_id=mask_id,
            camera=camera,
            step=step,
            z_range_m=z_range_m,
        )

    @tool(readonly=True)
    def finish(self) -> ToolResult:
        """End the episode; takes no arguments. Returns _finish, success, terminated (native task success) and truncated (environment step budget reached). Calling finish does not make a task succeed. Example: {}."""
        return ToolResult(
            data={
                "_finish": True,
                "success": self.native_success(),
                "terminated": self.native_success(),
                "truncated": self.env_steps >= self._max_episode_steps,
            }
        )


class RoboCasaPrimitives(ScenePrimitives):
    """World-pose servo over PandaOmron's native composite controller."""

    cameras = ("agentview", "agentview_right", "wrist")

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._meta = self.env._client.call("env.control_meta")
        self._open_gap = max(float(self._meta["gripper_max_gap"]), 1e-3)
        self._hold_opening = self.measure().opening

    def native_success(self) -> bool:
        self._success |= bool(self.env.check_success())
        return self._success

    def _finger_gap(self) -> float:
        return float(np.abs(self.env.gripper_qpos).sum())

    def measure(self) -> Measurement:
        return Measurement(
            self.env.eef_pos,
            Rotation.from_quat(self.env.eef_quat).as_matrix(),
            min(1.0, self._finger_gap() / self._open_gap),
        )

    def robot_state(self) -> dict[str, Any]:
        raw = self.env.current_raw_obs
        measurement = self.measure()
        return {
            "pose": {
                "xyz": np.round(measurement.xyz, 4).tolist(),
                "quaternion_wxyz": Rotation.from_matrix(measurement.rotation)
                .as_quat(scalar_first=True)
                .tolist(),
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
        native[6] = float(
            np.clip(4 * (self.measure().opening - self._hold_opening), -1, 1)
        )
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
    def move_to(
        self,
        xyz: Annotated[list[float], Field(min_length=3, max_length=3)],
        quaternion: Annotated[list[float], Field(min_length=4, max_length=4)]
        | None = None,
    ) -> ToolResult:
        """Servo the grasp point to an absolute world position, optionally setting its full orientation. The jaw opening is held. Motion follows a straight line without collision avoidance; choose clear waypoints. The result reports measured pose errors, independently of task success.

        Args:
            xyz: Absolute grasp-point position [x,y,z], world metres.
            quaternion: Optional unit orientation [w,x,y,z] in the world frame; omitted holds the measured orientation.
        """
        if self.ended:
            return ToolResult(data={"error": "episode_ended"})
        target = np.asarray(xyz, dtype=float)
        goal = (
            self.measure().rotation
            if quaternion is None
            else Rotation.from_quat(quaternion, scalar_first=True).as_matrix()
        )
        reason = "max_steps"
        for _ in range(self._config.max_motion_steps):
            current = self.measure()
            action, distance, angle = self._pose_action(current, target, goal)
            if (
                distance < self._config.position_tolerance_m
                and angle < self._config.rotation_tolerance_rad
            ):
                reason = None
                break
            if self.ended:
                reason = "episode_ended"
                break
            self._step(action)
        current = self.measure()
        return ToolResult(
            data={
                "status": "reached" if reason is None else "partial",
                **({"reason": reason} if reason else {}),
                "xyz": current.xyz.tolist(),
                "quaternion_wxyz": Rotation.from_matrix(current.rotation)
                .as_quat(scalar_first=True)
                .tolist(),
                "position_error_m": float(np.linalg.norm(target - current.xyz)),
                "rotation_error_rad": float(
                    Rotation.from_matrix(goal @ current.rotation.T).magnitude()
                ),
            }
        )

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


class RoboTwinPrimitives(ScenePrimitives):
    """Native planned one-arm or synchronized two-arm waypoint execution."""

    cameras = ("head", "left_wrist", "right_wrist")

    def annotate(self, image: np.ndarray, calibration: dict) -> np.ndarray:
        """Project measured TCPs into each view and label anatomical arm identity."""
        view = Image.fromarray(image).convert("RGB")
        draw = ImageDraw.Draw(view)
        extrinsic = np.linalg.inv(np.asarray(calibration["extrinsic_cam2world"]))
        intrinsic = np.asarray(calibration["intrinsic_K"])
        for arm, color in (("left", "cyan"), ("right", "yellow")):
            point = np.r_[self.env.last_info["robot_state"][f"{arm}_tcp_pose"][:3], 1.0]
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
        return (
            np.asarray(rgb),
            geometry.world_from_depth(depth, calibration),
            calibration,
        )

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
        result[:3, :3] = Rotation.from_quat(pose[3:], scalar_first=True).as_matrix()
        result[:3, 3] = pose[:3]
        return result

    @tool
    def move_to(
        self,
        arm: Literal["left", "right"],
        xyz: Annotated[list[float], Field(min_length=3, max_length=3)],
        quaternion: Annotated[list[float], Field(min_length=4, max_length=4)]
        | None = None,
        other_xyz: Annotated[list[float], Field(min_length=3, max_length=3)]
        | None = None,
        other_quaternion: Annotated[list[float], Field(min_length=4, max_length=4)]
        | None = None,
    ) -> ToolResult:
        """Plan a world-frame TCP pose for the selected arm. Optionally give other_xyz to move both arms synchronously in this one call. Unspecified orientations hold measured values and both grippers hold their openings. Native planning errors are returned without executing a partial pair of plans.

        Args:
            arm: Anatomical left or right arm, as labelled in the camera views.
            xyz: Absolute TCP position [x,y,z] in world metres.
            quaternion: Optional TCP orientation [w,x,y,z], unit quaternion.
            other_xyz: Optional absolute TCP position for the other arm.
            other_quaternion: Optional orientation for the other arm; requires other_xyz.
        """
        if other_quaternion is not None and other_xyz is None:
            return ToolResult(data={"error": "other_quaternion requires other_xyz"})
        state = self.env.last_info["robot_state"]
        targets = {arm: (xyz, quaternion)}
        if other_xyz is not None:
            targets["right" if arm == "left" else "left"] = (
                other_xyz,
                other_quaternion,
            )
        paths, goals = {}, {}
        for name, (position, orientation) in targets.items():
            tcp = self._matrix(state[f"{name}_tcp_pose"])
            eef = self._matrix(state[f"{name}_eef_pose"])
            desired = tcp.copy()
            desired[:3, 3] = position
            if orientation is not None:
                desired[:3, :3] = Rotation.from_quat(
                    orientation, scalar_first=True
                ).as_matrix()
            desired_eef = desired @ np.linalg.inv(np.linalg.inv(eef) @ tcp)
            goals[name] = desired
            pose = [
                *desired_eef[:3, 3],
                *Rotation.from_matrix(desired_eef[:3, :3]).as_quat(scalar_first=True),
            ]
            self._check_cancelled()
            plan = self.env.plan_arm_path(name, pose)
            if (
                plan["status"] != "Success"
                or plan.get("position") is None
                or not len(plan["position"])
            ):
                return ToolResult(
                    data={"status": "partial", "reason": "plan_failed", "arm": name}
                )
            paths[name] = np.asarray(plan["position"])
        length = max(len(path) for path in paths.values())
        actions = []
        initial = np.asarray(state["qpos_target14"], dtype=float)
        for index in range(length):
            action = initial.copy()
            for name, path in paths.items():
                joint_slice = slice(0, 6) if name == "left" else slice(7, 13)
                # Pair paths by normalized progress, finishing both on the last action.
                waypoint = round(index * (len(path) - 1) / max(1, length - 1))
                action[joint_slice] = path[waypoint]
            actions.append(action)
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
        other_opening: Annotated[float, Field(ge=0, le=1)] | None = None,
    ) -> ToolResult:
        """Set one normalized gripper opening while holding both arm poses. Optionally set the other gripper in the same synchronized call. 0 closes, 1 opens; an opening target or jaw contact is not proof that an object is held.

        Args:
            arm: Anatomical left or right arm.
            opening: Normalized opening, from 0 closed to 1 open.
            other_opening: Optional opening of the other gripper.
        """
        initial = np.asarray(
            self.env.last_info["robot_state"]["qpos_target14"], dtype=float
        )
        target = initial.copy()
        target[6 if arm == "left" else 13] = opening
        if other_opening is not None:
            target[13 if arm == "left" else 6] = other_opening
        result = self._execute(
            [
                initial + (target - initial) * fraction
                for fraction in np.linspace(0.1, 1, 10)
            ]
        )
        if not result.is_error:
            measured = self.env._client.call("env.measured_grippers")
            targets = {arm: opening}
            if other_opening is not None:
                targets["right" if arm == "left" else "left"] = other_opening
            reached = all(
                abs(measured[name]["opening"] - value) < self._config.opening_tolerance
                for name, value in targets.items()
            )
            result.data.update(
                status="reached" if reached else "partial", grippers=measured
            )
            if not reached:
                result.data["reason"] = "opening_differs"
        return result
