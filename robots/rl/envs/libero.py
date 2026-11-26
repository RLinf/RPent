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

"""LIBERO single-arm Panda general-frame adapter and in-process warm host."""

from __future__ import annotations

import itertools
import os
from dataclasses import dataclass
from typing import Annotated, Any

import numpy as np
from pydantic import Field
from scipy.spatial.transform import Rotation

from robots.rl import geometry
from robots.rl.geometry import Coordinates, grasp_axes
from robots.rl.perception import RecordedPerception
from rpent.tools import ToolResult, tool

SIM_CAMERAS: dict[str, str] = {"agentview": "agentview", "wrist": "robot0_eye_in_hand"}
IMAGE_SIZE = 1024


@dataclass(frozen=True)
class ControlConfig:
    """Backend servo limits; planners never set these."""

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
class Measurement:
    """Measured end-effector pose and normalized gripper opening."""

    xyz: np.ndarray
    rotation: np.ndarray
    opening: float

    def state(self) -> dict[str, Any]:
        return {
            "pose": {
                "xyz": [round(float(value), 4) for value in self.xyz],
                "orientation_rpy": geometry.rpy_of(self.rotation),
            },
            "opening": round(self.opening, 3),
        }


def _clip_norm(vector: np.ndarray, limit: float) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    return vector if norm <= limit else vector * (limit / norm)


def _stalled(errors: list[tuple[float, float]], config: ControlConfig) -> bool:
    if len(errors) <= config.stall_steps:
        return False
    (old_d, old_a), (d, a) = errors[-1 - config.stall_steps], errors[-1]
    return old_d - d < config.stall_progress_m and old_a - a < config.stall_progress_rad


class LocalLiberoEnv:
    """In-process LIBERO host exposing the same surface as LiberoEnvClient.

    Eliminates the extra env_server HTTP process and loopback RGB-D serialization.
    """

    def __init__(
        self,
        *,
        suite: str,
        task: int,
        seed: int,
        libero_type: str | None = None,
        max_episode_steps: int = 10000,
        record_artifacts: bool = False,
    ) -> None:
        from robots.libero.env_server import LiberoEnvFacade, make_env
        from robots.libero.suites import resolve_libero_type

        os.environ["LIBERO_TYPE"] = resolve_libero_type(suite, libero_type)
        # Skip per-step offscreen camera rendering during training physics loops;
        # OffScreenRenderEnv keeps has_offscreen_renderer=True so capture() still
        # renders RGB-D on demand via render_camera().
        raw_env = make_env(
            task_id=int(task),
            seed=int(seed),
            suite_name=suite,
            max_episode_steps=int(max_episode_steps),
            rl_control_state=True,
            use_camera_obs=bool(record_artifacts),
            camera_depths=False,
        )
        self._facade = LiberoEnvFacade(
            raw_env,
            meta={
                "suite": suite,
                "task": int(task),
                "seed": int(seed),
                "max_episode_steps": int(max_episode_steps),
            },
        )
        obs, info = self._facade.reset()
        self.last_obs: dict[str, Any] = obs
        self.last_info: dict[str, Any] = info
        self.terminated = False
        self.truncated = False

    def step(self, action: np.ndarray) -> None:
        obs, _rew, term, trunc, info = self._facade.step(action)
        self.last_obs = obs
        self.last_info = info
        self.terminated = bool(term)
        self.truncated = bool(trunc)

    def render_camera(
        self,
        camera_name: str,
        height: int = IMAGE_SIZE,
        width: int = IMAGE_SIZE,
        depth: bool = False,
    ) -> Any:
        return self._facade.render_camera(
            camera_name=camera_name, height=height, width=width, depth=depth
        )

    def get_camera_meta(
        self, camera_name: str, height: int = IMAGE_SIZE, width: int = IMAGE_SIZE
    ) -> dict[str, Any] | None:
        return self._facade.get_camera_meta(
            camera_name=camera_name, height=height, width=width
        )

    def get_task_language(self) -> str:
        return str(self._facade.get_task_language() or "")

    def close(self) -> None:
        self._facade.close()


class LiberoRLPrimitives(RecordedPerception):
    """Servo one LIBERO Franka Panda arm and query recorded RGB-D views."""

    cameras = tuple(SIM_CAMERAS)

    def __init__(
        self,
        *,
        env: Any,
        sam3_client: Any = None,
        state: Any = None,
        check_cancelled,
        config: ControlConfig = ControlConfig(),
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
            segment_min_score=config.segment_min_score,
        )
        self._config = config
        control = env.last_obs["control_state"]
        self.coordinates = Coordinates.from_control(control)
        self.tool_rotation = grasp_axes(
            control["jaw_slide_axes_site"][0], control["wrist_to_site_in_site"]
        )
        self._save_artifact("coordinates.json", self.coordinates.metadata(), None)
        self._open_gap = self._finger_gap()
        if self.record_artifacts and "main_images" in env.last_obs:
            self.frames = [np.asarray(env.last_obs["main_images"])]

    @property
    def ended(self) -> bool:
        return self.env.terminated or self.env.truncated

    def native_success(self) -> bool:
        return bool(self.env.terminated)

    def _ended_reason(self) -> str:
        return "terminated" if self.env.terminated else "truncated"

    def _finger_gap(self) -> float:
        states = np.asarray(self.env.last_obs["states"], dtype=np.float64)
        return float(abs(states[6]) + abs(states[7]))

    def _measure_native(self) -> Measurement:
        control = self.env.last_obs["control_state"]
        return Measurement(
            xyz=np.asarray(control["xyz"], dtype=np.float64),
            rotation=np.asarray(control["rotation"], dtype=np.float64),
            opening=min(1.0, self._finger_gap() / self._open_gap),
        )

    def measure(self) -> Measurement:
        native = self._measure_native()
        return Measurement(
            self.coordinates.points(native.xyz),
            self.coordinates.orientation(native.rotation @ self.tool_rotation),
            native.opening,
        )

    def robot_state(self) -> dict[str, Any]:
        return {
            **self.measure().state(),
            "finger_gap_m": round(self._finger_gap(), 5),
        }

    def capture(self, camera: str) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        sim_camera = SIM_CAMERAS[camera]
        rgb, depth = self.env.render_camera(
            sim_camera, IMAGE_SIZE, IMAGE_SIZE, depth=True
        )
        meta = self.env.get_camera_meta(sim_camera, IMAGE_SIZE, IMAGE_SIZE)
        if meta is None:
            raise RuntimeError(f"{sim_camera} camera metadata is unavailable")
        metric = geometry.metric_depth(depth, meta)[::-1]
        world = geometry.world_from_depth(metric, meta)
        valid = (
            np.isfinite(metric)
            & (metric > meta["depth_near"] * (1 + 1e-4))
            & (metric < meta["depth_far"] * (1 - 1e-5))
        )
        world[~valid] = np.nan
        calibration = {
            **meta,
            "coordinates": self.coordinates.metadata(),
            "tool_rotation_to_native": self.tool_rotation.tolist(),
        }
        return np.asarray(rgb)[::-1], world, calibration

    def _step(self, action: np.ndarray) -> None:
        self._check_cancelled()
        self.env.step(action)
        self.env_steps += 1
        if self.record_artifacts and "main_images" in self.env.last_obs:
            self.frames.append(np.asarray(self.env.last_obs["main_images"]))

    def _pose_action(
        self, current: Measurement, xyz: np.ndarray, rotation: np.ndarray
    ) -> tuple[np.ndarray, float, float]:
        position_error = xyz - current.xyz
        rotation_error = Rotation.from_matrix(rotation @ current.rotation.T).as_rotvec()
        action = np.zeros(7)
        control = self.env.last_obs["control_state"]
        frame = np.asarray(control["control_to_world_rotation"], dtype=np.float64)
        delta = np.concatenate(
            [
                frame.T @ _clip_norm(position_error, self._config.position_step_m),
                frame.T @ _clip_norm(rotation_error, self._config.rotation_step_rad),
            ]
        )
        low, high = np.asarray(control["output_min"]), np.asarray(control["output_max"])
        input_low, input_high = (
            np.asarray(control["input_min"]),
            np.asarray(control["input_max"]),
        )
        if np.any(high <= low):
            raise ValueError("native OSC controller has invalid output bounds")
        action[:6] = (delta - (high + low) / 2) * (input_high - input_low) / (
            high - low
        ) + (input_high + input_low) / 2
        if np.any(action[:6] < input_low - 1e-8) or np.any(
            action[:6] > input_high + 1e-8
        ):
            raise ValueError("servo step exceeds the native controller bounds")
        return (
            action,
            float(np.linalg.norm(position_error)),
            float(np.linalg.norm(rotation_error)),
        )

    @tool(
        json_schema_extra={
            "additionalProperties": False,
            "anyOf": [
                {"required": ["xyz"], "properties": {"xyz": {"type": "array"}}},
                {
                    "required": ["orientation_rpy"],
                    "properties": {"orientation_rpy": {"type": "array"}},
                },
            ],
        }
    )
    def move_to(
        self,
        xyz: Annotated[list[float], Field(min_length=3, max_length=3)] | None = None,
        orientation_rpy: Annotated[list[float], Field(min_length=3, max_length=3)]
        | None = None,
    ) -> ToolResult:
        """Servo the grip_site toward an absolute world pose, without collision avoidance. Supply xyz, orientation_rpy, or both; an omitted part keeps its measured value at the start of this action. The gripper keeps its previous opening target. Returns reached only within 4mm and 0.02rad; partial means the target was not reached and reports a reason. Use the returned measured pose, never assume the command succeeded. Example: {"xyz":[0.1,0,0.3],"orientation_rpy":[3.14159265,0,0]}.

        Args:
            xyz: Absolute grip-site position [x,y,z] in world metres; three finite numbers.
            orientation_rpy: Absolute grip-site [roll,pitch,yaw] in radians, fixed world-axis XYZ: R=Rz(yaw)@Ry(pitch)@Rx(roll). [pi,0,yaw] points the approach axis down. Omit to keep the complete measured rotation.
        """
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
        """Drive the gripper to a normalized opening while holding the end-effector pose. Closing on an object stops early and reports the measured opening; contact does not imply a stable grasp. The target stays in effect during later moves. Returns reached/partial/error, measured pose/opening, residual opening error and environment steps. Example: {"opening":1}.

        Args:
            opening: Target opening from 0 (closed) to 1 (fully open).
        """
        if self.ended:
            return ToolResult(data={"error": "episode_ended"})
        config = self._config
        hold = self._measure_native()
        hold_xyz = hold.xyz.copy()
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
            action, _, _ = self._pose_action(current, hold_xyz, hold.rotation)
            action[6] = 0.0 if within else float(-np.sign(error))
            self._step(action)
        return ToolResult(
            data={
                "status": "reached" if reason is None else "partial",
                **({} if reason is None else {"reason": reason}),
                "env_steps": steps,
                "opening_error": round(error, 3),
            }
        )
