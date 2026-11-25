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

"""RPC server owning one VLABench Franka environment."""

from __future__ import annotations

import argparse
import sys
import threading
import uuid
from pathlib import Path
from typing import Any

import numpy as np

# Support direct execution from an RPent checkout before package imports.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from robots.vlabench.sim.contracts import (
    DEFAULT_EPISODE_STEPS,
    DEFAULT_TRACK,
    VLABENCH_CAMERA_NAMES,
    VLABENCH_NATIVE_CAMERAS,
    env_identity,
)
from rpent.robots.components.env_facade_base import BaseEnvFacade
from rpent.utils.logging import get_logger
from rpent.utils.rpc.main_thread_serve import MainThreadServeMixin

logger = get_logger("vlabench_env_server")


class VLABenchEnvFacade(MainThreadServeMixin, BaseEnvFacade):
    """Expose task-safe sensors and pose/gripper motion over RPC.

    The MuJoCo EGL context must stay on one thread, so every dispatch runs on
    the main thread through ``MainThreadServeMixin``.
    """

    def __init__(
        self, env: Any, *, metadata: dict[str, Any], max_episode_steps: int
    ) -> None:
        self._env = env
        self._metadata = dict(metadata)
        self._max_episode_steps = int(max_episode_steps)
        self._control_steps = 0
        self._episode_id = "sim_" + uuid.uuid4().hex
        self._terminated = False
        self._truncated = False
        self._instruction: str | None = None
        self._motion_lock = threading.Lock()
        self._motion_id: str | None = None
        self._motion_started = False
        self._motion_cancelled = threading.Event()
        self._gravity_layout: dict[str, np.ndarray] | None = None
        self._gravity_stats = self._new_gravity_stats()
        super().__init__()

    def get_env_meta(self) -> dict[str, Any]:
        """Return immutable identity for endpoint compatibility checks."""
        return dict(self._metadata)

    def close(self) -> None:
        self._env.close()

    # --- section: episode state (reset, task status, budget) ---
    def reset(self) -> dict[str, Any]:
        """Start a new episode and return its identity."""
        self._env.reset()
        self._control_steps = 0
        self._episode_id = "sim_" + uuid.uuid4().hex
        self._terminated = False
        self._truncated = False
        self._instruction = self._env.task.get_instruction()
        self._gravity_stats = self._new_gravity_stats()
        return {
            "episode_id": self._episode_id,
            "task_name": self._metadata["task_name"],
        }

    def get_task_language(self) -> str:
        if self._instruction is None:
            self._instruction = self._env.task.get_instruction()
        return str(self._instruction or "")

    def _task_success(self) -> bool:
        return bool(self._env.task.should_terminate_episode(self._env.physics))

    def episode_budget(self) -> dict[str, int]:
        """Return the native control-step counters stored with each observation."""
        return {
            "step": self._control_steps,
            "step_limit": self._max_episode_steps,
            "remaining_steps": max(0, self._max_episode_steps - self._control_steps),
        }

    def task_status(self) -> dict[str, Any]:
        """Report native success and terminal flags without advancing the env."""
        success = self._task_success()
        return {
            "available": True,
            "episode_id": self._episode_id,
            "control_steps": self._control_steps,
            "step_limit": self._max_episode_steps,
            "episode_status": self.episode_budget(),
            "task_success": success,
            "terminated": bool(self._terminated or success),
            "truncated": bool(self._truncated),
        }

    def _require_active(self) -> None:
        status = self.task_status()
        if status["task_success"] or self._terminated or self._truncated:
            raise RuntimeError("VLABench episode is terminal")

    def _step(self, action: np.ndarray) -> Any:
        """Advance one native control step and update the terminal flags."""
        self._require_active()
        timestep = self._env.step(action)
        self._control_steps += 1
        success = self._task_success()
        if success or timestep.last():
            self._terminated = success
            self._truncated = not success
        elif self._control_steps >= self._max_episode_steps:
            self._truncated = True
        return timestep

    # --- section: sensors (sensor_snapshot, camera calibration) ---
    def _camera_id(self, alias: str) -> int:
        if alias not in VLABENCH_NATIVE_CAMERAS:
            raise ValueError(f"unsupported VLABench camera: {alias}")
        name = VLABENCH_NATIVE_CAMERAS[alias]
        camera_id = self._env.physics.model.name2id(name, "camera")
        if camera_id < 0:
            raise ValueError(f"VLABench task has no camera named {name!r}")
        return camera_id

    def _camera_calibration(
        self, camera_id: int, width: int, height: int
    ) -> tuple[np.ndarray, np.ndarray]:
        intrinsic, _ = self._env.get_camera_matrix(camera_id, width, height)
        data = self._env.physics.data
        rotation = np.asarray(data.cam_xmat[camera_id], dtype=float).reshape(3, 3)
        position = np.asarray(data.cam_xpos[camera_id], dtype=float)
        if (
            position.shape != (3,)
            or not np.isfinite(rotation).all()
            or not np.isfinite(position).all()
        ):
            raise RuntimeError("VLABench returned an invalid current camera pose")
        extrinsic = np.eye(4)
        # MuJoCo camera +Y/+Z point up/back; metric depth uses +Y down/+Z forward.
        extrinsic[:3, :3] = rotation @ np.diag([1.0, -1.0, -1.0])
        extrinsic[:3, 3] = position
        return np.asarray(intrinsic, dtype=float), extrinsic

    def robot_model_state(self) -> dict[str, Any]:
        """Return the Franka arm joints, finger widths and EEF pose in world."""
        from scipy.spatial.transform import Rotation

        physics, robot = self._env.physics, self._env.robot
        position = np.asarray(robot.get_end_effector_pos(physics), dtype=float)
        quat_wxyz = np.asarray(robot.get_end_effector_quat(physics), dtype=float)
        if quat_wxyz.shape != (4,) or not np.isfinite(quat_wxyz).all():
            raise RuntimeError("VLABench returned an invalid end-effector quaternion")
        pose = np.eye(4)
        pose[:3, :3] = Rotation.from_quat(quat_wxyz[[1, 2, 3, 0]]).as_matrix()
        pose[:3, 3] = position
        fingers = [
            float(
                np.asarray(
                    physics.bind(robot.mjcf_model.find("joint", name)).qpos
                ).reshape(-1)[0]
            )
            for name in ("finger_joint1", "finger_joint2")
        ]
        arm_qpos = np.asarray(robot.get_qpos(physics), dtype=float)
        if (
            arm_qpos.shape != (7,)
            or not np.isfinite(arm_qpos).all()
            or not np.isfinite(position).all()
            or not np.isfinite(fingers).all()
        ):
            raise RuntimeError("VLABench returned invalid Franka state")
        return {
            "schema_version": "rpent.vlabench_model_state.v1",
            "robot": "franka",
            "episode_id": self._episode_id,
            "control_steps": self._control_steps,
            "arm_qpos": arm_qpos,
            "gripper_qpos": np.asarray(fingers, dtype=float),
            "world_from_eef": pose,
        }

    def sensor_snapshot(
        self,
        cameras: tuple[str, ...] = VLABENCH_CAMERA_NAMES,
        height: int = 512,
        width: int = 512,
    ) -> dict[str, Any]:
        """Render calibrated RGB-D frames for the requested cameras."""
        if type(height) is not int or type(width) is not int or min(height, width) < 32:
            raise ValueError("camera height and width must be integers >= 32")
        if not cameras or len(set(cameras)) != len(cameras):
            raise ValueError("request nonempty unique VLABench cameras")
        physics = self._env.physics
        frames = []
        for alias in cameras:
            camera_id = self._camera_id(alias)
            rgb = np.asarray(
                physics.render(height=height, width=width, camera_id=camera_id),
                dtype=np.uint8,
            )
            depth = np.asarray(
                physics.render(
                    height=height, width=width, camera_id=camera_id, depth=True
                ),
                dtype=np.float32,
            )
            if rgb.shape != (height, width, 3) or depth.shape != (height, width):
                raise RuntimeError("VLABench render returned an unexpected shape")
            intrinsic, extrinsic = self._camera_calibration(camera_id, width, height)
            frames.append(
                {
                    "camera": alias,
                    "depth_encoding": "metric_top_down",
                    "rgb_top_down": rgb,
                    "depth_meters": np.where(
                        np.isfinite(depth) & (depth > 0), depth, 0
                    ),
                    "calibration": {
                        "intrinsic_K": intrinsic,
                        "extrinsic_cam2world": extrinsic,
                        "height": height,
                        "width": width,
                    },
                }
            )
        return {
            "episode_id": self._episode_id,
            "observation_id": "obs_" + uuid.uuid4().hex,
            "cameras": frames,
            "robot": {"model": self.robot_model_state()},
            "episode_status": self.episode_budget(),
        }

    def render_camera(
        self,
        camera_name: str,
        *,
        height: int = 512,
        width: int = 512,
        depth: bool = False,
    ) -> Any:
        return self._env.physics.render(
            height=height,
            width=width,
            camera_id=self._camera_id(camera_name),
            depth=depth,
        )

    def get_camera_meta(
        self, camera_name: str, *, height: int = 512, width: int = 512
    ) -> dict[str, Any]:
        intrinsic, extrinsic = self._camera_calibration(
            self._camera_id(camera_name), width, height
        )
        return {
            "intrinsic_K": intrinsic,
            "extrinsic_cam2world": extrinsic,
            "height": height,
            "width": width,
        }

    # --- section: motion (execute_pose, execute_gripper, cancellation) ---
    @staticmethod
    def _new_gravity_stats() -> dict[str, Any]:
        return {
            "method": "joint_bias_feedforward_via_position_target",
            "steps": 0,
            "force_limited_joint_steps": 0,
            "max_abs_compensation_torque_nm": 0.0,
        }

    def _gravity_layout_for_arm(self) -> dict[str, np.ndarray]:
        """Read the arm's position-actuator gains once and cache them."""
        if self._gravity_layout is not None:
            return self._gravity_layout
        physics, robot = self._env.physics, self._env.robot
        model = physics.model
        joints = robot.joints[: robot.n_dof]
        if len(joints) != 7:
            raise RuntimeError("VLABench Franka compensation needs 7 arm joints")
        dofs, gains, gears, limits = [], [], [], []
        for index, joint in enumerate(joints):
            joint_id = int(model.name2id(joint.full_identifier, "joint"))
            matching = np.flatnonzero(model.actuator_trnid[:, 0] == joint_id)
            if matching.tolist() != [index]:
                raise RuntimeError(
                    f"Expected one direct position actuator for {joint.full_identifier}"
                )
            actuator = int(matching[0])
            gain = float(model.actuator_gainprm[actuator, 0])
            bias = np.asarray(model.actuator_biasprm[actuator, :3], dtype=float)
            gear = float(model.actuator_gear[actuator, 0])
            if (
                not np.isfinite(gain)
                or gain <= 0
                or not np.isfinite(gear)
                or abs(gear) < 1e-8
                or not np.allclose(bias[:2], [0.0, -gain], atol=1e-8)
                or not np.isfinite(bias[2])
            ):
                raise RuntimeError(
                    f"{joint.full_identifier} is not an affine PD position actuator"
                )
            if bool(model.actuator_forcelimited[actuator]):
                lower, upper = model.actuator_forcerange[actuator]
                limits.append((float(lower), float(upper)))
            else:
                limits.append((-np.inf, np.inf))
            dofs.append(int(model.jnt_dofadr[joint_id]))
            gains.append(gain)
            gears.append(gear)
        self._gravity_layout = {
            "dof_addresses": np.asarray(dofs, dtype=int),
            "gains": np.asarray(gains, dtype=float),
            "gears": np.asarray(gears, dtype=float),
            "force_limits": np.asarray(limits, dtype=float),
        }
        return self._gravity_layout

    def _gravity_compensated_action(self, action: np.ndarray) -> np.ndarray:
        """Add the joint bias force to the arm targets of a 9-vector action."""
        command = np.asarray(action, dtype=float).copy()
        if command.shape != (9,) or not np.isfinite(command).all():
            raise ValueError("joint action must be a finite 9-vector")
        layout = self._gravity_layout_for_arm()
        bias = np.asarray(
            self._env.physics.data.qfrc_bias[layout["dof_addresses"]], dtype=float
        )
        if not np.isfinite(bias).all():
            raise RuntimeError("MuJoCo returned non-finite generalized bias forces")
        force = bias / layout["gears"]
        clipped = np.clip(
            force, layout["force_limits"][:, 0], layout["force_limits"][:, 1]
        )
        command[:7] += clipped / layout["gains"]
        stats = self._gravity_stats
        stats["steps"] += 1
        stats["force_limited_joint_steps"] += int(
            np.count_nonzero(np.abs(clipped - force) > 1e-9)
        )
        stats["max_abs_compensation_torque_nm"] = max(
            stats["max_abs_compensation_torque_nm"],
            float(np.max(np.abs(clipped * layout["gears"]))),
        )
        return command

    @staticmethod
    def _finger_target(gripper: Any, current: np.ndarray) -> np.ndarray:
        """Map -1 (open) .. +1 (close), or hold, to two finger joint targets."""
        if gripper == "hold" or (not isinstance(gripper, bool) and gripper == 0):
            return np.asarray(current, dtype=float).copy()
        if isinstance(gripper, bool) or not isinstance(gripper, (int, float)):
            raise ValueError("gripper must be -1..+1 or 'hold'")
        value = float(gripper)
        if not np.isfinite(value) or not -1 <= value <= 1:
            raise ValueError("gripper must be between -1 and +1")
        return np.full(2, 0.04 * (1.0 - value) / 2.0, dtype=float)

    @staticmethod
    def _stop_reason(status: dict[str, Any], completed: str) -> str:
        if status["task_success"]:
            return "native_success"
        if status["truncated"]:
            if status["episode_status"]["remaining_steps"] == 0:
                return "budget_exhausted"
            return "environment_terminal"
        return completed

    def _gravity_report(self, start: dict[str, Any]) -> dict[str, Any]:
        """Summarize the compensation applied since ``start`` was snapshotted."""
        stats = self._gravity_stats
        return {
            "method": stats["method"],
            "steps": stats["steps"] - start["steps"],
            "force_limited_joint_steps": (
                stats["force_limited_joint_steps"] - start["force_limited_joint_steps"]
            ),
            "max_abs_compensation_torque_nm": stats["max_abs_compensation_torque_nm"],
        }

    def execute_gripper(
        self, *, gripper: float, steps: int = 5, motion_id: str | None = None
    ) -> dict[str, Any]:
        """Hold the arm and drive the fingers for a bounded number of steps."""
        if type(steps) is not int or not 1 <= steps <= 1000:
            raise ValueError("steps must be an integer from 1 to 1000")
        self._require_active()
        cancelled = self._start_motion(motion_id)
        robot, physics = self._env.robot, self._env.physics
        start = dict(self._gravity_stats)
        arm_target = np.asarray(robot.get_qpos(physics), dtype=float)
        finger_target = self._finger_target(
            gripper, self.robot_model_state()["gripper_qpos"]
        )
        action = np.concatenate([arm_target, finger_target])
        used = 0
        interrupted = False
        for _ in range(steps):
            if cancelled.is_set():
                interrupted = True
                break
            self._step(self._gravity_compensated_action(action))
            used += 1
            if self._terminated or self._truncated:
                break
        status = self.task_status()
        return {
            "ok": used == steps,
            "interrupted": interrupted,
            "motion_id": motion_id,
            "commands_completed": used == steps,
            "requested_steps": steps,
            "episode_status": status["episode_status"],
            "stop_reason": self._stop_reason(
                status, "cancelled" if interrupted else "completed"
            ),
            "controller": "vlabench_franka_gripper",
            "steps_used": used,
            "gravity_compensation": self._gravity_report(start),
            "terminated": status["terminated"],
            "truncated": status["truncated"],
        }

    def execute_pose(
        self,
        *,
        xyz: list[float],
        quaternion_xyzw: list[float],
        gripper: Any = "hold",
        max_steps: int = 150,
        tol: float = 0.012,
        step_clip: float = 0.02,
        ori_tol: float = 0.05,
        motion_id: str | None = None,
    ) -> dict[str, Any]:
        """Servo the end effector toward a world pose with waypointed Franka IK."""
        from scipy.spatial.transform import Rotation

        target_xyz = np.asarray(xyz, dtype=float)
        target_quat = np.asarray(quaternion_xyzw, dtype=float)
        if target_xyz.shape != (3,) or not np.isfinite(target_xyz).all():
            raise ValueError("xyz must be a finite 3-vector")
        if target_quat.shape != (4,) or not np.isfinite(target_quat).all():
            raise ValueError("quaternion_xyzw must be a finite 4-vector")
        norm = float(np.linalg.norm(target_quat))
        if norm < 1e-8 or abs(norm - 1) > 0.01:
            raise ValueError("quaternion_xyzw must be unit length")
        target_rot = Rotation.from_quat(target_quat / norm)
        if (
            type(max_steps) is not int
            or not 1 <= max_steps <= 1000
            or not np.isfinite(tol)
            or tol <= 0
            or not np.isfinite(step_clip)
            or step_clip <= 0
            or not np.isfinite(ori_tol)
            or ori_tol <= 0
        ):
            raise ValueError("invalid VLABench servo limits")
        self._require_active()
        cancelled = self._start_motion(motion_id)
        robot, physics = self._env.robot, self._env.physics
        start = dict(self._gravity_stats)
        initial = self.robot_model_state()
        finger_target = self._finger_target(gripper, initial["gripper_qpos"])
        closing_steps = (
            min(max_steps, 5)
            if gripper not in ("hold", 0)
            and np.max(np.abs(finger_target - initial["gripper_qpos"])) > 1e-6
            else 0
        )
        used = 0
        ik_failed = False
        interrupted = False
        waypoint_xyz = waypoint_rot = arm_target = None
        waypoint_tol = max(0.004, min(0.008, step_clip * 0.25))
        waypoint_ori_tol = min(ori_tol, 0.02)
        last_error = None
        stalled = 0
        joint_names = [j.full_identifier for j in robot.joints[: robot.n_dof]]
        for _ in range(max_steps):
            if cancelled.is_set():
                interrupted = True
                break
            current_xyz = np.asarray(robot.get_end_effector_pos(physics), dtype=float)
            current_wxyz = np.asarray(robot.get_end_effector_quat(physics), dtype=float)
            current_rot = Rotation.from_quat(current_wxyz[[1, 2, 3, 0]])
            delta_xyz = target_xyz - current_xyz
            delta_rot = target_rot * current_rot.inv()
            if (
                np.linalg.norm(delta_xyz) <= tol
                and float(delta_rot.magnitude()) <= ori_tol
            ):
                if used >= closing_steps:
                    break
                hold = np.asarray(robot.get_qpos(physics), dtype=float)
                self._step(
                    self._gravity_compensated_action(
                        np.concatenate([hold, finger_target])
                    )
                )
                used += 1
                if self._terminated or self._truncated:
                    break
                continue

            waypoint_error = (
                float(np.linalg.norm(waypoint_xyz - current_xyz))
                if waypoint_xyz is not None
                else None
            )
            reached = waypoint_error is not None and (
                waypoint_error <= waypoint_tol
                and (waypoint_rot * current_rot.inv()).magnitude() <= waypoint_ori_tol
            )
            if waypoint_error is not None and last_error is not None:
                stalled = stalled + 1 if last_error - waypoint_error < 0.0002 else 0
            else:
                stalled = 0
            if waypoint_error is not None and stalled >= 2:
                reached = True
            if waypoint_xyz is None or reached:
                last_error, stalled = None, 0
                waypoint_xyz = current_xyz + np.clip(delta_xyz, -step_clip, step_clip)
                rotvec = delta_rot.as_rotvec()
                angle = float(np.linalg.norm(rotvec))
                if angle > 0.08:
                    rotvec *= 0.08 / angle
                waypoint_rot = Rotation.from_rotvec(rotvec) * current_rot
                ik_ok, arm_target = robot.get_qpos_from_ee_pos(
                    physics=physics,
                    pos=waypoint_xyz,
                    quat=waypoint_rot.as_quat()[[3, 0, 1, 2]],
                    inplace=False,
                    joint_names=joint_names,
                )
                if not ik_ok:
                    ik_failed = True
                    break
            last_error = waypoint_error
            action = np.concatenate(
                [np.asarray(arm_target, dtype=float), finger_target]
            )
            if action.shape != (9,) or not np.isfinite(action).all():
                raise RuntimeError("VLABench Franka IK returned an invalid target")
            if cancelled.is_set():
                interrupted = True
                break
            self._step(self._gravity_compensated_action(action))
            used += 1
            if self._terminated or self._truncated:
                break

        final = self.robot_model_state()["world_from_eef"]
        actual_xyz = np.asarray(final, dtype=float)[:3, 3]
        actual_rot = Rotation.from_matrix(np.asarray(final, dtype=float)[:3, :3])
        distance = float(np.linalg.norm(target_xyz - actual_xyz))
        angle_error = float((target_rot * actual_rot.inv()).magnitude())
        reached_goal = distance <= tol and angle_error <= ori_tol
        status = self.task_status()
        return {
            "ok": bool(
                reached_goal
                and not ik_failed
                and not interrupted
                and used >= closing_steps
            ),
            "interrupted": interrupted,
            "motion_id": motion_id,
            "controller": "vlabench_franka_ik",
            "steps_used": used,
            "max_steps": max_steps,
            "requested_steps": max_steps,
            "episode_status": status["episode_status"],
            "stop_reason": self._stop_reason(
                status,
                "cancelled"
                if interrupted
                else "reached"
                if reached_goal
                else "ik_failed"
                if ik_failed
                else "max_steps",
            ),
            "ik_failed": ik_failed,
            "final_eef_pos": actual_xyz.tolist(),
            "final_dist_m": distance,
            "final_orientation_error_rad": angle_error,
            "gravity_compensation": self._gravity_report(start),
            "terminated": status["terminated"],
            "truncated": status["truncated"],
        }

    def _register_rpc(self) -> None:
        super()._register_rpc()
        self._rpc.update(
            {
                "env.sensor_snapshot": self.sensor_snapshot,
                "env.robot_model_state": self.robot_model_state,
                "env.task_status": self.task_status,
                "env.execute_pose": self.execute_pose,
                "env.execute_gripper": self.execute_gripper,
                "env.begin_motion": self.begin_motion,
                "env.cancel_motion": self.cancel_motion,
            }
        )
        self._readonly_methods.update(
            {
                "env.sensor_snapshot",
                "env.robot_model_state",
                "env.task_status",
            }
        )

    def _dispatch_main_thread(
        self, method: str, args: tuple, kwargs: dict, *, session_id: Any = None
    ) -> Any:
        # Cancellation is the only call that must not wait behind a running
        # motion, and it never touches the simulator.
        if method == "env.cancel_motion":
            return self.cancel_motion(*args, **kwargs)
        return super()._dispatch_main_thread(
            method, args, kwargs, session_id=session_id
        )

    def step(self, flat_action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        """Advance one native step with a 9-dim joint action (arm 7 + fingers 2)."""
        action = np.asarray(flat_action, dtype=float)
        if action.shape != (9,) or not np.isfinite(action).all():
            raise ValueError("joint action must be a finite 9-vector")
        timestep = self._step(action)
        status = self.task_status()
        return (
            timestep.observation,
            0.0,
            status["terminated"],
            status["truncated"],
            status,
        )

    def chunk_step(
        self, actions: Any, *, return_all_frames: bool = False
    ) -> tuple[Any, np.ndarray, bool, bool, list[dict[str, Any]]]:
        """Advance several native steps, stopping at the first terminal flag."""
        array = np.asarray(actions, dtype=float)
        if array.ndim != 2 or array.shape[1] != 9 or not np.isfinite(array).all():
            raise ValueError("joint actions must have shape [N,9]")
        observations, rewards, infos = [], [], []
        for action in array:
            timestep = self._step(action)
            observations.append(timestep.observation)
            rewards.append(0.0)
            infos.append(self.task_status())
            if self._terminated or self._truncated:
                break
        observation = observations if return_all_frames else observations[-1]
        return (
            observation,
            np.asarray(rewards),
            self._terminated,
            self._truncated,
            infos,
        )

    def begin_motion(self) -> str:
        """Reserve a one-use motion token on the simulation thread."""
        with self._motion_lock:
            self._motion_id = uuid.uuid4().hex
            self._motion_started = False
            self._motion_cancelled = threading.Event()
            return self._motion_id

    def cancel_motion(self, *, motion_id: str) -> dict[str, Any]:
        """Request cancellation of the reserved motion; never touches the simulator."""
        with self._motion_lock:
            accepted = isinstance(motion_id, str) and motion_id == self._motion_id
            if accepted:
                self._motion_cancelled.set()
            return {"motion_id": motion_id, "cancel_requested": accepted}

    def _start_motion(self, motion_id: str | None) -> threading.Event:
        """Consume the motion token and return its cancellation event."""
        if motion_id is None:
            # In-process callers have no RPC token and cannot cancel.
            return threading.Event()
        with self._motion_lock:
            if motion_id != self._motion_id or self._motion_started:
                raise ValueError("stale or already consumed motion token")
            self._motion_started = True
            return self._motion_cancelled


def make_env(
    task_name: str,
    *,
    vlabench_root: Path,
    asset_root: Path,
    track: str,
    episode_index: int,
    seed: int,
    eval_unseen: bool,
    max_episode_steps: int | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Construct the VLABench task environment and its run provenance.

    VLABench imports stay inside this function so the module imports without
    the simulator installed.

    Returns:
        The environment and the metadata the server reports beyond the launch
        identity: effective step limit, official limit and the track file hash.
    """
    import hashlib
    import json
    import os
    import random
    import subprocess

    if not (vlabench_root / "configs" / "task_config.json").is_file():
        raise FileNotFoundError(f"VLABench configs not found under {vlabench_root}")
    for name in ("obj", "scenes"):
        link = vlabench_root / "assets" / name
        if not link.is_symlink() or link.resolve() != asset_root / name:
            raise ValueError(f"{link} must link to {asset_root / name}")

    track_path = vlabench_root / "configs" / "evaluation" / "tracks" / f"{track}.json"
    tracks = json.loads(track_path.read_text())
    if task_name not in tracks:
        raise ValueError(f"task {task_name!r} is not included in {track}")
    episodes = tracks[task_name]
    if type(episode_index) is not int or not 0 <= episode_index < len(episodes):
        raise ValueError(f"episode index must be in [0, {len(episodes) - 1}]")

    if max_episode_steps is not None and (
        type(max_episode_steps) is not int or max_episode_steps <= 0
    ):
        raise ValueError("max-episode-steps must be a positive integer")
    effective_steps = max_episode_steps or DEFAULT_EPISODE_STEPS

    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ["VLABENCH_ROOT"] = str(vlabench_root)
    sys.path.insert(0, str(vlabench_root.parent))

    # The package initialization sets resource paths, so it imports first.
    import VLABench  # noqa: F401

    os.environ["VLABENCH_ROOT"] = str(vlabench_root)
    import VLABench.robots  # noqa: F401
    import VLABench.tasks  # noqa: F401
    from VLABench.configs import name2config
    from VLABench.envs import load_env
    from VLABench.utils.utils import find_key_by_value

    task_config_path = vlabench_root / "configs" / "task_config.json"
    task_config_bytes = task_config_path.read_bytes()
    series = find_key_by_value(name2config, task_name)
    configured = (
        json.loads(task_config_bytes)
        .get(series, {})
        .get("evaluation", {})
        .get("max_episode_length")
    )
    official_limit = 200 if configured is None else configured
    if type(official_limit) is not int or official_limit <= 0:
        raise ValueError(f"invalid VLABench max_episode_length for {task_name}")

    random.seed(seed)
    np.random.seed(seed)
    env = load_env(
        task_name,
        robot="franka",
        time_limit=float("inf"),
        episode_config=episodes[episode_index],
        random_init=False,
        eval=eval_unseen,
        run_mode="eval",
    )
    commit = subprocess.run(
        ["git", "-C", str(vlabench_root.parent), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    ).stdout.strip()
    provenance = {
        "max_episode_steps": effective_steps,
        "episode_budget": {
            "official_step_limit": official_limit,
            "step_limit": effective_steps,
            "requested_override": max_episode_steps,
            "matches_official_limit": effective_steps == official_limit,
            "unit": "native_env_step",
            "source": {
                "file": "configs/task_config.json",
                "sha256": hashlib.sha256(task_config_bytes).hexdigest(),
                "task_series": series,
                "field": "evaluation.max_episode_length",
                "evaluator_default_used": configured is None,
            },
        },
        "track_file": track_path.name,
        "track_sha256": hashlib.sha256(track_path.read_bytes()).hexdigest(),
        "vlabench_commit": commit or None,
    }
    return env, provenance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transport", choices=("socket", "http"), default="http")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--task-name", required=True)
    parser.add_argument("--track", default=DEFAULT_TRACK)
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-unseen", action="store_true")
    parser.add_argument(
        "--vlabench-root",
        type=Path,
        required=True,
        help="Inner VLABench package directory containing configs/.",
    )
    parser.add_argument(
        "--asset-root",
        type=Path,
        required=True,
        help="External VLABench asset directory containing obj/ and scenes/.",
    )
    parser.add_argument(
        "--max-episode-steps",
        type=int,
        help=f"Custom run budget (default: {DEFAULT_EPISODE_STEPS}).",
    )
    parser.add_argument("--parent-watch", action="store_true")
    args = parser.parse_args()

    env, provenance = make_env(
        args.task_name,
        vlabench_root=args.vlabench_root.resolve(),
        asset_root=args.asset_root.resolve(),
        track=args.track,
        episode_index=args.episode_index,
        seed=args.seed,
        eval_unseen=args.eval_unseen,
        max_episode_steps=args.max_episode_steps,
    )
    metadata = {
        **env_identity(
            task_name=args.task_name,
            track=args.track,
            episode_index=args.episode_index,
            seed=args.seed,
        ),
        **provenance,
    }
    VLABenchEnvFacade(
        env,
        metadata=metadata,
        max_episode_steps=provenance["max_episode_steps"],
    ).serve(
        transport=args.transport,
        host=args.host,
        port=args.port,
        parent_watch=args.parent_watch,
    )


if __name__ == "__main__":
    main()
