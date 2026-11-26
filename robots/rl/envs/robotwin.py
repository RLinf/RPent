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

"""RoboTwin dual-arm general-frame adapter and in-process warm host."""

from __future__ import annotations

import gc
from typing import Annotated, Any, Literal

import numpy as np
from PIL import Image, ImageDraw
from pydantic import Field
from scipy.spatial.transform import Rotation

from robots.rl import geometry
from robots.rl.perception import RecordedPerception
from robots.robotwin.general import RobotwinFrame, robotwin_calibration
from rpent.tools import ToolResult, tool
from rpent.utils.transforms import quaternion_wxyz, rotation_from_wxyz

_POSITION_TOL_M = 0.004
_ROTATION_TOL_RAD = 0.02
_OPENING_TOL = 0.08


def planner_command_pose(
    planner: Any, actual_base_pose: Any, target_pose: Any
) -> list[float]:
    """Map a world goal through the native planner's nominal base convention."""
    pose_type = type(actual_base_pose)
    nominal_base = planner.robot_origion_pose * pose_type(
        -np.asarray(planner.frame_bias)
    )
    goal = (
        nominal_base
        * actual_base_pose.inv()
        * pose_type(target_pose[:3], target_pose[3:])
    )
    return [*goal.p, *goal.q]


def release_unused_batch_planners(env: Any) -> None:
    """Drop unused batch cuRobo planners to keep GPU memory tight."""
    import torch

    sub_env = env._sub_env(0)
    with sub_env.lock:
        robot = getattr(sub_env.task, "robot", None)
        if robot is not None:
            for arm in ("left", "right"):
                planner = getattr(robot, f"{arm}_planner", None)
                if planner is not None:
                    planner.motion_gen_batch = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class LocalRoboTwinEnv:
    """In-process RoboTwin host exposing the same interface as RoboTwinEnvClient.

    Runs directly inside a warm GPU worker process (zero extra env_server IPC hop).
    """

    def __init__(
        self,
        *,
        task_name: str,
        task_config: str,
        seed: int,
        assets_path: str,
        max_episode_steps: int = 10000,
        record_artifacts: bool = False,
    ) -> None:
        from robots.robotwin.env_server import _evaluation_language, make_env

        self._env = make_env(
            task_name=task_name,
            task_config=task_config,
            seed=seed,
            assets_path=assets_path,
            max_episode_steps=max_episode_steps,
        )
        self._task_name = task_name
        self._task_config = task_config
        self._seed = int(seed)
        self._max_episode_steps = int(max_episode_steps)
        self._record_artifacts = bool(record_artifacts)
        release_unused_batch_planners(self._env)

        _, info = self._env.reset(
            env_idx=[0], env_seeds=[self._seed], render_obs=self._record_artifacts
        )
        release_unused_batch_planners(self._env)
        actual = info["episode_status"]["actual_seed"]
        if actual != self._seed:
            raise RuntimeError(
                f"RoboTwin exact seed mismatch: requested {self._seed}, got {actual}"
            )
        published = _evaluation_language(task_config, task_name, self._seed)
        if published is not None:
            self._env.set_task_language(published, env_id=0)
        self.last_info: dict[str, Any] = info
        self.terminated = bool(info["episode_status"]["eval_success"])
        self.truncated = False

    def get_task_language(self) -> str:
        return self._env.get_task_language(env_id=0)

    def render_cameras(
        self, camera_names: tuple[str, ...], *, depth: bool = False
    ) -> dict[str, tuple[Any, dict[str, Any]]]:
        return self._env.render_cameras(camera_names, depth=depth, env_id=0)

    def render_camera(self, camera_name: str, *, depth: bool = False) -> Any:
        return self._env.render_camera(camera_name, depth=depth, env_id=0)

    def get_camera_meta(self, camera_name: str) -> dict[str, Any]:
        return self._env.get_camera_meta(camera_name, env_id=0)

    def coordinate_calibration(self) -> dict[str, Any]:
        sub_env = self._env._sub_env(0)
        with sub_env.lock:
            return robotwin_calibration(sub_env.task)

    def measured_grippers(self) -> dict[str, dict[str, float]]:
        sub_env = self._env._sub_env(0)
        with sub_env.lock:
            robot = sub_env.task.robot
            result = {}
            for side in ("left", "right"):
                entity = getattr(robot, f"{side}_entity")
                joint, multiplier, offset = getattr(robot, f"{side}_gripper")[0]
                index = entity.get_active_joints().index(joint)
                position = float(entity.get_qpos()[index])
                low, high = getattr(robot, f"{side}_gripper_scale")
                opening = ((position - offset) / multiplier - low) / (high - low)
                result[side] = {
                    "opening": float(np.clip(opening, 0.0, 1.0)),
                    "joint_position": position,
                    "commanded_opening": float(getattr(robot, f"{side}_gripper_val")),
                }
            return result

    def plan_arm_path(self, arm: str, target_pose: list[float]) -> dict[str, Any]:
        sub_env = self._env._sub_env(0)
        with sub_env.lock:
            robot = sub_env.task.robot
            planner = getattr(robot, f"{arm}_planner")
            base_name = planner.robot_config["robot_cfg"]["kinematics"]["base_link"]
            link = next(
                lnk
                for lnk in getattr(robot, f"{arm}_entity").get_links()
                if lnk.get_name() == base_name
            )
            cmd_pose = planner_command_pose(planner, link.get_pose(), target_pose)
        return self._env.plan_arm_path(0, arm, cmd_pose)

    def step_dense_arm(
        self, arm: Literal["left", "right"], position: np.ndarray, velocity: np.ndarray
    ) -> int:
        info = self._env.step_dense_arm(arm, position, velocity, env_id=0)
        status = info["episode_status"]
        self.terminated = bool(status["eval_success"])
        limit = status["step_lim"]
        self.truncated = limit is not None and status["take_action_cnt"] >= limit
        self.last_info = info
        return int(info["executed_actions"])

    def chunk_step(
        self, actions: np.ndarray, *, action_type: Literal["qpos", "ee"] = "qpos"
    ) -> tuple[Any, Any, Any, Any, dict[str, Any]]:
        obs_list, rew, term, trunc, info_list = self._env.chunk_step(
            np.asarray(actions, dtype=np.float64),
            action_type=action_type,
            render_obs=self._record_artifacts,
        )
        info = info_list[0]
        status = info["episode_status"]
        self.terminated = bool(status["eval_success"])
        limit = status["step_lim"]
        self.truncated = limit is not None and status["take_action_cnt"] >= limit
        self.last_info = info
        return (
            obs_list[0],
            np.asarray(rew)[0],
            bool(np.any(term[0])),
            bool(np.any(trunc[0])),
            info,
        )

    def close(self) -> None:
        # Keep SAPIEN Vulkan cache warm across episodes in persistent workers
        # (clear_cache=True tears down shared Vulkan state and can segfault on reuse).
        offload = getattr(self._env, "offload", None)
        if callable(offload):
            offload(clear_cache=False)
        elif callable(getattr(self._env, "close", None)):
            self._env.close(clear_cache=False)


class RoboTwinRLPrimitives(RecordedPerception):
    """Dual-arm Aloha general-frame control and RGB-D perception."""

    cameras = ("head", "left_wrist", "right_wrist")

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
        calibration = self._call_calibration()
        self.frame = RobotwinFrame.from_calibration(calibration)
        self.coordinates = self.frame.coordinates
        self._save_artifact("coordinates.json", self.frame.metadata(), None)

    def _call_calibration(self) -> dict[str, Any]:
        if hasattr(self.env, "coordinate_calibration"):
            return self.env.coordinate_calibration()
        return self.env._client.call("env.coordinate_calibration")

    def _call_measured_grippers(self) -> dict[str, dict[str, float]]:
        if hasattr(self.env, "measured_grippers"):
            return self.env.measured_grippers()
        return self.env._client.call("env.measured_grippers")

    def native_success(self) -> bool:
        self._success |= bool(self.env.last_info["episode_status"]["eval_success"])
        return self._success

    def robot_state(self) -> dict[str, Any]:
        native = self.env.last_info["robot_state"]
        state = self._call_measured_grippers()
        for arm, gripper in state.items():
            gripper["pose"] = self.frame.model_pose(native[f"{arm}_tcp_pose"], arm)
        return state

    def annotate(
        self,
        image: np.ndarray,
        calibration: dict[str, Any],
        state: dict[str, Any] | None = None,
    ) -> np.ndarray:
        """Project measured TCPs into each view and label LEFT (cyan) / RIGHT (yellow)."""
        state = self.robot_state() if state is None else state
        view = Image.fromarray(image).convert("RGB")
        draw = ImageDraw.Draw(view)
        extrinsic = np.linalg.inv(np.asarray(calibration["extrinsic_cam2world"]))
        intrinsic = np.asarray(calibration["intrinsic_K"])
        for arm, color in (("left", "cyan"), ("right", "yellow")):
            world_xyz = self.coordinates.world_position(
                np.asarray(state[arm]["pose"]["xyz"])
            )
            camera = extrinsic @ np.r_[world_xyz, 1.0]
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

    @staticmethod
    def _build_capture(
        rgb: np.ndarray, depth: np.ndarray, meta: dict[str, Any]
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        extrinsic = np.asarray(meta["extrinsic_cv"], dtype=float)
        if extrinsic.shape == (3, 4):
            extrinsic = np.vstack((extrinsic, [0, 0, 0, 1]))
        elif extrinsic.shape != (4, 4):
            raise ValueError("camera extrinsic must be a 3x4 or 4x4 transform")
        calibration = {
            "intrinsic_K": np.asarray(meta["intrinsic_K"]).tolist(),
            "extrinsic_cam2world": np.linalg.inv(extrinsic).tolist(),
            "height": depth.shape[0],
            "width": depth.shape[1],
        }
        depth_arr = np.asarray(depth, dtype=float)
        world = geometry.world_from_depth(depth_arr, calibration)
        world[~np.isfinite(depth_arr) | (depth_arr <= 0)] = np.nan
        return np.asarray(rgb), world, calibration

    def capture_all(self) -> dict[str, tuple[np.ndarray, np.ndarray, dict[str, Any]]]:
        render_all = getattr(self.env, "render_cameras", None)
        if callable(render_all):
            batch = render_all(self.cameras, depth=True)
            return {
                camera: self._build_capture(rgb, depth, meta)
                for camera, ((rgb, depth), meta) in batch.items()
            }
        return super().capture_all()

    def capture(self, camera: str) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        rgb, depth = self.env.render_camera(camera, depth=True)
        meta = self.env.get_camera_meta(camera)
        return self._build_capture(rgb, depth, meta)

    @staticmethod
    def _matrix(pose: Any) -> np.ndarray:
        pose = np.asarray(pose, dtype=float)
        result = np.eye(4)
        result[:3, :3] = rotation_from_wxyz(pose[3:]).as_matrix()
        result[:3, 3] = pose[:3]
        return result

    def _execute_arm_path(
        self, arm: Literal["left", "right"], position: np.ndarray, velocity: np.ndarray
    ) -> ToolResult:
        count = min(len(position), self._max_episode_steps - self.env_steps)
        if self.ended or count <= 0:
            return ToolResult(data={"error": "episode_ended"})
        for start in range(0, count, 64):
            self._check_cancelled()
            end = min(start + 64, count)
            self.env_steps += self.env.step_dense_arm(
                arm, position[start:end], velocity[start:end]
            )
            if self.ended:
                break
        return ToolResult(data={"status": "reached", "env_steps": self.env_steps})

    def _pose_errors(self, goals: dict[str, np.ndarray]) -> tuple[dict[str, Any], bool]:
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
            v["position_error_m"] < _POSITION_TOL_M
            and v["rotation_error_rad"] < _ROTATION_TOL_RAD
            for v in errors.values()
        )
        return errors, reached

    def _move_arm(
        self, arm: Literal["left", "right"], target: np.ndarray
    ) -> ToolResult:
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
        position = np.asarray(plan["position"], dtype=np.float64)
        velocity = np.asarray(plan["velocity"], dtype=np.float64)
        goals = {arm: target}
        result = self._execute_arm_path(arm, position, velocity)
        if result.is_error:
            return result
        settling_steps = 0
        errors, reached = self._pose_errors(goals)
        while not reached and not self.ended and settling_steps < 30:
            count = min(10, self._max_episode_steps - self.env_steps)
            if count <= 0:
                break
            hold = np.repeat(position[-1:], count, axis=0)
            before = self.env_steps
            result = self._execute_arm_path(arm, hold, np.zeros_like(hold))
            if result.is_error:
                return result
            settling_steps += self.env_steps - before
            errors, reached = self._pose_errors(goals)
        result.data.update(
            status="reached" if reached else "partial",
            errors=errors,
            settling_steps=settling_steps,
        )
        if not reached:
            result.data["reason"] = "not_at_target"
        return result

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
            xyz, orientation_rpy, native[f"{arm}_tcp_pose"], arm
        )
        return self._move_arm(arm, self._matrix(target))

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
        remaining = self._max_episode_steps - self.env_steps
        if self.ended or remaining <= 0:
            return ToolResult(data={"error": "episode_ended"})
        initial = np.asarray(
            self.env.last_info["robot_state"]["qpos_target14"], dtype=float
        )
        target = initial.copy()
        target[6 if arm == "left" else 13] = opening
        waypoints = np.asarray(
            [initial + (target - initial) * frac for frac in np.linspace(0.1, 1.0, 10)]
        )[:remaining]
        result = self.env.chunk_step(waypoints, action_type="qpos")
        self.env_steps += int(result[4]["executed_actions"])
        measured = self._call_measured_grippers()
        reached = abs(measured[arm]["opening"] - opening) < _OPENING_TOL
        data: dict[str, Any] = {
            "status": "reached" if reached else "partial",
            "env_steps": self.env_steps,
            "grippers": measured,
        }
        if not reached:
            data["reason"] = "opening_differs"
        return ToolResult(data=data)
