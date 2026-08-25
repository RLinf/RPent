"""RPent/agent runtime extension over the RLinf training ``RoboTwinEnv``."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from rlinf.envs.robotwin.robotwin_env import RoboTwinEnv

from robots.robotwin.robot_spec import RoboTwinActionType

__all__ = ["RoboTwinAgentEnv"]


def _camera_depth(camera: Any) -> np.ndarray:
    position = camera.get_picture("Position")
    invalid = position[..., 3] >= 1
    depth = (-position[..., 2]).astype(np.float32)
    depth[invalid] = np.nan
    return depth


def _camera_meta(camera: Any) -> dict[str, Any]:
    return {
        "intrinsic_K": np.asarray(camera.get_intrinsic_matrix()),
        "extrinsic_cv": np.asarray(camera.get_extrinsic_matrix()),
        "cam2world_gl": np.asarray(camera.get_model_matrix()),
        "width": int(camera.get_width()),
        "height": int(camera.get_height()),
    }


def _validate_actions(
    actions: Any, *, action_type: RoboTwinActionType
) -> np.ndarray:
    if action_type not in ("qpos", "ee"):
        raise ValueError("action_type must be 'qpos' or 'ee'")
    array = np.asarray(actions, dtype=np.float64)
    if array.ndim == 1:
        array = array[None, :]
    expected_dim = 14 if action_type == "qpos" else 16
    if array.ndim != 2 or array.shape[0] < 1 or array.shape[1] != expected_dim:
        raise ValueError(
            f"{action_type} actions must have shape [N,{expected_dim}], N >= 1"
        )
    if not np.isfinite(array).all():
        raise ValueError(f"{action_type} actions must contain only finite values")
    return array


def _execution_should_stop(status: dict[str, Any]) -> bool:
    step_limit = status["step_lim"]
    return status["eval_success"] or (
        step_limit is not None and status["take_action_cnt"] >= step_limit
    )


class RoboTwinAgentEnv(RoboTwinEnv):
    """RPent/agent runtime extension; preserves training RoboTwinEnv unchanged."""

    def _sub_env(self, env_id: int) -> Any:
        if not 0 <= env_id < len(self.venv.envs):
            raise IndexError(f"invalid RoboTwin env_id: {env_id}")
        return self.venv.envs[env_id]

    def _episode_status(self, sub_env: Any) -> dict[str, Any]:
        """Read native status; the caller must hold ``sub_env.lock``."""
        task = sub_env.task
        step_limit = getattr(task, "step_lim", None)
        if step_limit is None:
            task_args = getattr(sub_env, "args", None)
            if isinstance(task_args, dict):
                step_limit = task_args.get("step_lim")
            elif task_args is not None:
                step_limit = getattr(task_args, "step_lim", None)
        if step_limit is None:
            cfg_get = getattr(self.cfg, "get", None)
            step_limit = (
                cfg_get("max_episode_steps", None)
                if callable(cfg_get)
                else getattr(self.cfg, "max_episode_steps", None)
            )
        actual_seed = getattr(task, "ep_num", None)
        if actual_seed is None:
            actual_seed = getattr(sub_env, "env_seed", None)
        return {
            "eval_success": bool(getattr(task, "eval_success", False)),
            "take_action_cnt": int(getattr(task, "take_action_cnt", 0)),
            "step_lim": int(step_limit) if step_limit is not None else None,
            "actual_seed": int(actual_seed),
        }

    def chunk_step(
        self,
        actions: Any,
        *,
        action_type: RoboTwinActionType = "qpos",
        env_id: int = 0,
        return_all_frames: bool = False,
    ) -> tuple[list[Any], np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
        """Execute a chunk of native actions, returning a gym-style 5-tuple batched as ``[1, executed]``."""
        array = _validate_actions(actions, action_type=action_type)
        sub_env = self._sub_env(env_id)
        rewards: list[float] = []
        terminations: list[bool] = []
        truncations: list[bool] = []
        per_step: list[dict[str, Any]] = []
        frames: list[np.ndarray] = []
        executed = 0
        with sub_env.lock:
            for action in array:
                status = self._episode_status(sub_env)
                if _execution_should_stop(status):
                    break
                before = status["take_action_cnt"]
                sub_env.task.take_action(action, action_type=action_type)
                after = int(getattr(sub_env.task, "take_action_cnt", 0))
                if after <= before:
                    break
                executed += 1
                step_status = self._episode_status(sub_env)
                step_lim = step_status["step_lim"]
                budget = (
                    step_lim is not None and step_status["take_action_cnt"] >= step_lim
                )
                rewards.append(float(step_status["eval_success"]))
                terminations.append(bool(step_status["eval_success"]))
                truncations.append(bool(budget))
                per_step.append({"episode_status": step_status})
                if return_all_frames:
                    head_rgb = sub_env.task.get_obs()["observation"][
                        "head_camera"
                    ]["rgb"]
                    frames.append(
                        np.asarray(self.center_and_crop(head_rgb, center_crop=self.center_crop))
                    )
            episode_status = self._episode_status(sub_env)
            robot_state = self._robot_state(sub_env)
            if self.record_metrics:
                self._elapsed_steps[env_id] = int(episode_status["take_action_cnt"])
        # venv.get_obs() re-acquires sub_env.lock; read it outside the lock.
        observation = self._extract_obs_image(self.venv.get_obs())
        if return_all_frames:
            observation = {"frames": frames, "final": observation}
        info = {
            "action_type": action_type,
            "requested_actions": int(len(array)),
            "executed_actions": executed,
            "robot_state": robot_state,
            "episode_status": episode_status,
            "per_step": per_step,
        }
        return (
            [observation],
            np.asarray(rewards, dtype=np.float32).reshape(1, -1),
            np.asarray(terminations, dtype=bool).reshape(1, -1),
            np.asarray(truncations, dtype=bool).reshape(1, -1),
            [info],
        )

    def step(
        self,
        action: Any,
        *,
        action_type: RoboTwinActionType = "qpos",
        auto_reset: bool = False,
        env_id: int = 0,
    ) -> tuple[Any, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
        """Execute a single native action; the N=1 case of ``chunk_step``."""
        if auto_reset:
            raise NotImplementedError("RoboTwinAgentEnv does not support auto-reset")
        array = _validate_actions(action, action_type=action_type)
        if array.shape[0] != 1:
            raise ValueError("RoboTwinAgentEnv.step expects a single action")
        observation_list, rewards, terminated, truncated, info_list = self.chunk_step(
            array, action_type=action_type, env_id=env_id
        )
        return (
            observation_list[0],
            np.asarray(rewards).reshape(-1),
            np.asarray(terminated).reshape(-1),
            np.asarray(truncated).reshape(-1),
            info_list[0],
        )

    def reset(self, env_idx=None, env_seeds=None):
        """Reset and seed ``info`` with the initial robot state + episode status."""
        observation, info = super().reset(env_idx=env_idx, env_seeds=env_seeds)
        sub_env = self._sub_env(0)
        with sub_env.lock:
            info["robot_state"] = self._robot_state(sub_env)
            info["episode_status"] = self._episode_status(sub_env)
        return observation, info

    def _get_camera(self, sub_env: Any, camera_name: str) -> Any:
        """Resolve a native camera; the caller must hold ``sub_env.lock``."""
        cameras = sub_env.task.cameras
        result: dict[str, Any] = {}
        for camera, name in zip(cameras.static_camera_list, cameras.static_camera_name):
            if name == "head_camera":
                result["head"] = camera
                break
        if bool(getattr(cameras, "collect_wrist_camera", False)):
            result["left_wrist"] = cameras.left_camera
            result["right_wrist"] = cameras.right_camera
        try:
            return result[camera_name]
        except KeyError as error:
            available = sorted(result)
            raise ValueError(
                f"unknown RoboTwin camera {camera_name!r}; available={available}"
            ) from error

    def render_camera(
        self,
        camera_name: str,
        *,
        depth: bool = False,
        env_id: int = 0,
    ) -> Any:
        sub_env = self._sub_env(env_id)
        with sub_env.lock:
            native_observation = sub_env.task.get_obs()["observation"]
            camera_keys = {
                "head": "head_camera",
                "left_wrist": "left_camera",
                "right_wrist": "right_camera",
            }
            camera = self._get_camera(sub_env, camera_name)
            rgb = native_observation[camera_keys[camera_name]]["rgb"]
            if not depth:
                return rgb
            return rgb, _camera_depth(camera)

    def get_camera_meta(self, camera_name: str, env_id: int = 0) -> dict[str, Any]:
        sub_env = self._sub_env(env_id)
        with sub_env.lock:
            return _camera_meta(self._get_camera(sub_env, camera_name))

    def get_task_language(self, env_id: int = 0) -> str:
        sub_env = self._sub_env(env_id)
        with sub_env.lock:
            return str(sub_env.task.get_instruction())

    def _robot_state(self, sub_env: Any) -> dict[str, Any]:
        """Read the native robot state; the caller must hold ``sub_env.lock``."""
        robot = sub_env.task.robot
        left_target = robot.get_left_arm_jointState()
        right_target = robot.get_right_arm_jointState()
        left_real = robot.get_left_arm_real_jointState()
        right_real = robot.get_right_arm_real_jointState()
        return {
            "left_eef_pose": np.asarray(robot.get_left_ee_pose(), dtype=np.float64),
            "right_eef_pose": np.asarray(robot.get_right_ee_pose(), dtype=np.float64),
            "left_tcp_pose": np.asarray(robot.get_left_tcp_pose(), dtype=np.float64),
            "right_tcp_pose": np.asarray(robot.get_right_tcp_pose(), dtype=np.float64),
            "left_gripper": float(robot.get_left_gripper_val()),
            "right_gripper": float(robot.get_right_gripper_val()),
            "qpos_target14": np.asarray(left_target + right_target, dtype=np.float64),
            "arm_qpos_real12": np.asarray(
                left_real[:-1] + right_real[:-1], dtype=np.float64
            ),
        }

    def plan_arm_path(
        self,
        env_id: int,
        arm: Literal["left", "right"],
        target_pose: Any,
    ) -> dict[str, Any]:
        if arm not in ("left", "right"):
            raise ValueError("arm must be 'left' or 'right'")
        target = np.asarray(target_pose, dtype=np.float64)
        if target.shape != (7,):
            raise ValueError("target_pose must have shape (7,)")
        if not np.isfinite(target).all():
            raise ValueError("target_pose must contain only finite values")
        sub_env = self._sub_env(env_id)
        with sub_env.lock:
            planner = (
                sub_env.task.robot.left_plan_path
                if arm == "left"
                else sub_env.task.robot.right_plan_path
            )
            result = planner(target.tolist())
            return {
                "status": result.get("status", "Unknown"),
                "position": (
                    np.asarray(result["position"], dtype=np.float64)
                    if result.get("position") is not None
                    else None
                ),
                "velocity": (
                    np.asarray(result["velocity"], dtype=np.float64)
                    if result.get("velocity") is not None
                    else None
                ),
            }
