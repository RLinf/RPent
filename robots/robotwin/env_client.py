"""RPC client for one RLinf RoboTwin environment."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np

from rpent.tools.env_client_base import BaseEnvClient
from rpent.utils.rpc import RpcClient

READ_TIMEOUT_S = 120.0
STATE_CHANGE_TIMEOUT_S = 600.0
ActionType = Literal["qpos", "ee"]
_STATUS_KEYS = ("eval_success", "take_action_cnt", "step_lim", "actual_seed")
_CAMERA_NAMES = ("head", "left_wrist", "right_wrist")


class RoboTwinEnvClient(BaseEnvClient):
    """Client for one standard RLinf ``RoboTwinEnv`` instance."""

    _TIMEOUT_S = {**BaseEnvClient._TIMEOUT_S, "default": READ_TIMEOUT_S}

    def __init__(self, client: RpcClient, *, expected_meta: dict[str, Any]):
        self.terminated = False
        self.truncated = False
        self._expected_seed = int(expected_meta["seed"])
        super().__init__(client, expected_meta=expected_meta)

    def _read(
        self,
        method: str,
        *,
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        return self._client.call(
            f"env.{method}",
            kwargs=kwargs,
            timeout_s=READ_TIMEOUT_S,
        )

    def _require_common_active(self) -> None:
        if self.terminated or self.truncated:
            raise RuntimeError("RoboTwin common episode is terminal; reset is required")

    @staticmethod
    def _validate_action_request(action_type: Any, actions: Any) -> np.ndarray:
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

    @staticmethod
    def _validate_common_status(
        *,
        terminated: Any,
        truncated: Any,
        info: Any,
    ) -> dict[str, Any]:
        if not isinstance(info, dict):
            raise TypeError(f"common execution info must be a mapping, got {info!r}")
        status = info.get("episode_status")
        RoboTwinEnvClient._validate_status(status)
        term = bool(np.asarray(terminated).any())
        trunc = bool(np.asarray(truncated).any())
        step_lim = status["step_lim"]
        budget_exhausted = step_lim is not None and (
            status["take_action_cnt"] >= step_lim
        )
        if status["eval_success"] and not term:
            raise ValueError("native success was not reflected by terminated")
        if budget_exhausted and not trunc:
            raise ValueError("native action budget was not reflected by truncated")
        return status

    def _record_terminal_from_status(self, status: Any) -> None:
        self._validate_status(status)
        if status["eval_success"]:
            self.terminated = True
        step_lim = status["step_lim"]
        if step_lim is not None and status["take_action_cnt"] >= step_lim:
            self.truncated = True

    def reset(self) -> tuple[dict[str, Any], dict[str, Any]]:
        """Reset to the TaskRun seed and validate the native result."""
        observation, info = super().reset()
        if not isinstance(observation, dict):
            raise TypeError(
                f"RoboTwin reset observation must be a mapping, got {observation!r}"
            )
        self._validate_reset_result(info, self._expected_seed)
        self.terminated = False
        self.truncated = False
        return observation, info

    def step(
        self,
        action,
        *,
        action_type: ActionType = "qpos",
    ) -> tuple[Any, Any, Any, Any, dict[str, Any]]:
        flat = self._validate_action_request(action_type, action)
        if flat.shape[0] != 1:
            raise ValueError("RoboTwin common action must be a single action")
        flat = flat[0]
        self._require_common_active()
        result = self._client.call(
            "env.step",
            args=(flat,),
            kwargs={"action_type": action_type},
            timeout_s=STATE_CHANGE_TIMEOUT_S,
        )
        result = self._require_result_tuple(result, 5, "env.step")
        self.last_obs = result[0]
        _, _, terminated, truncated, info = result
        self._validate_common_status(
            terminated=terminated,
            truncated=truncated,
            info=info,
        )
        self.terminated |= bool(np.asarray(terminated).any())
        self.truncated |= bool(np.asarray(truncated).any())
        self._record_terminal_from_status(info["episode_status"])
        self.last_info = info
        return result

    def chunk_step(
        self,
        actions,
        *,
        action_type: ActionType = "qpos",
        return_all_frames: bool = False,
    ) -> tuple[Any, Any, Any, Any, dict[str, Any]]:
        array = self._validate_action_request(action_type, actions)
        if (
            return_all_frames
            and self.execution_capabilities.get("chunk_step_all_frames", True)
            is not True
        ):
            raise ValueError("env.chunk_step does not support return_all_frames=True")
        self._require_common_active()
        result = self._client.call(
            "env.chunk_step",
            args=(array,),
            kwargs={
                "action_type": action_type,
                "return_all_frames": return_all_frames,
            },
            timeout_s=STATE_CHANGE_TIMEOUT_S,
        )
        result = self._require_result_tuple(result, 5, "env.chunk_step")
        obs_field = result[0]
        if isinstance(obs_field, list):
            self.last_obs = obs_field[-1]
        else:
            self.last_obs = obs_field
        _, rewards, terminated, truncated, info = result
        self._validate_common_status(
            terminated=terminated,
            truncated=truncated,
            info=info,
        )
        executed = info.get("executed_actions")
        requested = info.get("requested_actions")
        per_step = info.get("per_step")
        lengths = [
            len(np.asarray(rewards).reshape(-1)),
            len(np.asarray(terminated).reshape(-1)),
            len(np.asarray(truncated).reshape(-1)),
        ]
        if (
            isinstance(executed, bool)
            or not isinstance(executed, int)
            or isinstance(requested, bool)
            or not isinstance(requested, int)
            or requested != len(array)
            or not 1 <= executed <= requested
            or lengths != [executed, executed, executed]
            or not isinstance(per_step, list)
            or len(per_step) != executed
        ):
            raise ValueError(f"invalid RoboTwin chunk result: {result!r}")
        self.terminated |= bool(np.asarray(terminated).any())
        self.truncated |= bool(np.asarray(truncated).any())
        self._record_terminal_from_status(info["episode_status"])
        self.last_info = info
        return result

    @staticmethod
    def _validate_status(status: Any) -> None:
        if not isinstance(status, dict):
            raise TypeError(f"episode_status must be a mapping, got {status!r}")
        missing = [key for key in _STATUS_KEYS if key not in status]
        if missing:
            raise ValueError(f"episode_status is missing {missing}: {status!r}")
        if not isinstance(status["eval_success"], bool):
            raise TypeError("episode_status.eval_success must be bool")
        for key in ("take_action_cnt", "actual_seed"):
            if isinstance(status[key], bool) or not isinstance(status[key], int):
                raise TypeError(f"episode_status.{key} must be int")
        if status["take_action_cnt"] < 0:
            raise ValueError("episode_status.take_action_cnt must be non-negative")
        step_lim = status["step_lim"]
        if step_lim is not None and (
            isinstance(step_lim, bool) or not isinstance(step_lim, int) or step_lim < 0
        ):
            raise ValueError(
                "episode_status.step_lim must be a non-negative int or null"
            )

    @classmethod
    def _validate_reset_result(cls, info: Any, expected_seed: int) -> None:
        if not isinstance(info, dict):
            raise TypeError(f"reset info must be a mapping, got {info!r}")
        cls._validate_status(info.get("episode_status"))
        if info.get("requested_seed") != expected_seed:
            raise ValueError(
                f"reset returned a different requested_seed: {info!r}"
            )
        if info["episode_status"]["actual_seed"] != expected_seed:
            raise ValueError(
                f"reset did not use the requested seed: {info!r}"
            )
        if not isinstance(info.get("instruction"), str):
            raise TypeError("reset instruction must be a string")

    def render_camera(self, camera_name: str, *, depth: bool = False) -> Any:
        if camera_name not in _CAMERA_NAMES:
            raise ValueError(
                f"unknown RoboTwin camera {camera_name!r}; "
                f"available={list(_CAMERA_NAMES)}"
            )
        if not isinstance(depth, bool):
            raise TypeError("RoboTwin camera depth flag must be bool")
        return self._read(
            "render_camera",
            kwargs={"camera_name": camera_name, "depth": depth},
        )

    def get_camera_meta(self, camera_name: str) -> dict[str, Any]:
        if camera_name not in _CAMERA_NAMES:
            raise ValueError(
                f"unknown RoboTwin camera {camera_name!r}; "
                f"available={list(_CAMERA_NAMES)}"
            )
        result = self._read(
            "get_camera_meta",
            kwargs={"camera_name": camera_name},
        )
        if not isinstance(result, dict):
            raise TypeError(f"RoboTwin camera metadata must be a mapping: {result!r}")
        return result

    def get_task_language(self) -> str:
        result = self._read("get_task_language")
        if not isinstance(result, str):
            raise TypeError(f"RoboTwin task language must be a string: {result!r}")
        return result

    def plan_arm_path(self, arm: str, target_pose) -> dict[str, Any]:
        return self._read(
            "plan_arm_path",
            kwargs={"arm": arm, "target_pose": target_pose},
        )
