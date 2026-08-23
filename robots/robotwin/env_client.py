"""RPC client for one RLinf RoboTwin environment."""

from __future__ import annotations

from typing import Any

import numpy as np

from robots.robotwin.env_spec import (
    ROBOTWIN_CAMERA_NAMES,
    ROBOTWIN_READ_TIMEOUT_S,
    ROBOTWIN_STATE_CHANGE_TIMEOUT_S,
    ROBOTWIN_STATUS_KEYS,
    RoboTwinActionType,
)
from rpent.tools.env_client_base import BaseEnvClient
from rpent.utils.rpc import RpcClient


class RoboTwinEnvClient(BaseEnvClient):
    """Client for one standard RLinf ``RoboTwinEnv`` instance."""

    _TIMEOUT_S = {**BaseEnvClient._TIMEOUT_S, "default": ROBOTWIN_READ_TIMEOUT_S}

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
            timeout_s=ROBOTWIN_READ_TIMEOUT_S,
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
    def _require_episode_status(info: Any) -> dict[str, Any]:
        """Validate the execution ``info`` mapping and return its ``episode_status``."""
        if not isinstance(info, dict):
            raise TypeError(f"common execution info must be a mapping, got {info!r}")
        status = info.get("episode_status")
        if not isinstance(status, dict):
            raise TypeError(f"episode_status must be a mapping, got {status!r}")
        missing = [key for key in ROBOTWIN_STATUS_KEYS if key not in status]
        if missing:
            raise ValueError(f"episode_status is missing {missing}: {status!r}")
        return status

    def reset(self) -> tuple[dict[str, Any], dict[str, Any]]:
        """Reset to the TaskRun seed and validate the native result."""
        observation, info = super().reset()
        if not isinstance(observation, dict):
            raise TypeError(
                f"RoboTwin reset observation must be a mapping, got {observation!r}"
            )
        status = self._require_episode_status(info)
        if status["actual_seed"] != self._expected_seed:
            raise ValueError(f"reset did not use the requested seed: {info!r}")
        if not isinstance(info.get("instruction"), str):
            raise TypeError("reset instruction must be a string")
        self.terminated = False
        self.truncated = False
        return observation, info

    def step(
        self,
        action,
        *,
        action_type: RoboTwinActionType = "qpos",
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
            timeout_s=ROBOTWIN_STATE_CHANGE_TIMEOUT_S,
        )
        result = self._require_result_tuple(result, 5, "env.step")
        _, _, terminated, truncated, info = result
        self._require_episode_status(info)
        self.last_obs = result[0]
        self.last_info = info
        self.terminated |= bool(np.asarray(terminated).any())
        self.truncated |= bool(np.asarray(truncated).any())
        return result

    def chunk_step(
        self,
        actions,
        *,
        action_type: RoboTwinActionType = "qpos",
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
            timeout_s=ROBOTWIN_STATE_CHANGE_TIMEOUT_S,
        )
        result = self._require_result_tuple(result, 5, "env.chunk_step")
        _, _, terminated, truncated, info = result
        self._require_episode_status(info)
        executed = info.get("executed_actions")
        if (
            isinstance(executed, bool)
            or not isinstance(executed, int)
            or not 1 <= executed <= len(array)
        ):
            raise ValueError(f"invalid RoboTwin chunk result: {result!r}")
        obs_field = result[0]
        if isinstance(obs_field, list):
            self.last_obs = obs_field[-1]
        else:
            self.last_obs = obs_field
        self.last_info = info
        self.terminated |= bool(np.asarray(terminated).any())
        self.truncated |= bool(np.asarray(truncated).any())
        return result

    def render_camera(self, camera_name: str, *, depth: bool = False) -> Any:
        if camera_name not in ROBOTWIN_CAMERA_NAMES:
            raise ValueError(
                f"unknown RoboTwin camera {camera_name!r}; "
                f"available={list(ROBOTWIN_CAMERA_NAMES)}"
            )
        if not isinstance(depth, bool):
            raise TypeError("RoboTwin camera depth flag must be bool")
        return self._read(
            "render_camera",
            kwargs={"camera_name": camera_name, "depth": depth},
        )

    def get_camera_meta(self, camera_name: str) -> dict[str, Any]:
        if camera_name not in ROBOTWIN_CAMERA_NAMES:
            raise ValueError(
                f"unknown RoboTwin camera {camera_name!r}; "
                f"available={list(ROBOTWIN_CAMERA_NAMES)}"
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
