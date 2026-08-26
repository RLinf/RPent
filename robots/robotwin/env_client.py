"""RPC client for one RLinf RoboTwin environment."""

from __future__ import annotations

from typing import Any

import numpy as np

from robots.robotwin.robot_spec import (
    ROBOTWIN_CAMERA_NAMES,
    ROBOTWIN_READ_TIMEOUT_S,
    ROBOTWIN_STATE_CHANGE_TIMEOUT_S,
    ROBOTWIN_STATUS_KEYS,
    RoboTwinActionType,
)
from rpent.robots.components.env_client_base import BaseEnvClient
from rpent.utils.rpc import RpcClient


class RoboTwinEnvClient(BaseEnvClient):
    """Client for one standard RLinf ``RoboTwinEnv`` instance."""

    _TIMEOUT_S = {**BaseEnvClient._TIMEOUT_S, "default": ROBOTWIN_READ_TIMEOUT_S}

    def __init__(self, client: RpcClient, *, expected_meta: dict[str, Any]):
        self.terminated = False
        self.truncated = False
        self._expected_seed = int(expected_meta["seed"])
        super().__init__(client, expected_meta=expected_meta)
        self.server_meta = dict(expected_meta)
        execution = self.server_meta.get("execution", {})
        self.execution_capabilities = (
            dict(execution) if isinstance(execution, dict) else {}
        )

    @staticmethod
    def _require_result_tuple(result: Any, size: int, method: str) -> tuple:
        if not isinstance(result, (list, tuple)) or len(result) != size:
            raise TypeError(
                f"{method} must return a {size}-item tuple, got {result!r}"
            )
        return tuple(result)

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
        result = self._client.call(
            "env.reset", timeout_s=self._TIMEOUT_S["env.reset"]
        )
        observation, info = self._require_result_tuple(result, 2, "env.reset")
        if not isinstance(observation, dict):
            raise TypeError(
                f"RoboTwin reset observation must be a mapping, got {observation!r}"
            )
        status = self._require_episode_status(info)
        if status["actual_seed"] != self._expected_seed:
            raise ValueError(f"reset did not use the requested seed: {info!r}")
        if not isinstance(info.get("instruction"), str):
            raise TypeError("reset instruction must be a string")
        self.last_obs = observation
        self.last_reset_info = dict(info)
        self.last_info = info
        self.terminated = False
        self.truncated = False
        return observation, info

    def step(
        self,
        action,
        *,
        action_type: RoboTwinActionType = "qpos",
    ) -> tuple[Any, Any, Any, Any, dict[str, Any]]:
        """Execute one RoboTwin action and update the cached episode state."""
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
        """Execute a RoboTwin action chunk and cache the final episode state."""
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
        n_executed = info.get("executed_actions")
        if (
            isinstance(n_executed, bool)
            or not isinstance(n_executed, int)
            or not 1 <= n_executed <= len(array)
        ):
            raise ValueError(f"invalid RoboTwin chunk result: {result!r}")
        obs_field = result[0]
        if return_all_frames:
            if (
                not isinstance(obs_field, dict)
                or "frames" not in obs_field
                or "final" not in obs_field
            ):
                raise TypeError(
                    "RoboTwin return_all_frames=True must return a "
                    "{'frames','final'} payload, got " + repr(obs_field)
                )
            self.last_obs = obs_field["final"]
        else:
            self.last_obs = obs_field
        self.last_info = info
        self.terminated |= bool(np.asarray(terminated).any())
        self.truncated |= bool(np.asarray(truncated).any())
        return result

    def render_camera(self, camera_name: str, *, depth: bool = False) -> Any:
        """Render one RoboTwin camera, optionally including depth."""
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
        """Return calibration metadata for one RoboTwin camera."""
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
        """Return the current RoboTwin task instruction."""
        result = self._read("get_task_language")
        if not isinstance(result, str):
            raise TypeError(f"RoboTwin task language must be a string: {result!r}")
        return result

    def plan_arm_path(self, arm: str, target_pose) -> dict[str, Any]:
        """Plan a native arm path to the target end-effector pose."""
        return self._read(
            "plan_arm_path",
            kwargs={"arm": arm, "target_pose": target_pose},
        )
