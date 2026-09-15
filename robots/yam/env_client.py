# Copyright 2026 The RPent Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""RPC client for the YAM real-robot env server."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np

from robots.yam.contracts import YAM_CAMERA_NAMES, YAM_STATUS_KEYS, validate_actions
from robots.yam.geometry import joint_limits_from_config
from robots.yam.servo import JointServoConfig
from rpent.robots.components.env_client_base import BaseEnvClient
from rpent.utils.rpc import RpcClient


class YamEnvClient(BaseEnvClient):
    """Client for one operator-supervised YAM env endpoint.

    Unlike the common base constructor, this client does not reset on connect.
    Reset changes the operator-approved episode, so construction performs
    metadata validation through the common base and then takes an explicit
    ``env.observe`` snapshot. Observe may initialize hardware.
    """

    def __init__(
        self,
        client: RpcClient,
        *,
        expected_meta: dict[str, Any],
    ) -> None:
        server_meta = client.call(
            "env.get_env_meta", timeout_s=self._TIMEOUT_S["default"]
        )
        if not isinstance(server_meta, dict):
            raise ValueError("YAM server metadata must be a mapping")
        fixed = deepcopy(server_meta)
        execution = fixed.get("execution")
        if not isinstance(execution, dict):
            raise ValueError("YAM server execution capabilities must be a mapping")
        if "joint_servo" in execution:
            JointServoConfig.from_config(execution.pop("joint_servo"))
        if "compact_control" in execution:
            if type(execution.pop("compact_control")) is not bool:
                raise ValueError("YAM compact_control capability must be boolean")
        bound_keys = {"joint_limit_min", "joint_limit_max"}
        provided_bounds = bound_keys.intersection(execution)
        if provided_bounds:
            if provided_bounds != bound_keys:
                raise ValueError("YAM server must advertise both joint limit bounds")
            joint_limits_from_config({key: execution.pop(key) for key in bound_keys})
        expected_fixed = deepcopy(expected_meta)
        for key in ("joint_servo", "compact_control", *bound_keys):
            expected_fixed.get("execution", {}).pop(key, None)
        if fixed != expected_fixed:
            raise ValueError(
                f"env_meta mismatch: expected={expected_fixed!r} actual={fixed!r}"
            )
        # The common handshake verifies this complete snapshot is still current.
        # Observe happens only after both fixed and optional contracts pass.
        super().__init__(
            client,
            expected_meta=server_meta,
            reset_on_connect=False,
        )
        self.server_meta = deepcopy(server_meta)
        execution = self.server_meta.get("execution", {})
        self.execution_capabilities = (
            dict(execution) if isinstance(execution, dict) else {}
        )
        self.observe()

    @staticmethod
    def _require_result_tuple(result: Any, size: int, method: str) -> tuple:
        if not isinstance(result, (list, tuple)) or len(result) != size:
            raise TypeError(f"{method} must return a {size}-item tuple, got {result!r}")
        return tuple(result)

    @staticmethod
    def _require_episode_status(info: Any) -> dict[str, Any]:
        if not isinstance(info, dict):
            raise TypeError(f"YAM info must be a mapping, got {info!r}")
        status = info.get("episode_status")
        if not isinstance(status, dict):
            raise TypeError(f"YAM episode_status must be a mapping, got {status!r}")
        missing = [key for key in YAM_STATUS_KEYS if key not in status]
        if missing:
            raise ValueError(f"YAM episode_status is missing {missing}: {status!r}")
        return status

    @staticmethod
    def _require_observation(observation: Any) -> dict[str, Any]:
        if not isinstance(observation, dict):
            raise TypeError(f"YAM observation must be a mapping, got {observation!r}")
        frames = observation.get("frames")
        if not isinstance(frames, dict):
            raise TypeError("YAM observation.frames must be a mapping")
        for name in YAM_CAMERA_NAMES:
            if name not in frames:
                raise ValueError(f"YAM observation is missing {name!r} camera")
        state = observation.get("state")
        if not isinstance(state, dict):
            raise TypeError("YAM observation.state must be a mapping")
        qpos = np.asarray(state.get("joint_position"), dtype=np.float64)
        if qpos.shape != (14,) or not np.isfinite(qpos).all():
            raise ValueError(
                "YAM observation.state.joint_position must be finite qpos14"
            )
        return observation

    def observe(self) -> tuple[dict[str, Any], dict[str, Any]]:
        result = self._client.call(
            "env.observe", timeout_s=self._TIMEOUT_S["env.render_camera"]
        )
        observation, info = self._require_result_tuple(result, 2, "env.observe")
        self.last_obs = self._require_observation(observation)
        self.last_info = info
        self._require_episode_status(info)
        return self.last_obs, info

    def _cache_control_state(self, observation: Any, info: Any) -> None:
        if not isinstance(observation, dict) or not isinstance(
            observation.get("state"), dict
        ):
            raise ValueError("YAM compact feedback requires a state mapping")
        qpos = np.asarray(observation["state"].get("joint_position"), dtype=np.float64)
        if qpos.shape != (14,) or not np.isfinite(qpos).all():
            raise ValueError("YAM compact feedback requires finite qpos14")
        self._require_episode_status(info)
        if not isinstance(info.get("robot_state"), dict):
            raise ValueError("YAM compact feedback requires robot_state")
        commanded = np.asarray(info.get("commanded_qpos"), dtype=np.float64)
        if commanded.shape != (14,) or not np.isfinite(commanded).all():
            raise ValueError("YAM compact feedback requires finite commanded_qpos")
        self.last_control_obs = observation
        self.last_control_info = info

    def read_control_state(self) -> tuple[dict[str, Any], dict[str, Any]]:
        if self.execution_capabilities.get("compact_control") is not True:
            raise RuntimeError("YAM server does not advertise compact_control")
        result = self._client.call(
            "env.read_control_state", timeout_s=self._TIMEOUT_S["default"]
        )
        observation, info = self._require_result_tuple(
            result, 2, "env.read_control_state"
        )
        self._cache_control_state(observation, info)
        return observation, info

    def control_step(self, action: Any, *, expected_episode_id: str) -> tuple:
        if self.execution_capabilities.get("compact_control") is not True:
            raise RuntimeError("YAM server does not advertise compact_control")
        flat = validate_actions(action)
        if flat.shape[0] != 1:
            raise ValueError("YAM control_step requires one qpos14 action")
        result = self._client.call(
            "env.control_step",
            args=(flat[0],),
            kwargs={"expected_episode_id": expected_episode_id},
            timeout_s=self._TIMEOUT_S["env.step"],
        )
        result = self._require_result_tuple(result, 5, "env.control_step")
        self._cache_control_state(result[0], result[4])
        return result

    def reset(self) -> tuple[dict[str, Any], dict[str, Any]]:
        result = self._client.call(
            "env.reset",
            timeout_s=self._TIMEOUT_S["env.reset"],
        )
        observation, info = self._require_result_tuple(result, 2, "env.reset")
        self.last_obs = self._require_observation(observation)
        self.last_info = info
        self._require_episode_status(info)
        return self.last_obs, info

    def step(
        self,
        action,
        *,
        action_type: str = "qpos",
        expected_episode_id: str | None = None,
    ) -> tuple[Any, Any, Any, Any, dict[str, Any]]:
        flat = validate_actions(action, action_type=action_type)
        if flat.shape[0] != 1:
            raise ValueError("YAM step requires one qpos14 action")
        result = self._client.call(
            "env.step",
            args=(flat[0],),
            kwargs={
                "action_type": action_type,
                "expected_episode_id": expected_episode_id
                or self.last_info["episode_status"]["episode_id"],
            },
            timeout_s=self._TIMEOUT_S["env.step"],
        )
        result = self._require_result_tuple(result, 5, "env.step")
        obs, _, _, _, info = result
        self.last_obs = self._require_observation(obs)
        self.last_info = info
        self._require_episode_status(info)
        return result

    def chunk_step(
        self,
        actions,
        *,
        action_type: str = "qpos",
        return_all_frames: bool = False,
        expected_episode_id: str | None = None,
    ) -> tuple[Any, Any, Any, Any, dict[str, Any]]:
        flat = validate_actions(actions, action_type=action_type)
        result = self._client.call(
            "env.chunk_step",
            args=(flat,),
            kwargs={
                "action_type": action_type,
                "return_all_frames": return_all_frames,
                "expected_episode_id": expected_episode_id
                or self.last_info["episode_status"]["episode_id"],
            },
            timeout_s=self._TIMEOUT_S["env.chunk_step"],
        )
        result = self._require_result_tuple(result, 5, "env.chunk_step")
        obs_field, _, _, _, info = result
        if isinstance(obs_field, list):
            if not obs_field:
                raise TypeError("YAM chunk_step returned no observations")
            self.last_obs = self._require_observation(obs_field[-1])
        elif isinstance(obs_field, dict):
            self.last_obs = self._require_observation(obs_field)
        else:
            raise TypeError(
                "YAM chunk_step must return an observation dict or a list of observations"
            )
        self.last_info = info
        self._require_episode_status(info)
        return result

    def render_camera(self, camera_name: str, *, depth: bool = False) -> Any:
        if camera_name not in YAM_CAMERA_NAMES:
            raise ValueError(
                f"unknown YAM camera {camera_name!r}; available={list(YAM_CAMERA_NAMES)}"
            )
        return super().render_camera(camera_name, depth=depth)

    def get_camera_meta(self, camera_name: str) -> dict[str, Any]:
        if camera_name not in YAM_CAMERA_NAMES:
            raise ValueError(
                f"unknown YAM camera {camera_name!r}; available={list(YAM_CAMERA_NAMES)}"
            )
        return super().get_camera_meta(camera_name)

    def get_task_language(self) -> str:
        result = super().get_task_language()
        if not isinstance(result, str):
            raise TypeError(f"YAM task language must be a string: {result!r}")
        return result

    def plan_arm_path(self, arm: str, target_pose) -> dict[str, Any]:
        return self._client.call(
            "env.plan_arm_path",
            kwargs={"arm": arm, "target_pose": target_pose},
            timeout_s=120.0,
        )

    def request_stop(self) -> dict[str, Any]:
        receipt = self._client.call(
            "env.request_stop",
            timeout_s=5.0,
        )
        if isinstance(receipt, dict) and receipt.get("stop_requested") is True:
            for name in ("last_info", "last_control_info"):
                info = getattr(self, name, None)
                if isinstance(info, dict) and isinstance(
                    info.get("episode_status"), dict
                ):
                    # This acknowledges the flag, not a completed hold or new joints.
                    info["episode_status"]["stop_requested"] = True
        return receipt
