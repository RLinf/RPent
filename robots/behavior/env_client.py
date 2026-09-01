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

"""RPC client for one BEHAVIOR environment."""

from __future__ import annotations

import base64
import copy
from typing import Any

import numpy as np

from robots.behavior.schemas import (
    ACTION_DIM,
    validate_action_chunk,
    validate_move_both_targets,
    validate_move_both_visual_hand_checks,
    validate_observe_request,
    validate_prepared_plan_id,
    validate_relative_navigation_motion,
)
from robots.behavior.terminal_success import validate_official_success_receipt
from rpent.robots.components.env_client_base import BaseEnvClient
from rpent.utils.rpc import RpcClient

_POST_SUCCESS_ALLOWED = frozenset(
    {
        "env.get_env_meta",
        "env.get_prepared_motion_status",
        "env.current_observation",
        "env.finalize_paused_runtime",
    }
)
_IMAGE_BYTE_FIELDS = frozenset(
    {
        "_depth_image_bytes",
        "_image_bytes",
        "_image_cam_bytes",
        "_image_nav_bytes",
        "_image_wrist_bytes",
    }
)


def _info_from_rpc_result(ret: Any) -> Any:
    if isinstance(ret, (tuple, list)):
        if len(ret) == 5:
            return ret[4]
        if len(ret) == 2:
            return ret[1]
    if isinstance(ret, dict):
        return ret.get("info", ret)
    return None


def _decode_observe_images(result: Any) -> Any:
    """Decode only public image fields returned by ``env.observe``."""

    if not isinstance(result, dict):
        return result
    decoded = dict(result)
    for field in _IMAGE_BYTE_FIELDS:
        payload = decoded.get(field)
        if (
            isinstance(payload, dict)
            and set(payload) == {"encoding", "data"}
            and payload.get("encoding") == "base64"
            and isinstance(payload.get("data"), str)
        ):
            decoded[field] = base64.b64decode(payload["data"], validate=True)
    return decoded


class BehaviorEnvClient(BaseEnvClient):
    """Remote implementation of the BEHAVIOR single-env protocol.

    Construction verifies only the requested immutable metadata and deliberately
    does not reset the simulator. Runtime initialization owns the first reset.
    """

    _TIMEOUT_S = {
        **BaseEnvClient._TIMEOUT_S,
        "env.reset": 1800.0,
        "env.step": 1800.0,
        "env.chunk_step": 1800.0,
        "env.current_observation": 120.0,
        "env.observe": 120.0,
        "env.pixel_to_world": 120.0,
        "env.move_to": 1800.0,
        "env.move_both_to": 1800.0,
        "env.get_prepared_motion_status": 30.0,
        "env.navigate_to": 1800.0,
        "env.rotate_wrist": 1800.0,
        "env.close": 120.0,
        "env.open": 120.0,
        "env.press": 1800.0,
        "env.save_robot_state_checkpoint": 120.0,
        "env.finalize_paused_runtime": 120.0,
    }

    def __init__(self, client: RpcClient, *, expected_meta: dict[str, Any]) -> None:
        # BaseEnvClient.__init__ performs an automatic reset and exact metadata
        # equality. BEHAVIOR reset is explicitly owned by runtime initialization.
        self._client = client
        self.last_obs: dict[str, Any] | None = None
        self.last_info: dict[str, Any] = {}
        self.episode_done = False
        self.total_env_steps = 0
        self.vla_endpoint: str | None = None
        self._official_success_latched = False
        self._official_success_receipt: dict[str, Any] | None = None
        server_meta = self._client.call(
            "env.get_env_meta",
            timeout_s=self._TIMEOUT_S["default"],
        )
        if not isinstance(server_meta, dict):
            raise RuntimeError(f"env_meta must be a mapping, got {type(server_meta)!r}")
        mismatches = {
            key: {"expected": expected, "actual": server_meta.get(key)}
            for key, expected in expected_meta.items()
            if server_meta.get(key) != expected
        }
        if mismatches:
            raise RuntimeError(f"env_meta mismatch: {mismatches!r}")
        self.server_meta = dict(server_meta)

    @staticmethod
    def _raw_success(info: Any) -> bool:
        done = info.get("done") if isinstance(info, dict) else None
        value = done.get("success") if isinstance(done, dict) else None
        return isinstance(value, (bool, np.bool_)) and bool(value)

    @staticmethod
    def _receipt_from_info(info: Any) -> dict[str, Any] | None:
        runtime = info.get("_rpent") if isinstance(info, dict) else None
        if not isinstance(runtime, dict):
            return None
        receipt = runtime.get("official_success_receipt")
        return copy.deepcopy(receipt) if isinstance(receipt, dict) else None

    def _latch_success_response(self, ret: Any) -> None:
        info = _info_from_rpc_result(ret)
        if not isinstance(info, dict):
            return
        runtime = info.get("_rpent")
        if isinstance(runtime, dict):
            steps = runtime.get("total_env_steps", runtime.get("global_env_steps"))
            if isinstance(steps, (int, np.integer)) and not isinstance(steps, bool):
                self.total_env_steps = max(self.total_env_steps, int(steps))
        if self._raw_success(info):
            self.episode_done = True
            self._official_success_latched = True
            self._official_success_receipt = validate_official_success_receipt(
                self._receipt_from_info(info)
            )

    def _rpc_call(
        self,
        method: str,
        *,
        args: tuple = (),
        kwargs: dict[str, Any] | None = None,
        timeout_s: float | None = None,
    ) -> Any:
        if self._official_success_latched and method not in _POST_SUCCESS_ALLOWED:
            raise RuntimeError(
                "raw task success is terminal; no further RPC is allowed"
            )
        ret = self._client.call(
            method,
            args=args,
            kwargs=kwargs or {},
            timeout_s=(
                timeout_s
                if timeout_s is not None
                else self._TIMEOUT_S.get(method, self._TIMEOUT_S["default"])
            ),
        )
        self._latch_success_response(ret)
        return ret

    @property
    def official_success_latched(self) -> bool:
        return self._official_success_latched

    @property
    def official_success_receipt(self) -> dict[str, Any] | None:
        return copy.deepcopy(self._official_success_receipt)

    def reset(self) -> tuple[dict[str, Any], dict[str, Any]]:
        ret = self._rpc_call("env.reset", timeout_s=self._TIMEOUT_S["env.reset"])
        if not isinstance(ret, (tuple, list)) or len(ret) != 2:
            raise TypeError("env.reset must return (observation, info)")
        obs, info = ret
        if not isinstance(obs, dict) or not isinstance(info, dict):
            raise TypeError("env.reset must return observation/info mappings")
        self.total_env_steps = 0
        self.episode_done = False
        self.last_obs = obs
        self.last_info = info
        return obs, info

    def step(self, action: Any) -> tuple[Any, Any, Any, Any, dict[str, Any]]:
        array = np.asarray(action, dtype=np.float32)
        if array.shape != (ACTION_DIM,) or not np.isfinite(array).all():
            raise ValueError(f"BEHAVIOR action must be finite [{ACTION_DIM}]")
        ret = self._rpc_call("env.step", args=(np.ascontiguousarray(array),))
        return self._note_gym_result(ret, "env.step")

    def chunk_step(
        self,
        actions: Any,
        *,
        return_all_frames: bool = False,
    ) -> tuple[Any, Any, Any, Any, dict[str, Any]]:
        action_array = validate_action_chunk(actions)
        ret = self._rpc_call(
            "env.chunk_step",
            args=(action_array,),
            kwargs={"return_all_frames": bool(return_all_frames)},
        )
        return self._note_gym_result(ret, "env.chunk_step")

    def _note_gym_result(self, ret: Any, method: str) -> tuple:
        if not isinstance(ret, (tuple, list)) or len(ret) != 5:
            raise TypeError(f"{method} must return a gym 5-tuple")
        obs, _reward, _terminated, _truncated, info = ret
        if not isinstance(info, dict):
            raise TypeError(f"{method} info must be a mapping")
        if isinstance(obs, list):
            if not obs:
                raise ValueError(f"{method} returned an empty observation list")
            self.last_obs = obs[-1]
        elif isinstance(obs, dict):
            self.last_obs = obs
        else:
            raise TypeError(f"{method} observation must be a mapping or list")
        self.last_info = info
        return tuple(ret)

    def current_observation(self) -> tuple[dict[str, Any], dict[str, Any]]:
        ret = self._rpc_call("env.current_observation")
        if not isinstance(ret, (tuple, list)) or len(ret) != 2:
            raise TypeError("env.current_observation must return (observation, info)")
        obs, info = ret
        if not isinstance(obs, dict) or not isinstance(info, dict):
            raise TypeError("env.current_observation returned invalid payload")
        self.last_obs = obs
        self.last_info = info
        return obs, info

    def get_camera_meta(self, camera_name: str, **kwargs: Any) -> dict[str, Any]:
        value = self._rpc_call(
            "env.get_camera_meta",
            kwargs={"camera_name": camera_name, **kwargs},
        )
        if not isinstance(value, dict):
            raise TypeError("env.get_camera_meta must return a mapping")
        return value

    def render_camera(self, camera_name: str, **kwargs: Any) -> np.ndarray:
        value = self._rpc_call(
            "env.render_camera",
            kwargs={"camera_name": camera_name, **kwargs},
        )
        image = np.asarray(value)
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3:
            raise TypeError("env.render_camera must return uint8[H,W,3]")
        return image

    def get_task_language(self) -> str:
        value = self._rpc_call("env.get_task_language")
        if not isinstance(value, str):
            raise TypeError("env.get_task_language must return a string")
        return value

    def observe(self, **kwargs: Any) -> dict[str, Any]:
        request = validate_observe_request(**kwargs)
        result = _decode_observe_images(self._rpc_call("env.observe", kwargs=request))
        if not isinstance(result, dict):
            raise TypeError("env.observe must return a mapping")
        return result

    def pixel_to_world(self, **kwargs: Any) -> dict[str, Any]:
        return self._rpc_call("env.pixel_to_world", kwargs=kwargs)

    def navigate_to(self, **kwargs: Any) -> dict[str, Any]:
        if "relative_motion" in kwargs and kwargs["relative_motion"] is not None:
            kwargs = {
                **kwargs,
                "relative_motion": validate_relative_navigation_motion(
                    kwargs["relative_motion"]
                ),
            }
        return self._rpc_call("env.navigate_to", kwargs=kwargs)

    def move_to(self, **kwargs: Any) -> dict[str, Any]:
        return self._rpc_call("env.move_to", kwargs=kwargs)

    def move_both_to(self, **kwargs: Any) -> dict[str, Any]:
        kwargs = {
            **kwargs,
            "targets": validate_move_both_targets(kwargs.get("targets")),
            "visual_hand_checks": validate_move_both_visual_hand_checks(
                kwargs.get("visual_hand_checks")
            ),
        }
        return self._rpc_call("env.move_both_to", kwargs=kwargs)

    def get_prepared_motion_status(self, *, prepared_plan_id: str) -> dict[str, Any]:
        return self._rpc_call(
            "env.get_prepared_motion_status",
            kwargs={"prepared_plan_id": validate_prepared_plan_id(prepared_plan_id)},
        )

    def rotate_wrist(self, **kwargs: Any) -> dict[str, Any]:
        return self._rpc_call("env.rotate_wrist", kwargs=kwargs)

    def close(self, **kwargs: Any) -> dict[str, Any]:
        return self._rpc_call("env.close", kwargs=kwargs)

    def open(self, **kwargs: Any) -> dict[str, Any]:
        return self._rpc_call("env.open", kwargs=kwargs)

    def press(self, **kwargs: Any) -> dict[str, Any]:
        return self._rpc_call("env.press", kwargs=kwargs)

    def save_robot_state_checkpoint(self, **kwargs: Any) -> dict[str, Any]:
        return self._rpc_call("env.save_robot_state_checkpoint", kwargs=kwargs)

    def finalize_paused_runtime(
        self, vla_status: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        return self._rpc_call(
            "env.finalize_paused_runtime",
            kwargs={"vla_status": vla_status},
        )

    def close_transport(self) -> None:
        close = getattr(self._client, "close", None)
        if callable(close):
            close()


__all__ = ["BehaviorEnvClient"]
