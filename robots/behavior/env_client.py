"""RPC client for one BEHAVIOR environment."""

from __future__ import annotations

import base64
import copy
import hashlib
import hmac
import json
from typing import Any

import numpy as np

from robots.behavior.schemas import (
    validate_action_chunk,
    validate_dashboard_command_id,
    validate_dashboard_control_capabilities,
    validate_dashboard_manual_command,
    validate_dashboard_plan_id,
    validate_dashboard_prepare_request,
    validate_move_both_targets,
    validate_move_both_visual_hand_checks,
    validate_observe_request,
    validate_relative_navigation_motion,
)
from rpent.utils.rpc import RpcClient

_TIMEOUT_S = {
    "default": 30.0,
    "env.reset": 1800.0,
    "env.current_observation": 120.0,
    "env.pi0_nav_pick_chunk_step": 1800.0,
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
    "env.dashboard_control_capabilities": 30.0,
    "env.dashboard_prepare_manual_command": 72.0,
    "env.dashboard_execute_prepared_command": 72.0,
    "env.dashboard_discard_prepared_command": 30.0,
    "env.dashboard_capture_views": 120.0,
    "env.dashboard_manual_command": 360.0,
    "env.dashboard_safe_stop": 30.0,
}
_POST_SUCCESS_ALLOWED = frozenset(
    {
        "env.get_env_meta",
        "env.get_prepared_motion_status",
        "env.current_observation",
        "env.finalize_paused_runtime",
        "env.dashboard_safe_stop",
    }
)


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _info_from_rpc_result(ret: Any) -> Any:
    if isinstance(ret, (tuple, list)):
        if len(ret) == 5:
            return ret[4]
        if len(ret) == 2:
            return ret[1]
    if isinstance(ret, dict):
        return ret.get("info", ret)
    return None


def _decode_bytes(value: Any) -> Any:
    if isinstance(value, dict):
        if set(value) == {"__bytes_b64__"} and isinstance(value["__bytes_b64__"], str):
            return base64.b64decode(value["__bytes_b64__"], validate=True)
        return {str(key): _decode_bytes(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_decode_bytes(item) for item in value]
    return value


class BehaviorEnvClient:
    """Remote implementation of the BEHAVIOR single-env protocol."""

    def __init__(self, client: RpcClient, *, expected_meta: dict[str, Any]) -> None:
        self._client = client
        self.episode_done = False
        self.total_env_steps = 0
        self.vla_endpoint: str | None = None
        self._official_success_latched = False
        self._official_success_receipt: dict[str, Any] | None = None
        server_meta = self._rpc_call("env.get_env_meta")
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
    def _canonical_receipt_bytes(value: dict[str, Any]) -> bytes:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")

    @classmethod
    def _valid_success_receipt(cls, value: Any) -> dict[str, Any] | None:
        if not isinstance(value, dict):
            return None
        required = {
            "schema_version",
            "source",
            "env_step",
            "raw_done",
            "receipt_sha256",
        }
        if not required.issubset(value):
            return None
        raw_done = value.get("raw_done")
        digest = value.get("receipt_sha256")
        if (
            value.get("schema_version") != 1
            or value.get("source") != 'info["done"]["success"]'
            or not isinstance(raw_done, dict)
            or raw_done.get("success") is not True
            or isinstance(value.get("env_step"), bool)
            or not isinstance(value.get("env_step"), int)
            or not isinstance(digest, str)
        ):
            return None
        material = {key: item for key, item in value.items() if key != "receipt_sha256"}
        expected = hashlib.sha256(cls._canonical_receipt_bytes(material)).hexdigest()
        if not hmac.compare_digest(digest, expected):
            return None
        return copy.deepcopy(value)

    @staticmethod
    def _receipt_from_info(info: Any) -> dict[str, Any] | None:
        runtime = info.get("_rpent") if isinstance(info, dict) else None
        if not isinstance(runtime, dict):
            return None
        direct = runtime.get("official_success_receipt")
        if isinstance(direct, dict):
            return copy.deepcopy(direct)
        monitor = runtime.get("pi0_nav_pick_monitor")
        if isinstance(monitor, dict) and isinstance(
            monitor.get("official_success_receipt"), dict
        ):
            return copy.deepcopy(monitor["official_success_receipt"])
        return None

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
            self._official_success_receipt = self._valid_success_receipt(
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
        ret = _decode_bytes(
            self._client.call(
                method,
                args=args,
                kwargs=kwargs or {},
                timeout_s=timeout_s or _TIMEOUT_S.get(method, _TIMEOUT_S["default"]),
            )
        )
        self._latch_success_response(ret)
        return ret

    @property
    def official_success_latched(self) -> bool:
        return self._official_success_latched

    @property
    def official_success_receipt(self) -> dict[str, Any] | None:
        return copy.deepcopy(self._official_success_receipt)

    def reset(self) -> tuple[dict[str, Any], Any]:
        ret = self._rpc_call("env.reset", timeout_s=_TIMEOUT_S["env.reset"])
        if not isinstance(ret, (tuple, list)) or len(ret) != 2:
            raise TypeError("env.reset must return (observation, info)")
        obs, info = ret
        if not isinstance(obs, dict):
            raise TypeError("env.reset observation must be a mapping")
        self.total_env_steps = 0
        self.last_obs = obs
        self.last_info = info
        return obs, info

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

    def pi0_nav_pick_chunk_step(
        self,
        actions: Any,
        *,
        chunk_index: int,
    ) -> tuple[Any, Any, Any, Any, dict[str, Any]]:
        action_array = validate_action_chunk(actions)
        ret = self._rpc_call(
            "env.pi0_nav_pick_chunk_step",
            args=(action_array,),
            kwargs={"chunk_index": int(chunk_index)},
            timeout_s=_TIMEOUT_S["env.pi0_nav_pick_chunk_step"],
        )
        if not isinstance(ret, (tuple, list)) or len(ret) != 5:
            raise TypeError("env.pi0_nav_pick_chunk_step must return a gym 5-tuple")
        obs, _reward, _terminated, _truncated, info = ret
        if isinstance(obs, dict):
            self.last_obs = obs
        self.last_info = info
        return tuple(ret)  # type: ignore[return-value]

    def observe(self, **kwargs: Any) -> dict[str, Any]:
        request = validate_observe_request(**kwargs)
        return self._rpc_call("env.observe", kwargs=request)

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
            kwargs={"prepared_plan_id": validate_dashboard_plan_id(prepared_plan_id)},
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

    def dashboard_control_capabilities(self) -> dict[str, Any]:
        return validate_dashboard_control_capabilities(
            self._rpc_call("env.dashboard_control_capabilities")
        )

    def dashboard_prepare_manual_command(self, **kwargs: Any) -> dict[str, Any]:
        return self._rpc_call(
            "env.dashboard_prepare_manual_command",
            kwargs=validate_dashboard_prepare_request(**kwargs),
        )

    def dashboard_execute_prepared_command(
        self,
        *,
        command_id: str,
        plan_id: str | None = None,
    ) -> dict[str, Any]:
        kwargs = {"command_id": validate_dashboard_command_id(command_id)}
        if plan_id is not None:
            kwargs["plan_id"] = validate_dashboard_plan_id(plan_id)
        return self._rpc_call(
            "env.dashboard_execute_prepared_command",
            kwargs=kwargs,
        )

    def dashboard_discard_prepared_command(
        self,
        *,
        command_id: str,
        plan_id: str | None = None,
    ) -> dict[str, Any]:
        kwargs = {"command_id": validate_dashboard_command_id(command_id)}
        if plan_id is not None:
            kwargs["plan_id"] = validate_dashboard_plan_id(plan_id)
        return self._rpc_call(
            "env.dashboard_discard_prepared_command",
            kwargs=kwargs,
        )

    def dashboard_capture_views(self, *, camera: str = "head") -> dict[str, Any]:
        validate_dashboard_manual_command(
            target="chassis", action="observe", camera=camera
        )
        return self._rpc_call("env.dashboard_capture_views", kwargs={"camera": camera})

    def dashboard_safe_stop(
        self,
        *,
        reason: str = "client_stop",
        stop_mode: str = "safe_stop",
    ) -> dict[str, Any]:
        return self._rpc_call(
            "env.dashboard_safe_stop",
            kwargs={"reason": str(reason), "stop_mode": str(stop_mode)},
        )

    def dashboard_manual_command(
        self,
        *,
        target: str,
        action: str,
        camera: str,
    ) -> dict[str, Any]:
        return self._rpc_call(
            "env.dashboard_manual_command",
            kwargs=validate_dashboard_manual_command(
                target=target,
                action=action,
                camera=camera,
            ),
        )

    def close_transport(self) -> None:
        close = getattr(self._client, "close", None)
        if callable(close):
            close()


__all__ = ["BehaviorEnvClient"]
