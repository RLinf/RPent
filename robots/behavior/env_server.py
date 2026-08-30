"""BEHAVIOR environment RPC adapter.

This server owns identity, CVD ordering, and RPC shape. It defaults to the
bundled adapter for the official RLinf ``BehaviorEnv``; the factory environment
variable remains an explicit testing/integration override.
"""

from __future__ import annotations

import argparse
import base64
import importlib
import os
import re
import sys
from http.server import HTTPServer
from pathlib import Path
from typing import Any, Callable

import numpy as np


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


if str(_repo_root()) not in sys.path:
    sys.path.insert(0, str(_repo_root()))

from robots.behavior.schemas import ACTION_DIM, DEFAULT_ACTION_CHUNK, validate_action_chunk
from robots.behavior.task_specs import get_task_spec
from robots.behavior.terminal_success import (
    make_raw_success_receipt,
    official_task_success,
)
from rpent.utils.rpc.http_rpc import _HttpRpcHandler


_ENV_METHODS = {
    "healthz",
    "env.get_env_meta",
    "env.reset",
    "env.current_observation",
    "env.pi0_nav_pick_chunk_step",
    "env.observe",
    "env.pixel_to_world",
    "env.navigate_to",
    "env.move_to",
    "env.move_both_to",
    "env.get_prepared_motion_status",
    "env.rotate_wrist",
    "env.close",
    "env.open",
    "env.press",
    "env.save_robot_state_checkpoint",
    "env.finalize_paused_runtime",
    "env.dashboard_control_capabilities",
    "env.dashboard_prepare_manual_command",
    "env.dashboard_execute_prepared_command",
    "env.dashboard_discard_prepared_command",
    "env.dashboard_capture_views",
    "env.dashboard_manual_command",
    "env.dashboard_safe_stop",
}


def _single_cuda_device(value: Any) -> str | None:
    if value in (None, ""):
        return None
    device = str(value)
    if re.fullmatch(r"[0-9]+", device) is None:
        raise ValueError("--cuda-device must be one physical GPU ordinal")
    return device


def _jsonable(value: Any) -> Any:
    if hasattr(value, "detach") and hasattr(value, "cpu") and hasattr(value, "numpy"):
        value = value.detach().cpu().numpy()
    if isinstance(value, bytes):
        return {"__bytes_b64__": base64.b64encode(value).decode("ascii")}
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


def _backend_factory_from_env() -> Any:
    spec = os.environ.get(
        "RPENT_BEHAVIOR_ENV_BACKEND_FACTORY",
        "robots.behavior.official_env_backend:create_backend",
    )
    module_name, sep, attr = spec.partition(":")
    if not sep or not module_name or not attr:
        raise RuntimeError(
            "RPENT_BEHAVIOR_ENV_BACKEND_FACTORY must be 'module:callable'"
        )
    factory = getattr(importlib.import_module(module_name), attr)
    if not callable(factory):
        raise RuntimeError("configured BEHAVIOR env backend factory is not callable")
    return factory


class BehaviorEnvFacade:
    """Thin checked adapter around a supplied live BEHAVIOR backend."""

    def __init__(self, *, meta: dict[str, Any], output_dir: Path) -> None:
        self._meta = dict(meta)
        self._output_dir = output_dir
        self._last_obs: dict[str, Any] | None = None
        self._last_info: dict[str, Any] = {}
        self._total_env_steps = 0
        self._official_success_receipt: dict[str, Any] | None = None
        factory = _backend_factory_from_env()
        self._backend = factory(meta=dict(meta), output_dir=output_dir)

    @property
    def total_env_steps(self) -> int:
        value = getattr(self._backend, "total_env_steps", self._total_env_steps)
        if isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)):
            return max(self._total_env_steps, int(value))
        return self._total_env_steps

    def _note_info(self, info: Any) -> dict[str, Any]:
        if not isinstance(info, dict):
            info = {}
        runtime = info.get("_rpent")
        if isinstance(runtime, dict):
            steps = runtime.get("total_env_steps", runtime.get("global_env_steps"))
            if isinstance(steps, (int, np.integer)) and not isinstance(steps, (bool, np.bool_)):
                self._total_env_steps = max(self._total_env_steps, int(steps))
        if official_task_success(info):
            self._official_success_receipt = make_raw_success_receipt(
                info,
                env_step=self.total_env_steps,
            )
        self._last_info = info
        return info

    def healthz(self) -> dict[str, Any]:
        return {"status": "ok", "pid": os.getpid(), **self._meta}

    def get_env_meta(self) -> dict[str, Any]:
        return dict(self._meta)

    def reset(self) -> tuple[dict[str, Any], dict[str, Any]]:
        if not hasattr(self._backend, "reset"):
            raise RuntimeError("backend does not expose reset()")
        ret = self._backend.reset()
        if isinstance(ret, (tuple, list)) and len(ret) == 2:
            obs, info = ret
        else:
            obs, info = ret, {}
        if not isinstance(obs, dict):
            raise TypeError("backend reset must return observation mapping")
        obs.setdefault("task_descriptions", self._meta["task_language"])
        self._last_obs = obs
        self._note_info(info)
        return obs, self._last_info

    def current_observation(self) -> tuple[dict[str, Any], dict[str, Any]]:
        method = getattr(self._backend, "current_observation", None)
        if callable(method):
            ret = method()
            if isinstance(ret, (tuple, list)) and len(ret) == 2:
                obs, info = ret
            else:
                obs, info = ret, self._last_info
            if not isinstance(obs, dict):
                raise TypeError("current_observation must return observation mapping")
            self._last_obs = obs
            self._note_info(info)
            return obs, self._last_info
        if self._last_obs is None:
            raise RuntimeError("no observation has been captured yet")
        return self._last_obs, self._last_info

    def pi0_nav_pick_chunk_step(
        self,
        actions: Any,
        *,
        chunk_index: int,
    ) -> tuple[Any, Any, bool, bool, dict[str, Any]]:
        action_array = validate_action_chunk(actions)
        method = getattr(self._backend, "pi0_nav_pick_chunk_step", None)
        if not callable(method):
            raise RuntimeError(
                "backend does not expose pi0_nav_pick_chunk_step(actions, chunk_index=...)"
            )
        ret = method(action_array, chunk_index=int(chunk_index))
        if not isinstance(ret, (tuple, list)) or len(ret) != 5:
            raise TypeError("pi0_nav_pick_chunk_step must return gym 5-tuple")
        obs, reward, terminated, truncated, info = ret
        if isinstance(obs, dict):
            self._last_obs = obs
        self._total_env_steps = max(self._total_env_steps, self._total_env_steps + action_array.shape[0])
        self._note_info(info)
        return obs, reward, bool(terminated), bool(truncated), self._last_info

    def _backend_call(self, public_name: str, **kwargs: Any) -> dict[str, Any]:
        method = getattr(self._backend, public_name, None)
        if not callable(method):
            raise RuntimeError(f"backend does not expose {public_name}()")
        ret = method(**kwargs)
        info = ret.get("info") if isinstance(ret, dict) else None
        if isinstance(info, dict):
            self._note_info(info)
        return _jsonable(ret)

    def finalize_paused_runtime(self, vla_status: dict[str, Any] | None = None) -> dict[str, Any]:
        method = getattr(self._backend, "finalize_paused_runtime", None)
        if callable(method):
            result = method(vla_status=vla_status)
            return _jsonable(result)
        return {
            "status": "ok",
            "task_success": official_task_success(self._last_info),
            "official_success_receipt": self._official_success_receipt,
            "vla_status": vla_status,
        }

    def dispatch(self, method: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        if method not in _ENV_METHODS:
            raise AttributeError(f"unknown BEHAVIOR env RPC method: {method}")
        if method == "healthz":
            return self.healthz()
        if method == "env.get_env_meta":
            return self.get_env_meta()
        if method == "env.reset":
            return self.reset()
        if method == "env.current_observation":
            return self.current_observation()
        if method == "env.pi0_nav_pick_chunk_step":
            return self.pi0_nav_pick_chunk_step(*args, **kwargs)
        if method == "env.finalize_paused_runtime":
            return self.finalize_paused_runtime(*args, **kwargs)
        public_name = method.removeprefix("env.")
        return self._backend_call(public_name, **kwargs)

    def shutdown(self) -> None:
        closer = getattr(self._backend, "close", None)
        if callable(closer):
            closer()


class BehaviorMainThreadHttpRpcServer(HTTPServer):
    """BEHAVIOR env RPC server that dispatches requests on the serving thread.

    OmniGibson/USD scene reset mutates simulator state that must stay on the
    process main thread. The shared HttpRpcServer uses ThreadingHTTPServer, so
    the BEHAVIOR env server keeps the same HTTP wire handler but serves requests
    serially from the thread running serve_forever().
    """

    allow_reuse_address = True

    def __init__(
        self,
        server_address: tuple[str, int],
        dispatch: Callable[[str, tuple[Any, ...], dict[str, Any]], Any],
    ) -> None:
        super().__init__(server_address, _HttpRpcHandler)
        self.dispatch = dispatch


def _build_meta(args: argparse.Namespace) -> dict[str, Any]:
    task_spec = get_task_spec(args.task_name)
    return {
        "runtime": "behavior_env",
        "task_name": task_spec.task_name,
        "task": int(args.task_index),
        "task_language": task_spec.task_language,
        "activity_definition_id": int(args.activity_definition_id),
        "activity_instance_id": int(args.activity_instance_id),
        "public_seed": int(args.public_seed),
        "scene_model": str(args.scene_model),
        "max_episode_steps": int(args.max_episode_steps),
        "action_dim": ACTION_DIM,
        "action_horizon": DEFAULT_ACTION_CHUNK,
        "official_success_path": ["info", "done", "success"],
        "behavior_repo": str(Path(args.behavior_repo).expanduser().resolve()),
        "activity_instance_dir": (
            None
            if not args.activity_instance_dir
            else str(Path(args.activity_instance_dir).expanduser().resolve())
        ),
        "rlinf_env_config_path": (
            None
            if not args.env_config_path
            else str(Path(args.env_config_path).expanduser().resolve())
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-name", required=True)
    parser.add_argument("--public-seed", type=int, required=True)
    parser.add_argument("--task-index", type=int, required=True)
    parser.add_argument("--activity-definition-id", type=int, required=True)
    parser.add_argument("--activity-instance-id", type=int, required=True)
    parser.add_argument("--scene-model", required=True)
    parser.add_argument("--max-episode-steps", type=int, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--behavior-repo", required=True)
    parser.add_argument("--activity-instance-dir", default=None)
    parser.add_argument("--env-config-path", default=None)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--cuda-device", default=None)
    parser.add_argument("--parent-watch", action="store_true")
    args = parser.parse_args()
    cuda_device = _single_cuda_device(args.cuda_device)
    if cuda_device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device
    os.environ["RPENT_RLINF_ROOT"] = str(Path(args.behavior_repo).expanduser().resolve())

    from rpent.utils.daemon import watch_parent_death

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    env = BehaviorEnvFacade(meta=_build_meta(args), output_dir=output_dir)
    server = BehaviorMainThreadHttpRpcServer((args.host, args.port), env.dispatch)
    if args.parent_watch:
        watch_parent_death(server.shutdown)
    try:
        server.serve_forever()
    finally:
        try:
            env.shutdown()
        finally:
            server.server_close()


if __name__ == "__main__":
    main()


__all__ = ["BehaviorEnvFacade", "BehaviorMainThreadHttpRpcServer", "main"]
