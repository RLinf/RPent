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

"""BEHAVIOR environment RPC adapter.

OmniGibson scene operations stay on the process main thread. The facade uses
the common environment RPC contract while ``serve`` keeps HTTP dispatch
serial, rather than using the shared threaded HTTP server.
"""

from __future__ import annotations

import argparse
import base64
import os
import re
import sys
import threading
from http.server import HTTPServer
from pathlib import Path
from typing import Any, Callable, Literal

import numpy as np


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


if str(_repo_root()) not in sys.path:
    sys.path.insert(0, str(_repo_root()))

from robots.behavior.schemas import (  # noqa: E402
    ACTION_DIM,
    DEFAULT_ACTION_CHUNK,
    validate_action_chunk,
)
from robots.behavior.task_specs import get_task_spec  # noqa: E402
from rpent.robots.components.env_facade_base import BaseEnvFacade  # noqa: E402
from rpent.utils.daemon import watch_parent_death  # noqa: E402
from rpent.utils.rpc.http_rpc import _HttpRpcHandler  # noqa: E402

_IMAGE_BYTE_FIELDS = frozenset(
    {
        "_depth_image_bytes",
        "_image_bytes",
        "_image_left_wrist_bytes",
        "_depth_left_wrist_bytes",
        "_image_right_wrist_bytes",
        "_depth_right_wrist_bytes",
    }
)


def _single_cuda_device(value: Any) -> str | None:
    if value in (None, ""):
        return None
    device = str(value)
    if re.fullmatch(r"[0-9]+", device) is None:
        raise ValueError("--cuda-device must be one physical GPU ordinal")
    return device


def _encode_observe_images(result: Any) -> Any:
    """Encode only public image byte fields at the ``env.observe`` boundary."""

    if not isinstance(result, dict):
        return result
    encoded = dict(result)
    for field in _IMAGE_BYTE_FIELDS:
        payload = encoded.get(field)
        if isinstance(payload, bytes):
            encoded[field] = {
                "encoding": "base64",
                "data": base64.b64encode(payload).decode("ascii"),
            }
    return encoded


class BehaviorEnvFacade(BaseEnvFacade):
    """Expose one official RLinf BEHAVIOR backend through common ENV RPC."""

    def __init__(self, *, backend: Any, meta: dict[str, Any]) -> None:
        self._backend = backend
        self._meta = dict(meta)
        self._closed = False
        super().__init__()

    def _register_rpc(self) -> None:
        super()._register_rpc()
        self._rpc.update(
            {
                "env.current_observation": self.current_observation,
                "env.observe": self.observe,
                "env.pixel_to_world": self.pixel_to_world,
                "env.navigate_to": self.navigate_to,
                "env.move_to": self.move_to,
                "env.rotate_wrist": self.rotate_wrist,
                "env.close_gripper": self.close_gripper,
                "env.open_gripper": self.open_gripper,
                "env.press": self.press,
                "env.finalize_paused_runtime": self.finalize_paused_runtime,
            }
        )
        self._readonly_methods.update(
            {
                "env.current_observation",
                "env.finalize_paused_runtime",
            }
        )

    def _call_backend(self, name: str, *args: Any, **kwargs: Any) -> Any:
        method = getattr(self._backend, name, None)
        if not callable(method):
            raise RuntimeError(f"backend does not expose {name}()")
        return method(*args, **kwargs)

    @property
    def total_env_steps(self) -> int:
        value = getattr(self._backend, "total_env_steps", 0)
        if isinstance(value, (int, np.integer)) and not isinstance(
            value, (bool, np.bool_)
        ):
            return max(0, int(value))
        return 0

    def get_env_meta(self) -> dict[str, Any]:
        return dict(self._meta)

    def reset(self) -> tuple[dict[str, Any], dict[str, Any]]:
        result = self._call_backend("reset")
        if not isinstance(result, (tuple, list)) or len(result) != 2:
            raise TypeError("backend reset must return (observation, info)")
        observation, info = result
        if not isinstance(observation, dict) or not isinstance(info, dict):
            raise TypeError("backend reset must return observation/info mappings")
        observation.setdefault("task_descriptions", self._meta["task_language"])
        return observation, info

    def step(self, action: Any) -> tuple[Any, Any, Any, Any, dict[str, Any]]:
        result = self._call_backend("step", action)
        return self._require_gym_result(result, "step")

    def chunk_step(
        self,
        actions: Any,
        *,
        return_all_frames: bool = False,
    ) -> tuple[Any, Any, Any, Any, dict[str, Any]]:
        action_array = validate_action_chunk(actions)
        result = self._call_backend(
            "chunk_step",
            action_array,
            return_all_frames=bool(return_all_frames),
        )
        return self._require_gym_result(result, "chunk_step")

    @staticmethod
    def _require_gym_result(result: Any, method: str) -> tuple:
        if not isinstance(result, (tuple, list)) or len(result) != 5:
            raise TypeError(f"env.{method} must return a gym 5-tuple")
        if not isinstance(result[4], dict):
            raise TypeError(f"env.{method} info must be a mapping")
        return tuple(result)

    def current_observation(self) -> tuple[dict[str, Any], dict[str, Any]]:
        result = self._call_backend("current_observation")
        if not isinstance(result, (tuple, list)) or len(result) != 2:
            raise TypeError("current_observation must return (observation, info)")
        observation, info = result
        if not isinstance(observation, dict) or not isinstance(info, dict):
            raise TypeError("current_observation returned invalid payload")
        return observation, info

    def get_task_language(self) -> str:
        value = self._call_backend("get_task_language")
        if not isinstance(value, str):
            raise TypeError("BEHAVIOR task language must be a string")
        return value

    def get_camera_meta(self, camera_name: str = "head", **kwargs: Any) -> dict:
        value = self._call_backend("get_camera_meta", camera_name, **kwargs)
        if not isinstance(value, dict):
            raise TypeError("BEHAVIOR camera metadata must be a mapping")
        return value

    def render_camera(self, camera_name: str = "head", **kwargs: Any) -> np.ndarray:
        image = np.asarray(self._call_backend("render_camera", camera_name, **kwargs))
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError(
                f"rendered RGB must be uint8[H,W,3], got {image.dtype}{image.shape}"
            )
        return np.ascontiguousarray(image)

    def observe(self, **kwargs: Any) -> dict[str, Any]:
        result = self._call_backend("observe", **kwargs)
        if not isinstance(result, dict):
            raise TypeError("env.observe must return a mapping")
        return _encode_observe_images(result)

    def pixel_to_world(self, **kwargs: Any) -> dict[str, Any]:
        return self._call_backend("pixel_to_world", **kwargs)

    def navigate_to(self, **kwargs: Any) -> dict[str, Any]:
        return self._call_backend("navigate_to", **kwargs)

    def move_to(self, **kwargs: Any) -> dict[str, Any]:
        return self._call_backend("move_to", **kwargs)

    def rotate_wrist(self, **kwargs: Any) -> dict[str, Any]:
        return self._call_backend("rotate_wrist", **kwargs)

    def close_gripper(self, **kwargs: Any) -> dict[str, Any]:
        return self._call_backend("close", **kwargs)

    def open_gripper(self, **kwargs: Any) -> dict[str, Any]:
        return self._call_backend("open", **kwargs)

    def press(self, **kwargs: Any) -> dict[str, Any]:
        return self._call_backend("press", **kwargs)

    def finalize_paused_runtime(
        self, vla_status: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        return self._call_backend("finalize_paused_runtime", vla_status=vla_status)

    def close(self) -> None:
        if self._closed:
            return
        closer = getattr(self._backend, "close", None)
        if callable(closer):
            closer()
        self._closed = True

    def serve(
        self,
        *,
        transport: Literal["socket", "http"],
        host: str,
        port: int,
        parent_watch: bool = False,
    ) -> None:
        if transport != "http":
            raise ValueError("BEHAVIOR env supports only HTTP RPC")
        server = BehaviorMainThreadHttpRpcServer((host, port), self._dispatch)
        bound_host, bound_port = server.server_address
        client_host = "127.0.0.1" if bound_host == "0.0.0.0" else bound_host
        print(f"RPC server listening on http://{client_host}:{bound_port}", flush=True)

        if parent_watch:
            watch_parent_death(self._shutdown_event.set)

        def stop_server() -> None:
            self._shutdown_event.wait()
            server.shutdown()

        stopper = threading.Thread(
            target=stop_server,
            name="behavior-env-stop",
            daemon=True,
        )
        stopper.start()
        try:
            server.serve_forever()
        finally:
            self._shutdown_event.set()
            server.server_close()
            self.close()
            stopper.join(timeout=5.0)


class BehaviorMainThreadHttpRpcServer(HTTPServer):
    """Serial HTTP RPC server whose handlers run on the serving thread."""

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
    os.environ["RPENT_RLINF_ROOT"] = str(
        Path(args.behavior_repo).expanduser().resolve()
    )

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    from robots.behavior.rlinf_env import OfficialBehaviorBackend

    meta = _build_meta(args)
    backend = OfficialBehaviorBackend(meta=meta, output_dir=output_dir)
    facade = BehaviorEnvFacade(backend=backend, meta=meta)
    facade.serve(
        transport="http",
        host=args.host,
        port=args.port,
        parent_watch=args.parent_watch,
    )


if __name__ == "__main__":
    main()


__all__ = ["BehaviorEnvFacade", "BehaviorMainThreadHttpRpcServer", "main"]
