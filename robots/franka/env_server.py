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

"""RPC server owning one RLinf single-Franka ``RealWorldEnv`` worker."""

from __future__ import annotations

import argparse
import dataclasses
import sys
import time
from collections.abc import Callable
from typing import Any

import numpy as np

from robots.franka.runtime_config import load_runtime_config
from rpent.robots.components.env_facade_base import BaseEnvFacade
from rpent.utils.config import get_repo_root, get_rlinf_repo_path
from rpent.utils.logging import get_logger

logger = get_logger("franka_env_server")

# Resolve the RLinf checkout before the deferred ``import rlinf`` executes.
RPENT_ROOT = get_repo_root()
RLINF_REPO_PATH = get_rlinf_repo_path() or (RPENT_ROOT.parent / "rlinf").resolve()
if str(RLINF_REPO_PATH) not in sys.path:
    sys.path.insert(0, str(RLINF_REPO_PATH))


def _to_numpy_tree(value: Any) -> Any:
    """Convert tensors and nested values into pickle-safe CPU/numpy data."""
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        return value.detach().cpu().numpy()
    if dataclasses.is_dataclass(value):
        return _to_numpy_tree(dataclasses.asdict(value))
    if isinstance(value, dict):
        return {key: _to_numpy_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_numpy_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_to_numpy_tree(item) for item in value)
    if isinstance(value, np.generic):
        return value.item()
    return value


class FrankaEnvFacade(BaseEnvFacade):
    """Expose the single-Franka ``env.*`` protocol from a Ray-backed worker."""

    _METHODS = (
        "get_env_meta",
        "reset",
        "get_robot_state",
        "get_observation",
        "get_camera_meta",
        "move_delta",
        "rotate_delta",
        "set_gripper",
        "chunk_step",
    )

    def __init__(self, backend: Any) -> None:
        self._backend = backend
        super().__init__()

    def _register_rpc(self) -> None:
        for name in self._METHODS:
            self._rpc[f"env.{name}"] = getattr(self._backend, name)
        self._readonly_methods.update(
            {
                "env.get_env_meta",
                "env.get_robot_state",
                "env.get_observation",
                "env.get_camera_meta",
            }
        )


class _RayBackend:
    """Turn RLinf Ray method handles into a synchronous facade backend."""

    def __init__(self, worker: Any) -> None:
        self.worker = worker

    def __getattr__(self, name: str):
        remote_method = getattr(self.worker, name)

        def invoke(*args, **kwargs):
            return remote_method(*args, **kwargs).wait()[0]

        return invoke


def _create_worker_class():
    """Build the Worker subclass only inside the RLinf server environment."""
    from rlinf.envs.realworld.realworld_env import RealWorldEnv
    from rlinf.scheduler import Worker
    from scipy.spatial.transform import Rotation as Rotation

    class FrankaEnvWorker(Worker):
        """Ray worker that owns the physical Franka and camera resources."""

        def __init__(self, cfg: Any, controller_config: dict[str, Any]):
            super().__init__()
            from robots.franka.rpent_env import (
                register_rpent_franka_env,
            )

            register_rpent_franka_env()
            self.cfg = cfg
            self.controller = dict(controller_config)
            self.env = RealWorldEnv(
                cfg.env.eval,
                num_envs=1,
                seed_offset=0,
                total_num_processes=1,
                worker_info=self.worker_info,
            )
            self.action_dim = int(self.env.action_space.shape[-1])
            env_config = self.env.env.call("get_wrapper_attr", "config")[0]
            self.action_scale = np.asarray(env_config.action_scale, dtype=np.float32)
            self.use_relative_frame = bool(cfg.env.eval.get("use_relative_frame", True))
            self.last_obs: dict[str, Any] | None = None

        def get_env_meta(self) -> dict[str, Any]:
            return {
                "ok": True,
                "action_dim": self.action_dim,
                "action_scale": self.action_scale.tolist(),
                "use_relative_frame": self.use_relative_frame,
            }

        def close_env(self) -> None:
            try:
                self.env.close()
            except Exception:
                pass

        def reset(self) -> dict[str, Any]:
            observation, info = self.env.reset()
            self.last_obs = observation
            return {
                "ok": True,
                "info": _to_numpy_tree(info),
                "robot_state": self.get_robot_state(),
            }

        def _ensure_obs(self) -> dict[str, Any]:
            if self.last_obs is None:
                raise RuntimeError(
                    "env.reset() has not been called; observation is unavailable. "
                    "Call reset before reading observations."
                )
            return self.last_obs

        @staticmethod
        def _strip_batch(value: Any) -> Any:
            array = _to_numpy_tree(value)
            if isinstance(array, np.ndarray) and array.ndim > 0 and array.shape[0] == 1:
                return array[0]
            if isinstance(array, list) and len(array) == 1:
                return array[0]
            return array

        def get_observation(self) -> dict[str, Any]:
            observation = self._ensure_obs()
            output = {
                key: self._strip_batch(value) for key, value in observation.items()
            }
            for key in ("extra_view_images", "extra_view_depths"):
                value = output.get(key)
                if isinstance(value, np.ndarray) and value.shape[0] == 1:
                    output[key] = value[0]
            return output

        def _raw_state(self) -> Any:
            return self.env.env.call("get_wrapper_attr", "_franka_state")[0]

        def _raw_tcp_pose(self) -> np.ndarray:
            return np.asarray(self._raw_state().tcp_pose, dtype=np.float32)

        def get_robot_state(self) -> dict[str, Any]:
            wrapped = self.get_observation().get("states")
            return {
                "raw_base_state": _to_numpy_tree(self._raw_state()),
                "wrapped_state_vector": _to_numpy_tree(wrapped),
                "action_dim": self.action_dim,
                "action_scale": self.action_scale.tolist(),
                "use_relative_frame": self.use_relative_frame,
            }

        def get_camera_meta(self) -> dict[str, Any] | None:
            try:
                metadata = self.env.env.call("get_camera_metadata")[0]
            except Exception as exc:
                return {"error": str(exc), "error_type": type(exc).__name__}
            metadata = _to_numpy_tree(metadata)
            main_key = self.cfg.env.eval.get("main_image_key")
            names = sorted(metadata.get("cameras", {}))
            extras = [name for name in names if name != main_key]
            metadata["observation_camera_map"] = {
                "main": main_key,
                **{f"extra_{index}": name for index, name in enumerate(extras)},
            }
            return metadata

        def _step_delta(
            self,
            delta_xyz: np.ndarray,
            delta_rpy: np.ndarray,
            *,
            frame: str,
            gripper: float | None = None,
        ) -> tuple[Any, Any, Any, Any, Any]:
            action = np.zeros(self.action_dim, dtype=np.float32)
            twist = np.zeros(6, dtype=np.float32)
            twist[:3] = delta_xyz / max(float(self.action_scale[0]), 1e-6)
            twist[3:6] = delta_rpy / max(float(self.action_scale[1]), 1e-6)
            if frame == "base" and self.use_relative_frame:
                from rlinf.envs.realworld.franka.utils import construct_adjoint_matrix

                twist = (
                    np.linalg.inv(construct_adjoint_matrix(self._raw_tcp_pose()))
                    @ twist
                )
            elif frame not in {"base", "eef"}:
                raise ValueError("frame must be 'base' or 'eef'")
            action[: min(6, self.action_dim)] = twist[: min(6, self.action_dim)]
            if gripper is not None and self.action_dim >= 7:
                action[-1] = float(gripper)
            result = self.env.step(action[None, :])
            self.last_obs = result[0]
            return result

        def move_delta(self, delta_xyz: Any) -> dict[str, Any]:
            requested = np.asarray(delta_xyz, dtype=np.float32)
            start = self._raw_tcp_pose()
            target = start[:3] + requested
            deadline = time.time() + self.controller["move_timeout_s"]
            max_iterations = max(
                self.controller["min_iterations"],
                int(np.ceil(np.max(np.abs(requested)) / self.action_scale[0]))
                * self.controller["iteration_multiplier"],
            )
            iterations = 0
            while iterations < max_iterations and time.time() < deadline:
                remaining = target - self._raw_tcp_pose()[:3]
                if np.linalg.norm(remaining) <= self.controller["move_tolerance_m"]:
                    break
                step = np.clip(
                    remaining,
                    -float(self.action_scale[0]),
                    float(self.action_scale[0]),
                )
                self._step_delta(step, np.zeros(3, dtype=np.float32), frame="base")
                iterations += 1
            final = self._raw_tcp_pose()
            error = float(np.linalg.norm(target - final[:3]))
            return {
                "ok": error <= self.controller["move_tolerance_m"],
                "requested_delta_xyz_base": requested.tolist(),
                "start_tcp_pose": start.tolist(),
                "final_tcp_pose": final.tolist(),
                "final_error_m": error,
                "steps_used": iterations,
            }

        def rotate_delta(self, delta_rpy: Any) -> dict[str, Any]:
            requested = np.asarray(delta_rpy, dtype=np.float32)
            start = self._raw_tcp_pose()
            target_rotation = Rotation.from_euler(
                "xyz", requested
            ) * Rotation.from_quat(start[3:])
            deadline = time.time() + self.controller["rotate_timeout_s"]
            max_iterations = max(
                self.controller["min_iterations"],
                int(np.ceil(np.linalg.norm(requested) / self.action_scale[1]))
                * self.controller["iteration_multiplier"],
            )
            iterations = 0
            error = float("inf")
            while iterations < max_iterations and time.time() < deadline:
                current = Rotation.from_quat(self._raw_tcp_pose()[3:])
                error_eef = current.inv().apply(
                    (target_rotation * current.inv()).as_rotvec()
                )
                error = float(np.linalg.norm(error_eef))
                if error <= self.controller["rotate_tolerance_rad"]:
                    break
                max_step = float(self.action_scale[1])
                if error > max_step:
                    error_eef *= max_step / error
                step_rpy = Rotation.from_rotvec(error_eef).as_euler("xyz")
                self._step_delta(
                    np.zeros(3, dtype=np.float32),
                    step_rpy.astype(np.float32),
                    frame="eef",
                )
                iterations += 1
            final = self._raw_tcp_pose()
            return {
                "ok": error <= self.controller["rotate_tolerance_rad"],
                "requested_delta_rpy_base": requested.tolist(),
                "start_tcp_pose": start.tolist(),
                "final_tcp_pose": final.tolist(),
                "final_error_rad": error,
                "steps_used": iterations,
            }

        def set_gripper(self, *, open: bool) -> dict[str, Any]:
            if self.action_dim < 7:
                return {"ok": False, "error": "action space has no gripper dimension"}
            deadline = time.time() + self.controller["gripper_timeout_s"]
            command = 1.0 if open else -1.0
            iterations = 0
            reached = False
            while (
                iterations < self.controller["gripper_max_iterations"]
                and time.time() < deadline
            ):
                self._step_delta(
                    np.zeros(3, dtype=np.float32),
                    np.zeros(3, dtype=np.float32),
                    frame="base",
                    gripper=command,
                )
                time.sleep(self.controller["gripper_settle_s"])
                state = _to_numpy_tree(self._raw_state())
                reached = bool(state.get("gripper_open")) == bool(open)
                iterations += 1
                if reached:
                    break
            return {
                "ok": reached,
                "target_gripper_open": bool(open),
                "steps_used": iterations,
                "robot_state": self.get_robot_state(),
            }

        def chunk_step(
            self,
            actions: Any,
            *,
            return_all_frames: bool = False,
        ) -> dict[str, Any]:
            observations = []
            terminated = False
            truncated = False
            last_info: Any = None
            for action in np.asarray(actions, dtype=np.float32):
                observation, _reward, term, trunc, info = self.env.step(action[None, :])
                self.last_obs = observation
                observations.append(self.get_observation())
                terminated = terminated or bool(np.asarray(_to_numpy_tree(term)).any())
                truncated = truncated or bool(np.asarray(_to_numpy_tree(trunc)).any())
                last_info = _to_numpy_tree(info)
                if terminated or truncated:
                    break
            return {
                "observation": observations if return_all_frames else observations[-1],
                "terminated": terminated,
                "truncated": truncated,
                "info": last_info,
            }

    return FrankaEnvWorker


def _launch_worker(
    cfg: Any,
    controller_config: dict[str, Any],
    *,
    create_worker_class: Callable[[], Any],
):
    from rlinf.scheduler import Cluster, ComponentPlacement

    worker_class = create_worker_class()
    cluster = Cluster(cluster_cfg=cfg.cluster)
    placement = ComponentPlacement(cfg, cluster).get_strategy("env")
    worker = worker_class.create_group(cfg, controller_config).launch(
        cluster=cluster,
        name="FrankaRPentEnvGroup",
        placement_strategy=placement,
    )
    worker.get_env_meta().wait()
    return worker


def main(
    *,
    create_worker_class: Callable[[], Any],
    load_runtime_config: Callable[..., Any],
) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--transport", choices=["http", "socket"], default="http")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--robot-config", default=None)
    parser.add_argument("--task-description", required=True)
    parser.add_argument("--parent-watch", action="store_true")
    parser.add_argument(
        "--print-config",
        action="store_true",
        help="Print the resolved RLinf config and exit without launching.",
    )
    args = parser.parse_args()

    runtime = load_runtime_config(
        args.robot_config,
        task_description=args.task_description,
    )
    if args.print_config:
        from omegaconf import OmegaConf

        print(OmegaConf.to_yaml(runtime.rlinf))
        return 0
    worker = _launch_worker(
        runtime.rlinf, runtime.controller, create_worker_class=create_worker_class
    )
    facade = FrankaEnvFacade(_RayBackend(worker))
    try:
        facade.serve(
            transport=args.transport,
            host=args.host,
            port=args.port,
            parent_watch=args.parent_watch,
        )
    finally:
        worker.close_env().wait()
    return 0


if __name__ == "__main__":
    raise SystemExit(
        main(
            create_worker_class=_create_worker_class,
            load_runtime_config=load_runtime_config,
        )
    )
