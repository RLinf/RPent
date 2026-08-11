"""RPC server owning one RLinf dual-Franka ``RealWorldEnv`` worker."""

from __future__ import annotations

import argparse
import dataclasses
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from rpent.utils.config import get_rlinf_repo_path
from rpent.utils.logging import get_logger
from rpent.utils.rpc import RpcFacade

RPENT_ROOT = Path(__file__).resolve().parents[2]
logger = get_logger("dual_franka_env_server")
DEFAULT_CONTROLLER_CONFIG = Path(__file__).with_name("controller_config.yaml")

_ARM_INDEX = {"left": 0, "right": 1}


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


def _matrix_to_rot6d(matrix: np.ndarray) -> np.ndarray:
    """Encode a 3x3 rotation as its first two columns (RLinf rot6d convention)."""
    matrix = np.asarray(matrix, dtype=np.float32)
    return np.concatenate([matrix[:, 0], matrix[:, 1]]).astype(np.float32)


def _pack_dual_action(
    left_xyz: np.ndarray,
    left_rot6d: np.ndarray,
    right_xyz: np.ndarray,
    right_rot6d: np.ndarray,
    *,
    left_grip: float = 0.0,
    right_grip: float = 0.0,
) -> np.ndarray:
    """Assemble a 20-D ``[L_xyz, L_rot6d, L_grip, R_xyz, R_rot6d, R_grip]`` action."""
    left = np.concatenate(
        [
            np.asarray(left_xyz, dtype=np.float32),
            np.asarray(left_rot6d, dtype=np.float32),
            np.array([left_grip], dtype=np.float32),
        ]
    )
    right = np.concatenate(
        [
            np.asarray(right_xyz, dtype=np.float32),
            np.asarray(right_rot6d, dtype=np.float32),
            np.array([right_grip], dtype=np.float32),
        ]
    )
    return np.concatenate([left, right]).astype(np.float32)


class DualFrankaEnvFacade(RpcFacade):
    """Expose explicit ``env.*`` methods from a local or Ray-backed worker."""

    _METHODS = {
        "ready",
        "reset",
        "get_robot_state",
        "get_observation",
        "get_camera_metadata",
        "move_delta",
        "rotate_delta",
        "set_gripper",
        "step_chunk",
    }

    def __init__(self, backend: Any) -> None:
        super().__init__()
        self._backend = backend

    def _dispatch(self, method: str, args: tuple, kwargs: dict) -> Any:
        if not method.startswith("env."):
            raise ValueError(f"unknown RPC method: {method!r}")
        name = method.removeprefix("env.")
        if name not in self._METHODS:
            raise ValueError(f"unknown dual-Franka env method: {name!r}")
        return getattr(self._backend, name)(*args, **kwargs)


class _RayBackend:
    """Turn RLinf Ray method handles into a synchronous facade backend."""

    def __init__(self, worker: Any) -> None:
        self.worker = worker

    def __getattr__(self, name: str):
        remote_method = getattr(self.worker, name)

        def invoke(*args, **kwargs):
            return remote_method(*args, **kwargs).wait()[0]

        return invoke


def _load_controller_config(path: str | Path | None) -> dict[str, Any]:
    from omegaconf import OmegaConf

    config_path = Path(path or DEFAULT_CONTROLLER_CONFIG).expanduser().resolve()
    raw = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    if not isinstance(raw, dict):
        raise ValueError(f"controller config must be a mapping: {config_path}")
    return {
        "move_timeout_s": float(raw["move"]["timeout_s"]),
        "move_tolerance_m": float(raw["move"]["tolerance_m"]),
        "move_max_step_m": float(raw["move"]["max_step_m"]),
        "rotate_timeout_s": float(raw["rotate"]["timeout_s"]),
        "rotate_tolerance_rad": float(raw["rotate"]["tolerance_rad"]),
        "rotate_max_step_rad": float(raw["rotate"]["max_step_rad"]),
        "iteration_multiplier": int(raw["servo"]["iteration_multiplier"]),
        "min_iterations": int(raw["servo"]["min_iterations"]),
        "gripper_settle_s": float(raw["gripper"]["settle_s"]),
        "gripper_timeout_s": float(raw["gripper"]["timeout_s"]),
        "gripper_max_iterations": int(raw["gripper"]["max_iterations"]),
    }


def _create_worker_class():
    """Build the Worker subclass only inside the RLinf server environment."""
    from rlinf.envs.realworld.realworld_env import RealWorldEnv
    from rlinf.scheduler import Worker
    from scipy.spatial.transform import Rotation as Rotation

    class DualFrankaEnvWorker(Worker):
        """Ray worker that owns both physical Franka arms and camera resources."""

        def __init__(self, cfg: Any, controller_config_path: str | None = None):
            super().__init__()
            self.cfg = cfg
            self.controller = _load_controller_config(controller_config_path)
            self.env = RealWorldEnv(
                cfg.env.eval,
                num_envs=1,
                seed_offset=0,
                total_num_processes=1,
                worker_info=self.worker_info,
            )
            self.action_dim = int(self.env.action_space.shape[-1])
            self.per_arm_dim = int(
                self.env.env.call("get_wrapper_attr", "PER_ARM_ACTION_DIM")[0]
            )
            self.gripper_idx = int(
                self.env.env.call("get_wrapper_attr", "GRIPPER_IDX_IN_ARM")[0]
            )
            if self.per_arm_dim != 10 or self.action_dim != 2 * self.per_arm_dim:
                raise ValueError(
                    "dual-Franka RPent bridge requires the TCP rot6d env "
                    f"(20-D); got action_dim={self.action_dim}, "
                    f"per_arm_dim={self.per_arm_dim}"
                )
            env_config = self.env.env.call("get_wrapper_attr", "config")[0]
            self.action_scale = np.asarray(env_config.action_scale, dtype=np.float32)
            self.last_obs: dict[str, Any] | None = None

        # ------------------------------------------------------------ lifecycle

        def ready(self) -> dict[str, Any]:
            return {
                "ok": True,
                "action_dim": self.action_dim,
                "per_arm_dim": self.per_arm_dim,
                "action_scale": self.action_scale.tolist(),
                "arms": ["left", "right"],
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

        # --------------------------------------------------------- observation

        def _ensure_obs(self) -> dict[str, Any]:
            if self.last_obs is None:
                observation, _ = self.env.reset()
                self.last_obs = observation
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
            value = output.get("extra_view_images")
            if (
                isinstance(value, np.ndarray)
                and value.ndim == 5
                and value.shape[0] == 1
            ):
                output["extra_view_images"] = value[0]
            return output

        # --------------------------------------------------------- arm state

        def _arm_states(self) -> tuple[Any, Any]:
            left = self.env.env.call("get_wrapper_attr", "_left_state")[0]
            right = self.env.env.call("get_wrapper_attr", "_right_state")[0]
            return left, right

        def _arm_poses(self) -> tuple[np.ndarray, np.ndarray]:
            left, right = self._arm_states()
            return (
                np.asarray(left.tcp_pose, dtype=np.float32),
                np.asarray(right.tcp_pose, dtype=np.float32),
            )

        def _arm_rot6d(self, pose: np.ndarray) -> np.ndarray:
            return _matrix_to_rot6d(Rotation.from_quat(pose[3:]).as_matrix())

        def _hold_action(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
            return _pack_dual_action(
                left[:3],
                self._arm_rot6d(left),
                right[:3],
                self._arm_rot6d(right),
            )

        def get_robot_state(self) -> dict[str, Any]:
            left, right = self._arm_states()
            return {
                "left_arm": _to_numpy_tree(left),
                "right_arm": _to_numpy_tree(right),
                "action_dim": self.action_dim,
                "per_arm_dim": self.per_arm_dim,
                "action_scale": self.action_scale.tolist(),
            }

        def get_camera_metadata(self) -> dict[str, Any] | None:
            try:
                specs_getter = self.env.env.call(
                    "get_wrapper_attr", "_all_camera_specs"
                )[0]
                specs = list(specs_getter())
            except Exception as exc:
                return {"error": str(exc), "error_type": type(exc).__name__}
            cameras = {
                name: {"serial": serial, "type": camera_type}
                for name, serial, camera_type in specs
            }
            main_key = self.cfg.env.eval.get("main_image_key")
            extras = [name for name, _, _ in specs if name != main_key]
            return {
                "cameras": cameras,
                "observation_camera_map": {
                    "main": main_key,
                    **{f"extra_{index}": name for index, name in enumerate(extras)},
                },
            }

        # --------------------------------------------------------- primitives

        def _arm_index(self, arm: str) -> int:
            index = _ARM_INDEX.get(str(arm).strip().lower())
            if index is None:
                raise ValueError("arm must be 'left' or 'right'")
            return index

        def move_delta(self, arm: str, delta_xyz: Any) -> dict[str, Any]:
            arm_idx = self._arm_index(arm)
            requested = np.asarray(delta_xyz, dtype=np.float32)
            left, right = self._arm_poses()
            start = left if arm_idx == 0 else right
            target_xyz = start[:3] + requested
            max_step = self.controller["move_max_step_m"]
            deadline = time.time() + self.controller["move_timeout_s"]
            max_iterations = max(
                self.controller["min_iterations"],
                int(np.ceil(np.max(np.abs(requested)) / max_step))
                * self.controller["iteration_multiplier"],
            )
            iterations = 0
            while iterations < max_iterations and time.time() < deadline:
                left, right = self._arm_poses()
                current = left if arm_idx == 0 else right
                remaining = target_xyz - current[:3]
                if np.linalg.norm(remaining) <= self.controller["move_tolerance_m"]:
                    break
                step_xyz = np.clip(remaining, -max_step, max_step)
                action = self._hold_action(left, right)
                base = arm_idx * self.per_arm_dim
                action[base : base + 3] = current[:3] + step_xyz
                self.env.step(action[None, :])
                iterations += 1
            left, right = self._arm_poses()
            final = left if arm_idx == 0 else right
            error = float(np.linalg.norm(target_xyz - final[:3]))
            return {
                "ok": error <= self.controller["move_tolerance_m"],
                "arm": ["left", "right"][arm_idx],
                "requested_delta_xyz": requested.tolist(),
                "start_tcp_pose": start.tolist(),
                "final_tcp_pose": final.tolist(),
                "final_error_m": error,
                "steps_used": iterations,
            }

        def rotate_delta(self, arm: str, delta_rpy: Any) -> dict[str, Any]:
            arm_idx = self._arm_index(arm)
            requested = np.asarray(delta_rpy, dtype=np.float32)
            left, right = self._arm_poses()
            start = left if arm_idx == 0 else right
            start_rot = Rotation.from_quat(start[3:])
            target_rot = Rotation.from_euler("xyz", requested) * start_rot
            max_step = self.controller["rotate_max_step_rad"]
            deadline = time.time() + self.controller["rotate_timeout_s"]
            max_iterations = max(
                self.controller["min_iterations"],
                int(np.ceil(np.linalg.norm(requested) / max_step))
                * self.controller["iteration_multiplier"],
            )
            iterations = 0
            error = float("inf")
            while iterations < max_iterations and time.time() < deadline:
                left, right = self._arm_poses()
                current = left if arm_idx == 0 else right
                current_rot = Rotation.from_quat(current[3:])
                error_rotvec = (target_rot * current_rot.inv()).as_rotvec()
                error = float(np.linalg.norm(error_rotvec))
                if error <= self.controller["rotate_tolerance_rad"]:
                    break
                if error > max_step:
                    error_rotvec = error_rotvec * (max_step / error)
                step_rot = Rotation.from_rotvec(error_rotvec) * current_rot
                action = self._hold_action(left, right)
                base = arm_idx * self.per_arm_dim
                action[base + 3 : base + 9] = _matrix_to_rot6d(step_rot.as_matrix())
                self.env.step(action[None, :])
                iterations += 1
            left, right = self._arm_poses()
            final = left if arm_idx == 0 else right
            return {
                "ok": error <= self.controller["rotate_tolerance_rad"],
                "arm": ["left", "right"][arm_idx],
                "requested_delta_rpy": requested.tolist(),
                "start_tcp_pose": start.tolist(),
                "final_tcp_pose": final.tolist(),
                "final_error_rad": error,
                "steps_used": iterations,
            }

        def set_gripper(self, arm: str, *, open: bool) -> dict[str, Any]:
            arm_idx = self._arm_index(arm)
            deadline = time.time() + self.controller["gripper_timeout_s"]
            command = 1.0 if open else -1.0
            iterations = 0
            reached = False
            while (
                iterations < self.controller["gripper_max_iterations"]
                and time.time() < deadline
            ):
                left, right = self._arm_poses()
                action = self._hold_action(left, right)
                action[arm_idx * self.per_arm_dim + self.gripper_idx] = command
                self.env.step(action[None, :])
                time.sleep(self.controller["gripper_settle_s"])
                left_state, right_state = self._arm_states()
                state = left_state if arm_idx == 0 else right_state
                reached = bool(getattr(state, "gripper_open")) == bool(open)
                iterations += 1
                if reached:
                    break
            return {
                "ok": reached,
                "arm": ["left", "right"][arm_idx],
                "target_gripper_open": bool(open),
                "steps_used": iterations,
                "robot_state": self.get_robot_state(),
            }

        def step_chunk(
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

    return DualFrankaEnvWorker


def _compose_config(rlinf_root: Path, config_name: str, overrides: list[str]):
    import hydra

    config_dir = rlinf_root / "examples" / "embodiment" / "config"
    previous = os.environ.get("EMBODIED_PATH")
    os.environ["EMBODIED_PATH"] = str(config_dir.parent)
    try:
        with hydra.initialize_config_dir(
            version_base="1.1",
            config_dir=str(config_dir),
        ):
            return hydra.compose(config_name=config_name, overrides=overrides)
    finally:
        if previous is None:
            os.environ.pop("EMBODIED_PATH", None)
        else:
            os.environ["EMBODIED_PATH"] = previous


def _launch_worker(cfg: Any, controller_config: str | None):
    from rlinf.scheduler import Cluster, ComponentPlacement

    worker_class = _create_worker_class()
    cluster = Cluster(cluster_cfg=cfg.cluster)
    placement = ComponentPlacement(cfg, cluster).get_strategy("env")
    worker = worker_class.create_group(cfg, controller_config).launch(
        cluster=cluster,
        name="DualFrankaRPentEnvGroup",
        placement_strategy=placement,
    )
    worker.ready().wait()
    return worker


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--transport", choices=["http", "socket"], default="http")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument(
        "--config-name", default="realworld_physical_agent_eval_dual_franka"
    )
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--controller-config", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--parent-watch", action="store_true")
    args = parser.parse_args()

    del args.output_dir
    rlinf_root = get_rlinf_repo_path() or (RPENT_ROOT.parent / "rlinf").resolve()
    if str(rlinf_root) not in sys.path:
        sys.path.insert(0, str(rlinf_root))
    cfg = _compose_config(rlinf_root, args.config_name, list(args.override))
    worker = _launch_worker(cfg, args.controller_config)
    facade = DualFrankaEnvFacade(_RayBackend(worker))
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
    raise SystemExit(main())
