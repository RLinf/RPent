"""RPC server owning one RLinf RoboTwin environment."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

# Support direct execution from an RPent checkout before package imports.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rpent.tools.env_facade_base import BaseEnvFacade
from rpent.utils.logging import get_logger

logger = get_logger("robotwin_env_server")

from omegaconf import OmegaConf  # noqa: E402
from robotwin.assets import validate_root  # noqa: E402
from robotwin.config import load_task_config  # noqa: E402

from robots.robotwin.robot_spec import RoboTwinActionType  # noqa: E402
from robots.robotwin.rlinf_env import RoboTwinAgentEnv  # noqa: E402


def _to_numpy_tree(value: Any) -> Any:
    """Convert RPC results to numpy arrays and plain Python values."""
    if hasattr(value, "detach") and hasattr(value, "cpu") and hasattr(value, "numpy"):
        return value.detach().cpu().numpy()
    if isinstance(value, dict):
        return {key: _to_numpy_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_numpy_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_to_numpy_tree(item) for item in value)
    if isinstance(value, np.generic):
        return value.item()
    return value


class RoboTwinEnvFacade(BaseEnvFacade):
    """Expose common and typed RoboTwin environment RPC contracts."""

    def __init__(self, env: Any, *, metadata: dict[str, Any]):
        self._env = env
        self._metadata = dict(metadata)
        super().__init__()

    def _strip_single_env_value(self, value: Any, name: str) -> Any:
        """Remove one leading single-environment batch dimension."""
        if value is None:
            return None
        if hasattr(value, "shape"):
            shape = tuple(value.shape)
            if not shape or shape[0] != 1:
                raise RuntimeError(f"{name} must have leading env dimension 1, got {shape}")
            return value[0]
        if isinstance(value, list):
            if len(value) != 1:
                raise RuntimeError(f"{name} must contain one environment, got {len(value)}")
            return value[0]
        raise RuntimeError(f"{name} is missing a single-environment batch dimension")

    def _strip_single_env_observation(self, observation: Any) -> dict[str, Any]:
        if not isinstance(observation, dict):
            raise TypeError(f"RoboTwin observation must be a mapping, got {observation!r}")
        return {
            key: self._strip_single_env_value(value, f"observation.{key}")
            for key, value in observation.items()
        }

    def _strip_single_signal(self, value: Any, name: str) -> Any:
        if hasattr(value, "detach") and hasattr(value, "cpu") and hasattr(value, "numpy"):
            value = value.detach().cpu().numpy()
        array = np.asarray(value).reshape(-1)
        if array.size != 1:
            raise RuntimeError(f"{name} must contain one environment, got {array.shape}")
        return array[0].item()

    def _register_rpc(self) -> None:
        self._rpc.update(
            {
                "env.get_env_meta": self.get_env_meta,
                "env.reset": self.reset,
                "env.step": self.step,
                "env.chunk_step": self.chunk_step,
                "env.render_camera": self.render_camera,
                "env.get_camera_meta": self.get_camera_meta,
                "env.get_task_language": self.get_task_language,
                "env.plan_arm_path": self.plan_arm_path,
            }
        )
        self._readonly_methods.update(
            {
                "env.get_env_meta",
                "env.render_camera",
                "env.get_camera_meta",
                "env.get_task_language",
            }
        )

    def get_env_meta(self) -> dict[str, Any]:
        """Return immutable identity for endpoint compatibility checks."""
        return dict(self._metadata)

    def reset(self) -> tuple[dict[str, Any], dict[str, Any]]:
        seed = int(self._metadata["seed"])
        observation, info = self._env.reset(env_idx=[0], env_seeds=[seed])
        episode_status = info["episode_status"]
        if episode_status["actual_seed"] != seed:
            raise RuntimeError(
                "RoboTwin exact seed mismatch: "
                f"requested {seed}, initialized {episode_status['actual_seed']}"
            )
        instruction = self._env.get_task_language(0)
        info["requested_seed"] = seed
        info["instruction"] = instruction
        return self._strip_single_env_observation(observation), info

    def step(
        self, action, *, action_type: RoboTwinActionType = "qpos"
    ) -> tuple[Any, Any, Any, Any, dict[str, Any]]:
        expected_dim = 14 if action_type == "qpos" else 16
        array = np.asarray(action, dtype=np.float64)
        if array.shape != (expected_dim,):
            raise ValueError(
                f"RoboTwin common action must have shape ({expected_dim},)"
            )
        if not np.isfinite(array).all():
            raise ValueError("RoboTwin common action must contain only finite values")
        observation, reward, terminated, truncated, info = self._env.step(
            array, action_type=action_type
        )
        return (
            self._strip_single_env_observation(observation),
            self._strip_single_signal(reward, "reward"),
            bool(self._strip_single_signal(terminated, "terminated")),
            bool(self._strip_single_signal(truncated, "truncated")),
            info,
        )

    def chunk_step(
        self,
        actions,
        *,
        action_type: RoboTwinActionType = "qpos",
        return_all_frames: bool = False,
    ) -> tuple[Any, Any, Any, Any, dict[str, Any]]:
        expected_dim = 14 if action_type == "qpos" else 16
        array = np.asarray(actions, dtype=np.float64)
        if array.ndim != 2 or array.shape[0] < 1 or array.shape[1] != expected_dim:
            raise ValueError(
                f"RoboTwin common actions must have shape [N,{expected_dim}], N >= 1"
            )
        if not np.isfinite(array).all():
            raise ValueError("RoboTwin common actions must contain only finite values")
        observation_list, rewards, terminated, truncated, info_list = (
            self._env.chunk_step(
                array, action_type=action_type, return_all_frames=return_all_frames
            )
        )
        if len(observation_list) != 1 or len(info_list) != 1:
            raise RuntimeError(
                "RoboTwin chunk_step must return one environment, got "
                f"{len(observation_list)} obs / {len(info_list)} info"
            )
        observation = observation_list[0]
        if return_all_frames:
            observation = {
                "frames": observation["frames"],
                "final": self._strip_single_env_observation(observation["final"]),
            }
        else:
            observation = self._strip_single_env_observation(observation)
        return (
            observation,
            np.asarray(rewards)[0],
            np.asarray(terminated, dtype=bool)[0],
            np.asarray(truncated, dtype=bool)[0],
            info_list[0],
        )

    def render_camera(self, camera_name: str, depth: bool = False) -> Any:
        return self._env.render_camera(camera_name, depth=depth, env_id=0)

    def get_camera_meta(self, camera_name: str) -> dict[str, Any]:
        return self._env.get_camera_meta(camera_name, env_id=0)

    def get_task_language(self) -> str:
        return self._env.get_task_language(env_id=0)

    def plan_arm_path(self, arm: str, target_pose) -> dict[str, Any]:
        return self._env.plan_arm_path(0, arm, target_pose)

    def _dispatch(self, method: str, args: tuple, kwargs: dict) -> Any:
        return _to_numpy_tree(super()._dispatch(method, args, kwargs))


def build_env_cfg(
    *,
    task_name: str,
    task_config: str,
    seed: int,
    assets_path: str,
    max_episode_steps: int = 10000,
) -> Any:
    """Build a single-env RLinf config from packaged RoboTwin resources."""
    native_task_config = OmegaConf.create(load_task_config(task_config))
    step_limit = int(max_episode_steps)
    native_task_config.task_name = task_name
    native_task_config.task_config = task_config
    native_task_config.step_lim = step_limit
    native_task_config.ckpt_setting = "hybrid_lingbot"
    native_task_config.policy_name = "hybrid_lingbot"
    native_task_config.planner_backend = "curobo"
    native_task_config.eval_video_log = False
    native_task_config.render_freq = 0

    return OmegaConf.create(
        {
            "env_type": "robotwin",
            "initial_env_seeds": [int(seed)],
            "auto_reset": False,
            "ignore_terminations": False,
            "reward_coef": 1.0,
            "use_custom_reward": True,
            "use_rel_reward": True,
            "center_crop": False,
            "seed": seed,
            "group_size": 1,
            "use_fixed_reset_state_ids": True,
            "max_steps_per_rollout_epoch": step_limit,
            "max_episode_steps": step_limit,
            "is_eval": True,
            "assets_path": assets_path,
            "seeds_path": None,
            "video_cfg": {
                "save_video": False,
                "info_on_video": False,
                "video_base_dir": None,
            },
            "enable_offload": False,
            "task_config": native_task_config,
        }
    )


def make_env(
    task_name: str,
    task_config: str,
    seed: int,
    assets_path: str,
    max_episode_steps: int = 10000,
) -> RoboTwinAgentEnv:
    """Construct the only simulator owner used by an RPent run."""
    assets_identity = validate_root(assets_path)
    resolved_assets_path = Path(assets_identity["root"])
    os.environ["ROBOTWIN_ASSETS_PATH"] = str(resolved_assets_path)
    os.environ["ROBOTWIN_ASSETS_ROOT"] = str(resolved_assets_path)
    logger.info("RoboTwin assets: %s", assets_identity)
    print(
        f"robotwin_assets {json.dumps(assets_identity, sort_keys=True)}",
        flush=True,
    )
    cfg = build_env_cfg(
        task_name=task_name,
        task_config=task_config,
        seed=seed,
        assets_path=str(resolved_assets_path),
        max_episode_steps=max_episode_steps,
    )
    return RoboTwinAgentEnv(
        cfg=cfg,
        num_envs=1,
        seed_offset=0,
        total_num_processes=1,
        worker_info=None,
        record_metrics=False,
    )


def main() -> None:
    from robots.robotwin.robot_spec import ROBOTWIN_TASK_CONFIGS

    parser = argparse.ArgumentParser()
    parser.add_argument("--transport", choices=["socket", "http"], default="http")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--task-name", required=True)
    parser.add_argument(
        "--task-config",
        choices=ROBOTWIN_TASK_CONFIGS,
        required=True,
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--max-episode-steps",
        type=int,
        default=10000,
        help="Episode action budget for the RPent RoboTwin agent runtime.",
    )
    parser.add_argument("--assets-path", required=True)
    parser.add_argument("--parent-watch", action="store_true")
    args = parser.parse_args()

    env = make_env(
        args.task_name,
        args.task_config,
        args.seed,
        args.assets_path,
        args.max_episode_steps,
    )
    from robots.robotwin.robot_spec import env_runtime_contract

    facade = RoboTwinEnvFacade(
        env,
        metadata=env_runtime_contract(
            task_name=args.task_name,
            task_config=args.task_config,
            seed=args.seed,
            max_episode_steps=args.max_episode_steps,
        ),
    )
    try:
        facade.serve(
            transport=args.transport,
            host=args.host,
            port=args.port,
            parent_watch=args.parent_watch,
        )
    finally:
        env.offload(clear_cache=True)


if __name__ == "__main__":
    main()
