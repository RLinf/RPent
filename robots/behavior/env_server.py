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

"""Serve the RLinf BEHAVIOR environment through main-thread RPent RPC."""

from __future__ import annotations

import argparse
import importlib
import math
import os
import signal
import sys
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from robots.behavior.env_client import (
    env_runtime_contract,
    validate_native_actions,
)
from robots.behavior.tasks import TASK_LANGUAGE, TASK_NAMES
from rpent.evaluation import write_json_atomic
from rpent.robots.components.env_facade_base import BaseEnvFacade
from rpent.utils.logging import get_logger
from rpent.utils.rpc.main_thread_serve import MainThreadServeMixin

logger = get_logger("behavior_env_server")


class BehaviorEnvFacade(MainThreadServeMixin, BaseEnvFacade):
    """Expose a borrowed native environment; its owner closes it after serving.

    All environment and planning methods run on the constructing thread.
    ``on_shutdown`` may only signal cancellation, never call the simulator.
    """

    def __init__(
        self,
        backend: Any = None,
        *,
        task_id: int,
        task_language: str | None = None,
        on_shutdown: Callable[[], None] | None = None,
        motion_planner: Any = None,
        motion_model: Callable[[], Any] | None = None,
    ):
        if backend is None:
            raise RuntimeError(
                "BEHAVIOR requires an injected environment backend; "
                "the launcher must construct and own the RLinf environment"
            )
        self._backend = backend
        self._task_id = task_id
        self._task_language = task_language
        self._on_shutdown = on_shutdown
        self._motion_planner = motion_planner
        self._motion_model = motion_model
        super().__init__()

    def _builtin_dispatch(self, method: str, args: tuple, kwargs: dict) -> Any:
        if method == "shutdown" and self._on_shutdown is not None:
            self._on_shutdown()
        return super()._builtin_dispatch(method, args, kwargs)

    def _register_rpc(self) -> None:
        super()._register_rpc()
        self._rpc["env.get_observation"] = self._backend.get_observation
        self._readonly_methods.add("env.get_observation")
        self._rpc["env.get_motion_snapshot"] = self._backend.get_motion_snapshot
        self._readonly_methods.add("env.get_motion_snapshot")
        self._rpc["env.get_session_status"] = self._backend.session_status
        self._readonly_methods.add("env.get_session_status")
        # Share the action dispatch lock so planning cannot race an Env action.
        self._rpc["env.plan_arm_path"] = self.plan_arm_path
        # Serialize the first lazy model load with other Env operations.
        self._rpc["env.get_arm_joint_limits"] = self.get_arm_joint_limits

    def get_arm_joint_limits(self) -> dict[str, list[list[float]]]:
        """Read native joint limits without loading CuRobo or advancing simulation."""
        if self._motion_model is None:
            raise RuntimeError("BEHAVIOR Env has no motion model provider")
        model = self._motion_model()
        return {arm: model.joint_limits(arm).tolist() for arm in ("left", "right")}

    def get_env_meta(self) -> dict[str, Any]:
        """Return the native RLinf adapter identity and reset ownership."""
        return env_runtime_contract(self._task_id)

    def reset(self) -> tuple:
        """Explicitly reset the owned episode; connections do not reset it."""
        return self._backend.reset()

    def plan_arm_path(
        self,
        arm: str,
        target_pose: Any,
        *,
        expected_observation_token: dict[str, int] | None = None,
        allow_orientation_relax: bool = False,
    ) -> dict[str, Any]:
        """Plan from the current public observation without advancing simulation."""
        if arm not in ("left", "right"):
            raise ValueError("arm must be left or right")
        pose = np.asarray(target_pose)
        if (
            pose.shape != (7,)
            or pose.dtype.kind not in "fiu"
            or not np.isfinite(pose).all()
            or not np.isclose(np.linalg.norm(pose[3:]), 1.0, atol=1e-4, rtol=0)
        ):
            raise ValueError("target_pose must be finite robot-base xyz + unit xyzw")
        if self._motion_planner is None:
            return {
                "status": "Unavailable",
                "code": "motion_planner_unavailable",
                "error": "a CuRobo R1Pro planner is not configured",
            }
        if self._backend.session_status()["ended"] is True:
            return {
                "status": "Unavailable",
                "code": "session_ended",
                "error": "the official episode has ended",
            }
        snapshot = self._backend.get_motion_snapshot()
        if expected_observation_token is not None and (
            expected_observation_token != snapshot["observation_token"]
        ):
            return {
                "status": "Unavailable",
                "code": "stale_observation",
                "error": "public planning frame changed",
            }
        result = self._motion_planner.plan_arm_path(
            arm=arm,
            target_pose=pose.astype(np.float64),
            observation=snapshot["observation"],
            motion_context=snapshot["motion_context"],
            allow_orientation_relax=bool(allow_orientation_relax),
        )
        return {**result, "observation_token": snapshot["observation_token"]}

    def step(self, action: Any) -> tuple:
        """Delegate one action and return backend feedback without synthesis."""
        array = validate_native_actions(action, chunk=False)
        return self._backend.chunk_step(array[None])

    def chunk_step(
        self,
        actions: Any,
        *,
        return_all_frames: bool = False,
        expected_observation_token: dict[str, int] | None = None,
    ) -> tuple:
        """Delegate a chunk; completion is the injected backend's responsibility."""
        kwargs = {"return_all_frames": return_all_frames}
        if expected_observation_token is not None:
            kwargs["expected_observation_token"] = expected_observation_token
        return self._backend.chunk_step(
            validate_native_actions(actions, chunk=True),
            **kwargs,
        )

    def get_camera_meta(self, camera_name: str | None = None) -> dict:
        """Return calibration paired with the latest confirmed frame."""
        return self._backend.get_camera_meta(camera_name)

    def render_camera(self, camera_name: str = "head") -> np.ndarray:
        """Return one camera from the latest confirmed frame."""
        return self._backend.render_camera(camera_name)

    def get_task_language(self) -> str | None:
        """Return the task instruction supplied by the launcher."""
        return self._task_language


def build_parser() -> argparse.ArgumentParser:
    """Expose the RLinf environment configuration and existing RPC options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-config", type=Path, required=True)
    parser.add_argument(
        "--env-api",
        choices=("behavior-env", "simulator"),
        help="Use RLinf BehaviorEnv or the in-process BehaviorSimulator. "
        "Existing simulator checkouts keep their native motion support.",
    )
    parser.add_argument("--task-id", type=int, choices=tuple(TASK_NAMES), required=True)
    parser.add_argument("--task-language")
    parser.add_argument(
        "--instance-split", choices=("train", "public_test", "hidden_test")
    )
    parser.add_argument("--instance-id", type=int)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--seed", type=int)
    for name in ("action-frequency", "rendering-frequency", "physics-frequency"):
        parser.add_argument(f"--{name}", type=float)
    parser.add_argument("--camera-size", type=int)
    parser.add_argument("--torch-num-threads", type=int, default=4)
    parser.add_argument(
        "--env-wrapper", choices=("rgb", "rgbd", "default"), default="rgbd"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--transport", choices=("socket", "http"), default="socket")
    parser.add_argument("--parent-watch", action="store_true")
    parser.add_argument("--motion-backend", choices=("none", "curobo"), default="none")
    parser.add_argument("--motion-device", default="cuda:0")
    return parser


def build_config(args: argparse.Namespace) -> Any:
    """Compose the existing RLinf YAML, then apply explicit operator overrides."""
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    source = args.env_config.expanduser().resolve(strict=True)
    with initialize_config_dir(version_base=None, config_dir=str(source.parent)):
        cfg = compose(config_name=source.stem)
    overrides = {
        "total_num_envs": 1,
        "auto_reset": False,
        "ignore_terminations": False,
        "torch_num_threads": args.torch_num_threads,
        "omni_config.env.automatic_reset": False,
        "omni_config.env.env_wrapper": "rgb"
        if args.env_wrapper == "rgbd"
        else args.env_wrapper,
        "omni_config.task.activity_name": TASK_NAMES[args.task_id],
        "video_cfg.save_video": False,
        "video_cfg.video_base_dir": str(args.output_dir / "video"),
    }
    if args.env_api == "behavior-env":
        if args.instance_split or OmegaConf.select(
            cfg, "omni_config.task.instance_split"
        ):
            raise ValueError(
                "BehaviorEnv selects the split through omni_config.task.activity_instance_dir "
                "in --env-config. Set the path for your split there and omit --instance-split."
            )
        if OmegaConf.select(cfg, "omni_config.robots[0].proprio_obs") is not None:
            raise ValueError(
                "Use the official 2026 proprio layout: remove the robot proprio_obs override."
            )
        overrides.update(use_2026_eval_proprio_and_rooms=True, num_env_subprocess=1)
    optional = {
        "seed": args.seed,
        "max_episode_steps": args.max_steps,
        "omni_config.task.instance_split": args.instance_split,
        "omni_config.task.activity_instance_id": args.instance_id,
    }
    for rate in ("action_frequency", "rendering_frequency", "physics_frequency"):
        optional[f"omni_config.env.{rate}"] = getattr(args, rate)
    overrides.update(
        {key: value for key, value in optional.items() if value is not None}
    )
    if args.camera_size is not None:
        for camera in ("head", "wrist"):
            overrides[f"omni_config.camera.{camera}_resolution"] = [
                args.camera_size
            ] * 2
    for key, value in overrides.items():
        OmegaConf.update(cfg, key, value, force_add=True)
    if args.env_wrapper == "rgbd":
        # RPent uses its segmentation service; native semantic/instance
        # annotators are unnecessary for its RGB-D geometry tools.
        OmegaConf.update(
            cfg,
            "omni_config.robots.0.obs_modalities",
            ["rgb", "proprio", "depth_linear"],
            force_add=True,
        )
    for key in ("max_episode_steps", "torch_num_threads"):
        if int(cfg[key]) <= 0:
            raise ValueError(f"{key} must be positive")
    for name in ("action_frequency", "rendering_frequency", "physics_frequency"):
        value = float(cfg.omni_config.env[name])
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be positive and finite")
    return cfg


def main(argv: list[str] | None = None) -> None:
    """Construct one native environment, serve RPC, and save before SDK shutdown."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if not 0 < args.port < 65536:
        parser.error("--port must be between 1 and 65535")
    if args.camera_size is not None and args.camera_size <= 0:
        parser.error("--camera-size must be positive")
    args.task_language = (
        TASK_LANGUAGE[args.task_id]
        if args.task_language is None
        else args.task_language.strip()
    )
    if not args.task_language:
        parser.error("--task-language must not be empty")
    from omegaconf import OmegaConf

    from robots.behavior.rlinf_env import PublicBehaviorEnv, RlinfBehaviorEnv

    if args.env_api is None:
        package = importlib.import_module("rlinf.envs.sim.behavior")
        args.env_api = (
            "simulator" if hasattr(package, "BehaviorSimulator") else "behavior-env"
        )
    public = args.env_api == "behavior-env"
    if public and args.motion_backend == "curobo":
        parser.error("CuRobo needs the in-process robot exposed by --env-api simulator")
    cfg = build_config(args)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, args.output_dir / "env_config.yaml", resolve=True)
    result_path = args.output_dir / "env-result.json"
    result: dict[str, Any] = {
        "server_status": "starting",
        "task_id": args.task_id,
        "env_api": args.env_api,
    }
    write_json_atomic(result_path, result)
    simulator = None
    env = None
    motion_model = None
    owned_ray = None

    def get_motion_model() -> Any:
        nonlocal motion_model
        if motion_model is None:
            from robots.behavior.motion import R1ProMotionModel

            motion_model = R1ProMotionModel(simulator.get_robot)
        return motion_model

    try:
        if args.motion_backend == "curobo":
            # OG pins Warp 1.12; Kit bundles older modules. Load the CuRobo
            # compatibility namespaces before Kit's importer can mix versions.
            for namespace in ("warp.context", "warp.torch", "warp.types"):
                importlib.import_module(namespace)
        if public:
            import ray
            from rlinf.envs.sim.behavior.behavior_env import BehaviorEnv

            # RLinf creates CPU-scheduled Ray actors that render on the service GPU.
            if not ray.is_initialized():
                gpu_env = {"RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1"}
                if "CUDA_VISIBLE_DEVICES" in os.environ:
                    gpu_env["CUDA_VISIBLE_DEVICES"] = os.environ["CUDA_VISIBLE_DEVICES"]
                ray.init(
                    address="local",
                    num_cpus=2,
                    include_dashboard=False,
                    runtime_env={"env_vars": gpu_env},
                )
                owned_ray = ray
            simulator = BehaviorEnv(
                cfg,
                num_envs=1,
                seed_offset=0,
                total_num_processes=1,
                worker_info=SimpleNamespace(group_world_size=1),
            )
            env = PublicBehaviorEnv(
                simulator, task_id=args.task_id, task_language=args.task_language
            )
        else:
            from rlinf.envs.sim.behavior import BehaviorSimulator

            simulator = BehaviorSimulator(cfg, num_envs=1, seed=int(cfg.seed))
            env = RlinfBehaviorEnv(
                simulator, task_id=args.task_id, task_language=args.task_language
            )
        env.reset()
        motion_planner = None
        if args.motion_backend == "curobo":
            from robots.behavior.motion import R1ProMotionPlanner

            motion_planner = R1ProMotionPlanner(
                model=get_motion_model(),
                task_id=args.task_id,
                device=args.motion_device,
            )
        facade = BehaviorEnvFacade(
            env,
            task_id=args.task_id,
            task_language=args.task_language,
            on_shutdown=env.stop,
            motion_planner=motion_planner,
            motion_model=None if public else get_motion_model,
        )

        def stop_owned(signum: int, frame: Any) -> None:
            env.stop()
            facade._shutdown_event.set()

        previous = {
            sig: signal.signal(sig, stop_owned)
            for sig in (signal.SIGTERM, signal.SIGINT)
        }
        try:
            result["server_status"] = "ready"
            write_json_atomic(result_path, result)
            facade.serve(
                transport=args.transport,
                host=args.host,
                port=args.port,
                parent_watch=args.parent_watch,
            )
            result["server_status"] = "stopped"
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)
    except BaseException as error:
        result.update(server_status="error", error=f"{type(error).__name__}: {error}")
        logger.exception("BEHAVIOR environment service failed")
        raise
    finally:
        # OmniGibson closes the dedicated process. Persist failure/scores and
        # close auxiliary resources first; no Python cleanup can follow it.
        try:
            if env is not None:
                result["session"] = env.session_status()
            if simulator is not None and not public:
                result["metrics"] = simulator.get_metrics()
            if motion_model is not None:
                motion_model.close()
        except Exception as error:
            result.update(
                server_status="error", cleanup_error=f"{type(error).__name__}: {error}"
            )
            logger.exception("BEHAVIOR final state collection failed")
        write_json_atomic(result_path, result)
        sys.stdout.flush()
        sys.stderr.flush()
        try:
            if simulator is not None:
                simulator.close()
        finally:
            if owned_ray is not None:
                owned_ray.shutdown()


if __name__ == "__main__":
    main()
