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

"""Internal BEHAVIOR Env facade and an official-policy bridge service.

The service forwards policy observations and actions; the separately launched
official evaluator owns simulation, reset and task scoring.
"""

from __future__ import annotations

import argparse
import asyncio
import math
import signal
import threading
from collections.abc import Callable
from typing import Any

import numpy as np

from robots.behavior.env_client import (
    env_runtime_contract,
    validate_native_actions,
)
from robots.behavior.tasks import TASK_NAMES
from rpent.robots.components.env_facade_base import BaseEnvFacade
from rpent.utils.logging import get_logger

logger = get_logger("behavior_env_server")


class BehaviorEnvFacade(BaseEnvFacade):
    """Expose a borrowed backend through RPent's standard Env RPC routes.

    Args:
        backend: Bridge providing observation, session-status and motion-snapshot
            readers, and ``chunk_step`` Gym 5-tuples. Unknown reward and terminal
            feedback must remain None.
        task_id: BEHAVIOR task served by this single-environment adapter.
        on_shutdown: Optional owner callback that cancels bridge waits before
            the standard shutdown route acquires the Env dispatch lock. A
            borrowed backend is not closed unless its owner supplies this.
        motion_planner: Optional borrowed planner implementing
            ``plan_arm_path(arm=..., target_pose=..., observation=...)``.
            It receives public observations, never the simulator or task state.
        motion_model: Lazy access to the static model owned by the launcher.
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
                "the launcher must construct and own the evaluator bridge"
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
        """Read static limits without requiring CuRobo or advancing simulation."""
        if self._motion_model is None:
            raise RuntimeError("BEHAVIOR Env has no motion model provider")
        model = self._motion_model()
        return {arm: model.joint_limits(arm).tolist() for arm in ("left", "right")}

    def get_env_meta(self) -> dict[str, Any]:
        """Return the internal adapter contract, not official protocol metadata."""
        return env_runtime_contract(self._task_id)

    def reset(self) -> None:
        """Reject autonomous reset; the official evaluator owns reset."""
        raise RuntimeError("BEHAVIOR reset is owned by the official evaluator")

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

    def get_task_language(self) -> str | None:
        """Return the task instruction supplied by the launcher."""
        return self._task_language


async def _serve(args: argparse.Namespace) -> None:
    from robots.behavior.eval.evaluator import EvaluatorStepBridge, serve_websocket

    bridge = EvaluatorStepBridge(
        simulation_timeout_s=args.simulation_timeout_s,
        action_timeout_s=args.action_timeout_s,
        chunk_timeout_s=args.chunk_timeout_s,
    )
    motion_planner = None
    motion_model = None

    def get_motion_model() -> Any:
        nonlocal motion_model
        if motion_model is None:
            from robots.behavior.motion import R1ProMotionModel

            motion_model = R1ProMotionModel()
        return motion_model

    if args.motion_backend == "curobo":
        from robots.behavior.motion import R1ProMotionPlanner

        motion_planner = R1ProMotionPlanner(
            model=get_motion_model(),
            task_id=args.task_id,
            device=args.motion_device,
        )
    facade = BehaviorEnvFacade(
        bridge,
        task_id=args.task_id,
        task_language=args.task_language,
        on_shutdown=bridge.stop,
        motion_planner=motion_planner,
        motion_model=get_motion_model,
    )
    errors: list[BaseException] = []

    def serve_rpc() -> None:
        try:
            facade.serve(
                transport=args.transport,
                host=args.host,
                port=args.port,
                parent_watch=args.parent_watch,
            )
        except BaseException as error:
            errors.append(error)
            facade._shutdown_event.set()

    def stop_owned() -> None:
        bridge.stop()
        facade._shutdown_event.set()

    loop = asyncio.get_running_loop()
    worker = threading.Thread(target=serve_rpc, name="behavior-env-rpc", daemon=True)
    installed_signals = []
    try:
        async with serve_websocket(
            bridge,
            host=args.policy_host,
            port=args.policy_port,
            task_id=args.task_id,
            metadata={
                "robot": "behavior",
                "task_language": args.task_language,
                "termination_owner": "official_evaluator",
            },
        ) as server:
            for signum in (signal.SIGTERM, signal.SIGINT):
                loop.add_signal_handler(signum, stop_owned)
                installed_signals.append(signum)
            worker.start()
            task = asyncio.create_task(server.serve_forever())
            try:
                while not facade._shutdown_event.is_set() and not task.done():
                    await asyncio.sleep(0.1)
                if task.done():
                    await task
            finally:
                stop_owned()
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
    finally:
        stop_owned()
        if worker.ident is not None:
            await asyncio.to_thread(worker.join, 5.0)
        for signum in installed_signals:
            loop.remove_signal_handler(signum)
        if worker.is_alive():
            raise RuntimeError("BEHAVIOR Env RPC thread did not stop")
        if motion_model is not None:
            motion_model.close()
    if errors:
        raise RuntimeError("BEHAVIOR Env RPC service failed") from errors[0]


def main(argv: list[str] | None = None) -> None:
    """Parse configuration and serve Env RPC plus the official WebSocket."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task-id", type=int, choices=tuple(TASK_NAMES), metavar="0..99", required=True
    )
    parser.add_argument("--task-language", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--transport", choices=("socket", "http"), default="socket")
    parser.add_argument("--policy-host", default="127.0.0.1")
    parser.add_argument("--policy-port", type=int, required=True)
    parser.add_argument("--simulation-timeout-s", type=float, default=600.0)
    parser.add_argument("--action-timeout-s", type=float, default=180.0)
    parser.add_argument("--chunk-timeout-s", type=float, default=1200.0)
    parser.add_argument("--parent-watch", action="store_true")
    parser.add_argument("--motion-backend", choices=("none", "curobo"), default="none")
    parser.add_argument("--motion-device", default="cuda:0")
    args = parser.parse_args(argv)
    if not args.task_language.strip():
        parser.error("--task-language must not be empty")
    if not all(0 < port < 65536 for port in (args.port, args.policy_port)):
        parser.error("ports must be between 1 and 65535")
    if args.port == args.policy_port:
        parser.error("internal Env and official policy services require distinct ports")
    for name in ("simulation_timeout_s", "action_timeout_s", "chunk_timeout_s"):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive and finite")
    if args.chunk_timeout_s < args.simulation_timeout_s:
        parser.error("chunk timeout must cover the simulation inactivity timeout")
    logger.info("Starting BEHAVIOR bridge for task %s", args.task_id)
    asyncio.run(_serve(args))


if __name__ == "__main__":
    main()
