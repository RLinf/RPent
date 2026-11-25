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

"""RPC client and server launcher for one VLABench Franka environment."""

from __future__ import annotations

import argparse
import sys
import threading
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from robots.vlabench.sim.contracts import check_env_meta, validate_motion_receipt
from rpent.robots.components.env_client_base import BaseEnvClient
from rpent.robots.runtime import try_spawn_server, try_wait_server
from rpent.utils.config import get_repo_root, resolve_cuda_device

if TYPE_CHECKING:
    from rpent.dashboard.events import DashboardEventSink
    from rpent.utils.daemon import ProcessDaemon
    from rpent.utils.rpc import RpcClient

#: Seconds to wait for a launched environment server, which loads the scene
#: and renders before it answers; an external endpoint is already running.
LAUNCHED_SERVER_TIMEOUT_S = 900.0
EXTERNAL_SERVER_TIMEOUT_S = 300.0


class VLABenchEnvClient(BaseEnvClient):
    """Client for one VLABench environment server.

    Motions are dispatched under a one-use token so a cancellation reaches the
    motion that is actually running. An acknowledged cancellation is not motion
    completion: the original receipt is always awaited, and a motion that may
    have executed is never retried.
    """

    _TIMEOUT_S = {
        **BaseEnvClient._TIMEOUT_S,
        "env.reset": 300.0,
        "env.execute_pose": 300.0,
        "env.execute_gripper": 300.0,
        "env.sensor_snapshot": 120.0,
    }

    def __init__(
        self,
        client: RpcClient,
        *,
        task_name: str,
        track: str,
        episode_index: int,
        seed: int,
    ) -> None:
        self.terminated = False
        self.truncated = False
        # The server also reports provenance only it can compute, so compare
        # the launch identity here and let the base class pin the full meta.
        meta = check_env_meta(
            client.call("env.get_env_meta", timeout_s=self._TIMEOUT_S["default"]),
            task_name=task_name,
            track=track,
            episode_index=episode_index,
            seed=seed,
        )
        super().__init__(client, expected_meta=meta)
        self.server_meta = meta

    def reset(self) -> dict[str, Any]:
        """Start a new episode and clear the terminal flags."""
        result = self._client.call("env.reset", timeout_s=self._TIMEOUT_S["env.reset"])
        self.terminated = False
        self.truncated = False
        self.last_obs = result
        return result

    def sensor_snapshot(
        self, *, cameras: list[str], height: int = 512, width: int = 512
    ) -> dict[str, Any]:
        """Render calibrated RGB-D frames and the robot state."""
        return self._client.call(
            "env.sensor_snapshot",
            kwargs={"cameras": cameras, "height": height, "width": width},
            timeout_s=self._TIMEOUT_S["env.sensor_snapshot"],
        )

    def robot_model_state(self) -> dict[str, Any]:
        return self._client.call("env.robot_model_state", timeout_s=30.0)

    def task_status(self) -> dict[str, Any]:
        return self._client.call("env.task_status", timeout_s=30.0)

    def get_task_language(self) -> str:
        return self._client.call("env.get_task_language", timeout_s=30.0)

    def execute_pose(
        self,
        *,
        xyz: list[float],
        quaternion_xyzw: list[float],
        gripper: Any,
        max_steps: int,
        tol: float,
        step_clip: float,
        ori_tol: float,
        check_cancelled: Callable[[], None] | None = None,
    ) -> dict[str, Any]:
        """Servo the end effector toward a finite world pose."""
        values = np.asarray([*xyz, *quaternion_xyzw], dtype=float)
        if values.shape != (7,) or not np.isfinite(values).all():
            raise ValueError("finite VLABench pose required")
        result = self._execute_motion(
            "env.execute_pose",
            kwargs={
                "xyz": values[:3].tolist(),
                "quaternion_xyzw": values[3:].tolist(),
                "gripper": gripper,
                "max_steps": int(max_steps),
                "tol": float(tol),
                "step_clip": float(step_clip),
                "ori_tol": float(ori_tol),
            },
            check_cancelled=check_cancelled,
        )
        self._update_terminal(result)
        return result

    def execute_gripper(
        self,
        *,
        gripper: float,
        steps: int,
        check_cancelled: Callable[[], None] | None = None,
    ) -> dict[str, Any]:
        """Hold the arm and drive the fingers for a bounded number of steps."""
        result = self._execute_motion(
            "env.execute_gripper",
            kwargs={"gripper": gripper, "steps": int(steps)},
            check_cancelled=check_cancelled,
        )
        self._update_terminal(result)
        return result

    def _execute_motion(
        self,
        method: str,
        *,
        kwargs: dict[str, Any],
        check_cancelled: Callable[[], None] | None,
    ) -> dict[str, Any]:
        if check_cancelled is not None:
            check_cancelled()
        motion_id = self._client.call("env.begin_motion", timeout_s=30.0)
        if not isinstance(motion_id, str) or not motion_id:
            raise RuntimeError("invalid VLABench motion token")
        done = threading.Event()
        errors: list[Exception] = []

        def cancel() -> None:
            reply = self._client.call(
                "env.cancel_motion",
                kwargs={"motion_id": motion_id},
                timeout_s=5.0,
            )
            if reply != {"motion_id": motion_id, "cancel_requested": True}:
                raise RuntimeError(
                    "VLABench did not acknowledge the cancellation request"
                )

        def monitor() -> None:
            while not done.wait(0.02):
                try:
                    check_cancelled()
                except Exception:
                    try:
                        cancel()
                    except Exception as exc:
                        errors.append(exc)
                    return

        watcher = None
        if check_cancelled is not None:
            watcher = threading.Thread(
                target=monitor, name="vlabench-cancel", daemon=True
            )
            watcher.start()
        try:
            result = self._client.call(
                method,
                kwargs={**kwargs, "motion_id": motion_id},
                timeout_s=self._TIMEOUT_S[method],
            )
            validate_motion_receipt(result, method, kwargs, motion_id)
        except Exception:
            try:
                cancel()
            except Exception:
                pass  # Keep the original unknown-execution error.
            raise
        finally:
            done.set()
            if watcher is not None:
                watcher.join()
        if errors:
            raise RuntimeError(
                "VLABench cancellation failed; execution state unknown"
            ) from errors[0]
        return result

    def _update_terminal(self, result: dict[str, Any]) -> None:
        self.terminated = result["terminated"]
        self.truncated = result["truncated"]


def build_env_client(args: argparse.Namespace, rpc: RpcClient) -> VLABenchEnvClient:
    """Connect to a running server and check that it matches the requested task."""
    return VLABenchEnvClient(
        rpc,
        task_name=args.task_name,
        track=args.track,
        episode_index=args.episode_index,
        seed=args.seed,
    )


def spawn_env_server(
    args: argparse.Namespace, output_dir: Path
) -> tuple[ProcessDaemon | None, RpcClient]:
    """Launch the environment server, or connect to ``--env-endpoint``.

    The server runs in the VLABench interpreter, which may differ from the one
    running RPent, so it is started as a separate process.
    """
    from rpent.utils.daemon import ProcessDaemon, pick_free_port
    from rpent.utils.rpc import make_rpc_client
    from rpent.utils.rpc.http_rpc import HttpRpcClient

    if args.env_endpoint is not None:
        return None, make_rpc_client(args.env_endpoint)
    host, port = "127.0.0.1", pick_free_port()
    command = [
        args.vlabench_python or sys.executable,
        str(get_repo_root() / "robots" / "vlabench" / "sim" / "env_server.py"),
        "--task-name",
        args.task_name,
        "--track",
        args.track,
        "--episode-index",
        str(args.episode_index),
        "--seed",
        str(args.seed),
        "--vlabench-root",
        str(Path(args.vlabench_root).expanduser()),
        "--asset-root",
        str(Path(args.vlabench_asset_root).expanduser()),
        "--transport",
        "http",
        "--host",
        host,
        "--port",
        str(port),
        "--parent-watch",
    ]
    if args.eval_unseen:
        command.append("--eval-unseen")
    if args.max_episode_steps is not None:
        command += ["--max-episode-steps", str(args.max_episode_steps)]
    device = resolve_cuda_device(args, "env")
    daemon = ProcessDaemon(
        "vlabench_env_server",
        command,
        env_overrides=(
            {"CUDA_VISIBLE_DEVICES": str(device)} if device is not None else {}
        ),
        log_path=str(output_dir / "vlabench_env_server.log"),
    )
    rpc = HttpRpcClient(f"http://{host}:{port}")
    daemon.start()
    return daemon, rpc


def init_runtime(
    args: argparse.Namespace,
    output_dir: Path,
    dashboard_events: DashboardEventSink,
    components: set[str] | None,
) -> tuple[list[ProcessDaemon], dict[str, Any]]:
    """Start the environment server, or only ``components`` when given.

    Returns:
        The daemons this call owns and the keyword arguments of the toolkit.
    """
    available = {"env"}
    selected = available if components is None else components
    unknown = selected.difference(available)
    if unknown:
        raise ValueError(f"unknown VLABench runtime components: {sorted(unknown)}")

    owned: dict[str, ProcessDaemon] = {}
    runtime_kwargs: dict[str, Any] = {}
    if "env" in selected:
        daemon, rpc = try_spawn_server(
            owned,
            dashboard_events,
            "env",
            lambda: spawn_env_server(args, output_dir),
        )
        runtime_kwargs["env"] = try_wait_server(
            owned,
            dashboard_events,
            "env",
            rpc,
            daemon,
            LAUNCHED_SERVER_TIMEOUT_S
            if daemon is not None
            else EXTERNAL_SERVER_TIMEOUT_S,
            post_fn=lambda: build_env_client(args, rpc),
        )
    return list(owned.values()), runtime_kwargs
