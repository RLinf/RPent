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

"""Shared policy adapter for XPolicyLab WebSocket inference.

Exposes ``vla.predict`` for existing callers. Opt-in stateful mode uses RPC
sessions with ``vla.reset_session``, ``vla.update_obs(obs, step)`` and
``vla.get_action``. ``vla.end_episode`` releases history ownership without
disconnecting the transport. One session owns the upstream model at a time.
Policy observations are passed through unchanged.
"""

from __future__ import annotations

import argparse
import atexit
import json
import os
import signal
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from rpent.robots.components.vla_facade_base import BaseVLAFacade
from rpent.utils.daemon import ProcessDaemon
from rpent.utils.logging import get_logger

logger = get_logger("vla_server")


def add_connection_args(
    parser: argparse.ArgumentParser, *, required: bool = False
) -> None:
    """Register the upstream WebSocket connection and evaluation identity."""
    parser.add_argument("--task", required=required)
    parser.add_argument("--evaluation-id")
    parser.add_argument("--policy-server-url", required=required)


def add_backend_args(parser: argparse.ArgumentParser) -> None:
    """Register XPolicyLab launch and WebSocket configuration."""
    add_connection_args(parser)
    for name in (
        "bench",
        "ckpt",
        "env-cfg-type",
        "action-type",
        "policy-root",
    ):
        parser.add_argument(f"--{name}")
    parser.add_argument("--policy-gpu", type=int, default=None)
    parser.add_argument("--policy-port", type=int, default=0)
    parser.add_argument(
        "--stateful",
        action="store_true",
        help="Require an exclusive RPC session and explicit per-step observations",
    )
    parser.add_argument(
        "--output-dir", default=os.getcwd(), help="Directory for policy server logs"
    )


def _wait_for_port(host: str, port: int, timeout_s: float = 900) -> None:
    import socket

    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=2):
                return
        except OSError:
            time.sleep(2)
    raise RuntimeError(f"policy server did not become ready on {host}:{port}")


def _spawn_policy_server(args: argparse.Namespace) -> ProcessDaemon:
    """Launch Pi_05, preserving inherited GPU visibility unless overridden."""
    pi05_root = args.policy_root
    if not pi05_root:
        raise ValueError("--policy-root is required for a local XPolicyLab server")
    launcher = os.path.join(pi05_root, "setup_eval_policy_server.sh")
    if not os.path.exists(launcher):
        raise RuntimeError(f"policy launcher not found: {launcher}")
    cmd = [
        "bash",
        launcher,
        args.bench,
        args.task,
        args.ckpt,
        args.env_cfg_type,
        args.action_type,
        "0",
        str(args.policy_gpu),
        "uv",
        str(args.policy_port),
        "localhost",
    ]
    overrides = None
    if args.policy_gpu is None:
        # The shell launcher always assigns CUDA_VISIBLE_DEVICES. Use its
        # Python entry point with the configured policy interpreter to retain
        # the caller's mask, including an unset or empty value.
        root = Path(pi05_root).resolve().parents[1]
        robot_info = json.loads((root / "utils/robot/_robot_info.json").read_text())[
            args.env_cfg_type
        ]
        action_dim = sum(robot_info["arm_dim"]) + sum(robot_info["ee_dim"])
        cmd = [
            sys.executable,
            "-u",
            str(root / "setup_policy_server.py"),
            "--config_path",
            str(Path(pi05_root) / "deploy.yml"),
            "--overrides",
            f"port={args.policy_port}",
            "host=localhost",
            f"bench_name={args.bench}",
            f"task_name={args.task}",
            f"ckpt_name={args.ckpt}",
            f"env_cfg_type={args.env_cfg_type}",
            "seed=0",
            f"policy_name={Path(pi05_root).name}",
            f"action_type={args.action_type}",
            f"action_dim={action_dim}",
        ]
        overrides = {
            "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.3",
            "PYTHONWARNINGS": "ignore::UserWarning",
        }
    logger.info("spawning policy server: %s", " ".join(cmd))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    daemon = ProcessDaemon(
        name="xpolicylab_policy",
        cmd=cmd,
        env_overrides=overrides,
        log_path=str(output_dir / "vla_server.log"),
        cwd=os.getcwd(),
    )
    try:
        daemon.start()
        _wait_for_port("localhost", args.policy_port)
    except BaseException:
        daemon.stop()
        raise
    return daemon


def _connect_policy(url: str, args: argparse.Namespace):
    from client_server.ws.model_client import WsModelClient

    return WsModelClient(
        url=url,
        evaluation_id=args.evaluation_id
        or datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        trial_id=f"{args.task}-vla",
        action_case_id=f"{args.task}_case",
        repeat_index=None,
    )


class XPolicyLabVLAFacade(BaseVLAFacade):
    """Adapt XPolicyLab policies with optional exclusive episode history."""

    def __init__(self, args: argparse.Namespace) -> None:
        if not args.task:
            raise ValueError("--task is required for XPolicyLab metadata")
        if not args.policy_server_url and not all(
            (
                args.bench,
                args.ckpt,
                args.env_cfg_type,
                args.action_type,
                args.policy_port,
            )
        ):
            raise ValueError(
                "Local XPolicyLab requires bench, ckpt, env-cfg-type, action-type and policy-port"
            )
        self._policy_daemon = None
        self._model_client = None
        self._stateful = getattr(args, "stateful", False)
        self._owner: str | None = None
        self._next_step = 0
        self._inferred_step = -1
        self._episode_ready = False
        self._close_lock = threading.Lock()
        atexit.register(self.close)
        try:
            policy_url = args.policy_server_url
            if not policy_url:
                self._policy_daemon = _spawn_policy_server(args)
                policy_url = f"ws://localhost:{args.policy_port}"
            logger.info("connecting to policy server: %s", policy_url)
            self._model_client = self._connect_policy(policy_url, args)
            self._ws_lock = threading.Lock()
            super().__init__(enable_sessions=self._stateful)
        except BaseException:
            self.close()
            raise
        logger.info("policy client connected")

    def _connect_policy(self, url: str, args: argparse.Namespace) -> Any:
        return _connect_policy(url, args)

    def _register_rpc(self) -> None:
        super()._register_rpc()
        self._rpc["reset"] = self.reset
        if self._stateful:
            self._rpc.update(
                {
                    "vla.reset_session": self.reset_session,
                    "vla.update_obs": self.update_obs,
                    "vla.get_action": self.get_action,
                    "vla.end_episode": self.end_episode,
                }
            )

    def reset(self, *, session_id: str | None = None) -> dict[str, Any]:
        if self._stateful:
            raise ValueError("Use vla.reset_session in stateful mode")
        with self._ws_lock:
            self._model_client.call(func_name="reset")
        return {"ok": True}

    def predict(
        self,
        obs: dict[str, Any],
        options: dict | None = None,
        *,
        session_id: str | None = None,
    ) -> Any:
        if self._stateful:
            raise ValueError("Use vla.update_obs and vla.get_action in stateful mode")
        del options
        with self._ws_lock:
            self._model_client.call(func_name="update_obs", obs=obs)
            result = self._model_client.call(func_name="get_action")
        if isinstance(result, dict) and "actions" in result:
            return result["actions"]
        return result

    def reset_session(self, *, session_id: str) -> dict[str, Any]:
        """Start an episode for the sole owner of this policy instance."""
        with self._ws_lock:
            if self._owner is not None and self._owner != session_id:
                raise RuntimeError("Policy history is owned by another session")
            self._owner = session_id
            self._episode_ready = False
            self._model_client.call(func_name="reset")
            self._next_step = 0
            self._inferred_step = -1
            self._episode_ready = True
        return {"ok": True}

    def _require_episode(self, session_id: str) -> None:
        if self._owner != session_id:
            raise RuntimeError("Call vla.reset_session to acquire policy history")
        if not self._episode_ready:
            raise RuntimeError(
                "Policy state is uncertain; reset the episode before use"
            )

    def _episode_call(self, **kwargs) -> Any:
        # A failed response may follow a successful model mutation. Never retry
        # or continue a history whose contents are no longer known.
        try:
            return self._model_client.call(**kwargs)
        except Exception:
            self._episode_ready = False
            raise

    def update_obs(self, obs: dict, step: int, *, session_id: str) -> dict:
        """Append exactly one control observation, starting at episode step zero."""
        with self._ws_lock:
            self._require_episode(session_id)
            if type(step) is not int or step != self._next_step:
                raise ValueError(
                    f"Expected observation step {self._next_step}, got {step}"
                )
            self._episode_call(func_name="update_obs", obs=obs)
            self._next_step += 1
        return {"ok": True}

    def get_action(self, *, session_id: str, include_info: bool = False) -> Any:
        """Infer once; optionally retain the upstream result envelope."""
        if type(include_info) is not bool:
            raise ValueError("include_info must be a boolean")
        with self._ws_lock:
            self._require_episode(session_id)
            current_step = self._next_step - 1
            if current_step <= self._inferred_step:
                raise RuntimeError("A new observation is required before inference")
            result = self._episode_call(func_name="get_action")
            self._inferred_step = current_step
        if not include_info and isinstance(result, dict) and "actions" in result:
            return result["actions"]
        return result

    def _on_session_drop(self, session_id: str) -> None:
        # The next owner must reset before any observation or inference.
        with self._ws_lock:
            if self._owner == session_id:
                self._owner = None
                self._episode_ready = False

    def end_episode(self, *, session_id: str) -> dict:
        """Release episode ownership while retaining the reusable RPC connection."""
        self._on_session_drop(session_id)
        return {"ok": True}

    def close(self) -> None:
        """Release owned policy processes even when the WebSocket close fails."""
        with self._close_lock:
            # Stop the GPU owner first; a broken WebSocket must not delay it.
            if self._policy_daemon is not None:
                self._policy_daemon.stop(timeout=5.0)
                self._policy_daemon = None
            if self._model_client is not None:
                try:
                    self._model_client.close()
                except Exception:  # noqa: BLE001
                    logger.warning("XPolicyLab client close failed", exc_info=True)
                self._model_client = None
            atexit.unregister(self.close)


def main() -> None:
    """Run the XPolicyLab adapter and clean up its owned policy process."""
    parser = argparse.ArgumentParser()
    add_backend_args(parser)
    parser.add_argument("--transport", choices=["socket", "http"], default="http")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument(
        "--parent-watch",
        action="store_true",
        help="watch parent process via stdin pipe and exit when it dies",
    )
    parser.add_argument(
        "--cuda-device",
        type=int,
        default=None,
        help="GPU device exposed through CUDA_VISIBLE_DEVICES.",
    )
    args = parser.parse_args()

    if args.cuda_device is not None:
        target = str(args.cuda_device)
        prev = os.environ.get("CUDA_VISIBLE_DEVICES")
        if prev is not None and prev != target:
            logger.warning(
                "CUDA_VISIBLE_DEVICES=%s is already set; overriding with --cuda-device=%s",
                prev,
                args.cuda_device,
            )
        os.environ["CUDA_VISIBLE_DEVICES"] = target

    def terminate(signum, frame):
        raise SystemExit(128 + signum)

    # ProcessDaemon.stop sends SIGTERM; unwind startup/serve and its cleanup.
    previous = signal.signal(signal.SIGTERM, terminate)
    facade = None
    try:
        facade = XPolicyLabVLAFacade(args)
        facade.serve(
            transport=args.transport,
            host=args.host,
            port=args.port,
            parent_watch=args.parent_watch,
            **({"session_sweep_s": 60.0} if args.stateful else {}),
        )
    finally:
        try:
            if facade is not None:
                facade.close()
        finally:
            signal.signal(signal.SIGTERM, previous)


if __name__ == "__main__":
    main()
