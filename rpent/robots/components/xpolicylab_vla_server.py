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

Exposes the shared :class:`BaseVLAFacade` wire protocol (``vla.predict``) over
the XPolicyLab WebSocket Pi_05 policy server. It spawns the policy server via
the configured launcher and connects with the XPolicyLab model client. Policy
observations are passed through unchanged.
"""

from __future__ import annotations

import argparse
import atexit
import os
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from rpent.robots.components.vla_facade_base import BaseVLAFacade
from rpent.utils.daemon import ProcessDaemon
from rpent.utils.logging import get_logger

logger = get_logger("vla_server")


def add_backend_args(parser: argparse.ArgumentParser) -> None:
    """Register XPolicyLab launch and WebSocket configuration."""
    for name in (
        "task",
        "bench",
        "ckpt",
        "env-cfg-type",
        "action-type",
        "policy-root",
        "evaluation-id",
    ):
        parser.add_argument(f"--{name}")
    parser.add_argument("--policy-gpu", type=int, default=0)
    parser.add_argument("--policy-port", type=int, default=0)
    parser.add_argument("--policy-server-url")
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
    """Launch the Pi_05 policy server via the configured launcher."""
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
    logger.info("spawning policy server: %s", " ".join(cmd))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    daemon = ProcessDaemon(
        name="xpolicylab_policy",
        cmd=cmd,
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
    """Adapt the XPolicyLab Pi_05 WebSocket policy to the shared VLA facade."""

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
        self._close_lock = threading.Lock()
        atexit.register(self.close)
        try:
            policy_url = args.policy_server_url
            if not policy_url:
                self._policy_daemon = _spawn_policy_server(args)
                policy_url = f"ws://localhost:{args.policy_port}"
            logger.info("connecting to policy server: %s", policy_url)
            self._model_client = _connect_policy(policy_url, args)
            self._ws_lock = threading.Lock()
            super().__init__()
        except BaseException:
            self.close()
            raise
        logger.info("policy client connected")

    def _register_rpc(self) -> None:
        super()._register_rpc()
        self._rpc["reset"] = self.reset

    def reset(self) -> dict[str, Any]:
        with self._ws_lock:
            self._model_client.call(func_name="reset")
        return {"ok": True}

    def predict(self, obs: dict[str, Any], options: dict | None = None) -> Any:
        del options
        with self._ws_lock:
            self._model_client.call(func_name="update_obs", obs=obs)
            result = self._model_client.call(func_name="get_action")
        if isinstance(result, dict) and "actions" in result:
            return result["actions"]
        return result

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
