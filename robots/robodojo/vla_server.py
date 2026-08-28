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

"""RPC server wrapping the RoboDojo Pi_05 policy server (XPolicyLab ws).

Exposes the shared :class:`BaseVLAFacade` wire protocol (``vla.predict``) over
the XPolicyLab WebSocket Pi_05 policy server. It spawns the policy server via
the workspace launcher and connects with the XPolicyLab model client. Policy
observations are passed through unchanged (no brightness calibration).
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

# Support direct execution from an RPent checkout before package imports.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rpent.robots.components.vla_facade_base import BaseVLAFacade  # noqa: E402

parser = argparse.ArgumentParser(description="RoboDojo Pi_05 policy RPC server")
parser.add_argument("--task", default="put_bottles_into_dustbin")
parser.add_argument("--bench", default="RoboDojo")
parser.add_argument("--ckpt", default="RoboDojo-sim-arx_x5-joint-0")
parser.add_argument("--env-cfg-type", default="arx_x5")
parser.add_argument("--action-type", default="joint")
parser.add_argument("--policy-gpu", type=int, default=0)
parser.add_argument(
    "--policy-port",
    type=int,
    default=0,
    help="Spawn a local Pi_05 policy server on this port",
)
parser.add_argument(
    "--policy-server-url",
    default=None,
    help="Connect to an already-running policy server",
)
parser.add_argument("--host", default="127.0.0.1")
parser.add_argument("--port", type=int, default=0)
parser.add_argument("--parent-pid", type=int, default=None)
parser.add_argument("--transport", choices=["http", "socket"], default="http")
parser.add_argument("--parent-watch", action="store_true")
_args = parser.parse_args()


def _log(msg: str) -> None:
    print(f"[vla_server] {msg}", flush=True)


if _args.parent_pid is not None:

    def _watch_parent(pid: int) -> None:
        while True:
            try:
                os.kill(pid, 0)
            except OSError:
                os._exit(0)
            time.sleep(2)

    threading.Thread(
        target=_watch_parent, args=(_args.parent_pid,), daemon=True
    ).start()


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


def _spawn_policy_server() -> str:
    """Launch the Pi_05 policy server via the workspace launcher."""
    pi05_root = os.environ.get(
        "ROBODOJO_PI05_POLICY_ROOT",
        "/home/admin/robodojo_pro6000_ws/src/RoboDojo/XPolicyLab/policy/Pi_05",
    )
    launcher = os.path.join(pi05_root, "setup_eval_policy_server.sh")
    if not os.path.exists(launcher):
        raise RuntimeError(f"policy launcher not found: {launcher}")
    cmd = [
        "bash",
        launcher,
        _args.bench,
        _args.task,
        _args.ckpt,
        _args.env_cfg_type,
        _args.action_type,
        "0",
        str(_args.policy_gpu),
        "uv",
        str(_args.policy_port),
        "localhost",
    ]
    _log("spawning policy server: " + " ".join(cmd))
    proc = subprocess.Popen(
        cmd,
        env=os.environ.copy(),
        stdout=open("/tmp/rpent_pi05_policy_server.log", "ab"),
        stderr=subprocess.STDOUT,
    )
    _wait_for_port("localhost", _args.policy_port)
    _log("policy server ready (pid=%d)" % proc.pid)
    return f"ws://localhost:{_args.policy_port}"


def _connect_policy(url: str):
    from client_server.ws.model_client import WsModelClient

    return WsModelClient(
        url=url,
        evaluation_id=os.environ.get(
            "ROBODOJO_RUN_ID", datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        ),
        trial_id=f"{_args.task}-vla",
        action_case_id=f"{_args.task}_case",
        repeat_index=None,
    )


class RoboDojoVLAFacade(BaseVLAFacade):
    """Adapt the XPolicyLab Pi_05 WebSocket policy to the shared VLA facade."""

    def __init__(self) -> None:
        policy_url = _args.policy_server_url or _spawn_policy_server()
        _log(f"connecting to policy server: {policy_url}")
        self._model_client = _connect_policy(policy_url)
        self._ws_lock = threading.Lock()
        _log("policy client connected")
        super().__init__()

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
        try:
            self._model_client.close()
        except Exception:  # noqa: BLE001
            pass


def main() -> None:
    facade = RoboDojoVLAFacade()
    facade.serve(
        transport=_args.transport,
        host=_args.host,
        port=_args.port,
        parent_watch=_args.parent_watch,
    )


if __name__ == "__main__":
    main()
