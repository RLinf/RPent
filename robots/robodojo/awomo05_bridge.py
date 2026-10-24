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

"""Stateful RPent bridge for a separately managed official Awomo05 service."""

from __future__ import annotations

import argparse
import os
import signal
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from robots.robodojo.awomo05_contract import SESSION_METHODS, validate_model_meta
from rpent.robots.components.xpolicylab_vla_server import (
    XPolicyLabVLAFacade,
    add_connection_args,
)
from rpent.utils.config import get_repo_root
from rpent.utils.daemon import ProcessDaemon, pick_free_port
from rpent.utils.rpc.http_rpc import HttpRpcClient


class Awomo05Facade(XPolicyLabVLAFacade):
    """Adapt the shared session protocol to the official Awomo05 service."""

    def __init__(self, args: argparse.Namespace) -> None:
        if not args.policy_server_url:
            raise ValueError("Awomo05 bridge requires --policy-server-url")
        args.stateful = True
        super().__init__(args)
        try:
            try:
                meta = self._model_client.call(func_name="get_model_meta")
            except Exception as exc:
                raise RuntimeError(
                    "Upstream Awomo05 must expose get_model_meta; start it with "
                    "robots/robodojo/awomo05_policy_server.py"
                ) from exc
            self._meta = {
                **validate_model_meta(meta),
                "stateful": True,
                "rpc_methods": list(SESSION_METHODS),
                "action_info": True,
            }
        except BaseException:
            self.close()
            raise

    def _register_rpc(self) -> None:
        super()._register_rpc()
        self._rpc["vla.get_model_meta"] = self.get_model_meta
        self._readonly_methods.add("vla.get_model_meta")

    def get_model_meta(self, *, session_id: str | None = None) -> dict:
        return dict(self._meta)

    def _connect_policy(self, url: str, args: argparse.Namespace) -> Any:
        # Inject an observer through the official synchronous client's supported
        # client argument. Keep its encoding, request IDs, and state handling.
        from client_server.ws.model_client import WsModelClient
        from client_server.ws.protocol.client import (
            PolicyEvalClient,
            PolicyEvalClientConfig,
        )

        class ObservedClient(PolicyEvalClient):
            async def call(self, *args, **kwargs):
                response = await super().call(*args, **kwargs)
                self.call_info = {
                    k: v for k, v in response.payload.items() if k != "result"
                }
                return response

        evaluation_id = args.evaluation_id or datetime.now().isoformat()
        self._observed_client = ObservedClient(
            PolicyEvalClientConfig(url=url, evaluation_id=evaluation_id)
        )
        return WsModelClient(
            url=url,
            evaluation_id=evaluation_id,
            trial_id=f"{args.task}-vla",
            action_case_id=f"{args.task}_case",
            client=self._observed_client,
        )

    def _episode_call(self, **kwargs):
        started = time.perf_counter()
        result = super()._episode_call(**kwargs)
        if kwargs["func_name"] != "get_action":
            return result
        # All metadata belongs to this inference, inside the episode lock. No
        # second RPC or cached action copy is needed to retrieve it.
        return {
            "actions": result["actions"]
            if isinstance(result, dict) and "actions" in result
            else result,
            "info": {
                "upstream": self._observed_client.call_info,
                "model": {k: v for k, v in result.items() if k != "actions"}
                if isinstance(result, dict)
                else {},
                "bridge_rpc_s": time.perf_counter() - started,
            },
        }


def spawn_bridge(
    args: argparse.Namespace, output_dir: Path
) -> tuple[ProcessDaemon, HttpRpcClient]:
    """Start only the CPU bridge, borrowing an independently managed model server."""
    interpreter = Path(args.pi05_python).expanduser()
    if not interpreter.is_file():
        raise ValueError(f"Policy bridge Python not found: {interpreter}")
    root = (
        Path(args.xpolicylab_root).expanduser()
        if args.xpolicylab_root
        else (
            Path(args.source_root).expanduser() / "XPolicyLab"
            if args.source_root
            else None
        )
    )
    if root is None or not (root / "client_server/ws/model_client.py").is_file():
        raise ValueError(
            "Awomo05 WebSocket connection requires --xpolicylab-root or --source-root"
        )
    root = root.resolve()
    host, port = "127.0.0.1", pick_free_port()
    daemon = ProcessDaemon(
        name="robodojo_awomo05_bridge",
        cmd=[
            str(interpreter),
            "-u",
            "-m",
            "robots.robodojo.awomo05_bridge",
            "--task",
            args.task,
            "--policy-server-url",
            args.policy_server_url,
            "--host",
            host,
            "--port",
            str(port),
            "--parent-watch",
        ],
        env_overrides={
            "PYTHONPATH": os.pathsep.join(
                [str(get_repo_root()), str(root.parent), str(root)]
                + ([os.environ["PYTHONPATH"]] if os.environ.get("PYTHONPATH") else [])
            )
        },
        log_path=str(output_dir / "awomo05_bridge.log"),
    )
    try:
        daemon.start()
    except BaseException:
        daemon.stop()
        raise
    return daemon, HttpRpcClient(f"http://{host}:{port}", enable_sessions=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_connection_args(parser, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--parent-watch", action="store_true")
    args = parser.parse_args()

    def terminate(signum, frame):
        raise SystemExit(128 + signum)

    previous = signal.signal(signal.SIGTERM, terminate)
    facade = None
    try:
        facade = Awomo05Facade(args)
        facade.serve(
            transport="http",
            host=args.host,
            port=args.port,
            parent_watch=args.parent_watch,
            session_sweep_s=60.0,
        )
    finally:
        try:
            if facade is not None:
                facade.close()
        finally:
            signal.signal(signal.SIGTERM, previous)


if __name__ == "__main__":
    main()
