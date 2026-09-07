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

"""Shared helpers for the RPC dispatch tests."""

from __future__ import annotations

import socket
import threading
import time
from contextlib import contextmanager

from rpent.utils.rpc.http_rpc import HttpRpcClient, HttpRpcServer
from rpent.utils.rpc.socket_rpc import SocketRpcClient, SocketRpcServer

TRANSPORTS = ["socket", "http"]


@contextmanager
def _server_and_client(facade, transport, *, enable_sessions=False):
    """Serve ``facade._dispatch`` on a real transport server; yield a client.

    The plain (concurrent) dispatch path, equivalent to ``RpcFacade.serve``.
    """
    if transport == "socket":
        server = SocketRpcServer(("127.0.0.1", 0), facade._dispatch)
    else:
        server = HttpRpcServer(("127.0.0.1", 0), facade._dispatch)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    port = server.server_address[1]
    try:
        if transport == "socket":
            client = SocketRpcClient("127.0.0.1", port, enable_sessions=enable_sessions)
        else:
            client = HttpRpcClient(
                f"http://127.0.0.1:{port}", enable_sessions=enable_sessions
            )
        yield client
    finally:
        server.shutdown()
        server.server_close()


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@contextmanager
def _serve_in_thread(facade, transport, *, enable_sessions):
    """Run ``facade.serve`` (main-thread dispatch) and yield a client."""
    port = _free_port()

    def run():
        facade.serve(
            transport=transport,
            host="127.0.0.1",
            port=port,
            parent_watch=False,
            session_sweep_s=60.0,  # required when sessions are enabled
        )

    t = threading.Thread(target=run, daemon=True)
    t.start()
    deadline = time.monotonic() + 10.0
    while True:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                break
        except OSError:
            if time.monotonic() > deadline:
                raise TimeoutError("server did not become ready")
            time.sleep(0.01)
    if transport == "socket":
        client = SocketRpcClient("127.0.0.1", port, enable_sessions=enable_sessions)
    else:
        client = HttpRpcClient(
            f"http://127.0.0.1:{port}", enable_sessions=enable_sessions
        )
    try:
        yield client
    finally:
        try:
            client.call("shutdown", timeout_s=2.0)
        except Exception:
            pass
        t.join(timeout=5.0)
