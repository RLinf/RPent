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

"""Shared fixtures for the RPC dispatch tests."""

from __future__ import annotations

import socket
import threading
import time
from contextlib import contextmanager

import pytest

from rpent.utils.daemon import pick_free_port
from rpent.utils.rpc.http_rpc import HttpRpcClient, HttpRpcServer
from rpent.utils.rpc.socket_rpc import SocketRpcClient, SocketRpcServer

TRANSPORTS = ["socket", "http"]


@pytest.fixture(params=TRANSPORTS)
def transport(request):
    return request.param


@pytest.fixture
def make_server_and_client():
    """Serve ``facade._dispatch`` on a real transport server; yield a client.

    The plain (concurrent) dispatch path, equivalent to ``RpcFacade.serve``.
    """

    @contextmanager
    def _ctx(facade, transport, *, enable_sessions=False):
        if transport == "socket":
            server = SocketRpcServer(("127.0.0.1", 0), facade._dispatch)
        else:
            server = HttpRpcServer(("127.0.0.1", 0), facade._dispatch)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        port = server.server_address[1]
        try:
            if transport == "socket":
                client = SocketRpcClient(
                    "127.0.0.1", port, enable_sessions=enable_sessions
                )
            else:
                client = HttpRpcClient(
                    f"http://127.0.0.1:{port}", enable_sessions=enable_sessions
                )
            yield client
        finally:
            server.shutdown()
            server.server_close()

    return _ctx


@pytest.fixture
def make_serve_in_thread():
    """Run ``facade.serve`` (main-thread dispatch) and yield a client."""

    @contextmanager
    def _ctx(facade, transport, *, enable_sessions):
        port = pick_free_port()

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

    return _ctx
