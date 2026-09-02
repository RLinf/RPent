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

"""Transport handlers must not force a ``session_id`` kwarg onto dispatch.

``Sam3Facade._dispatch`` takes no ``session_id`` parameter; before the
fix, the http/socket handlers unconditionally called
``dispatch(method, args, kwargs, session_id=...)`` and every call raised
``TypeError``. The handlers now only forward ``session_id`` when the client
actually sent one (session-aware clients).
"""

from __future__ import annotations

import os
import threading
from contextlib import contextmanager

import pytest

from rpent.utils.rpc.http_rpc import HttpRpcClient, HttpRpcServer
from rpent.utils.rpc.rpc_facade import RpcFacade
from rpent.utils.rpc.socket_rpc import SocketRpcClient, SocketRpcServer


@pytest.fixture(autouse=True)
def _bypass_system_proxy():
    """Keep urllib off the host's system proxy.

    ``urllib.request.urlopen`` consults the macOS system proxy (e.g. a local
    Clash at 127.0.0.1:7892); routed through it, calls to 127.0.0.1 get a
    502 / empty body. ``no_proxy`` forces a direct connection so the HTTP
    transport is deterministic in tests.
    """
    os.environ["no_proxy"] = "*"
    os.environ["NO_PROXY"] = "*"
    yield


class SessionlessFacade(RpcFacade):
    """Mirrors ``Sam3Facade``: ``_dispatch`` has no ``session_id`` parameter."""

    def _dispatch(self, method, args, kwargs):
        if method == "ping":
            return {"pong": args[0] if args else None}
        raise ValueError(f"unknown RPC method: {method!r}")


class SessionAwareFacade(RpcFacade):
    """``_dispatch`` accepts ``session_id`` (like a session-enabled facade)."""

    def __init__(self):
        super().__init__()
        self.received_sessions: list[str | None] = []

    def _dispatch(self, method, args, kwargs, *, session_id=None):
        if method == "ping":
            self.received_sessions.append(session_id)
            return {"pong": args[0] if args else None}
        raise ValueError(f"unknown RPC method: {method!r}")


TRANSPORTS = ["socket", "http"]


@pytest.fixture(params=TRANSPORTS)
def transport(request):
    return request.param


@contextmanager
def _server_and_client(facade, transport, *, enable_sessions=False):
    """Serve ``facade._dispatch`` on the real transport server and yield a client."""
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


def test_sessionless_dispatch_over_transports(transport):
    facade = SessionlessFacade()
    with _server_and_client(facade, transport) as client:
        result = client.call("ping", args=("hello",))
        assert result == {"pong": "hello"}


def test_session_id_still_forwarded_when_present(transport):
    facade = SessionAwareFacade()
    with _server_and_client(facade, transport, enable_sessions=True) as client:
        assert client._session_id is not None
        result = client.call("ping", args=(42,))
        assert result == {"pong": 42}
        assert facade.received_sessions == [client._session_id]
