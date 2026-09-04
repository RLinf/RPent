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

"""RPC dispatch tests for session-aware facades (server keeps per-client state).

Both serve paths (plain and main-thread) over socket / http run the full
lifecycle: register -> call (the server injects the caller's ``session_id``
into the handler) -> close (fires ``_on_session_drop``) -> the dropped
session is rejected on the next call.
"""

from __future__ import annotations

import pytest

from rpent.utils.rpc.client_utils import wait_for_ready
from rpent.utils.rpc.main_thread_serve import MainThreadServeMixin
from rpent.utils.rpc.rpc_client import RpcError
from rpent.utils.rpc.rpc_facade import RpcFacade
from tests.utils.rpc._rpc_test_helpers import (
    TRANSPORTS,
    _serve_in_thread,
    _server_and_client,
)


@pytest.fixture(params=TRANSPORTS)
def transport(request):
    return request.param


class SessionFacade(RpcFacade):
    """``enable_sessions=True``; records injected sessions + drops."""

    def __init__(self):
        super().__init__(enable_sessions=True)
        self._rpc["ping"] = self.ping
        self.received_sessions: list[str | None] = []
        self.dropped_sessions: list[str] = []

    def ping(self, value, *, session_id=None):
        self.received_sessions.append(session_id)
        return {"pong": value}

    def _on_session_drop(self, session_id):
        self.dropped_sessions.append(session_id)


class MTWSessionFacade(MainThreadServeMixin, RpcFacade):
    """Main-thread served, ``enable_sessions=True``; records sessions + drops."""

    def __init__(self):
        super().__init__(enable_sessions=True)
        self._rpc["ping"] = self.ping
        self.received_sessions: list[str | None] = []
        self.dropped_sessions: list[str] = []

    def ping(self, value, *, session_id=None):
        self.received_sessions.append(session_id)
        return {"pong": value}

    def _on_session_drop(self, session_id):
        self.dropped_sessions.append(session_id)


def _assert_lifecycle(client, facade):
    assert client._session_id is not None
    wait_for_ready(client, timeout_s=10.0)  # registers the session

    assert client.call("ping", args=("hi",)) == {"pong": "hi"}
    # the server injects the caller's session id into the handler
    assert facade.received_sessions == [client._session_id]

    client.call("session.close")
    # closing the session fires the cleanup hook
    assert facade.dropped_sessions == [client._session_id]

    # an unregistered session is rejected on the next call
    with pytest.raises(RpcError, match="session not found"):
        client.call("ping", args=("hi",))


def test_session_lifecycle_over_transports(transport):
    facade = SessionFacade()
    with _server_and_client(facade, transport, enable_sessions=True) as client:
        _assert_lifecycle(client, facade)


def test_main_thread_session_lifecycle_over_transports(transport):
    facade = MTWSessionFacade()
    with _serve_in_thread(facade, transport, enable_sessions=True) as client:
        _assert_lifecycle(client, facade)
