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

"""RPC dispatch tests for session-less facades (no per-client server state).

Covers both serve paths over socket / http:

- plain ``RpcFacade`` dispatch: an overridden ``_dispatch`` with no
  ``session_id`` parameter (Sam3-style) and a default facade with
  ``enable_sessions=False``;
- ``MainThreadServeMixin`` single-thread dispatch with
  ``enable_sessions=False``.
"""

from __future__ import annotations

from rpent.utils.rpc.main_thread_serve import MainThreadServeMixin
from rpent.utils.rpc.rpc_facade import RpcFacade


class OverrideDispatchFacade(RpcFacade):
    """Sam3-style: ``_dispatch`` overridden with NO ``session_id`` parameter."""

    def _dispatch(self, method, args, kwargs):
        if method == "ping":
            return {"pong": args[0] if args else None}
        raise ValueError(f"unknown RPC method: {method!r}")


class PlainFacade(RpcFacade):
    """Default ``RpcFacade._dispatch`` with ``enable_sessions=False``."""

    def __init__(self):
        super().__init__()
        self._rpc["ping"] = self.ping

    def ping(self, value):
        return {"pong": value}


class MainThreadPlainFacade(MainThreadServeMixin, PlainFacade):
    """Use the session-less handlers through main-thread dispatch."""


def test_override_dispatch_over_transports(transport, make_server_and_client):
    facade = OverrideDispatchFacade()
    with make_server_and_client(facade, transport) as client:
        assert client.call("ping", args=("hello",)) == {"pong": "hello"}


def test_plain_facade_over_transports(transport, make_server_and_client):
    facade = PlainFacade()
    with make_server_and_client(facade, transport) as client:
        assert client.call("ping", args=(42,)) == {"pong": 42}


def test_main_thread_wo_session_over_transports(transport, make_serve_in_thread):
    facade = MainThreadPlainFacade()
    with make_serve_in_thread(facade, transport, enable_sessions=False) as client:
        assert client.call("ping", args=(7,)) == {"pong": 7}
