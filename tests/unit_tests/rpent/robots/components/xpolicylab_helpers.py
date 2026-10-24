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

"""Loopback transport for testing shared sessions and concrete policy clients."""

import threading
from contextlib import contextmanager
from types import SimpleNamespace

from rpent.robots.components.xpolicylab_vla_client import XPolicyLabVLAClient
from rpent.utils.rpc import make_rpc_client, wait_for_ready
from rpent.utils.rpc.http_rpc import HttpRpcServer
from rpent.utils.rpc.socket_rpc import SocketRpcServer


@contextmanager
def serve_policy(facade, transport="http"):
    server_cls = HttpRpcServer if transport == "http" else SocketRpcServer
    server = server_cls(("127.0.0.1", 0), facade._dispatch)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    endpoint = f"{transport}://127.0.0.1:{server.server_address[1]}"
    clients = []

    def connect(client_type=XPolicyLabVLAClient):
        rpc = make_rpc_client(endpoint, enable_sessions=True)
        clients.append(rpc)
        wait_for_ready(rpc, timeout_s=2)
        return client_type(rpc)

    try:
        yield SimpleNamespace(connect=connect, facade=facade, endpoint=endpoint)
    finally:
        for client in clients:
            client.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
        facade.close()
        assert not thread.is_alive()
