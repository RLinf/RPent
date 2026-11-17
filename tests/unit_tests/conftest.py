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

"""Shared logging isolation and loopback RPC fixtures."""

import threading
from contextlib import contextmanager

import pytest

from rpent.utils import logging as run_logging
from rpent.utils.rpc.http_rpc import HttpRpcClient, HttpRpcServer
from rpent.utils.rpc.socket_rpc import SocketRpcClient, SocketRpcServer

TRANSPORTS = ["socket", "http"]


@pytest.fixture(autouse=True)
def isolate_run_logging(monkeypatch):
    """Restore process-wide logging after planner, CLI, or robot tests."""
    logger = run_logging.get_logger()
    handlers = list(logger.handlers)
    level, propagate = logger.level, logger.propagate
    monkeypatch.setattr(run_logging, "_log_initialized", run_logging._log_initialized)
    monkeypatch.setattr(run_logging, "_output_dir", run_logging._output_dir)
    try:
        yield
    finally:
        for handler in logger.handlers:
            if handler not in handlers:
                handler.close()
        logger.handlers[:] = handlers
        logger.setLevel(level)
        logger.propagate = propagate


@pytest.fixture(params=TRANSPORTS)
def transport(request):
    return request.param


@pytest.fixture
def make_server_and_client():
    """Serve a facade on a loopback transport and yield its client."""

    @contextmanager
    def _ctx(facade, transport, *, enable_sessions=False):
        if transport == "socket":
            server = SocketRpcServer(("127.0.0.1", 0), facade._dispatch)
        else:
            server = HttpRpcServer(("127.0.0.1", 0), facade._dispatch)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
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
            thread.join(timeout=2)

    return _ctx
