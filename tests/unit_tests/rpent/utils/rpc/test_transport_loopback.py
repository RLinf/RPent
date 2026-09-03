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

from __future__ import annotations

import os
import socket
import threading
import time
import urllib.request
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Literal

import numpy as np
import pytest

from rpent.utils.daemon import pick_free_port
from rpent.utils.rpc import (
    RpcClient,
    RpcError,
    RpcFacade,
    make_rpc_client,
    wait_for_ready,
)
from rpent.utils.rpc.http_rpc import HttpRpcClient, _is_direct_url

Transport = Literal["http", "socket"]
PROXY_ENVIRONMENT_VARIABLES = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "NO_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
    "no_proxy",
)


@pytest.fixture(autouse=True)
def _isolate_proxy_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in PROXY_ENVIRONMENT_VARIABLES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(urllib.request, "_opener", None)


class ContractFacade(RpcFacade):
    """Small real facade used to exercise both production transports."""

    def __init__(self) -> None:
        super().__init__()
        self.closed = threading.Event()
        self.delay_finished = threading.Event()
        self._rpc.update(
            {
                "combine": self.combine,
                "delay": self.delay,
                "fail": self.fail,
            }
        )
        self._readonly_methods.update(self._rpc)

    @staticmethod
    def combine(
        values: np.ndarray,
        *,
        scale: float,
        metadata: dict,
    ) -> dict:
        return {
            "values": values * scale,
            "metadata": metadata,
        }

    def delay(self, seconds: float) -> str:
        try:
            time.sleep(seconds)
            return "finished"
        finally:
            self.delay_finished.set()

    @staticmethod
    def fail(message: str) -> None:
        raise RuntimeError(message)

    def close(self) -> None:
        self.closed.set()


@dataclass(frozen=True)
class RunningFacade:
    client: RpcClient
    facade: ContractFacade
    port: int


def _port_accepts_connections(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.settimeout(0.1)
        return probe.connect_ex(("127.0.0.1", port)) == 0


@contextmanager
def _running_facade(
    transport: Transport,
    *,
    client_host: str = "127.0.0.1",
) -> Iterator[RunningFacade]:
    facade = ContractFacade()
    port = pick_free_port()
    thread = threading.Thread(
        target=facade.serve,
        kwargs={
            "transport": transport,
            "host": "127.0.0.1",
            "port": port,
        },
        daemon=True,
    )
    thread.start()
    client = make_rpc_client(f"{transport}://{client_host}:{port}")

    try:
        wait_for_ready(client, timeout_s=3.0, poll_interval_s=0.01)
        yield RunningFacade(client=client, facade=facade, port=port)
    finally:
        if thread.is_alive():
            try:
                client.call("shutdown", timeout_s=1.0)
            except Exception:
                facade._shutdown_event.set()
        thread.join(timeout=3.0)
        client.close()

    assert not thread.is_alive(), f"{transport} RPC server did not stop"
    assert facade.closed.is_set(), f"{transport} facade was not closed"
    assert not _port_accepts_connections(port), (
        f"{transport} RPC server did not release port {port}"
    )


@pytest.mark.parametrize("transport", ["http", "socket"])
def test_transport_round_trips_nested_numpy_payloads(transport: Transport) -> None:
    original = np.arange(6, dtype=np.float32).reshape(2, 3)

    with _running_facade(transport) as running:
        result = running.client.call(
            "combine",
            args=(original,),
            kwargs={
                "scale": 2.5,
                "metadata": {
                    "count": np.int64(6),
                    "valid": np.bool_(True),
                    "labels": ["left", "right"],
                },
            },
        )

        np.testing.assert_array_equal(result["values"], original * 2.5)
        assert result["metadata"] == {
            "count": 6,
            "valid": True,
            "labels": ["left", "right"],
        }
        result["values"][0, 0] = -1
        assert original[0, 0] == 0


def _configure_dead_proxy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:1")
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:1")
    monkeypatch.delenv("NO_PROXY", raising=False)
    monkeypatch.delenv("no_proxy", raising=False)
    monkeypatch.setattr(urllib.request, "_opener", None)


class _RecordingProxyHandler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args) -> None:  # noqa: A002
        return

    def do_POST(self) -> None:
        content_length = int(self.headers.get("Content-Length", 0))
        if content_length:
            self.rfile.read(content_length)
        self.server.request_targets.append(self.path)  # type: ignore[attr-defined]
        body = b'{"ok": true, "result": {"status": "proxied"}}'
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@contextmanager
def _running_http_proxy() -> Iterator[tuple[int, list[str]]]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _RecordingProxyHandler)
    request_targets: list[str] = []
    server.request_targets = request_targets  # type: ignore[attr-defined]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    try:
        yield int(server.server_port), request_targets
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3.0)

    assert not thread.is_alive()


@pytest.mark.parametrize("host", ["127.0.0.1", "localhost"])
def test_http_local_hosts_bypass_proxy_without_mutating_environment(
    host: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure_dead_proxy(monkeypatch)
    proxy_environment = {
        name: value
        for name in ("HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY", "no_proxy")
        if (value := os.environ.get(name)) is not None
    }

    with _running_facade("http", client_host=host) as running:
        assert running.client.call("healthz", timeout_s=1.0) == {"status": "ok"}

    assert {
        name: value
        for name in ("HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY", "no_proxy")
        if (value := os.environ.get(name)) is not None
    } == proxy_environment


@pytest.mark.parametrize("host", ["service.example.invalid", "127.0.0.2"])
def test_other_http_hosts_use_environment_proxy(
    host: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _running_http_proxy() as (proxy_port, request_targets):
        monkeypatch.setenv("HTTP_PROXY", f"http://127.0.0.1:{proxy_port}")
        monkeypatch.delenv("NO_PROXY", raising=False)
        monkeypatch.delenv("no_proxy", raising=False)
        monkeypatch.setattr(urllib.request, "_opener", None)
        client = HttpRpcClient(f"http://{host}:8123")

        assert client.call("healthz", timeout_s=1.0) == {"status": "proxied"}

    assert request_targets == [f"http://{host}:8123/call"]


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("http://127.0.0.1:8000", True),
        ("http://localhost:8000", True),
        ("http://LOCALHOST:8000", True),
        ("http://127.0.0.2:8000", False),
        ("http://[::1]:8000", False),
        ("http://service.example.invalid:8000", False),
    ],
)
def test_http_direct_host_classification(url: str, expected: bool) -> None:
    assert _is_direct_url(url) is expected


@pytest.mark.parametrize("transport", ["http", "socket"])
def test_transport_preserves_remote_error_context(transport: Transport) -> None:
    with _running_facade(transport) as running:
        with pytest.raises(RpcError, match="offline failure") as exc_info:
            running.client.call("fail", args=("offline failure",))

        assert exc_info.value.method == "fail"
        assert exc_info.value.server_traceback is not None
        assert "RuntimeError: offline failure" in exc_info.value.server_traceback


@pytest.mark.parametrize("transport", ["http", "socket"])
def test_transport_timeout_does_not_wedge_the_server(transport: Transport) -> None:
    with _running_facade(transport) as running:
        expected_error = RpcError if transport == "http" else TimeoutError
        with pytest.raises(expected_error):
            running.client.call("delay", args=(0.2,), timeout_s=0.01)

        assert running.client.call("healthz", timeout_s=1.0) == {"status": "ok"}
        assert running.facade.delay_finished.wait(timeout=1.0)
