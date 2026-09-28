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
import sys
import threading
import time
import urllib.request
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Literal

import numpy as np
import pytest

from robots.behavior.dino_v2.encoder import DINOV2_DIMENSION
from robots.behavior.dino_v2.server import BehaviorDinoFacade
from robots.behavior.env_server import BehaviorEnvFacade
from rpent.utils.daemon import ProcessDaemon, pick_free_port
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
                    "score": np.float32(1.5),
                    "labels": ["left", "right"],
                },
            },
        )

        np.testing.assert_array_equal(result["values"], original * 2.5)
        metadata = result["metadata"]
        # Numpy scalars keep their exact dtype on both transports: the
        # socket transport natively via pickle, the HTTP transport via
        # the ``__npscalar__`` tag.
        assert isinstance(metadata["count"], np.int64)
        assert metadata["count"] == 6
        assert isinstance(metadata["valid"], np.bool_)
        assert metadata["valid"] == np.bool_(True)
        assert isinstance(metadata["score"], np.float32)
        assert metadata["score"] == np.float32(1.5)
        assert metadata["labels"] == ["left", "right"]
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


class _ThreadRecordingBehaviorEnvFacade(BehaviorEnvFacade):
    def __init__(self) -> None:
        super().__init__(backend=object(), meta={"task_language": "test"})
        self.serve_thread_id: int | None = None
        self.business_thread_id: int | None = None

    def serve(self, **kwargs: Any) -> None:
        self.serve_thread_id = threading.get_ident()
        super().serve(**kwargs)

    def get_env_meta(self) -> dict[str, Any]:
        self.business_thread_id = threading.get_ident()
        return super().get_env_meta()


def test_behavior_facades_use_default_healthz_and_registered_metadata() -> None:
    facade = BehaviorEnvFacade(backend=object(), meta={"task_language": "test"})
    dino = BehaviorDinoFacade(
        encoder=object(),
        meta={"runtime": "behavior_dino", "dimension": DINOV2_DIMENSION},
    )

    assert facade._dispatch("healthz", (), {}) == {"status": "ok"}
    assert facade._dispatch("env.get_env_meta", (), {}) == {"task_language": "test"}
    assert "env.close_gripper" in facade._rpc
    assert "env.open_gripper" in facade._rpc
    assert "env.close" not in facade._rpc
    assert "env.open" not in facade._rpc
    with pytest.raises(ValueError, match="requires primitive arguments"):
        facade.close_gripper()
    assert dino._dispatch("healthz", (), {}) == {"status": "ok"}
    dino_meta = dino._dispatch("dino.get_meta", (), {})
    assert dino_meta["runtime"] == "behavior_dino"
    assert dino_meta["dimension"] == DINOV2_DIMENSION
    assert isinstance(dino_meta["pid"], int)


def test_behavior_env_facade_serve_dispatches_business_calls_on_serving_thread() -> (
    None
):
    facade = _ThreadRecordingBehaviorEnvFacade()
    port = pick_free_port()
    thread = threading.Thread(
        target=facade.serve,
        kwargs={
            "transport": "http",
            "host": "127.0.0.1",
            "port": port,
        },
        daemon=True,
    )
    thread.start()
    client = HttpRpcClient(f"http://127.0.0.1:{port}")

    try:
        deadline = time.monotonic() + 3.0
        while True:
            try:
                assert client.call("healthz", timeout_s=0.5) == {"status": "ok"}
                break
            except Exception:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(0.01)

        assert client.call("env.get_env_meta", timeout_s=1.0) == {
            "task_language": "test"
        }
        assert facade.business_thread_id == facade.serve_thread_id
        assert facade.business_thread_id != threading.get_ident()
        assert client.call("shutdown", timeout_s=1.0) == {"ok": True}
    finally:
        client.close()
        facade._shutdown_event.set()
        thread.join(timeout=3.0)

    assert not thread.is_alive()
    assert facade._closed
    assert not _port_accepts_connections(port)


def test_env_main_sigterm_uses_owning_thread_cleanup(tmp_path):
    port = pick_free_port()
    child_args = [
        "--task-name",
        "turning_on_radio",
        "--public-seed",
        "0",
        "--task-index",
        "0",
        "--activity-definition-id",
        "0",
        "--activity-instance-id",
        "242",
        "--scene-model",
        "test",
        "--max-episode-steps",
        "32",
        "--output-dir",
        str(tmp_path),
        "--behavior-repo",
        str(tmp_path),
        "--port",
        str(port),
        "--parent-watch",
    ]
    child_code = """
import signal
import threading
from robots.behavior import env_server, rlinf_env

closed_on = []
class Backend:
    def __init__(self, **kwargs):
        pass
    def close(self):
        closed_on.append(threading.get_ident())

rlinf_env.OfficialBehaviorBackend = Backend
env_server._build_meta = lambda args: {}
previous = signal.getsignal(signal.SIGTERM)
env_server.main()
assert closed_on == [threading.get_ident()]
assert signal.getsignal(signal.SIGTERM) == previous
print("OWNING_THREAD_CLEANUP_OK", flush=True)
"""
    log_path = tmp_path / "parent_watch.log"
    daemon = ProcessDaemon(
        name="behavior_parent_watch",
        cmd=[sys.executable, "-c", child_code, *child_args],
        log_path=str(log_path),
    )
    client = HttpRpcClient(f"http://127.0.0.1:{port}")
    try:
        daemon.start()
        wait_for_ready(client, timeout_s=10.0, poll_interval_s=0.01, daemon=daemon)
        daemon.stop()
        log = log_path.read_text()
        assert daemon.poll() == 0, log
        assert "OWNING_THREAD_CLEANUP_OK" in log
        assert "Fatal Python error" not in log
    finally:
        client.close()
        daemon.stop()
