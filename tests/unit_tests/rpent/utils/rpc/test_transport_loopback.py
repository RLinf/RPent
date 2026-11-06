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

import io
import os
import pickle
import socket
import threading
import time
import urllib.request
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from typing import Literal

import numpy as np
import pytest

from rpent.robots.components.env_client_base import BaseEnvClient
from rpent.robots.components.env_facade_base import BaseEnvFacade
from rpent.session.records import Payload
from rpent.utils.daemon import pick_free_port
from rpent.utils.rpc import (
    RpcClient,
    RpcError,
    RpcFacade,
    make_rpc_client,
    wait_for_ready,
)
from rpent.utils.rpc.http_rpc import HttpRpcClient, _is_direct_url
from rpent.utils.rpc.socket_rpc import _write_frame

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


def test_robodojo_camera_meta_roundtrip(transport, make_server_and_client):
    from robots.robodojo.env_server import RoboDojoEnvFacade

    camera = {
        "intrinsic_matrix": np.eye(3, dtype=np.float32),
        "extrinsic_matrix": np.eye(4, dtype=np.float64),
        "color": np.zeros((2, 3, 3), dtype=np.uint8),
    }
    env = SimpleNamespace(
        eval_fair=False,
        get_camera_meta=lambda *args: {
            "camera_name": "cam_head",
            "height": 2,
            "width": 3,
            **{key: camera[key] for key in ("intrinsic_matrix", "extrinsic_matrix")},
        },
    )
    facade = RoboDojoEnvFacade(env)
    with make_server_and_client(facade, transport) as client:
        result = client.call("env.get_camera_meta", kwargs={"camera_name": "cam_head"})
    assert result["camera_name"] == "cam_head"
    assert (result["height"], result["width"]) == (2, 3)
    for key in ("intrinsic_matrix", "extrinsic_matrix"):
        np.testing.assert_array_equal(result[key], camera[key])
        assert result[key].dtype == camera[key].dtype


def test_numpy_array_frame_loads_without_numpy2_numeric_module() -> None:
    # NumPy 1.26.4 on the model host lacks numpy._core.numeric. A NumPy 2
    # protocol-5 array request therefore closes the socket before dispatch.
    class ModelHostUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if module == "numpy._core.numeric":
                raise ModuleNotFoundError("No module named 'numpy._core.numeric'")
            return super().find_class(module, name)

    frames = io.BytesIO()
    values = np.arange(24, dtype=np.uint8).reshape(2, 4, 3)
    _write_frame(frames, {"image": values})
    decoded = ModelHostUnpickler(io.BytesIO(frames.getvalue()[4:])).load()
    np.testing.assert_array_equal(decoded["image"], values)


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
@pytest.mark.parametrize("shape", [(2, 3), (480, 640, 3)])
def test_transport_round_trips_nested_numpy_payloads(
    transport: Transport, shape: tuple[int, ...]
) -> None:
    original = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)

    with _running_facade(transport) as running:
        result = running.client.call(
            "combine",
            args=(original,),
            kwargs={
                "scale": 2.5,
                "metadata": {
                    "count": np.int64(original.size),
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
        assert metadata["count"] == original.size
        assert isinstance(metadata["valid"], np.bool_)
        assert metadata["valid"] == np.bool_(True)
        assert isinstance(metadata["score"], np.float32)
        assert metadata["score"] == np.float32(1.5)
        assert metadata["labels"] == ["left", "right"]
        result["values"].flat[0] = -1
        assert original.flat[0] == 0


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


class _CaptureEnv(BaseEnvFacade):
    def __init__(self, tuple_size=5):
        self.tuple_size = tuple_size
        super().__init__()

    def get_env_meta(self):
        return {}

    def _observation(self):
        return {
            "state": np.array([self.position], dtype=np.float32),
            "image": np.full((2, 3, 3), self.count, dtype=np.uint8),
        }

    def reset(self):
        self.position = 0.0
        self.count = 0
        return self._observation()

    def step(self, action, *, capture=False):
        before = self._observation() if capture else None
        if action[0] < 0:
            raise ValueError("environment step failed")
        native_action = np.clip(action, 0, 1).astype(np.float32)
        self.position += float(native_action[0])
        self.count += 1
        obs = self._observation()
        reward = self.position
        done = self.count == 2
        info = {"count": self.count}
        result = (obs, reward, done, info)
        if self.tuple_size == 5:
            result = (obs, reward, done, False, info)
        if not capture:
            return result
        return {
            "result": result,
            "before_observation": asdict(
                Payload("test.observation", before, completeness="complete")
            ),
            "steps": [
                {
                    "action": asdict(
                        Payload("test.action", native_action, completeness="complete")
                    ),
                    "after_observation": asdict(
                        Payload("test.observation", obs, completeness="complete")
                    ),
                    "result": asdict(
                        Payload(
                            "test.result",
                            {"reward": reward, "done": done, "info": info},
                            completeness="complete",
                        )
                    ),
                }
            ],
        }

    def chunk_step(self, actions, *, return_all_frames=False):
        results = []
        for action in actions:
            result = self.step(action)
            results.append(result)
            if result[2]:
                break
        obs = [result[0] for result in results]
        result = (
            obs if return_all_frames else obs[-1],
            np.array([result[1] for result in results]),
            np.array([result[2] for result in results]),
            {"executed_actions": len(results)},
        )
        if self.tuple_size == 5:
            result = (*result[:3], np.zeros(len(results), dtype=bool), result[-1])
        return result


@pytest.mark.parametrize("tuple_size", [4, 5])
@pytest.mark.parametrize("value", [0.5, 2.0])
def test_step_capture_preserves_result_and_executed_action(
    transport, make_server_and_client, tuple_size, value
):
    facade = _CaptureEnv(tuple_size)
    action = np.array([value], dtype=np.float64)
    with make_server_and_client(facade, transport) as rpc:
        env = BaseEnvClient(rpc, expected_meta={})
        original = env.step(action)
        original_obs = env.last_obs
        env.reset()
        captured = env.step(action, capture=True)

    np.testing.assert_equal(captured["result"], original)
    np.testing.assert_equal(env.last_obs, original_obs)
    before = captured["before_observation"]
    assert before["format"] == "test.observation"
    np.testing.assert_array_equal(before["data"]["state"], [0.0])
    (step,) = captured["steps"]
    assert step["action"]["format"] == "test.action"
    assert step["action"]["completeness"] == "complete"
    assert step["action"]["data"].dtype == np.float32
    np.testing.assert_array_equal(step["action"]["data"], [min(value, 1)])
    after = step["after_observation"]
    assert after["format"] == "test.observation"
    assert after["data"]["state"].dtype == np.float32
    np.testing.assert_array_equal(after["data"]["state"], [min(value, 1)])
    assert after["data"]["image"].dtype == np.uint8
    np.testing.assert_array_equal(
        after["data"]["image"], np.full((2, 3, 3), 1, dtype=np.uint8)
    )
    assert step["result"]["format"] == "test.result"
    assert step["result"]["data"]["info"] == {"count": 1}


@pytest.mark.parametrize("tuple_size", [4, 5])
@pytest.mark.parametrize("all_frames", [False, True])
def test_chunk_returns_executed_frames_and_caches_final_observation(
    transport, make_server_and_client, tuple_size, all_frames
):
    facade = _CaptureEnv(tuple_size)
    with make_server_and_client(facade, transport) as rpc:
        env = BaseEnvClient(rpc, expected_meta={})
        result = env.chunk_step(
            np.array([[0.5], [2.0], [0.75]]), return_all_frames=all_frames
        )

    assert facade.count == 2
    assert len(result) == tuple_size
    assert result[-1] == {"executed_actions": 2}
    np.testing.assert_array_equal(result[1], [0.5, 1.5])
    np.testing.assert_array_equal(result[2], [False, True])
    observations = result[0] if all_frames else [result[0]]
    assert len(observations) == (2 if all_frames else 1)
    for count, obs in zip((1, 2) if all_frames else (2,), observations):
        np.testing.assert_array_equal(obs["state"], [0.5 if count == 1 else 1.5])
        np.testing.assert_array_equal(
            obs["image"], np.full((2, 3, 3), count, dtype=np.uint8)
        )
    assert env.last_obs is observations[-1]


def test_failed_chunk_raises_without_updating_cached_observation(
    transport, make_server_and_client
):
    facade = _CaptureEnv()
    with make_server_and_client(facade, transport) as rpc:
        env = BaseEnvClient(rpc, expected_meta={})
        before = env.last_obs
        with pytest.raises(RpcError, match="environment step failed"):
            env.chunk_step(np.array([[0.5], [-1.0]]))
        assert facade.count == 1
        assert env.last_obs is before


def test_disabled_capture_supports_existing_facade_signatures(
    transport, make_server_and_client
):
    class ExistingEnv(_CaptureEnv):
        def step(self, action):
            return super().step(action)

        def chunk_step(self, actions, *, return_all_frames=False):
            results = [self.step(action) for action in actions]
            return results[-1]

    with make_server_and_client(ExistingEnv(), transport) as rpc:
        env = BaseEnvClient(rpc, expected_meta={})
        assert env.step(np.array([0.5]), capture=False)[1] == 0.5
        assert env.chunk_step(np.array([[0.5]]))[1] == 1.0
