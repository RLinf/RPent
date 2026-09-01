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

import socket
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
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

Transport = Literal["http", "socket"]


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
def _running_facade(transport: Transport) -> Iterator[RunningFacade]:
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
    client = make_rpc_client(f"{transport}://127.0.0.1:{port}")

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
