"""Offline tests for the import-light Franka RPC facade."""

from __future__ import annotations

import threading

import numpy as np
import pytest

from robots.franka.env_client import FrankaEnvClient
from robots.franka.env_server import FrankaEnvFacade, _to_numpy_tree
from rpent.utils.rpc.http_rpc import HttpRpcClient, HttpRpcServer


class FakeBackend:
    def ready(self):
        return {"ok": True}

    def move_delta(self, delta_xyz):
        return {"delta_xyz": np.asarray(delta_xyz)}


def test_facade_dispatches_only_explicit_env_methods():
    facade = FrankaEnvFacade(FakeBackend())

    assert facade._dispatch("env.ready", (), {}) == {"ok": True}
    result = facade._dispatch("env.move_delta", (), {"delta_xyz": [1, 2, 3]})
    np.testing.assert_array_equal(result["delta_xyz"], [1, 2, 3])
    with pytest.raises(ValueError, match="unknown Franka env method"):
        facade._dispatch("env.delete_everything", (), {})


def test_to_numpy_tree_converts_nested_numpy_scalars_and_dataclasses():
    value = {"items": [np.float32(1.5), (np.int64(2),)]}

    assert _to_numpy_tree(value) == {"items": [1.5, (2,)]}


def test_franka_client_and_facade_round_trip_numpy_over_http():
    facade = FrankaEnvFacade(FakeBackend())
    server = HttpRpcServer(("127.0.0.1", 0), facade._dispatch)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    try:
        client = FrankaEnvClient(HttpRpcClient(f"http://{host}:{port}"))
        result = client.move_delta([0.01, 0.0, -0.02])
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    np.testing.assert_allclose(result["delta_xyz"], [0.01, 0.0, -0.02])
