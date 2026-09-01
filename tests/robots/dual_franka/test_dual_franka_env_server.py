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

"""Offline tests for the import-light dual-Franka RPC facade and action math."""

from __future__ import annotations

import threading

import numpy as np
import pytest

from robots.dual_franka.env_client import DualFrankaEnvClient
from robots.dual_franka.env_server import (
    _matrix_to_rot6d,
    _pack_dual_action,
)
from robots.franka.env_server import FrankaEnvFacade, _to_numpy_tree
from rpent.utils.rpc.http_rpc import HttpRpcClient, HttpRpcServer


class FakeBackend:
    def ready(self):
        return {"ok": True}

    def move_delta(self, arm, delta_xyz):
        return {"arm": arm, "delta_xyz": np.asarray(delta_xyz)}


def test_facade_dispatches_only_explicit_env_methods():
    facade = FrankaEnvFacade(FakeBackend())

    assert facade._dispatch("env.ready", (), {}) == {"ok": True}
    result = facade._dispatch(
        "env.move_delta", (), {"arm": "left", "delta_xyz": [1, 2, 3]}
    )
    assert result["arm"] == "left"
    np.testing.assert_array_equal(result["delta_xyz"], [1, 2, 3])
    with pytest.raises(ValueError, match="unknown Franka env method"):
        facade._dispatch("env.delete_everything", (), {})


def test_matrix_to_rot6d_uses_first_two_columns():
    np.testing.assert_allclose(_matrix_to_rot6d(np.eye(3)), [1, 0, 0, 0, 1, 0])


def test_pack_dual_action_has_20d_layout_with_gripper_slots():
    rot6d = _matrix_to_rot6d(np.eye(3))
    action = _pack_dual_action(
        [0.1, 0.2, 0.3], rot6d, [0.4, 0.5, 0.6], rot6d, left_grip=1.0, right_grip=-1.0
    )

    assert action.shape == (20,)
    assert action.dtype == np.float32
    np.testing.assert_allclose(action[:3], [0.1, 0.2, 0.3])
    assert action[9] == 1.0
    np.testing.assert_allclose(action[10:13], [0.4, 0.5, 0.6])
    assert action[19] == -1.0


def test_to_numpy_tree_converts_nested_numpy_scalars_and_dataclasses():
    value = {"items": [np.float32(1.5), (np.int64(2),)]}

    assert _to_numpy_tree(value) == {"items": [1.5, (2,)]}


def test_dual_franka_client_and_facade_round_trip_numpy_over_http():
    facade = FrankaEnvFacade(FakeBackend())
    server = HttpRpcServer(("127.0.0.1", 0), facade._dispatch)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    try:
        client = DualFrankaEnvClient(HttpRpcClient(f"http://{host}:{port}"))
        result = client.move_delta("right", [0.01, 0.0, -0.02])
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    assert result["arm"] == "right"
    np.testing.assert_allclose(result["delta_xyz"], [0.01, 0.0, -0.02])
