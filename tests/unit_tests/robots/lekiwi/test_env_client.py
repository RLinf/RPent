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

"""Offline tests for the LeKiwi RPC client contract."""

from __future__ import annotations

import numpy as np

from robots.lekiwi.env_client import LeKiwiEnvClient


class FakeRpcClient:
    """Record calls and return deterministic method-tagged payloads."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple, dict, float | None]] = []

    def call(
        self,
        method: str,
        args: tuple = (),
        kwargs: dict | None = None,
        *,
        timeout_s: float | None = None,
    ):
        payload = kwargs or {}
        self.calls.append((method, args, payload, timeout_s))
        return {"method": method, **payload}


def test_lekiwi_client_uses_explicit_env_methods():
    rpc = FakeRpcClient()
    client = LeKiwiEnvClient(rpc)

    client.drive([0.1, 0.0, 5.0])
    client.hold([0.2, 0.0, 0.0], ttl_s=0.75)
    client.stop()

    calls_by_method = {
        method: payload for method, _args, payload, _timeout in rpc.calls
    }
    assert set(calls_by_method) == {
        "env.get_env_meta",
        "env.reset",
        "env.drive",
        "env.hold",
        "env.stop",
    }
    drive = calls_by_method["env.drive"]
    np.testing.assert_allclose(
        drive["velocity"], np.array([0.1, 0.0, 5.0], dtype=np.float32)
    )
    assert drive["velocity"].dtype == np.float32
    assert drive["duration_s"] == 1.0
    hold = calls_by_method["env.hold"]
    np.testing.assert_allclose(
        hold["velocity"], np.array([0.2, 0.0, 0.0], dtype=np.float32)
    )
    assert hold["ttl_s"] == 0.75


def test_lekiwi_client_drive_passes_requested_duration():
    rpc = FakeRpcClient()
    client = LeKiwiEnvClient(rpc)

    client.drive([0.0, -0.2, 0.0], duration_s=0.5)

    _method, _args, payload, _timeout = rpc.calls[-1]
    assert payload["duration_s"] == 0.5


def test_lekiwi_client_backfills_states_from_last_observation():
    rpc = FakeRpcClient()
    rpc_responses = {
        "env.get_env_meta": {"action_dim": 3},
        "env.reset": {"ok": True, "states": np.zeros(3, dtype=np.float32)},
        "env.get_observation": {},
        "env.get_robot_state": {"velocity": [0.0, 0.0, 0.0]},
    }

    def call(method, args=(), kwargs=None, *, timeout_s=None):
        return rpc_responses[method]

    rpc.call = call
    client = LeKiwiEnvClient(rpc)

    observation = client.get_observation()
    robot_state = client.get_robot_state()

    np.testing.assert_array_equal(observation["states"], np.zeros(3))
    np.testing.assert_array_equal(robot_state["wrapped_state_vector"], np.zeros(3))
