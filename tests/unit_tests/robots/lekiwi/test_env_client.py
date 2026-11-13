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
import pytest

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

    client.drive([0.2, 0.0, 0.0], 2)
    client.rotate(90)
    client.stop()

    methods = [method for method, *_ in rpc.calls]
    assert methods == [
        "env.get_env_meta",
        "env.reset",
        "env.drive",
        "env.rotate",
        "env.stop",
    ]
    drive, rotate = rpc.calls[2:4]
    assert drive[2] == {"velocity": [0.2, 0.0, 0.0], "duration_s": 2}
    assert rotate[2] == {"angle_deg": 90}
    assert drive[3] > 11
    assert rotate[3] > 61


def test_lekiwi_client_preserves_server_state_without_stale_backfill():
    rpc = FakeRpcClient()
    rpc_responses = {
        "env.get_env_meta": {"action_dim": 3},
        "env.reset": {"ok": True, "states": np.zeros(3, dtype=np.float32)},
        "env.get_observation": {},
        "env.get_robot_state": {"velocity": [0.2, 0.0, 0.0]},
    }

    def call(method, args=(), kwargs=None, *, timeout_s=None):
        return rpc_responses[method]

    rpc.call = call
    client = LeKiwiEnvClient(rpc)

    observation = client.get_observation()
    robot_state = client.get_robot_state()

    assert observation == {}
    assert robot_state == {"velocity": [0.2, 0.0, 0.0]}
    assert client.last_obs == observation


@pytest.mark.parametrize(
    "method,args", [("drive", ([0.1, 0, 0], 1)), ("rotate", (90,))]
)
def test_motion_call_returns_final_result_in_one_request(method, args):
    rpc = FakeRpcClient()
    client = LeKiwiEnvClient(rpc, reset_on_connect=False)
    calls = []
    result = {"status": "succeeded", "stop_confirmed": True}

    def call(name, args=(), kwargs=None, *, timeout_s=None):
        calls.append(name)
        return result

    rpc.call = call
    assert getattr(client, method)(*args) is result
    assert calls == [f"env.{method}"]


def test_motion_failure_does_not_retry_or_send_a_late_stop():
    rpc = FakeRpcClient()
    client = LeKiwiEnvClient(rpc, reset_on_connect=False)
    calls = []

    def call(method, args=(), kwargs=None, *, timeout_s=None):
        calls.append(method)
        raise TimeoutError("response unavailable")

    rpc.call = call
    with pytest.raises(TimeoutError, match="response unavailable"):
        client.rotate(90)
    assert calls == ["env.rotate"]
