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

"""Offline tests for the dual-Franka RPC client contract."""

from __future__ import annotations

import numpy as np

from robots.dual_franka.env_client import DualFrankaEnvClient


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


def test_dual_franka_client_uses_explicit_env_methods():
    rpc = FakeRpcClient()
    client = DualFrankaEnvClient(rpc)

    client.reset()
    client.move_delta("left", [0.01, 0.0, -0.02])
    client.rotate_delta("right", [0.0, 0.0, 0.1])
    client.set_gripper("left", open=True)
    client.step_chunk(np.zeros((2, 20), dtype=np.float64))

    assert [call[0] for call in rpc.calls] == [
        "env.ready",
        "env.reset",
        "env.move_delta",
        "env.rotate_delta",
        "env.set_gripper",
        "env.step_chunk",
    ]
    assert rpc.calls[2][2]["arm"] == "left"
    np.testing.assert_allclose(
        rpc.calls[2][2]["delta_xyz"],
        np.array([0.01, 0.0, -0.02], dtype=np.float32),
    )
    assert rpc.calls[3][2]["arm"] == "right"
    assert rpc.calls[4][2] == {"arm": "left", "open": True}
    assert rpc.calls[5][2]["actions"].dtype == np.float32
