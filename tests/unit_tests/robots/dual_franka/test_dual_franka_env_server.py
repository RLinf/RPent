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

"""Offline test for the dual-Franka RPC facade/client round-trip over HTTP."""

from __future__ import annotations

import threading

import numpy as np

from robots.dual_franka.env_client import DualFrankaEnvClient
from robots.dual_franka.env_server import DualFrankaEnvFacade
from rpent.utils.rpc.http_rpc import HttpRpcClient, HttpRpcServer


class FakeBackend:
    def get_env_meta(self):
        return {"ok": True}

    def reset(self):
        return {"ok": True}

    def get_robot_state(self):
        return {}

    def get_observation(self):
        return {}

    def get_camera_meta(self):
        return {}

    def move_delta(self, arm, delta_xyz):
        return {"arm": arm, "delta_xyz": np.asarray(delta_xyz)}

    def rotate_delta(self, arm, delta_rpy):
        return {"arm": arm, "delta_rpy": np.asarray(delta_rpy)}

    def set_gripper(self, arm, *, open):
        return {"arm": arm, "open": open}

    def chunk_step(self, actions, *, return_all_frames=False):
        return {"actions": np.asarray(actions), "return_all_frames": return_all_frames}

    def recover_joint_posture(self, *, reason="", return_to_start=True):
        return {"ok": True, "reason": reason, "return_to_start": return_to_start}


def test_dual_franka_client_and_facade_round_trip_numpy_over_http():
    facade = DualFrankaEnvFacade(FakeBackend())
    server = HttpRpcServer(("127.0.0.1", 0), facade._dispatch)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    try:
        client = DualFrankaEnvClient(HttpRpcClient(f"http://{host}:{port}"))
        result = client.move_delta("right", [0.01, 0.0, -0.02])
        recovery = client.recover_joint_posture(reason="joint drift")
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    assert result["arm"] == "right"
    np.testing.assert_allclose(result["delta_xyz"], [0.01, 0.0, -0.02])
    assert recovery == {
        "ok": True,
        "reason": "joint drift",
        "return_to_start": True,
    }
