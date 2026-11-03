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

"""The R2 env server driving RLinf's simulated controller through the RPent client."""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from robots.r2.env_client import R2EnvClient
from rpent.utils.rpc import make_rpc_client
from tests.e2e_tests.common import publish_check, require_depth, require_rgb


@pytest.mark.timeout(300)
def test_environment_component(r2_mock_endpoint: str) -> None:
    # Connecting resets the robot: both arms move to the rest pose.
    client = R2EnvClient(make_rpc_client(r2_mock_endpoint))
    assert client.meta["cameras"] == ["head", "wrist_left", "wrist_right"]

    observation = client.observe()
    for name, view in observation["views"].items():
        require_depth(view["depth_mm"], f"{name} depth", require_rgb(view["rgb"], name))
    state = observation["robot_state"]
    assert state["health"]["healthy"] is True
    start = np.asarray(state["arms"]["left"]["tcp_xyz"])

    moved = client.move_delta("left", [0.05, 0.0, 0.03])
    assert moved["ok"] is True, moved
    np.testing.assert_allclose(
        moved["final_tcp"][:3], start + [0.05, 0.0, 0.03], atol=0.01
    )
    looked = client.look(0.0, -0.4)
    assert looked["ok"] is True and looked["neck"][1] == pytest.approx(-0.4, abs=0.01)
    gripped = client.set_gripper("right", 0.3)
    assert gripped["ok"] is True
    assert gripped["measured_opening"] == pytest.approx(0.3, abs=0.02)

    # A stop from another connection ends a running move early.
    moving: dict = {}
    mover = threading.Thread(
        target=lambda: moving.update(client.move_to("left", [0.3, 0.25, 0.85]))
    )
    mover.start()
    time.sleep(0.5)
    assert make_rpc_client(r2_mock_endpoint).call("env.request_stop") == {
        "ok": True,
        "stopped": True,
    }
    mover.join(timeout=30.0)
    assert moving["ok"] is False and moving["stopped"] is True, moving
    stopped_at = np.asarray(moving["final_tcp"][:3])
    assert np.linalg.norm(stopped_at - [0.3, 0.25, 0.85]) > 0.02

    home = client.go_home()
    assert home["ok"] is True, home
    np.testing.assert_allclose(
        home["robot_state"]["arms"]["left"]["tcp_xyz"], start, atol=0.01
    )
    publish_check(
        "r2_environment_component",
        {"moved": moved["final_tcp"], "stopped_at": stopped_at.tolist()},
    )
