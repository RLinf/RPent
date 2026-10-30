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
"""The R2 runtime hook, client, and env server over a real local HTTP transport."""

from __future__ import annotations

import threading
from argparse import Namespace

import numpy as np
import pytest

from robots.r2 import robot_spec
from robots.r2.diagnostics import check
from robots.r2.env_server import R2EnvFacade
from rpent.dashboard.events import NullDashboardEventSink
from rpent.utils.daemon import pick_free_port
from rpent.utils.rpc import make_rpc_client


class FakeAgentEnv:
    def __init__(self, protocol: int = 1) -> None:
        self.protocol = protocol
        self.calls: list[tuple] = []
        self.closed = False

    def get_env_meta(self):
        return {"robot": "r2", "protocol": self.protocol, "base_motion": False}

    def reset(self):
        self.calls.append(("reset",))
        return {"ok": True}

    def observe(self):
        return {
            "views": {"head": {"depth_mm": np.arange(6, dtype=np.uint16).reshape(2, 3)}}
        }

    def robot_state(self):
        return {"frame": "base_link"}

    def get_camera_meta(self):
        return {"cameras": {}}

    def move_delta(self, arm, delta_xyz):
        self.calls.append(("move_delta", arm, delta_xyz))
        return {"ok": True}

    def close(self):
        self.closed = True

    def __getattr__(self, name):
        return lambda **kwargs: {"ok": True, name: kwargs}


@pytest.fixture
def served():
    def serve(env):
        port = pick_free_port()
        facade = R2EnvFacade(env)
        thread = threading.Thread(
            target=facade.serve,
            kwargs={"transport": "http", "host": "127.0.0.1", "port": port},
            daemon=True,
        )
        thread.start()
        started.append((port, thread))
        return f"http://127.0.0.1:{port}"

    started: list = []
    yield serve
    for port, thread in started:
        make_rpc_client(f"http://127.0.0.1:{port}").call("shutdown", timeout_s=5)
        thread.join(timeout=5)


def _args(endpoint):
    return Namespace(env_endpoint=endpoint, dashboard=False)


def test_runtime_attaches_to_the_robot_server_without_owning_it(served):
    env = FakeAgentEnv()
    daemons, runtime_kwargs = robot_spec._init_runtime(
        _args(served(env)), None, NullDashboardEventSink(), None
    )

    client = runtime_kwargs["env"]
    depth = client.observe()["views"]["head"]["depth_mm"]
    client.move_delta("left", [0.0, 0.0, 0.01])

    assert daemons == []
    assert depth.dtype == np.uint16 and depth.tolist() == [[0, 1, 2], [3, 4, 5]]
    assert env.calls == [("reset",), ("move_delta", "left", [0.0, 0.0, 0.01])]


def test_client_refuses_a_server_with_another_protocol(served):
    with pytest.raises(RuntimeError, match="speaking protocol 1"):
        robot_spec._init_runtime(
            _args(served(FakeAgentEnv(protocol=2))),
            None,
            NullDashboardEventSink(),
            None,
        )


def test_runtime_rejects_unknown_components_and_a_missing_endpoint():
    with pytest.raises(ValueError, match="unknown R2 runtime components"):
        robot_spec._init_runtime(_args(None), None, NullDashboardEventSink(), {"vla"})
    with pytest.raises(ValueError, match="--env-endpoint is required"):
        robot_spec._init_runtime(_args(None), None, NullDashboardEventSink(), None)


def test_task_language_replaces_the_instruction(tmp_path):
    config = robot_spec._parse_config(
        Namespace(task_id=1, task_language="pick up the red cup", output_dir=tmp_path)
    )

    assert config.prompt_vars["instruction"] == "pick up the red cup"
    assert config.task_desc == {
        "task_id": 1,
        "task_name": "tabletop_pick_place",
        "instruction": "pick up the red cup",
    }


class ObservedAgentEnv(FakeAgentEnv):
    def get_env_meta(self):
        return {**super().get_env_meta(), "limits": {"max_delta_m": 0.1}}

    def observe(self):
        return {
            "robot_state": {
                "health": {"healthy": True},
                "arms": {"left": {"tcp_xyz": [0.1, 0.2, 0.7]}},
            },
            "views": {
                "head": {
                    "rgb": np.zeros((4, 6, 3), dtype=np.uint8),
                    "depth_mm": np.array([[0, 1, 1], [1, 1, 1]], dtype=np.uint16),
                },
                "wrist_left": {"error": "camera 'wrist_left' has delivered no frame"},
            },
            "camera_meta": {
                "cameras": {
                    "head": {"intrinsics": None, "T_base_camera": np.eye(4).tolist()}
                }
            },
        }


def test_diagnostics_reports_health_and_cameras_without_moving(served, tmp_path):
    env = ObservedAgentEnv()

    report = check(served(env), tmp_path / "check")

    assert env.calls == []
    assert report["health"] == {"healthy": True}
    assert report["tcp"] == {"left": [0.1, 0.2, 0.7]}
    assert report["cameras"]["head"] == {
        "image_shape": [4, 6],
        "valid_depth": 0.833,
        "calibrated": False,
        "posed": True,
    }
    assert "error" in report["cameras"]["wrist_left"]
    assert (tmp_path / "check" / "head.png").exists()
