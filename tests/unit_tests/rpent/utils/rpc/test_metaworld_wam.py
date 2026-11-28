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

"""MetaWorld platform adapter through the real HTTP/socket model boundary."""

import argparse
from unittest.mock import Mock

import numpy as np
import pytest

from robots.metaworld import robot_spec
from robots.metaworld.toolkit import MetaWorldToolkit
from robots.metaworld.wam_client import MetaWorldWAMClient
from rpent.robots.components.am_facade_base import BaseWAMFacade
from tests.unit_tests.robots.metaworld.test_wam import metadata, observation


class _SawyerWAM(BaseWAMFacade):
    def __init__(self, uses_sessions, **meta):
        self.requests = []
        self.resets = []
        super().__init__(
            {
                **metadata(),
                "backend": "test",
                "checkpoint": "test",
                "uses_sessions": uses_sessions,
                **meta,
            },
            adapter=(lambda x: x, lambda x: x),
        )

    def predict_native(self, request, *, session_id=None):
        self.requests.append(request)
        return {"actions": np.zeros((2, 4)), "future_observation": "not evidence"}

    def reset_native(self, *, session_id=None):
        self.resets.append(session_id)


@pytest.mark.parametrize("uses_sessions", [False, True])
def test_model_transport_reset_and_session_release(
    transport, make_server_and_client, uses_sessions
):
    facade = _SawyerWAM(uses_sessions)
    with make_server_and_client(facade, transport) as rpc:
        endpoint = (
            rpc._base_url if transport == "http" else f"socket://{rpc.host}:{rpc.port}"
        )
        model = MetaWorldWAMClient.connect(endpoint)
        result = model.predict(observation())
        assert result["actions"].shape == (2, 4)
        assert facade.requests[0]["instruction"] == "reach"
        assert set(facade.requests[0]["state"]) == {"eef_position", "gripper_opening"}
        model.close()
        assert len(facade.resets) == (2 if uses_sessions else 0)


class _Events:
    enabled = False

    def __init__(self):
        self.events = []

    def emit(self, event):
        self.events.append((event.component, event.status))


def _endpoint(rpc, transport):
    return rpc._base_url if transport == "http" else f"socket://{rpc.host}:{rpc.port}"


def _args(*argv):
    parser = argparse.ArgumentParser()
    robot_spec._add_cli_args(parser)
    return parser.parse_args(list(argv))


def test_runtime_rejects_incompatible_worker_before_starting_env(
    transport, make_server_and_client, monkeypatch, tmp_path
):
    facade = _SawyerWAM(False, camera_roles=["wrist"])
    spawn = Mock()
    monkeypatch.setattr(robot_spec, "_spawn", spawn)
    events = _Events()
    with make_server_and_client(facade, transport) as rpc:
        args = _args("--am-endpoint", _endpoint(rpc, transport))
        with pytest.raises(RuntimeError, match="primary camera"):
            robot_spec._init_runtime(args, tmp_path, events)
    spawn.assert_not_called()
    assert events.events == [("wam", "starting"), ("wam", "failed")]


@pytest.mark.parametrize("uses_sessions", [False, True])
def test_runtime_releases_worker_session_when_env_fails(
    transport, make_server_and_client, monkeypatch, tmp_path, uses_sessions
):
    facade = _SawyerWAM(uses_sessions)
    monkeypatch.setattr(robot_spec, "_spawn", Mock(side_effect=OSError("no simulator")))
    events = _Events()
    with make_server_and_client(facade, transport) as rpc:
        args = _args("--am-endpoint", _endpoint(rpc, transport))
        with pytest.raises(RuntimeError, match="no simulator"):
            robot_spec._init_runtime(args, tmp_path, events)
    assert len(facade.resets) == (2 if uses_sessions else 0)
    assert events.events == [
        ("wam", "starting"),
        ("wam", "ready"),
        ("env", "starting"),
        ("env", "failed"),
        ("wam", "pending"),
    ]


@pytest.mark.parametrize("endpoint", [None, "http://127.0.0.1:1"])
def test_dashboard_lists_wam_component_only_with_endpoint(endpoint, tmp_path):
    args = _args(*(["--am-endpoint", endpoint] if endpoint else []))
    spec = robot_spec._resolve_dashboard(args)
    names = [component["name"] for component in spec["runtime_components"]]
    assert names == (["env", "wam"] if endpoint else ["env"])
    assert all(c["scope"] == "unique" for c in spec["runtime_components"])
    assert ("wam_act" in spec["primitives"]) == bool(endpoint)
    assert robot_spec._init_runtime(args, tmp_path, _Events(), set()) == ([], {})
    if not endpoint:
        with pytest.raises(ValueError, match="wam"):
            robot_spec._init_runtime(args, tmp_path, _Events(), {"wam"})


def test_toolkit_releases_worker_when_construction_fails(tmp_path):
    from rpent.dashboard.events import NullDashboardEventSink
    from rpent.memory import MemoryManager
    from tests.unit_tests.robots.metaworld.test_contracts import toolkit

    env = toolkit().env
    env.last_obs = {**observation(), "steps": 0, "camera": {}}
    env.get_obs = Mock(side_effect=RuntimeError("state capture failed"))
    wam = Mock()
    with pytest.raises(RuntimeError, match="state capture failed"):
        MetaWorldToolkit(
            runtime_kwargs={"env_client": env, "wam_client": wam},
            dashboard_events=NullDashboardEventSink(),
            memory=MemoryManager(tmp_path / "memory"),
            state_output_dir=tmp_path / "run",
        )
    wam.close.assert_called_once()
    env.step.assert_not_called()
