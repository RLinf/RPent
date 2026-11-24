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

import numpy as np
import pytest

from robots.metaworld.toolkit import MetaWorldToolkit
from robots.metaworld.wam_client import MetaWorldWAMClient
from rpent.robots.components.am_facade_base import BaseWAMFacade
from tests.unit_tests.robots.metaworld.test_wam import metadata, observation


class _SawyerWAM(BaseWAMFacade):
    def __init__(self, uses_sessions):
        self.requests = []
        self.resets = []
        super().__init__(
            {
                **metadata(),
                "backend": "test",
                "checkpoint": "test",
                "uses_sessions": uses_sessions,
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


def test_toolkit_releases_session_when_contract_rejected(monkeypatch, tmp_path):
    from unittest.mock import Mock

    from rpent.dashboard.events import NullDashboardEventSink
    from rpent.memory import MemoryManager
    from tests.unit_tests.robots.metaworld.test_contracts import toolkit

    obs = {**observation(), "steps": 0, "depth": np.ones((3, 4)), "camera": {}}
    env = toolkit().env
    env.last_obs = obs
    env.get_obs = lambda: obs
    rpc = Mock()
    rpc.call.return_value = {**metadata(), "camera_roles": ["wrist"]}
    monkeypatch.setattr("rpent.utils.rpc.make_rpc_client", lambda *a, **kw: rpc)
    monkeypatch.setattr("rpent.utils.rpc.wait_for_ready", lambda *a, **kw: None)
    with pytest.raises(ValueError, match="primary"):
        MetaWorldToolkit(
            runtime_kwargs={"env_client": env, "am_endpoint": "test"},
            dashboard_events=NullDashboardEventSink(),
            memory=MemoryManager(tmp_path / "memory"),
            state_output_dir=tmp_path / "run",
        )
    rpc.close.assert_called_once()
    env.step.assert_not_called()
