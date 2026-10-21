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

"""A session-based joint controller exercises the same RPC as stateless Cosmos."""

from contextlib import ExitStack

import numpy as np
import pytest

from rpent.robots.components.am_client_base import BaseWAMClient
from rpent.robots.components.am_contracts import WAMModelMeta, WAMPrediction
from rpent.robots.components.am_facade_base import BaseWAMFacade
from rpent.utils.rpc import RpcError, wait_for_ready


class _JointWAM(BaseWAMFacade):
    def __init__(self):
        self.history = {}
        super().__init__(
            WAMModelMeta(
                backend="test_joint_wam",
                checkpoint="test",
                control={
                    "embodiment": "test_arm",
                    "action_space": "test.joint_position.v1",
                    "action_schema": [f"joint_{i}" for i in range(7)] + ["gripper"],
                    "action_type": None,
                },
                camera_roles=("exterior_left", "exterior_right", "wrist"),
                state_schema={"joint_position": 7, "gripper_position": 1},
                uses_sessions=True,
                chunk_size=None,
            ),
            adapter=(lambda request: request, lambda result: result),
        )

    def predict_native(self, request, *, session_id=None):
        count = self.history.get(session_id, 0) + 1
        self.history[session_id] = count
        # Variable horizons and dimensions must survive both transports.
        return WAMPrediction(actions=np.full((count, 8), count, np.float32))

    def reset_native(self, *, session_id=None):
        self.history.pop(session_id, None)


def test_session_history_isolation_reset_and_cleanup(transport, make_server_and_client):
    facade = _JointWAM()
    request = {
        "instruction": "move",
        "embodiment": "test_arm",
        "action_space": "test.joint_position.v1",
        "images": {
            role: np.zeros((2, 3, 3), np.uint8)
            for role in facade._model_meta["camera_roles"]
        },
        "state": {"joint_position": np.zeros(7), "gripper_position": np.zeros(1)},
    }
    with ExitStack() as stack:
        clients = []
        for _ in range(2):
            rpc = stack.enter_context(
                make_server_and_client(facade, transport, enable_sessions=True)
            )
            wait_for_ready(rpc, timeout_s=5)
            clients.append(BaseWAMClient(rpc))
        first, second = clients
        assert first.model_meta["uses_sessions"]
        assert first.predict(request)["actions"].shape == (1, 8)
        assert first.predict(request)["actions"].shape == (2, 8)
        assert second.predict(request)["actions"].shape == (1, 8)
        first.reset()
        assert first.predict(request)["actions"].shape == (1, 8)
        assert second.predict(request)["actions"].shape == (2, 8)
        first.close()
        assert len(facade.history) == 1
        facade.close()
        assert facade.history == {}
        with pytest.raises(RpcError, match="session not found"):
            second.predict(request)
        second.close()
