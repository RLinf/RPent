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

"""Exercise Cosmos raw observations across the real RPent transports."""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from robots.libero.cosmos_policy_client import CosmosPolicyClient
from robots.libero.cosmos_policy_server import CosmosPolicyFacade
from rpent.robots.components.vla_facade_base import BaseVLAFacade


def test_cosmos_policy_roundtrip(transport, make_server_and_client) -> None:
    facade = CosmosPolicyFacade.__new__(CosmosPolicyFacade)
    BaseVLAFacade.__init__(facade)
    facade._cfg = SimpleNamespace(seed=1, num_denoising_steps_action=5)
    facade._model = facade._dataset_stats = None
    facade._get_action = Mock(return_value={"actions": np.zeros((16, 7))})
    raw = {
        "agentview_image": np.zeros((256, 256, 3), np.uint8),
        "robot0_eye_in_hand_image": np.ones((256, 256, 3), np.uint8),
        "robot0_gripper_qpos": np.array([0.04, -0.04]),
        "robot0_eef_pos": np.array([0.1, 0.2, 0.3]),
        "robot0_eef_quat": np.array([0, 0, 0, 1.0]),
        "task_descriptions": "pick up the bowl",
    }
    with make_server_and_client(facade, transport) as rpc:
        actions = CosmosPolicyClient(rpc).predict(raw, {"mode": "eval"})
    assert actions.shape == (16, 7)
    assert actions.dtype == np.float32
    received = facade._get_action.call_args.args[3]
    np.testing.assert_allclose(
        received["proprio"], [0.04, -0.04, 0.1, 0.2, 0.3, 0, 0, 0, 1]
    )
