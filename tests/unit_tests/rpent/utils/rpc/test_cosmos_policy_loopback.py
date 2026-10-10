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
import pytest

from robots.libero.wam_client import LiberoWAMClient, libero_request
from rpent.robots.components.cosmos_policy.adapter import make_adapter
from rpent.robots.components.cosmos_policy.server import CosmosPolicyFacade
from rpent.robots.components.wam_facade_base import BaseWAMFacade


@pytest.fixture
def raw_obs():
    image = np.zeros((256, 256, 3), dtype=np.uint8)
    image[0] = [1, 2, 3]
    image[-1] = [4, 5, 6]
    return {
        "agentview_image": image,
        "robot0_eye_in_hand_image": image + 10,
        "robot0_gripper_qpos": np.array([0.04, -0.04]),
        "robot0_eef_pos": np.array([0.1, 0.2, 0.3]),
        "robot0_eef_quat": np.array([0, 0, 0, -1.0]),
        "task_descriptions": "pick up the bowl",
        "segmentation": object(),
    }


@pytest.fixture
def facade():
    facade = CosmosPolicyFacade.__new__(CosmosPolicyFacade)
    capabilities, adapter = make_adapter("libero", "test")
    BaseWAMFacade.__init__(facade, capabilities, adapter=adapter)
    facade._cfg = SimpleNamespace(seed=1, num_denoising_steps_action=5)
    facade._model = facade._dataset_stats = None
    facade._predict_future = facade._cached_instructions_only = False
    facade._text_embeddings = {}
    facade._get_action = Mock(
        return_value={
            "actions": np.zeros((16, 7)),
            "future_image_predictions": {"image": np.zeros((2, 2, 3), np.uint8)},
            "value_prediction": 0.6,
        }
    )
    return facade


@pytest.mark.parametrize("predict_future", [False, True])
def test_cosmos_policy_roundtrip(
    transport, make_server_and_client, raw_obs, facade, predict_future
) -> None:
    facade._capabilities, facade._adapter = make_adapter(
        "libero", "test", predict_future=predict_future
    )
    facade._predict_future = predict_future
    with make_server_and_client(facade, transport) as rpc:
        result = LiberoWAMClient(rpc).predict_result(raw_obs, {"mode": "eval"})
    assert result.actions.shape == (16, 7)
    assert result.actions.dtype == np.float32
    assert (result.future_observation is not None) == predict_future
    assert result.value == (0.6 if predict_future else None)
    args, kwargs = facade._get_action.call_args
    assert args[4] == "pick up the bowl"
    assert kwargs == {
        "seed": 1,
        "num_denoising_steps_action": 5,
        "generate_future_state_and_value_in_parallel": predict_future,
    }
    received = args[3]
    np.testing.assert_array_equal(received["primary_image"][0, 0], [4, 5, 6])
    np.testing.assert_array_equal(received["wrist_image"][0, 0], [14, 15, 16])
    np.testing.assert_array_equal(raw_obs["agentview_image"][0, 0], [1, 2, 3])
    np.testing.assert_allclose(
        received["proprio"], [0.04, -0.04, 0.1, 0.2, 0.3, 0, 0, 0, -1]
    )


@pytest.mark.parametrize(
    "key,value",
    [
        ("agentview_image", np.zeros((128, 128, 3), np.uint8)),
        ("robot0_eye_in_hand_image", np.zeros((256, 256, 3), np.float32)),
        ("robot0_eef_quat", np.zeros(3)),
        ("robot0_gripper_qpos", np.array([np.nan, 0])),
    ],
)
def test_invalid_observation(raw_obs, key, value, facade):
    with pytest.raises(ValueError):
        facade.predict(libero_request({**raw_obs, key: value}))
    facade._get_action.assert_not_called()


@pytest.mark.parametrize(
    "actions", [np.zeros((1, 16, 7)), np.zeros((0, 7)), np.full((16, 7), np.inf)]
)
def test_client_rejects_invalid_chunks(raw_obs, actions, facade):
    def call(method, args=(), **kwargs):
        if method == "wam.capabilities":
            return facade.get_capabilities()
        result = facade.predict(args[0])
        return {**result, "actions": actions}

    client = LiberoWAMClient(Mock(call=call))
    with pytest.raises(ValueError):
        client.predict(raw_obs)


def test_cached_only_worker_rejects_uncached_instruction(raw_obs, facade):
    facade._cached_instructions_only = True
    with pytest.raises(ValueError, match="precomputed T5"):
        facade.predict(libero_request(raw_obs))
    facade._get_action.assert_not_called()
