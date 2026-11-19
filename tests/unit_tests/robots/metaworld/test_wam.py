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

"""Model contracts and bounded execution without simulator dependencies."""

from unittest.mock import Mock

import numpy as np
import pytest

from robots.metaworld.control import METAWORLD_CONTROL
from robots.metaworld.wam_client import MetaWorldWAMClient
from tests.unit_tests.robots.metaworld.test_contracts import toolkit


def metadata():
    return {
        "control": dict(METAWORLD_CONTROL),
        "camera_roles": ["primary"],
        "state_schema": {"eef_position": 3, "gripper_opening": 1},
        "chunk_size": None,
        "uses_sessions": False,
    }


def observation():
    return {
        "rgb": np.zeros((3, 4, 3), np.uint8),
        "eef_position": [0.0, 0.0, 0.0],
        "gripper_opening": 0.5,
        "instruction": "reach",
        "terminated": False,
        "truncated": False,
    }


@pytest.mark.parametrize(
    "change",
    [
        {"control": {**METAWORLD_CONTROL, "embodiment": "libero_7d"}},
        {"camera_roles": ["wrist"]},
        {"state_schema": {"joint_positions": 7}},
        {"joint_target_history_shape": (2, 4)},
    ],
)
def test_rejects_incompatible_worker_before_prediction(change):
    rpc = Mock()
    rpc.call.return_value = {**metadata(), **change}
    with pytest.raises(ValueError):
        MetaWorldWAMClient(rpc)
    assert rpc.call.call_count == 1


@pytest.mark.parametrize(
    "actions",
    [
        np.zeros((2, 7)),
        np.zeros((0, 4)),
        np.full((2, 4), np.nan),
        np.full((2, 4), 1.000001),
    ],
)
def test_rejects_bad_chunk_before_any_step(actions):
    rpc = Mock()
    rpc.call.side_effect = [metadata(), {"actions": actions}]
    model = MetaWorldWAMClient(rpc)
    tk = toolkit()
    tk._wam = model
    tk.env.get_obs = Mock(return_value=observation())
    with pytest.raises(ValueError):
        tk.wam_act()
    tk.env.step.assert_not_called()


def test_chunks_refresh_observations_and_stop_at_terminal():
    rpc = Mock()
    rpc.call.side_effect = [metadata(), {"actions": np.zeros((3, 4))}]
    tk = toolkit()
    tk._wam = MetaWorldWAMClient(rpc)
    obs = observation()
    tk.env.get_obs = lambda: obs

    def step(action):
        obs["terminated"] = True
        return obs, 0, True, False, {}

    tk.env.step.side_effect = step
    assert tk.wam_act(max_chunks=3).data["steps"] == 1
    assert tk.env.step.call_count == 1
    request = rpc.call.call_args.kwargs["args"][0]
    assert request["images"]["primary"].shape == (3, 4, 3)
    assert request["state"]["gripper_opening"].shape == (1,)
    assert set(request["state"]) == {"eef_position", "gripper_opening"}
    assert request["instruction"] == "reach"


def test_wam_checks_cancellation_before_predicting():
    from rpent.tools.toolkit import ToolCancelled

    tk = toolkit()
    tk._wam = Mock()
    tk.raise_if_cancelled.side_effect = ToolCancelled("stop")
    with pytest.raises(ToolCancelled):
        tk.wam_act()
    tk._wam.predict.assert_not_called()
    tk.env.step.assert_not_called()
