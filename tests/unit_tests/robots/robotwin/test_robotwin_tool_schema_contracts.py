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

"""Robot-specific schema contracts for RoboTwin tools."""

from unittest.mock import Mock

import numpy as np
import pytest

from robots.robotwin import tools
from robots.robotwin.primitives import RoboTwinPrimitives
from rpent.tools import iter_tools


@pytest.mark.parametrize("action_type,dim", [("qpos", 14), ("ee", 16)])
def test_wam_executes_selected_control_and_refreshes_observation(action_type, dim):
    status = {"eval_success": False, "step_lim": 20, "take_action_cnt": 0}
    env = Mock(
        terminated=False,
        truncated=False,
        last_info={
            "episode_status": status,
            "robot_state": {"qpos_target14": np.zeros(14)},
        },
    )
    env.get_task_language.return_value = "native task"
    env.render_camera.return_value = np.zeros((8, 8, 3), np.uint8)

    def step(actions, **kwargs):
        assert kwargs["action_type"] == action_type
        status["take_action_cnt"] += len(actions)
        env.last_info["robot_state"] = {
            "qpos_target14": np.full(14, status["take_action_cnt"])
        }
        return {}, 0, False, False, {**env.last_info, "executed_actions": len(actions)}

    env.chunk_step.side_effect = step
    env.reset.return_value = ({}, {})
    model = Mock(action_type=action_type, predict=Mock(return_value=np.zeros((3, dim))))
    primitive = RoboTwinPrimitives(
        env=env, model=model, seed=0, check_cancelled=lambda: None, policy_kind="wam"
    )
    primitive.reset()
    model.reset.assert_called_once()
    assert primitive.wam_act("subtask").data["executed_steps"] == 3
    assert primitive.wam_act().data["executed_steps"] == 3
    calls = model.predict.call_args_list
    assert [call.args[0]["task_language"] for call in calls] == [
        "subtask",
        "native task",
    ]
    np.testing.assert_array_equal(
        calls[1].args[0]["robot_state"]["qpos_target14"], np.full(14, 3)
    )


def test_perception_schemas_use_the_same_view_coordinate_space() -> None:
    by_name = {spec.name: spec for spec in iter_tools(tools)}

    for tool_name, coordinate_name in (
        ("sample_world_xyz", "pixels"),
        ("query_world_map", "bbox"),
    ):
        schema = by_name[tool_name].input_schema
        assert schema["required"] == ["view", coordinate_name]
        assert schema["properties"]["view"]["type"] == "string"
