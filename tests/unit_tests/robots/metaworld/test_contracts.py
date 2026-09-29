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

"""Behavioral contracts without importing the optional simulator."""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from robots.metaworld.config import validate
from robots.metaworld.env_server import MetaWorldEnvFacade
from robots.metaworld.toolkit import MetaWorldToolkit, back_project_pixel
from rpent.tools.toolkit import ToolCancelled


@pytest.mark.parametrize(
    "kwargs",
    [
        {"task": "missing"},
        {"seed": -1},
        {"max_episode_steps": 0},
        {"camera": "missing"},
    ],
)
def test_launch_validation(kwargs):
    settings = {
        "task": "reach-v3",
        "seed": 0,
        "max_episode_steps": 500,
        "camera": "corner2",
    }
    settings.update(kwargs)
    with pytest.raises(ValueError):
        validate(**settings)


def test_projection_honors_camera_axes_and_world_rotation():
    obs = {
        "depth": np.full((3, 3), 2.0),
        "camera": {
            "intrinsics": [[2, 0, 1], [0, 2, 1], [0, 0, 1]],
            "rotation_world_from_camera": [[0, -1, 0], [1, 0, 0], [0, 0, 1]],
            "position_world": [1, 2, 3],
        },
    }
    np.testing.assert_allclose(back_project_pixel(obs, 2, 1), [2, 2, 1])


@pytest.mark.parametrize(
    "row,col,depth",
    [
        (-1, 0, 1),
        (3, 0, 1),
        (0, 3, 1),
        (True, 0, 1),
        (1.5, 0, 1),
        (0, 0, 0),
        (0, 0, float("nan")),
        (0, 0, float("inf")),
    ],
)
def test_projection_rejects_invalid_pixels_and_depth(row, col, depth):
    with pytest.raises(ValueError):
        back_project_pixel({"depth": np.full((3, 3), depth)}, row, col)


def facade():
    env = MetaWorldEnvFacade.__new__(MetaWorldEnvFacade)
    env._terminated = env._truncated = env._success = False
    env._steps = 0
    env._video_writer = None
    env._meta = {"max_episode_steps": 2}
    env._env = Mock()
    env._env.step.return_value = ({}, 0.0, False, False, {"success": False})
    env.get_obs = lambda **kwargs: {}
    return env


@pytest.mark.parametrize(
    "action",
    [
        [0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 1.01],
        [float("nan"), 0, 0, 0],
    ],
)
def test_invalid_actions_do_not_advance_native_simulation(action):
    env = facade()
    with pytest.raises(ValueError):
        env.step(action)
    env._env.step.assert_not_called()


def test_budget_and_success_stop_subsequent_actions():
    env = facade()
    assert not env.step([0, 0, 0, 0])[3]
    assert env.step([0, 0, 0, 0])[3]
    with pytest.raises(RuntimeError):
        env.step([0, 0, 0, 0])
    assert env._env.step.call_count == 2
    env = facade()
    env._env.step.return_value = ({}, 1.0, False, False, {"success": True})
    assert env.step([0, 0, 0, 0])[2]
    assert env.is_success()
    with pytest.raises(RuntimeError):
        env.step([0, 0, 0, 0])


def toolkit():
    tk = MetaWorldToolkit.__new__(MetaWorldToolkit)
    tk.obs = {
        "eef_position": np.zeros(3),
        "gripper_opening": 1.0,
        "terminated": False,
        "truncated": False,
    }

    def step(action):
        tk.obs = {
            **tk.obs,
            "eef_position": tk.obs["eef_position"] + np.array(action[:3]) * 0.01,
        }
        return tk.obs, 0.0, False, False, {}

    tk.env = SimpleNamespace(step=Mock(side_effect=step))
    tk.raise_if_cancelled = Mock()
    return tk


def test_motion_respects_budget_and_preserves_gripper_command():
    tk = toolkit()
    result = tk.move_to([0.2, 0, 0], gripper=-1, max_steps=3)
    assert result["steps"] == 3
    assert not result["reached"]
    assert tk.env.step.call_count == 3
    for call in tk.env.step.call_args_list:
        assert call.args[0] == [1, 0, 0, -1]


def test_motion_stops_on_episode_end_and_on_cancellation():
    tk = toolkit()
    tk.env.step.side_effect = lambda action: (
        {**tk.obs, "truncated": True},
        0,
        False,
        True,
        {},
    )
    assert tk.move_to([0.2, 0, 0], 1, 50)["steps"] == 1
    tk = toolkit()
    tk.raise_if_cancelled.side_effect = ToolCancelled("cancelled")
    with pytest.raises(ToolCancelled):
        tk.move_to([0.2, 0, 0], 1, 50)
    tk.env.step.assert_not_called()


@pytest.mark.parametrize(
    "target,gripper,budget",
    [
        ([0, 0], 0, 1),
        ([float("inf"), 0, 0], 0, 1),
        ([0, 0, 0], 2, 1),
        ([0, 0, 0], 0, 101),
    ],
)
def test_bad_motion_input_cannot_act(target, gripper, budget):
    tk = toolkit()
    with pytest.raises(ValueError):
        tk.move_to(target, gripper, budget)
    tk.env.step.assert_not_called()
