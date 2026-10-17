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

"""Observable position/yaw control contracts without a simulator."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from robots.libero.tools import LiberoPrimitives


class MotionEnv:
    def __init__(self, yaw=0.0, terminal=None):
        self.xyz = np.zeros(3)
        self.yaw = yaw
        self.actions = []
        self.terminated = False
        self.truncated = False
        self.terminal = terminal

    def observation(self):
        return {"states": np.concatenate((self.xyz, np.zeros(5)))}

    def raw_obs(self):
        rotation = Rotation.from_euler("z", self.yaw) * Rotation.from_euler("x", np.pi)
        return {"robot0_eef_quat": rotation.as_quat()}

    def step(self, action):
        self.actions.append(action.copy())
        self.xyz += action[:3] * 0.05
        self.yaw += float(action[5]) * 0.1
        self.terminated = self.terminal == "terminated"
        self.truncated = self.terminal == "truncated"
        return self.observation(), 0, self.terminated, self.truncated, {}


def primitives(env):
    driver = LiberoPrimitives(
        env=env, model=None, sam3_client=None, check_cancelled=lambda: None
    )
    driver.set_obs(env.observation())
    return driver


@pytest.mark.parametrize("target_xyz", [[0, 0, 0], [0.05, 0, 0]])
def test_move_to_finishes_yaw_even_after_position_converges(target_xyz):
    env = MotionEnv()
    result = primitives(env).move_to(target_xyz, target_yaw=0.8).data
    assert len(env.actions) > 1
    assert abs(env.yaw - 0.8) < 0.02
    assert np.linalg.norm(env.xyz - np.asarray(target_xyz)) < 0.012
    assert abs(result["final_yaw_error_rad"]) < 0.02


def test_move_to_takes_shortest_yaw_path_across_pi():
    env = MotionEnv(yaw=np.pi - 0.03)
    result = primitives(env).move_to([0, 0, 0], target_yaw=-np.pi + 0.03).data
    assert len(env.actions) == 1
    assert env.actions[0][5] > 0
    assert abs(result["final_yaw_error_rad"]) < 0.02


def test_position_only_move_does_not_read_orientation():
    env = MotionEnv()

    def unexpected():
        pytest.fail("Position-only control must not require quaternion observations")

    env.raw_obs = unexpected
    result = primitives(env).move_to([0.05, 0, 0]).data
    assert result["final_dist_m"] < 0.012
    assert all(action[5] == 0 for action in env.actions)
    assert "final_yaw_error_rad" not in result


@pytest.mark.parametrize("terminal", ["terminated", "truncated"])
def test_yaw_control_stops_at_native_terminal_flags(terminal):
    env = MotionEnv(terminal=terminal)
    result = primitives(env).move_to([0, 0, 0], target_yaw=0.8).data
    assert len(env.actions) == 1
    assert result[terminal] is True


def test_yaw_control_respects_budget_and_requested_tolerance():
    env = MotionEnv()
    result = primitives(env).move_to([0, 0, 0], target_yaw=0.8, max_steps=2).data
    assert len(env.actions) == 2
    assert result["final_yaw_error_rad"] > 0.5
    result = primitives(env).move_to([0, 0, 0], target_yaw=0.21, yaw_tol=0.03).data
    assert len(env.actions) == 2
    assert abs(result["final_yaw_error_rad"]) < 0.03


@pytest.mark.parametrize("yaw_tol", [0, -0.1, np.nan, np.inf])
def test_invalid_yaw_tolerance_fails_before_actions(yaw_tol):
    env = MotionEnv()
    with pytest.raises(ValueError, match="yaw_tol"):
        primitives(env).move_to([0, 0, 0], target_yaw=0.8, yaw_tol=yaw_tol)
    assert not env.actions


class TiltingMotionEnv(MotionEnv):
    def __init__(self, horizontal_projection, *, tilt_after=0, terminal=None):
        super().__init__(terminal=terminal)
        self.pitch = np.arccos(horizontal_projection)
        self.tilt_after = tilt_after

    def raw_obs(self):
        if len(self.actions) < self.tilt_after:
            return super().raw_obs()
        rotation = Rotation.from_euler("z", self.yaw) * Rotation.from_euler(
            "y", self.pitch
        )
        return {"robot0_eef_quat": rotation.as_quat()}


@pytest.mark.parametrize("projection", [0.0, 0.5e-6, -0.5e-6])
@pytest.mark.parametrize("max_steps", [0, 80])
def test_vertical_x_axis_returns_error_even_at_target(projection, max_steps):
    env = TiltingMotionEnv(projection)
    result = (
        primitives(env).move_to([0, 0, 0], target_yaw=0.0, max_steps=max_steps).data
    )
    assert result["name"] == "move_to"
    assert "nearly vertical" in result["error"]
    assert "1e-6" in result["error"]
    assert "final_yaw_error_rad" not in result
    assert not env.actions


@pytest.mark.parametrize(
    ("max_steps", "terminal"),
    [(80, None), (1, None), (80, "terminated"), (80, "truncated")],
)
def test_yaw_becoming_degenerate_stops_without_reporting_convergence(
    max_steps, terminal
):
    env = TiltingMotionEnv(0.0, tilt_after=1, terminal=terminal)
    result = (
        primitives(env).move_to([0, 0, 0], target_yaw=0.1, max_steps=max_steps).data
    )
    assert result["name"] == "move_to"
    assert "nearly vertical" in result["error"]
    assert "final_yaw_error_rad" not in result
    assert len(env.actions) == 1


def test_yaw_projection_above_threshold_is_accepted():
    env = TiltingMotionEnv(2e-6)
    result = primitives(env).move_to([0, 0, 0], target_yaw=0.0).data
    assert "error" not in result
    assert result["final_yaw_error_rad"] == 0.0
    assert not env.actions
