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

"""Offline position-hold behavior for LIBERO wrist and pitch rotation."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from robots.libero.tools import LiberoPrimitives

_DISTURBANCE = np.array([0.02, 0.0, 0.0])


class _FakeEnv:
    """Applies OSC scales: translation * 0.05 m, rotation * 0.5 rad."""

    def __init__(self, on_step, xyz=None):
        self.xyz = np.array([0.0, 0.0, 0.4] if xyz is None else xyz, dtype=np.float64)
        self.rot = Rotation.from_euler("x", np.pi)
        self.on_step = on_step
        self.actions: list[np.ndarray] = []
        self.terminated = False
        self.truncated = False

    def raw_obs(self) -> dict[str, np.ndarray]:
        return {"robot0_eef_quat": np.asarray(self.rot.as_quat(), dtype=np.float64)}

    def observation(self) -> dict[str, np.ndarray]:
        states = np.zeros(8, dtype=np.float32)
        states[:3] = self.xyz
        return {"states": states}

    def reset(self):
        return self.observation(), {}

    def step(self, action):
        recorded = np.asarray(action, dtype=np.float64).copy()
        self.actions.append(recorded)
        self.on_step(self, recorded)
        return self.observation(), 0.0, self.terminated, self.truncated, {}


def _primitives(env: _FakeEnv) -> LiberoPrimitives:
    return LiberoPrimitives(
        env=env,
        model=SimpleNamespace(),
        sam3_client=SimpleNamespace(),
        check_cancelled=lambda: None,
    )


def _call(env: _FakeEnv, method: str, **kwargs):
    primitives = _primitives(env)
    primitives.reset()
    return getattr(primitives, method)(**kwargs)


def _noop(env: _FakeEnv, action: np.ndarray) -> None:
    del env, action


def _servo(env: _FakeEnv, action: np.ndarray) -> None:
    env.xyz = env.xyz + action[:3] * 0.05 + _DISTURBANCE
    env.rot = (
        Rotation.from_euler("x", float(action[3]) * 0.5)
        * Rotation.from_euler("z", float(action[5]) * 0.5)
        * env.rot
    )


def _walk(env: _FakeEnv, action: np.ndarray) -> None:
    env.xyz = env.xyz + np.array([0.1, 0.0, 0.0])
    env.rot = (
        Rotation.from_euler("x", float(action[3]) * 0.5)
        * Rotation.from_euler("z", float(action[5]) * 0.5)
        * env.rot
    )


@pytest.mark.parametrize(
    ("method", "delta_name", "axis"),
    [
        ("rotate_wrist", "delta_yaw", 5),
        ("rotate_pitch", "delta_pitch", 3),
    ],
)
def test_rotation_holds_position_and_uses_half_radian_scale(method, delta_name, axis):
    env = _FakeEnv(_servo)
    result = _call(env, method, **{delta_name: 0.25})

    issued = [float(action[axis]) for action in env.actions]
    # clip(0.10 / 0.5), clip(0.10 / 0.5), clip(0.05 / 0.5); / 0.10 would be 1, 1, 0.5.
    assert issued == pytest.approx([0.2, 0.2, 0.1])
    assert result["converged"] is True
    assert result["position_hold_ok"] is True
    assert result["max_pos_drift_m"] <= 0.03


@pytest.mark.parametrize(
    ("method", "delta_name", "axis"),
    [
        ("rotate_wrist", "delta_yaw", 5),
        ("rotate_pitch", "delta_pitch", 3),
    ],
)
def test_rotation_scale_stays_inside_the_action_range(method, delta_name, axis):
    at_clip = _FakeEnv(_noop)
    _call(at_clip, method, **{delta_name: 0.10}, max_steps=1)
    assert at_clip.actions[0][axis] == pytest.approx(0.2)

    over_scale = _FakeEnv(_noop)
    _call(over_scale, method, **{delta_name: 1.0}, step_clip=1.0, max_steps=1)
    command = float(over_scale.actions[0][axis])
    assert -1.0 <= command <= 1.0
    assert command == pytest.approx(1.0)


@pytest.mark.parametrize("method", ["rotate_wrist", "rotate_pitch"])
def test_translation_command_returns_to_the_entry_xyz(method):
    start = np.array([0.1, -0.2, 0.3])
    offsets = (
        np.array([0.04, 0.0, 0.0]),
        np.array([0.02, 0.03, -0.01]),
    )

    def on_step(env, action):
        del action
        step_index = len(env.actions) - 1
        if step_index < len(offsets):
            env.xyz = start + offsets[step_index]

    env = _FakeEnv(on_step, xyz=start)
    delta_name = "delta_yaw" if method == "rotate_wrist" else "delta_pitch"
    _call(env, method, **{delta_name: 1.0}, max_steps=3)

    np.testing.assert_allclose(
        env.actions[1][:3], np.clip(-offsets[0] / 0.05, -1.0, 1.0), atol=1e-5
    )
    np.testing.assert_allclose(
        env.actions[2][:3], np.clip(-offsets[1] / 0.05, -1.0, 1.0), atol=1e-5
    )


def test_rotate_pitch_commands_yaw_toward_the_entry_yaw():
    def on_step(env, action):
        del action
        env.rot = Rotation.from_euler("z", 0.2) * env.rot

    env = _FakeEnv(on_step)
    _call(env, "rotate_pitch", delta_pitch=0.5, max_steps=2)

    assert env.actions[1][5] < 0


@pytest.mark.parametrize(
    ("method", "delta_name"),
    [
        ("rotate_wrist", "delta_yaw"),
        ("rotate_pitch", "delta_pitch"),
    ],
)
def test_position_hold_failure_has_no_error_key(method, delta_name):
    env = _FakeEnv(_walk)
    result = _call(env, method, **{delta_name: 0.25})

    assert result["converged"] is True
    assert result["position_hold_ok"] is False
    assert result["max_pos_drift_m"] > 0.03
    assert "error" not in result


@pytest.mark.parametrize(
    ("method", "delta_name"),
    [
        ("rotate_wrist", "delta_yaw"),
        ("rotate_pitch", "delta_pitch"),
    ],
)
def test_rotation_step_budget_reports_not_converged(method, delta_name):
    env = _FakeEnv(_noop)
    result = _call(env, method, **{delta_name: 0.5}, max_steps=4)

    assert result["converged"] is False
    assert result["steps_used"] == 4
    assert len(env.actions) == 4


@pytest.mark.parametrize(
    ("method", "message"),
    [
        ("rotate_wrist", "need target_yaw or delta_yaw"),
        ("rotate_pitch", "need target_pitch or delta_pitch"),
    ],
)
def test_missing_rotation_target_does_not_step(method, message):
    env = _FakeEnv(_noop)
    result = _call(env, method)

    assert result == {"name": method, "error": message}
    assert env.actions == []
