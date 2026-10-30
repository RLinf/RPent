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


"""Feedback history and request isolation without simulator/model dependencies."""

from __future__ import annotations

from collections import deque
from types import SimpleNamespace

import numpy as np
import pytest

from robots.robotwin.env_client import RoboTwinEnvClient
from robots.robotwin.primitives import RoboTwinPrimitives
from robots.robotwin.uniwam_client import UniWAMClient
from robots.robotwin.uniwam_server import UniWAMFacade


class FeedbackRPC:
    """Return native state feedback distinct from the requested action values."""

    def __init__(self):
        self.count = 0
        self.limit = 100
        self.with_history = True
        self.last_result = None

    def info(self, states=()):
        result = {
            "instruction": "Use the hammer to beat the block.",
            "robot_state": {"qpos_target14": np.full(14, self.count)},
            "episode_status": {
                "actual_seed": 7,
                "eval_success": False,
                "take_action_cnt": self.count,
                "step_lim": self.limit,
            },
            "executed_actions": len(states),
            "per_step": [
                {"qpos_target14": state} if self.with_history else {}
                for state in states
            ],
        }
        return result

    def call(self, method, args=(), kwargs=None, **unused):
        if method == "env.get_env_meta":
            return {"seed": 7}
        if method == "env.reset":
            self.count = 0
            return {}, self.info()
        if method == "env.get_task_language":
            return self.info()["instruction"]
        if method == "env.render_camera":
            return np.zeros((240, 320, 3), dtype=np.uint8)
        assert method in {"env.step", "env.chunk_step"}
        count = 1 if method == "env.step" else len(args[0])
        states = []
        for _ in range(min(count, self.limit - self.count)):
            self.count += 1
            states.append(np.full(14, self.count))
        self.last_result = {}, 0.0, False, self.count >= self.limit, self.info(states)
        return self.last_result


def test_history_covers_step_chunk_truncation_and_reset():
    rpc = FeedbackRPC()
    env = RoboTwinEnvClient(rpc, expected_meta={"seed": 7})
    np.testing.assert_array_equal(env.qpos_history, np.zeros((1, 14)))
    env.step(np.full(14, -99))
    env.chunk_step(np.full((20, 14), -99))
    np.testing.assert_array_equal(env.qpos_history[:, 0], np.arange(6, 22))
    rpc.last_result[4]["per_step"][-1]["qpos_target14"][:] = -77
    snapshot = env.qpos_history
    snapshot[:] = -55
    assert env.qpos_history[-1, 0] == 21
    rpc.limit = 24
    env.chunk_step(np.zeros((16, 14)))
    np.testing.assert_array_equal(env.qpos_history[-3:, 0], [22, 23, 24])
    before = env.qpos_history
    env.render_camera("head")
    np.testing.assert_array_equal(env.qpos_history, before)
    env.reset()
    np.testing.assert_array_equal(env.qpos_history, np.zeros((1, 14)))


def test_legacy_feedback_does_not_fabricate_history():
    rpc = FeedbackRPC()
    rpc.with_history = False
    env = RoboTwinEnvClient(rpc, expected_meta={"seed": 7})
    env.chunk_step(np.zeros((3, 16)), action_type="ee")
    assert env.last_info["executed_actions"] == 3
    with pytest.raises(RuntimeError, match="per-step qpos"):
        _ = env.qpos_history


@pytest.mark.parametrize("operation", ["step", "chunk_step", "reset"])
def test_failed_mutation_invalidates_history(monkeypatch, operation):
    rpc = FeedbackRPC()
    env = RoboTwinEnvClient(rpc, expected_meta={"seed": 7})

    def fail(*args, **kwargs):
        raise TimeoutError("execution outcome unknown")

    monkeypatch.setattr(rpc, "call", fail)
    with pytest.raises(TimeoutError):
        if operation == "reset":
            env.reset()
        else:
            getattr(env, operation)(np.zeros(14))
    with pytest.raises(RuntimeError):
        _ = env.qpos_history


@pytest.mark.parametrize("operation", ["step", "chunk_step", "reset"])
def test_invalid_feedback_blocks_history_until_reset(monkeypatch, operation):
    rpc = FeedbackRPC()
    env = RoboTwinEnvClient(rpc, expected_meta={"seed": 7})
    original_info = rpc.info

    def invalid_info(states=()):
        info = original_info(states)
        state = info["per_step"][-1] if states else info["robot_state"]
        state["qpos_target14"] = ["bad"] * 14
        return info

    with monkeypatch.context() as patch:
        patch.setattr(rpc, "info", invalid_info)
        with pytest.raises(ValueError):
            if operation == "reset":
                env.reset()
            else:
                actions = (
                    np.zeros((2, 14)) if operation == "chunk_step" else np.zeros(14)
                )
                getattr(env, operation)(actions)
    env.step(np.zeros(14))
    with pytest.raises(RuntimeError, match="Complete history is unavailable"):
        _ = env.qpos_history
    env.reset()
    np.testing.assert_array_equal(env.qpos_history, np.zeros((1, 14)))


class Policy:
    def __init__(self):
        self.history = deque(maxlen=16)
        self.seen = []

    def record_executed_qpos(self, obs):
        self.history.append(np.array(obs["joint_action"]["vector"], copy=True))

    def update_obs(self, obs):
        if not self.history:
            self.record_executed_qpos(obs)

    def set_instruction(self, instruction):
        self.instruction = instruction

    def get_action(self):
        self.seen.append(np.stack(self.history))
        return np.zeros((16, 14))


def test_facade_replaces_stale_history_for_interleaved_requests():
    policy = Policy()
    facade = UniWAMFacade(policy, lambda p: p.history.clear())

    def request(history):
        return {
            "history": history,
            "instruction": "native instruction",
            "joint_action": {"vector": np.full(14, 99)},
        }

    for value in (1, 2, 1):
        policy.history.append(np.full(14, -99))
        result = facade._dispatch(
            "vla.predict", (request(np.full((3, 14), value)), None), {}
        )
        assert result.shape == (16, 14)
        np.testing.assert_array_equal(policy.seen[-1], np.full((3, 14), value))
    with pytest.raises(ValueError, match="finite"):
        facade.predict(request(np.full((1, 14), np.nan)))


def test_client_maps_cameras_and_native_state():
    captured = []
    rpc = SimpleNamespace(
        call=lambda *a, **kw: captured.append((a, kw)) or np.zeros((16, 14))
    )
    client = UniWAMClient(rpc)
    views = {
        name: {"rgb": np.full((2, 2, 3), i)}
        for i, name in enumerate(("head", "left_wrist", "right_wrist"))
    }
    client.infer(
        {
            "views": views,
            "robot_state": {"qpos_target14": np.arange(14)},
            "task_language": "exact native text",
            "qpos_history": np.ones((2, 14)),
        }
    )
    payload = captured[0][1]["args"][0]
    assert payload["instruction"] == "exact native text"
    np.testing.assert_array_equal(payload["joint_action"]["vector"], np.arange(14))
    assert payload["observation"]["right_camera"]["rgb"][0, 0, 0] == 2


def test_analytic_intervention_is_in_next_policy_request():
    rpc = FeedbackRPC()
    env = RoboTwinEnvClient(rpc, expected_meta={"seed": 7})
    histories = []

    def infer(obs):
        histories.append(obs["qpos_history"])
        return np.zeros((16, 14))

    primitives = RoboTwinPrimitives(
        env=env,
        model=SimpleNamespace(infer=infer),
        seed=7,
        check_cancelled=lambda: None,
        vla_backend="uniwam",
    )
    primitives.uniwam_act(chunks=1)
    primitives.apply_qpos_updates([{"arm": "left", "gripper": 1.0}])
    primitives.uniwam_act(chunks=2)
    np.testing.assert_array_equal(histories[1][:, 0], np.arange(2, 18))
    np.testing.assert_array_equal(histories[2][:, 0], np.arange(18, 34))
    assert rpc.count == 49
