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

"""Exercise Awomo action-cache ownership through the production Awomo HTTP bridge and client."""

import argparse
from types import SimpleNamespace

import numpy as np
import pytest

from robots.robodojo.awomo05 import Awomo05Client
from robots.robodojo.awomo05_bridge import Awomo05Facade
from robots.robodojo.awomo05_contract import STATE_FIELDS
from robots.robodojo.execution import ExecutionSyncError
from robots.robodojo.stateful_policy import StatefulPolicyEnv
from rpent.robots.components.xpolicylab_vla_server import add_connection_args
from rpent.tools.toolkit import ToolCancelled
from rpent.utils.rpc import RpcError
from tests.unit_tests.robots.robodojo.fakes import FakeEnv, metadata
from tests.unit_tests.rpent.robots.components.xpolicylab_helpers import serve_policy


class Model:
    """Three-action native joint replies with observable upstream history."""

    def __init__(self):
        self.history = []
        self.inferences = []
        self.resets = 0
        self.failure = None
        self.closed = False
        self.call_info = {}

    def call(self, func_name, obs=None):
        if func_name == "get_model_meta":
            return {**metadata(), "replan_steps": 3}
        if func_name == "reset":
            self.resets += 1
            self.history.clear()
        elif func_name == "update_obs":
            assert set(obs["state"]) == set(STATE_FIELDS)
            assert obs["instruction"] == "stack the blocks"
            frame = int(obs["state"]["left_arm_joint_state"][0])
            for camera in obs["vision"].values():
                assert set(camera) == {"color"}
                np.testing.assert_array_equal(
                    camera["color"], np.full((2, 2, 3), frame, np.uint8)
                )
            self.history.append(frame)
        elif func_name == "get_action":
            self.inferences.append(list(self.history))
            self.call_info = {"inference": len(self.inferences)}
        else:
            raise AssertionError(func_name)
        if self.failure == func_name:
            raise RuntimeError("lost response after model mutation")
        if func_name == "get_action":
            return {
                "actions": [
                    {
                        key: np.full(size, self.history[-1] + i, dtype=float)
                        for key, size in STATE_FIELDS.items()
                    }
                    for i in range(1, 4)
                ]
            }

    def close(self):
        self.closed = True


@pytest.fixture
def live_bridge(monkeypatch):
    model = Model()

    def connect_model(self, url, args):
        self._observed_client = model
        return model

    monkeypatch.setattr(Awomo05Facade, "_connect_policy", connect_model)
    parser = argparse.ArgumentParser()
    add_connection_args(parser, required=True)
    facade = Awomo05Facade(
        parser.parse_args(["--task", "stack", "--policy-server-url", "ws://fake"])
    )
    with serve_policy(facade) as service:
        yield SimpleNamespace(
            model=model, connect=lambda: service.connect(Awomo05Client)
        )
    assert model.closed


def execution(live_bridge, native=None, cancel=lambda: None):
    native = native or FakeEnv()
    return StatefulPolicyEnv(
        native,
        live_bridge.connect(),
        cancel,
    )


@pytest.mark.parametrize("cancel_at", [0, 1, 3])
def test_cancelled_chunk_resumes_without_repeating_inference(live_bridge, cancel_at):
    native = FakeEnv()
    model = live_bridge.model
    cancelled = True

    def check():
        if cancelled and native.frame == cancel_at and model.inferences:
            # At 0 and 3, wait for the new chunk's inference to return. At 1,
            # cancel after one acknowledged action from the first chunk.
            if cancel_at == 1 or model.inferences[-1][-1] == cancel_at:
                raise ToolCancelled("cancelled before action")

    env = execution(live_bridge, native, check)
    try:
        with pytest.raises(ToolCancelled):
            env.run(8)
        assert env.last_run["stop_reason"] == "cancelled"
        assert env.last_run["executed_steps"] == native.frame == cancel_at
        assert env.last_run["pending_actions"] == (0 if cancel_at == 1 else 3)
        assert model.history == list(range(cancel_at + 1))
        with pytest.raises(RuntimeError, match="resume a pause"):
            env.run(1)
        cancelled = False
        env.resume_execution()
        result = env.run(4)
        assert result["executed_steps"] == 4
        assert result["observation_step"] == native.frame == cancel_at + 4
        assert model.history == list(range(cancel_at + 5))
        assert native._history_step == cancel_at + 4
        assert [step for step, _ in native._rgb_history] == model.history
        assert [history[-1] for history in model.inferences] == {
            0: [0, 3],
            1: [0, 1, 4],
            3: [0, 3, 6],
        }[cancel_at]
        assert model.resets == 1
    finally:
        env.close()


def test_cancelled_unstarted_chunk_cannot_resume_uncertain_execution(live_bridge):
    native = FakeEnv()
    model = live_bridge.model
    cancelled = True

    def check():
        if cancelled and model.inferences:
            raise ToolCancelled("cancelled after inference")

    env = execution(live_bridge, native, check)
    try:
        with pytest.raises(ToolCancelled):
            env.run(2)
        cancelled = False
        env.resume_execution()

        def fail():
            raise RuntimeError("response lost")

        native.after_step = fail
        with pytest.raises(RuntimeError, match="response lost"):
            env.run(1)
        assert native.frame == 1
        assert env.last_run["executed_steps"] == 0
        assert not env._pending
        assert model.history == [0]
        assert native._history_step == 0
        native.after_step = lambda: None
        env.resume_execution()
        with pytest.raises(RuntimeError, match="unavailable"):
            env.run(1)
        assert native.frame == 1
        assert model.inferences == [[0]]
        assert model.resets == 1
    finally:
        env.close()


@pytest.mark.parametrize("change", ["advance", "reset", "instruction", "cancel"])
def test_context_changes_discard_old_actions_without_implicit_model_reset(
    live_bridge, change
):
    native = FakeEnv()
    cancelled = False

    def check():
        if cancelled:
            raise RuntimeError("cancelled")

    env = execution(live_bridge, native, check)
    env.run(2)
    if change == "advance":
        native.frame += 1
    elif change == "reset":
        native._native_reset()  # Bypass RPent to simulate an external reset.
    elif change == "instruction":
        native.get_task_language = lambda: "new instruction"
    else:
        cancelled = True
    frame = native.frame
    with pytest.raises(RuntimeError):
        env.run(1)
    assert not env._pending
    assert native.frame == frame
    assert live_bridge.model.resets == 1
    env.close()


def test_reset_starts_fresh_action_chunk(live_bridge):
    env = execution(live_bridge)
    env.run(1)
    env.reset()
    env.run(1)
    assert live_bridge.model.history == [0, 1]
    assert live_bridge.model.inferences == [[0], [0]]
    assert live_bridge.model.resets == 2
    env.close()


@pytest.mark.parametrize("failure", ["update_obs", "get_action"])
def test_ambiguous_model_failure_requires_new_episode(live_bridge, failure):
    native = FakeEnv()
    policy = live_bridge.connect()
    env = StatefulPolicyEnv(native, policy, lambda: None)
    live_bridge.model.failure = failure
    with pytest.raises(
        ExecutionSyncError if failure == "update_obs" else RpcError,
        match="lost response",
    ):
        env.run(3)
    frame = native.frame
    with pytest.raises(RuntimeError, match="unavailable"):
        env.run(3)
    with pytest.raises(RuntimeError, match="unavailable"):
        env.step({})
    with pytest.raises(RpcError, match="uncertain"):
        policy.get_action()
    assert native.frame == frame
    live_bridge.model.failure = None
    env.reset()
    env.run(1)
    assert live_bridge.model.history == [0, 1]


@pytest.mark.parametrize("failure", ["reset", "update_obs"])
def test_initial_observation_failure_releases_session(live_bridge, failure):
    live_bridge.model.failure = failure
    with pytest.raises(ExecutionSyncError, match="lost response"):
        execution(live_bridge)
    live_bridge.model.failure = None
    env = execution(live_bridge)
    env.run(1)
    assert live_bridge.model.history == [0, 1]


def test_toolkit_runs_policy_and_scripted_gripper_with_one_history(
    live_bridge, monkeypatch, tmp_path
):
    from robots.robodojo.toolkit import RoboDojoToolkit
    from rpent.dashboard.events import NullDashboardEventSink
    from rpent.memory import MemoryManager
    from rpent.utils import logging

    monkeypatch.setattr(logging, "_output_dir", tmp_path)
    native = FakeEnv()
    policy = live_bridge.connect()
    toolkit = RoboDojoToolkit(
        primitives_kwargs={
            "env": native,
            "vla_client": policy,
            "vla_backend": "awomo05",
            "vla_agent": True,
        },
        dashboard_events=NullDashboardEventSink(),
        memory=MemoryManager(tmp_path / "memory"),
    )
    try:
        names = {spec.name for spec in toolkit.list_tools()}
        assert "run_policy" in names
        assert "pi0_pick" not in names
        for budget in (1, 1, 2):
            result = toolkit.execute_tool("run_policy", {"max_steps": budget})
            assert not result.is_error, result.data
            assert result.data["log"]["result"]["executed_steps"] == budget
            assert not toolkit.execute_tool("view_env_state", {}).is_error
        toolkit.execute_tool("set_gripper", {"arm": "right", "val": 0.0, "steps": 1})
        result = toolkit.execute_tool("run_policy", {"max_steps": 1})
        assert not result.is_error, result.data
        assert result.data["log"]["result"]["executed_steps"] == 1
        assert live_bridge.model.history == list(range(7))
        assert live_bridge.model.inferences == [[0], list(range(4)), list(range(6))]
        assert live_bridge.model.resets == 1
        inferences = toolkit._primitives.env.last_run["inferences"]
        assert len(inferences) == 1
        assert inferences[0]["info"]["upstream"] == {"inference": 3}
        assert not toolkit.execute_tool(
            "finish", {"status": "stuck", "summary": "bounded integration"}
        ).is_error
    finally:
        toolkit.close()
    assert not native.closed  # Runtime retains environment ownership.
    # Toolkit closes its episode, while Runtime can reuse both connections.
    assert native._execution is None
    next_env = StatefulPolicyEnv(native, policy, lambda: None)
    try:
        next_env.run(1)
        assert live_bridge.model.history == [6, 7]
        assert live_bridge.model.resets == 2
    finally:
        next_env.release()
    assert not native.closed
