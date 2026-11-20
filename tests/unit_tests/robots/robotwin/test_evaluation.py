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

"""Exercise direct policy and native episode evidence through the actual adapters."""

from __future__ import annotations

import asyncio
import json
import time
from argparse import Namespace
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from robots.robotwin import direct_eval
from robots.robotwin.env_client import RoboTwinEnvClient
from robots.robotwin.evaluation import RoboTwinTrial
from robots.robotwin.robot_spec import get_robot_spec
from tests.unit_tests.robots.robotwin.test_uniwam_contracts import FeedbackRPC


class NativeRPC(FeedbackRPC):
    def __init__(self, success):
        super().__init__()
        self.success = success
        self.sealed = False

    def info(self, states=()):
        data = super().info(states)
        data["episode_id"] = "unique-episode"
        data["episode_status"]["eval_success"] = self.success if self.count else False
        return data

    def call(self, method, args=(), kwargs=None, **unused):
        if method == "env.seal_episode":
            assert kwargs["episode_id"] == "unique-episode"
            self.sealed = True
            return {**self.info(), "sealed": True}
        if method in ("env.step", "env.chunk_step"):
            assert kwargs["episode_id"] == "unique-episode" and not self.sealed
        return super().call(method, args, kwargs, **unused)


@pytest.mark.parametrize(
    ("backend", "family"), [("lingbot", "vla"), ("fast-wam", "wam")]
)
def test_direct_policy_reports_one_native_trial(monkeypatch, tmp_path, backend, family):
    rpc = NativeRPC(success=True)
    rpc.limit = 3
    env = RoboTwinEnvClient(rpc, expected_meta={"seed": 7})
    fake_model = SimpleNamespace(
        history_length=0,
        action_type="qpos",
        reset=lambda: None,
        infer=lambda obs: np.zeros((3, 16)),
        predict=lambda obs: np.zeros((3, 14)),
    )
    runtime = {
        "env": env,
        "model": fake_model,
        "policy_kind": family,
        "policy_backend": backend,
        "seed": 7,
    }
    spec = replace(get_robot_spec(), init_runtime=lambda *args: ([], runtime))
    monkeypatch.setattr(direct_eval, "get_robot_spec", lambda: spec)
    assert (
        direct_eval.main(
            [
                "--task-name",
                "hammer",
                "--seed",
                "7",
                "--output-dir",
                str(tmp_path),
                "--am-backend",
                backend,
                "--am-endpoint",
                "http://localhost:9999",
                "--max-episode-steps",
                "3",
            ]
        )
        == 0
    )
    [path] = tmp_path.glob("metrics/*/episode_metrics.json")
    record = json.loads(path.read_text())
    assert rpc.sealed and record["status"] == "success"
    assert record["trial_id"] == "robotwin/demo_randomized/hammer/7"
    assert record["method"] == f"direct_{family}/{backend}"
    assert record["verdict_source"] == "RoboTwin.TASK_ENV.eval_success"
    assert record["timing"]["run_s"] >= record["timing"]["success_s"]
    assert record["instruction_version"].startswith("sha256:")


def test_missing_native_is_not_false(tmp_path):
    args = Namespace(
        seed=7,
        task_name="hammer",
        task_config="demo_randomized",
        planner="api",
        model="test",
        max_turns=2,
        max_episode_steps=3,
        planner_timeout_s=None,
        trial_wall_s=0.5,
        am_endpoint="http://localhost:9999",
    )
    rpc = NativeRPC(success=None)
    env = RoboTwinEnvClient(rpc, expected_meta={"seed": 7})
    trial = RoboTwinTrial(args, SimpleNamespace(output_dir=tmp_path))
    trial.start({"env": env})
    planner = SimpleNamespace(timeout_s=10)
    trial.prepare_planner(planner)
    assert 0 < planner.timeout_s <= 0.5
    # The environment no longer exposes its native success signal.
    rpc.count = 1
    trial.finalize(result=SimpleNamespace(stats={}, error=None))
    data = json.loads(trial.trial.path.read_text())
    assert data["status"] == "unresolved"
    assert data["native_verdict"] is None
    assert rpc.sealed and data["timing"]["run_s"] is not None


@pytest.mark.timeout(45)
def test_wall_budget_interrupts_an_idle_api_planner_and_blocks_actions(tmp_path):
    from pydantic_ai.models.function import FunctionModel

    from rpent.planner.api_loop import ApiAgentLoop
    from rpent.utils.logging import init_output_dir
    from tests.unit_tests.rpent.planner.test_api_contracts import Events, RobotToolkit

    events_at = []

    async def idle_model(messages, info):
        events_at.append(("request", time.monotonic()))
        try:
            await asyncio.sleep(10)
            events_at.append(("late_response", time.monotonic()))
            yield "late"
        finally:
            events_at.append(("closed", time.monotonic()))

    init_output_dir(tmp_path)
    events = Events()
    toolkit = RobotToolkit(events)
    planner = ApiAgentLoop(
        model=FunctionModel(stream_function=idle_model),
        dashboard_events=events,
        timeout_s=20,
    )
    args = Namespace(
        seed=7,
        task_name="hammer",
        task_config="demo_randomized",
        planner="api",
        model="offline",
        max_turns=2,
        max_episode_steps=10,
        planner_timeout_s=20,
        trial_wall_s=0.2,
        am_endpoint="http://localhost:9999",
    )
    rpc = NativeRPC(success=False)
    env = RoboTwinEnvClient(rpc, expected_meta={"seed": 7})
    trial = RoboTwinTrial(args, SimpleNamespace(output_dir=tmp_path))
    trial.start({"env": env})
    trial.prepare_planner(planner)
    started = time.monotonic()
    result = planner.solve(
        system_prompt="Run one trial.",
        user_message="Do the task.",
        toolkit=toolkit,
        max_turns=2,
    )
    assert "timed out" in result.error, (
        result.error,
        started,
        time.monotonic(),
        events_at,
    )
    assert all(name != "late_response" for name, _ in events_at), (started, events_at)
    # The first SDK construction may block synchronously during cold imports.
    # Its elapsed time is retained, and no late request may enter the model.
    assert all(
        timestamp <= trial.deadline
        for name, timestamp in events_at
        if name == "request"
    )
    assert all(timestamp - started < 5 for _, timestamp in events_at)
    with pytest.raises(TimeoutError):
        env.chunk_step(np.zeros((2, 14)))
    trial.finalize(toolkit=toolkit, result=result)
    data = json.loads(trial.trial.path.read_text())
    assert not toolkit.calls and rpc.count == 0 and rpc.sealed
    assert data["status"] == "failure" and data["stop_reason"] == "wall_budget"
    assert 0.2 <= data["timing"]["stop_s"] <= data["timing"]["run_s"]


@pytest.mark.parametrize(
    ("component", "expected", "reason"),
    [
        ("env", "unresolved", "environment_initialization_failed"),
        ("wam", "failure", "method_initialization_failed"),
        ("vla", "failure", "method_initialization_failed"),
        ("unknown", "unresolved", "runtime_initialization_failed"),
    ],
)
def test_startup_component_does_not_certify_infrastructure(
    tmp_path, component, expected, reason
):
    args = Namespace(
        seed=7,
        task_name="hammer",
        task_config="demo_randomized",
        max_episode_steps=4,
        am_endpoint="http://localhost:9999",
    )
    attempt = RoboTwinTrial(args, SimpleNamespace(output_dir=tmp_path), direct=True)
    error = RuntimeError(
        f"[{component}] wait / client connect failed: invalid configuration"
    )
    error.__cause__ = ValueError("invalid configured input")
    attempt.finalize(error=True, startup_error=error)
    record = json.loads(attempt.trial.path.read_text())
    assert (record["status"], record["stop_reason"], record["error_type"]) == (
        expected,
        reason,
        "ValueError",
    )
    assert record["native_verdict"] is None and record["timing"]["run_s"] is None
