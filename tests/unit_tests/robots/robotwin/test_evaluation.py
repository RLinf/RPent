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
from threading import Event
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


@pytest.mark.parametrize("sealed_value", [False, True, None])
def test_late_seal_only_backdates_a_latched_negative(
    tmp_path, monkeypatch, sealed_value
):
    """A late success lacks in-budget evidence; a latched false covers the deadline."""
    from rpent.evaluation import runtime

    clock = [10.0]
    monkeypatch.setattr(runtime.time, "monotonic", lambda: clock[0])
    args = Namespace(
        seed=7,
        task_name="hammer",
        task_config="demo_randomized",
        max_episode_steps=10,
        trial_wall_s=1000,
        am_endpoint="http://localhost:9999",
    )
    trial = RoboTwinTrial(args, SimpleNamespace(output_dir=tmp_path), direct=True)
    env = RoboTwinEnvClient(NativeRPC(success=None), expected_meta={"seed": 7})
    trial.start({"env": env})
    clock[0] += 1000.01
    trial.close_admission("wall_budget")
    trial.native_update(sealed_value, source=trial.native_source, sealed=True)
    trial._timer.cancel()
    trial._timer.join()
    trial.trial.quiesce(at=clock[0])
    record = json.loads(trial.trial.finish().read_text())
    assert record["status"] == "failure"
    assert record["native_verdict"] is (False if sealed_value is False else None)
    assert record["timing"]["success_s"] is None


def test_failed_memory_setup_keeps_the_resolved_method_and_missing_time(
    tmp_path, monkeypatch
):
    from rpent.evaluation.runtime import prepare_run_trial

    def missing_memory(args, config):
        raise OSError("test corpus is unavailable")

    monkeypatch.setenv("CODEX_MODEL", "pinned-model")
    spec = replace(get_robot_spec(), prepare_memory=missing_memory)
    args = Namespace(
        seed=0, task_config="suite", task_name="hammer", planner="codex", model=None
    )
    with pytest.raises(OSError, match="corpus is unavailable"):
        prepare_run_trial(
            args, spec, SimpleNamespace(output_dir=tmp_path), automatic=True
        )
    [path] = tmp_path.glob("metrics/*/episode_metrics.json")
    row = json.loads(path.read_text())
    assert row["method"] == "agentic/codex/pinned-model/lingbot"
    assert (row["status"], row["stop_reason"], row["error_type"]) == (
        "unresolved",
        "memory_initialization_failed",
        "OSError",
    )
    assert row["timing"]["run_s"] is None and row["native_verdict"] is None


@pytest.mark.parametrize("loses_reply", [True, False], ids=["timeout", "remote-error"])
def test_model_rpc_requires_a_completion_reply_for_runtime(
    tmp_path, transport, make_server_and_client, loses_reply
):
    """A sealed environment and an open model request have distinct end times."""
    from rpent.utils.rpc import RpcError, RpcFacade

    worker_started, allow_finish, worker_finished = Event(), Event(), Event()

    def predict():
        worker_started.set()
        try:
            if loses_reply:
                assert allow_finish.wait(10)
                return []
            raise ValueError("invalid observation")
        finally:
            worker_finished.set()

    worker = RpcFacade()
    worker._rpc["vla.predict"] = predict
    args = Namespace(seed=7, task_name="hammer", task_config="demo_randomized")
    env_rpc = NativeRPC(success=False)
    env = RoboTwinEnvClient(env_rpc, expected_meta={"seed": 7})
    trial = RoboTwinTrial(args, SimpleNamespace(output_dir=tmp_path), direct=True)
    trial.start({"env": env})
    try:
        with make_server_and_client(worker, transport) as client:
            with pytest.raises((RpcError, TimeoutError)):
                with trial.trial.tokens.unmetered_call(
                    model="test-policy", module="action_model", stage="execution"
                ):
                    client.call("vla.predict", timeout_s=0.1 if loses_reply else 2)
            assert worker_started.wait(2)
            trial.finalize(error=True)
            data = json.loads(trial.trial.path.read_text())
            assert env_rpc.sealed and data["native_verdict"] is False
            assert data["status"] == "failure" and data["stop_reason"] == "method_error"
            timing = data["timing"]
            assert timing["observed_s"] >= timing["stop_s"] >= 0
            if loses_reply:
                assert not worker_finished.is_set()
                assert timing["run_s"] is None and timing["tail_s"] is None
                [request] = timing["unconfirmed_model_requests"]
                [model_call] = [
                    e
                    for e in data["token_ledger"]["entries"]
                    if e["module"] == "action_model"
                ]
                assert request == {
                    "request_id": model_call["request_id"],
                    "module": "action_model",
                    "model": "test-policy",
                    "rpc_method": "vla.predict",
                    "cause": "TimeoutError",
                }
            else:
                assert (
                    worker_finished.is_set()
                    and "unconfirmed_model_requests" not in timing
                )
                assert timing["run_s"] >= timing["stop_s"] and timing["tail_s"] >= 0
    finally:
        allow_finish.set()
    assert worker_finished.wait(2)
