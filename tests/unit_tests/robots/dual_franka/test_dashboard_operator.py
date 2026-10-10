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

"""Web operator contracts using real Dashboard, toolkit and memory components."""

import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from robots.dual_franka import robot_spec
from robots.dual_franka.toolkit import DualFrankaToolkit
from rpent.dashboard.events import NullDashboardEventSink
from rpent.dashboard.operator import DashboardOperator
from rpent.dashboard.server import DashboardServer
from rpent.dashboard.state import DashboardState
from rpent.memory import MemoryManager
from rpent.planner.base import PlannerResult
from tests.unit_tests.robots.dual_franka.test_exploration import FakeEnv


def call(toolkit, name, **kwargs):
    return toolkit.execute_tool(name, kwargs).data


def wait_for_request(operator):
    deadline = time.monotonic() + 3
    while operator.snapshot() is None and time.monotonic() < deadline:
        time.sleep(0.01)
    assert operator.snapshot() is not None


@pytest.fixture
def pending_reset(tmp_path, monkeypatch, dual_franka_robot_config):
    from robots.franka.runtime_config import set_robot_config_path

    set_robot_config_path(dual_franka_robot_config)
    monkeypatch.setattr("robots.dual_franka.toolkit.get_output_dir", lambda: tmp_path)
    env = FakeEnv()
    state = DashboardState(
        output_dir=tmp_path, dashboard_spec=robot_spec.DUAL_FRANKA_DASHBOARD_SPEC
    )
    state.shared_services_ready()
    state.request_task({"task_id": 3})
    state.wait_for_task(0)
    operator = DashboardOperator(tmp_path)
    state.operator = operator
    toolkit = DualFrankaToolkit(
        runtime_kwargs={"env": env, "model": None, "task_description": "test"},
        dashboard_events=NullDashboardEventSink(),
        memory=MemoryManager(
            tmp_path / "memory", memory_access="inbox_write", inbox_cell_tag="test"
        ),
        mode="evaluation",
        state_output_dir=tmp_path,
        operator_input=state.operator,
    )
    state.bind_toolkit(toolkit)
    with (
        TestClient(DashboardServer(state=state)._app) as client,
        ThreadPoolExecutor() as pool,
    ):
        future = pool.submit(call, toolkit, "request_scene_reset", reason="web reset")
        try:
            wait_for_request(state.operator)
            pending = client.get("/api/session/state").json()["operator"]
            yield toolkit, env, state, client, pending, future
        finally:
            operator.close()
            toolkit.close()
            state.unbind_toolkit(toolkit)


def message(pending, text):
    return {
        "text": text,
        "operator_context": {
            "generation": pending["generation"],
            "request_id": pending["pending"]["id"],
        },
    }


@pytest.mark.parametrize("answer", ["done", "abort"])
def test_reset_waits_for_one_matching_reply(pending_reset, answer):
    toolkit, env, _, client, pending, future = pending_reset
    reply = message(pending, f"/{answer} scene checked")
    for context in (
        {"generation": -1},
        {**reply["operator_context"], "request_id": "stale"},
    ):
        assert (
            client.post(
                "/api/session/messages",
                json={"text": "/done", "operator_context": context},
            ).status_code
            == 422
        )
        assert not future.done() and env.resets == 0
    assert client.post("/api/session/messages", json=reply).status_code == 202
    result = future.result(3)
    assert env.resets == int(answer == "done")
    if answer == "done":
        assert toolkit.state.latest_record().result["operator_notes"] == "scene checked"
    else:
        assert result["code"] == "tool_cancelled"
    assert client.post("/api/session/messages", json=reply).status_code == 422
    assert env.resets == int(answer == "done")


def test_verdict_closes_only_current_task(pending_reset):
    toolkit, env, state, client, pending, future = pending_reset
    assert (
        client.post("/api/session/messages", json=message(pending, "/done")).status_code
        == 202
    )
    future.result(3)
    reply = message(pending, "/success checked")
    assert client.post("/api/session/messages", json=reply).status_code == 202
    assert call(toolkit, "open_gripper", arm="right")["motion_refused"]
    toolkit.finalize_direct_verdict()
    assert toolkit.solved()
    assert client.post("/api/session/messages", json=reply).status_code == 422
    state.unbind_toolkit(toolkit)
    state.complete_task(state="cancelled")
    assert client.get("/api/session/state").json()["state"] == "succeeded"
    state.request_task({"task_id": 4})
    state.wait_for_task(0)
    state.bind_toolkit(toolkit)
    assert client.post("/api/session/messages", json=reply).status_code == 409
    assert env.resets == 1


def test_late_verdict_preserves_queued_task(pending_reset):
    toolkit, _, state, client, pending, _ = pending_reset
    assert (
        client.post(
            "/api/session/messages", json=message(pending, "/rpent-task 4")
        ).status_code
        == 202
    )
    assert (
        client.post(
            "/api/session/messages", json=message(pending, "/success")
        ).status_code
        == 422
    )
    assert not toolkit.direct_verdict_requested
    state.unbind_toolkit(toolkit)
    state.complete_task(state="cancelled")
    assert state.wait_for_task(0).request == {"task_id": 4}


def test_cancel_releases_operator_wait_without_reset(pending_reset):
    toolkit, env, state, _, _, future = pending_reset
    toolkit.cancel_active_and_wait()
    future.result(3)
    assert env.resets == 0 and state.operator.snapshot() is None


@pytest.mark.parametrize("verdict", ["success", "failure", "abort"])
def test_task_uses_web_operator_and_publishes_only_confirmed_success(
    tmp_path, monkeypatch, dual_franka_robot_config, verdict
):
    from rpent.cli import dashboard, main

    parser = main._build_argparser()
    robot_spec._add_cli_args(parser, use_dashboard=True)
    args = parser.parse_args(
        [
            "--robot",
            "dual_franka",
            "--dashboard",
            "--explore",
            "--task-id",
            "3",
            "--planner",
            "codex",
            "--auto-merge-memory",
            "--robot-config",
            str(dual_franka_robot_config),
            "--memory-dir",
            str(tmp_path / "memory"),
            "--output-dir",
            str(tmp_path / "run"),
        ]
    )
    env = FakeEnv()
    spec = replace(
        robot_spec.get_robot_spec(),
        init_runtime=lambda *a: (
            [],
            {"env": env, "model": None, "task_description": "test"},
        ),
    )
    state = DashboardState(output_dir=tmp_path, dashboard_spec=spec.dashboard)
    state.shared_services_ready()
    state.request_task({"task_id": 3})
    claimed = state.wait_for_task(0)

    def solve(**kwargs):
        toolkit = kwargs["toolkit"]
        assert env.resets == 0
        with ThreadPoolExecutor() as pool:
            future = pool.submit(call, toolkit, "request_scene_reset", reason="test")
            wait_for_request(state.operator)
            pending = state.operator_snapshot()
            state.operator_reply(
                pending["generation"], pending["pending"]["id"], "done"
            )
            future.result(3)
        call(toolkit, "move_delta", arm="right", delta_xyz=[0, 0, 0.01])
        state.operator_finish(pending["generation"], verdict, "checked")
        return PlannerResult()

    monkeypatch.setattr(
        dashboard, "build_planner", lambda *a, **kw: SimpleNamespace(solve=solve)
    )
    assert (
        dashboard._run_dashboard_task(
            args=args,
            robot_spec=spec,
            state=state,
            claimed=claimed,
            shared_runtime_kwargs={},
            unique_components={"env"},
            session_root=tmp_path,
        )
        is None
    )
    assert env.resets == 1 and len(env.moves) == 1
    for filename in ("dual_franka_t3.json", "dual_franka_t3_recipe.jsonl"):
        assert (tmp_path / "memory/task-specific" / filename).is_file() is (
            verdict == "success"
        )
    if verdict != "abort":
        evidence = list((tmp_path / "memory").rglob(f"*-{verdict}.json"))
        assert len(evidence) == 1
        assert (
            json.loads(evidence[0].read_text())["verdict"]["operator_verdict"]
            == verdict
        )
