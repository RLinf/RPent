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

"""Attended Dashboard reset, verdict and memory lifecycle with fake hardware."""

import json

import pytest

from robots.dual_franka.toolkit import DualFrankaToolkit
from rpent.dashboard.events import NullDashboardEventSink
from rpent.memory import MemoryManager
from tests.unit_tests.robots.dual_franka.test_exploration import FakeEnv


class Env(FakeEnv):
    def get_robot_state(self):
        return {**super().get_robot_state(), "coordinate_frame": "right_base"}


@pytest.fixture
def build(tmp_path, monkeypatch, dual_franka_robot_config):
    from robots.franka.runtime_config import set_robot_config_path

    set_robot_config_path(dual_franka_robot_config)
    monkeypatch.setattr("robots.dual_franka.toolkit.get_output_dir", lambda: tmp_path)
    created = []

    def make(*, confirm=True):
        root = tmp_path / str(len(created))
        env, replies = Env(), ["done"]
        toolkit = DualFrankaToolkit(
            runtime_kwargs={"env": env, "model": None, "task_description": "test"},
            dashboard_events=NullDashboardEventSink(),
            memory=MemoryManager(
                root / "memory", memory_access="inbox_write", inbox_cell_tag="test"
            ),
            mode="evaluation",
            state_output_dir=root,
            operator_input=lambda *a, **kw: replies.pop(0),
        )
        created.append(toolkit)
        if confirm:
            result = call(toolkit, "request_scene_reset", reason="operator ready")
            assert result["operator_lifecycle"]["scene_ready"]
        return toolkit, env, replies

    yield make
    for toolkit in created:
        toolkit.close()


def call(toolkit, tool, **kwargs):
    return toolkit.execute_tool(tool, kwargs).data


@pytest.mark.parametrize("reply", ["done", "abort"])
@pytest.mark.parametrize("shortcut", [False, True])
@pytest.mark.parametrize("notes", ["", "calibration board still attached"])
def test_dashboard_operator_reset_is_request_scoped(
    build, tmp_path, reply, shortcut, notes
):
    import time
    from concurrent.futures import ThreadPoolExecutor

    from fastapi.testclient import TestClient

    from robots.dual_franka.robot_spec import DUAL_FRANKA_DASHBOARD_SPEC
    from rpent.dashboard.operator import DashboardOperator
    from rpent.dashboard.server import DashboardServer
    from rpent.dashboard.state import DashboardState

    t, env, _ = build(confirm=False)
    state = DashboardState(
        output_dir=tmp_path / "web", dashboard_spec=DUAL_FRANKA_DASHBOARD_SPEC
    )
    state.shared_services_ready()
    state.request_task({"task_id": 3})
    state.wait_for_task(0)
    state.operator = DashboardOperator(tmp_path / "web")
    t._operator_input = state.operator
    state.bind_toolkit(t)
    with (
        TestClient(DashboardServer(state=state)._app) as client,
        ThreadPoolExecutor() as pool,
    ):
        future = pool.submit(call, t, "request_scene_reset", reason="web reset")
        try:
            deadline = time.monotonic() + 3
            while state.operator.snapshot() is None and time.monotonic() < deadline:
                time.sleep(0.01)
            response = client.get("/api/session/state")
            assert response.status_code == 200
            data = response.json()["operator"]
            assert data["enabled"] and data["active"]
            assert data["pending"]["choices"] == ["done", "abort"]
            assert env.resets == 0
            message = {
                "text": "/done"
                if shortcut and reply == "done"
                else f"/operator {data['pending']['id']} {reply}",
                "operator_context": {
                    "generation": data["generation"],
                    "request_id": data["pending"]["id"],
                },
            }
            if notes:
                message["text"] += f" {notes}"
            invalid_messages = [
                (
                    {"text": message["text"]},
                    "No active operator channel",
                ),
                (
                    {**message, "text": "/operator"},
                    "Usage: /operator",
                ),
                (
                    {**message, "operator_context": {"generation": 99}},
                    "expired operator task",
                ),
                (
                    {**message, "text": f"/operator stale-request {reply}"},
                    "No matching pending operator request",
                ),
                (
                    {**message, "text": f"/operator {data['pending']['id']} success"},
                    "invalid operator response",
                ),
            ]
            invalid_messages.extend(
                (
                    {**message, "operator_context": invalid_context},
                    "operator_context must be an object",
                )
                for invalid_context in ([], "invalid", 123, True)
            )
            if shortcut and reply == "done":
                invalid_messages.append(
                    (
                        {
                            **message,
                            "operator_context": {
                                **message["operator_context"],
                                "request_id": "stale-request",
                            },
                        },
                        "No matching pending operator request",
                    )
                )
            for invalid_message, error in invalid_messages:
                rejected = client.post("/api/session/messages", json=invalid_message)
                assert rejected.status_code == 422
                assert error in rejected.json()["error"]
                assert not future.done() and env.resets == 0
                assert (
                    client.get("/api/session/state").json()["operator"]["pending"]
                    == data["pending"]
                )
            assert client.post("/api/session/messages", json=message).status_code == 202
            result = future.result(timeout=3)
            assert env.resets == (1 if reply == "done" else 0)
            if reply == "abort":
                assert result["operator_aborted"]
                finish = call(t, "finish", status="failure", summary="operator abort")
                assert finish["_finish"] and finish["operator_aborted"]
                assert not t.solved()
            assert (
                client.get("/api/session/state").json()["operator"]["pending"] is None
            )
            duplicate = client.post("/api/session/messages", json=message)
            assert duplicate.status_code == 422
            assert "No matching pending operator request" in duplicate.json()["error"]
            assert env.resets == (1 if reply == "done" else 0)
        finally:
            state.operator.close()
    state.unbind_toolkit(t)


@pytest.mark.parametrize("verdict", ["success", "failure", "abort"])
def test_dashboard_terminal_verdict_closes_only_current_task(build, tmp_path, verdict):
    from fastapi.testclient import TestClient

    from robots.dual_franka.robot_spec import DUAL_FRANKA_DASHBOARD_SPEC
    from rpent.dashboard.operator import DashboardOperator
    from rpent.dashboard.server import DashboardServer
    from rpent.dashboard.state import DashboardState

    t, _, _ = build()
    state = DashboardState(
        output_dir=tmp_path / "web", dashboard_spec=DUAL_FRANKA_DASHBOARD_SPEC
    )
    state.shared_services_ready()
    state.request_task({"task_id": 3})
    state.wait_for_task(0)
    state.operator = DashboardOperator(tmp_path / "web")
    state.bind_toolkit(t)
    with TestClient(DashboardServer(state=state)._app) as client:
        message = {
            "text": f"/{verdict} test",
            "operator_context": {"generation": 1},
        }
        assert (
            client.post(
                "/api/session/messages",
                json=message,
            ).status_code
            == 202
        )
        assert t.direct_verdict_requested
        assert "motion_refused" in call(t, "open_gripper", arm="right")
        t.finalize_direct_verdict()
        assert t.solved() is (verdict == "success")
        duplicate = client.post("/api/session/messages", json=message)
        assert duplicate.status_code == 422
        assert "no matching active operator task" in duplicate.json()["error"]
        assert client.get("/api/session/state").json()["operator"]["verdict"] == verdict
        state.unbind_toolkit(t)
        state.complete_task(state="cancelled")
        assert (
            state._task_state
            == {"success": "succeeded", "failure": "failed", "abort": "cancelled"}[
                verdict
            ]
        )
        state.request_task({"task_id": 3})
        state.wait_for_task(0)
        assert state.operator_verdict is None
        next_toolkit, _, _ = build()
        state.operator = DashboardOperator(tmp_path / "next-web")
        state.bind_toolkit(next_toolkit)
        try:
            late = client.post("/api/session/messages", json=message)
            assert late.status_code == 422
            assert "expired operator task" in late.json()["error"]
            assert not next_toolkit.direct_verdict_requested
            snapshot = client.get("/api/session/state").json()["operator"]
            assert snapshot["generation"] == 2
            assert snapshot["verdict"] is None
        finally:
            state.unbind_toolkit(next_toolkit)


def test_dashboard_cancel_releases_pending_operator_wait(build, tmp_path):
    import time
    from concurrent.futures import ThreadPoolExecutor

    from rpent.dashboard.operator import DashboardOperator

    t, env, _ = build(confirm=False)
    t._operator_input = DashboardOperator(tmp_path)
    with ThreadPoolExecutor() as pool:
        future = pool.submit(call, t, "request_scene_reset", reason="wait")
        deadline = time.monotonic() + 3
        while t._operator_input.snapshot() is None and time.monotonic() < deadline:
            time.sleep(0.01)
        t.cancel_active_and_wait()
        future.result(timeout=3)
        assert env.resets == 0
        assert t._operator_input.snapshot() is None


@pytest.mark.parametrize(
    "verdict,auto_merge,planner_error,replace_task",
    [
        ("success", True, None, False),
        ("success", False, None, False),
        ("success", True, "planner failed", False),
        ("failure", True, None, False),
        ("abort", True, None, False),
        (None, True, None, True),
        ("success", True, None, True),
    ],
)
@pytest.mark.parametrize("explore", [False, True])
def test_dashboard_task_orchestration_uses_web_operator(
    tmp_path,
    monkeypatch,
    verdict,
    auto_merge,
    planner_error,
    replace_task,
    explore,
    dual_franka_robot_config,
):
    import time
    from concurrent.futures import ThreadPoolExecutor
    from dataclasses import replace

    from robots.dual_franka import robot_spec
    from rpent.cli import dashboard, main
    from rpent.dashboard.state import DashboardState
    from rpent.planner.base import PlannerResult

    parser = main._build_argparser()
    robot_spec._add_cli_args(parser, use_dashboard=True)
    args = parser.parse_args(
        [
            "--robot",
            "dual_franka",
            "--dashboard",
            "--task-id",
            "3",
            "--robot-config",
            str(dual_franka_robot_config),
            "--planner",
            "codex",
            "--memory-profile",
            "local",
            "--memory-dir",
            str(tmp_path / "memory"),
            "--auto-merge-memory" if auto_merge else "--no-auto-merge-memory",
        ]
    )
    env = Env()

    args.explore = explore

    def runtime(_task_args, *unused):
        assert env.resets == 0
        return [], {
            "env": env,
            "model": None,
            "task_description": "test",
        }

    spec = replace(robot_spec.get_robot_spec(), init_runtime=runtime)
    state = DashboardState(output_dir=tmp_path, dashboard_spec=spec.dashboard)
    state.shared_services_ready()
    state.request_task({"task_id": 3})
    claimed = state.wait_for_task(0)
    merge_calls = []
    merge_memory = MemoryManager.merge_memory

    def merge(manager, **kwargs):
        merge_calls.append(kwargs)
        return merge_memory(manager, **kwargs)

    monkeypatch.setattr(MemoryManager, "merge_memory", merge)

    class Planner:
        def solve(self, **kwargs):
            t = kwargs["toolkit"]
            assert kwargs["dashboard_interaction"] is state
            assert t._attended and env.resets == 0
            assert "error" in call(t, "move_delta", arm="right", delta_xyz=[0, 0, 0.01])
            with ThreadPoolExecutor() as pool:
                future = pool.submit(
                    call, t, "request_scene_reset", reason="fake web test"
                )
                try:
                    deadline = time.monotonic() + 3
                    while (
                        state.operator.snapshot() is None
                        and time.monotonic() < deadline
                    ):
                        time.sleep(0.01)
                    data = state.operator_snapshot()
                    state.operator_reply(
                        data["generation"], data["pending"]["id"], "done"
                    )
                    future.result(3)
                finally:
                    if not future.done():
                        state.operator.close()
            assert env.resets == 1
            if verdict is not None:
                state.operator_finish(1, verdict, "fake operator verdict")
            if replace_task:
                state.request_task({"task_id": 4})
            return PlannerResult(messages=[], stats={}, error=planner_error)

    monkeypatch.setattr(dashboard, "build_planner", lambda *a, **kw: Planner())
    error = dashboard._run_dashboard_task(
        args=args,
        robot_spec=spec,
        state=state,
        claimed=claimed,
        shared_runtime_kwargs={},
        unique_components={"env"},
        session_root=tmp_path,
    )
    assert error == planner_error
    assert env.resets == 1
    if explore and verdict in {"success", "failure"}:
        evidence = list((tmp_path / "memory").rglob(f"*-{verdict}.json"))
        assert len(evidence) == 1
        assert (
            json.loads(evidence[0].read_text())["verdict"]["operator_verdict"]
            == verdict
        )
    should_merge = (
        explore
        and verdict == "success"
        and auto_merge
        and planner_error is None
        and not replace_task
    )
    assert len(merge_calls) == int(should_merge)
    for name in ("dual_franka_t3.json", "dual_franka_t3_recipe.jsonl"):
        assert (tmp_path / "memory" / "task-specific" / name).is_file() is should_merge
    state.complete_task(state="cancelled", error=error)
    assert state._task_state == (
        "cancelled"
        if error
        else {
            "success": "succeeded",
            "failure": "failed",
            "abort": "cancelled",
            None: "cancelled",
        }[verdict]
    )
    if replace_task:
        replacement = state.wait_for_task(0)
        assert replacement is not None
        assert replacement.number == 2
        assert replacement.request == {"task_id": 4}
