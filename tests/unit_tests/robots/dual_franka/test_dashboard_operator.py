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

"""Offline check of an operator-controlled task through the web API."""

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from types import SimpleNamespace

from fastapi.testclient import TestClient

from robots.dual_franka import robot_spec
from rpent.cli import dashboard, main
from rpent.dashboard.server import DashboardServer
from rpent.dashboard.state import DashboardState
from rpent.planner.base import PlannerResult
from tests.unit_tests.robots.dual_franka.test_exploration import FakeEnv


def test_web_operator_confirms_reset_and_completes_task(
    tmp_path, monkeypatch, dual_franka_robot_config
):
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
        with (
            TestClient(DashboardServer(state=state)._app) as client,
            ThreadPoolExecutor() as pool,
        ):
            future = pool.submit(
                toolkit.execute_tool, "request_scene_reset", {"reason": "test"}
            )
            deadline = time.monotonic() + 3
            while state.operator.snapshot() is None and time.monotonic() < deadline:
                time.sleep(0.01)
            operator = client.get("/api/session/state").json()["operator"]
            assert operator["pending"] is not None and env.resets == 0
            context = {
                "generation": operator["generation"],
                "request_id": operator["pending"]["id"],
            }
            for stale in (
                {**context, "generation": context["generation"] - 1},
                {**context, "request_id": "expired"},
            ):
                assert client.post(
                    "/api/session/messages",
                    json={"text": "/done", "operator_context": stale},
                ).is_error
            assert env.resets == 0
            assert (
                client.post(
                    "/api/session/messages",
                    json={"text": "/done", "operator_context": context},
                ).status_code
                == 202
            )
            assert not future.result(3).is_error
            toolkit.execute_tool(
                "move_delta", {"arm": "right", "delta_xyz": [0, 0, 0.01]}
            )
            assert (
                client.post(
                    "/api/session/messages",
                    json={"text": "/success", "operator_context": context},
                ).status_code
                == 202
            )
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
    assert (tmp_path / "memory/task-specific/dual_franka_t3_recipe.jsonl").is_file()
