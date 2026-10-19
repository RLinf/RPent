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

"""Offline Web transport with the real runtime connector, toolkit and runner."""

from concurrent.futures import ThreadPoolExecutor
from threading import Event
from types import SimpleNamespace

import pytest

from robots.dual_franka import robot_spec
from robots.dual_franka.tasks import get_dual_franka_task
from rpent.cli import dashboard
from rpent.cli.main import _build_argparser
from rpent.dashboard.server import DashboardServer
from rpent.dashboard.state import DashboardState
from rpent.planner.base import PlannerResult
from tests.unit_tests.robots.dual_franka.test_exploration import FakeEnv


@pytest.mark.parametrize(
    "explore,verdict",
    [(False, v) for v in ("success", "failure", "continue")]
    + [(True, v) for v in ("success", "failure", "abort", "replace", "shutdown")],
)
@pytest.mark.parametrize("auto_merge", [True, False])
def test_web_confirmation_and_task_completion(
    tmp_path, monkeypatch, dual_franka_robot_config, explore, verdict, auto_merge
):
    fake_env = FakeEnv()

    def rpc_call(name, *, kwargs=None, **options):
        if name == "env.get_env_meta":
            return {"explicit_reset_only": True}
        if name == "env.get_observation":
            return {**fake_env.get_observation(), "states": []}
        if name == "env.move_delta":
            return fake_env.move_delta(kwargs["arm"], kwargs["delta_xyz"])
        return getattr(fake_env, name.removeprefix("env."))()

    rpc = SimpleNamespace(call=rpc_call)
    monkeypatch.setattr(robot_spec, "try_spawn_server", lambda *a: (None, rpc))
    monkeypatch.setattr(robot_spec, "try_wait_server", lambda *a, **kw: kw["post_fn"]())
    spec = robot_spec.get_robot_spec()
    parser = _build_argparser()
    spec.add_cli_args(parser, use_dashboard=True)
    args = parser.parse_args(
        ["--robot", "dual_franka", "--dashboard", "--task-id", "1"]
        + ([] if auto_merge else ["--no-auto-merge-memory"])
    )
    args.robot_config = str(dual_franka_robot_config)
    args.memory_dir, args.memory_profile = str(tmp_path / "memory"), "local"
    args.explore, args.explore_sessions = explore, 2
    tag = f"dual_franka_{get_dual_franka_task(1).name}"
    old_task = tmp_path / "memory/task-specific"
    old_task.mkdir(parents=True)
    for name in ("dual_franka_t1.json", "dual_franka_t1_recipe.jsonl"):
        (old_task / name).write_text("old task record")
    state = DashboardState(output_dir=tmp_path, dashboard_spec=spec.dashboard)
    state.shared_services_ready()
    state.request_task({"task_id": 1})
    claimed = state.wait_for_task(0)
    ready, changed, calls = Event(), state.operator_changed, []

    def notify():
        changed()
        if state.operator and state.operator.snapshot():
            ready.set()

    monkeypatch.setattr(state, "operator_changed", notify)

    def solve(**kwargs):
        calls.append(True)
        toolkit, reader = kwargs["toolkit"], state.operator
        call = toolkit.execute_tool
        motion = {"arm": "right", "delta_xyz": [0, 0, 0.01]}
        if explore:
            inbox = toolkit.memory.root / "_internal/inbox" / tag
            inbox.mkdir(parents=True, exist_ok=True)
            (inbox / "new_global_strategy_lesson.md").write_text(
                "---\nid: lesson\nscope: global\nkind: strategy\ntitle: Lesson\n"
                "applies_when: Same setup\nconfidence: single-shot\n"
                f"evidence:\n  cells: [{tag}]\n---\nObserved attempt.\n"
            )
        assert fake_env.resets == 0
        assert "request_scene_reset" in {t.name for t in toolkit.list_tools()}
        endpoint = next(
            r.endpoint
            for r in DashboardServer(state=state)._app.routes
            if r.path == "/api/session/messages" and "POST" in r.methods
        )

        def context():
            snapshot = state.snapshot()
            op = snapshot["operator"]
            return {
                "generation": snapshot["task_generation"],
                "request_id": (op["pending"] or {}).get("id"),
            }

        def send(text, ctx=None):
            payload = {"text": text, "operator_context": ctx or context()}
            return endpoint(payload).status_code

        with ThreadPoolExecutor() as pool:
            try:
                assert call("move_delta", motion).is_error
                future = pool.submit(call, "request_scene_reset", {"reason": "test"})
                assert ready.wait(3)
                ctx = context()
                for stale in (
                    {**ctx, "generation": ctx["generation"] - 1},
                    {**ctx, "request_id": "expired"},
                ):
                    assert send("/done", stale) == 422
                assert send("/success", ctx) == 422
                assert fake_env.resets == 0
                assert send("/done", ctx) == 202
                assert not future.result(3).is_error
                assert send("/done", ctx) == 422
                assert fake_env.resets == 1
                state.set_planner_activity("idle", accepting_input=True)
                assert send("move slowly") == 202
                assert not call("move_delta", motion).is_error
                ready.clear()
                pending = pool.submit(call, "request_operator_verdict", {})
                assert ready.wait(3)
                ctx = context()
                assert send("/success", {**ctx, "request_id": "expired"}) == 422
                if verdict in {"replace", "shutdown"}:
                    if verdict == "replace":
                        assert send("/rpent-task 2") == 202
                        assert send("/success") == 422
                    else:
                        state.request_shutdown()
                    toolkit.cancel_active_and_wait()
                else:
                    if not explore:
                        assert send("/done") == 422
                        assert send("/abort") == 422
                    assert send("/" + verdict, ctx) == 202
                    assert send("/" + verdict, ctx) == 422
                response = pending.result(3)
                assert state.task_replacement_requested is (
                    explore and verdict != "shutdown"
                )
                if not explore:
                    assert not response.is_error and response.data["status"] == verdict
                    assert not toolkit.direct_verdict_requested
                    status = "failure" if verdict == "failure" else "success"
                    finish = call("finish", {"status": status, "summary": "test"})
                    if verdict == "continue":
                        assert finish.is_error
                        return PlannerResult()
                    assert not finish.is_error and finish.data["_finish"]
                    assert not finish.data.get("operator_finished")
                    return PlannerResult(finish_result=finish.data)
            finally:
                reader.close()
        return PlannerResult()

    monkeypatch.setattr(
        dashboard, "build_planner", lambda *a, **kw: SimpleNamespace(solve=solve)
    )
    result = dashboard._run_dashboard_task(
        args=args,
        robot_spec=spec,
        state=state,
        claimed=claimed,
        shared_runtime_kwargs={"model": None},
        unique_components={"env"},
        session_root=tmp_path,
    )
    assert result is None
    assert len(calls) == 1 and len(fake_env.moves) == 1 and state.operator is None
    assert (state.wait_for_task(0) is not None) is (verdict == "replace")
    assert (tmp_path / "memory/global/lesson.md").is_file() == (
        explore and auto_merge and verdict in {"success", "failure"}
    )
    for name in (f"{tag}.json", f"{tag}_recipe.jsonl"):
        assert (tmp_path / "memory/task-specific" / name).is_file() == (
            explore and auto_merge and verdict == "success"
        )
    for name in ("dual_franka_t1.json", "dual_franka_t1_recipe.jsonl"):
        assert (old_task / name).read_text() == "old task record"
