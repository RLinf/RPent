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

"""Runner recovery and finalization preserve structured stop evidence."""

import json
import sys
from types import SimpleNamespace

import pytest

from robots.libero.result import finalize_libero_result
from rpent.cli import dashboard
from rpent.cli import main as cli
from rpent.dashboard.state import ClaimedTask, DashboardState
from rpent.planner.base import PlannerResult
from rpent.robots import PromptBundle, RobotSpec, RunConfig

_SAFE_STOP = {
    "reason": "backend_pause",
    "category": "cancelled",
    "execution_certain": True,
    "result_delivery_complete": True,
    "cleanup_complete": True,
    "allow_new_session": True,
}


@pytest.mark.parametrize("entrypoint", ["cli", "dashboard"])
@pytest.mark.parametrize(
    "stop_info,planner_error,expected_sessions,fail_second",
    [
        (None, None, 2, False),
        (None, "Planner timed out", 2, False),
        ({}, None, 1, False),
        ({}, "Planner timed out", 1, False),
        ({**_SAFE_STOP, "execution_certain": False}, "Planner timed out", 1, False),
        ({**_SAFE_STOP, "allow_new_session": False}, None, 1, False),
        (_SAFE_STOP, None, 2, False),
        (_SAFE_STOP, "Planner timed out", 2, False),
        (_SAFE_STOP, None, 2, True),
    ],
)
def test_runners_require_stop_evidence_and_preserve_finalization_fields(
    tmp_path,
    monkeypatch,
    entrypoint,
    stop_info,
    planner_error,
    expected_sessions,
    fail_second,
):
    captured = []
    solve_calls = []
    returned_stats = []
    usage = {"tool_calls": 0, "total_input_tokens": 17}

    class Toolkit:
        memory = None
        closed = False

        def cancel_active_and_wait(self):
            pass

        def solved(self):
            return False

        def evaluation_metrics(self, *, success):
            assert success is False
            return {}

        def close(self):
            self.closed = True

    toolkit = Toolkit()

    def solve(**kwargs):
        solve_calls.append(kwargs["user_message"])
        if fail_second and len(solve_calls) == 2:
            raise RuntimeError("second session failed")
        stats = {**usage, "stop_reason": "budget", "stop_info": stop_info}
        returned_stats.append(stats)
        return PlannerResult(
            stats=stats,
            error=planner_error,
            stop_info=stop_info,
            stop_reason="request_limit" if stop_info is not None else None,
        )

    def finalize(context):
        assert toolkit.closed
        captured.append(context)

    def add_cli_args(parser, use_dashboard):
        parser.add_argument("--explore-sessions", type=int, default=2)

    spec = RobotSpec(
        name="libero",
        prompts=PromptBundle(
            system=lambda variables: "system", user=lambda variables: "task"
        ),
        add_cli_args=add_cli_args,
        parse_config=lambda args: RunConfig(
            recipe_tag="task", output_dir=tmp_path / "run", prompt_vars={}, task_desc={}
        ),
        init_runtime=lambda *args: ([], {}),
        finalize_run=finalize,
        supports_exploration=True,
    )
    argv = [
        "--robot",
        "libero",
        "--planner",
        "api",
        "--model",
        "test",
        "--memory-profile",
        "local",
        "--memory-dir",
        str(tmp_path / "memory"),
        "--explore",
        "--output-dir",
        str(tmp_path / "run"),
    ]
    runner = cli if entrypoint == "cli" else dashboard
    monkeypatch.setattr(
        runner, "build_planner", lambda *a, **kw: SimpleNamespace(solve=solve)
    )
    monkeypatch.setattr(runner, "get_toolkit", lambda *a, **kw: toolkit)
    if entrypoint == "cli":
        monkeypatch.setattr(cli, "get_robot_spec", lambda name: spec)
        monkeypatch.setattr(cli, "enumerate_robots", lambda: ("libero",))
        monkeypatch.setattr(sys, "argv", ["rpent", *argv])
        assert cli.main() == (1 if planner_error or fail_second else 0)
    else:
        args = cli._build_argparser().parse_args(argv)
        args.explore_sessions = 2
        state = DashboardState(
            output_dir=tmp_path,
            dashboard_spec={
                "task": {
                    "command": "/task",
                    "usage": "/task",
                    "fields": (),
                    "display": "task",
                    "output_slug": "task",
                },
                "runtime_components": (),
                "primitives": (),
            },
        )
        error = dashboard._run_dashboard_task(
            args=args,
            robot_spec=spec,
            state=state,
            claimed=ClaimedTask(number=1, request={}, output_dir=tmp_path / "run"),
            shared_runtime_kwargs={},
            unique_components=set(),
            session_root=tmp_path,
        )
        assert error == ("second session failed" if fail_second else planner_error)
    assert len(solve_calls) == expected_sessions
    assert len(captured) == 1
    assert captured[0].stop_info == (
        {"category": "error"} if fail_second else stop_info
    )
    assert captured[0].stop_reason == (
        None if fail_second or stop_info is None else "request_limit"
    )
    assert all(
        stats == {**usage, "stop_reason": "budget", "stop_info": stop_info}
        for stats in returned_stats
    )
    result = json.loads(finalize_libero_result(captured[0]).read_text())
    if fail_second:
        assert captured[0].stats == usage
        assert result["termination_reason"] == "planner_error"
        assert result["termination_category"] == "error"
        assert result["stop_info"]["category"] == "error"
        assert result["stop_info"]["execution_certain"] is None
    else:
        assert captured[0].stats == returned_stats[-1]
        assert result["termination_reason"] == (
            (stop_info or {}).get("reason")
            or ("request_limit" if stop_info is not None else "budget")
        )


@pytest.mark.parametrize(
    "entrypoint,event",
    [
        (entrypoint, event)
        for entrypoint in ("cli", "dashboard")
        for event in (
            "timeout",
            "drain",
            "metrics",
            "close",
            "legacy_close",
            "daemon",
            "planner_and_close",
            "interrupt",
            "result_write",
        )
    ]
    + [
        ("dashboard", event)
        for event in (
            "replace",
            "shutdown",
            "fatal",
            "success",
            "failure",
            "abort",
        )
    ],
)
def test_runner_finalization_and_capture_share_terminal_boundary(
    tmp_path, monkeypatch, entrypoint, event
):
    from rpent.session import EnvState

    captured = []
    original_stop = {
        **_SAFE_STOP,
        "reason": "timeout",
        "category": "timeout",
    }
    original_result = PlannerResult(
        error="planner timed out after 1s",
        stop_reason="timeout",
        stop_info=None if event == "legacy_close" else original_stop,
        stats={"stop_info": original_stop} if event == "legacy_close" else {},
    )
    state = DashboardState(
        output_dir=tmp_path,
        dashboard_spec={
            "task": {
                "command": "/task",
                "usage": "/task",
                "fields": (),
                "display": "task",
                "output_slug": "task",
            },
            "runtime_components": (),
            "primitives": (),
        },
    )
    state.shared_services_ready()
    state.request_task({})
    claimed = state.wait_for_task(timeout=0)

    class Toolkit:
        memory = None
        direct_verdict_requested = False
        closed = False

        def __init__(self):
            self.state = EnvState(tmp_path / "trajectory")

        def cancel_active_and_wait(self):
            if event == "drain":
                raise RuntimeError("drain failed")

        def solved(self):
            return False

        def evaluation_metrics(self, *, success):
            if event == "metrics":
                raise RuntimeError("metrics failed")
            return {}

        def close(self):
            self.closed = True
            if event in ("close", "legacy_close", "planner_and_close"):
                raise RuntimeError("close failed")

        def request_direct_verdict(self, verdict):
            self.direct_verdict_requested = True
            return True

        def finalize_direct_verdict(self):
            return {
                "operator_finished": True,
                "operator_verdict": event,
                "operator_aborted": event == "abort",
            }

    toolkit = Toolkit()
    if event == "interrupt":
        toolkit.memory = SimpleNamespace(
            merge_memory=lambda **kwargs: pytest.fail(
                "interrupted run published memory"
            )
        )

    class Daemon:
        stopped = False

        def stop(self):
            self.stopped = True
            if event == "daemon":
                raise RuntimeError("daemon failed")

    daemon = Daemon()

    def solve(**kwargs):
        if event == "planner_and_close":
            raise ValueError("model failed")
        if event == "interrupt":
            raise KeyboardInterrupt
        if event == "replace":
            state.request_task({})
        elif event == "shutdown":
            state.request_shutdown()
        elif event == "fatal":
            state.fail_session("shared service failed")
        elif event in ("success", "failure", "abort"):
            state.submit_input(
                f"/{event}",
                operator_context={"generation": claimed.number, "request_id": None},
            )
        return original_result

    def finalize(context):
        assert toolkit.closed and daemon.stopped
        captured.append(context)
        if event == "result_write":
            raise OSError("result write failed")
        return finalize_libero_result(context)

    spec = RobotSpec(
        name="libero",
        prompts=PromptBundle(
            system=lambda variables: "system", user=lambda variables: "task"
        ),
        add_cli_args=lambda parser, use_dashboard: parser.add_argument(
            "--auto-merge-memory", action="store_true"
        ),
        parse_config=lambda args: RunConfig(
            recipe_tag="task", output_dir=tmp_path / "run", prompt_vars={}, task_desc={}
        ),
        init_runtime=lambda *args: ([daemon], {}),
        finalize_run=finalize,
        supports_exploration=True,
        is_real_robot=entrypoint == "dashboard",
        supports_human_interactive_exploration=entrypoint == "dashboard",
    )
    argv = [
        "--robot",
        "libero",
        "--planner",
        "api",
        "--model",
        "test",
        "--memory-profile",
        "local",
        "--memory-dir",
        str(tmp_path / "memory"),
        "--output-dir",
        str(tmp_path / "run"),
        "--record-trajectory",
    ]
    runner = cli if entrypoint == "cli" else dashboard
    monkeypatch.setattr(
        runner, "build_planner", lambda *a, **kw: SimpleNamespace(solve=solve)
    )
    monkeypatch.setattr(runner, "get_toolkit", lambda *a, **kw: toolkit)
    if entrypoint == "cli":
        monkeypatch.setattr(cli, "get_robot_spec", lambda name: spec)
        monkeypatch.setattr(cli, "enumerate_robots", lambda: ("libero",))
        if event == "interrupt":
            argv += ["--explore", "--auto-merge-memory"]
        monkeypatch.setattr(sys, "argv", ["rpent", *argv])
        assert cli.main() == 1
    else:
        args = cli._build_argparser().parse_args([*argv, "--explore"])
        args.auto_merge_memory = event == "interrupt"
        error = dashboard._run_dashboard_task(
            args=args,
            robot_spec=spec,
            state=state,
            claimed=claimed,
            shared_runtime_kwargs={},
            unique_components=set(),
            session_root=tmp_path,
        )

    if event == "result_write":
        assert not (tmp_path / "run" / "result.json").exists()
        if entrypoint == "dashboard":
            assert "result finalization failed: OSError: result write failed" in error
        return
    context = captured[0]
    record = json.loads((tmp_path / "run" / "result.json").read_text())
    task = toolkit.state.get_task()
    assert task.stop_reason == record["termination_reason"]
    assert record["termination_category"] == (
        "timeout"
        if event == "timeout"
        else "finish"
        if event in ("success", "failure")
        else "cancelled"
        if event in ("interrupt", "replace", "shutdown", "abort")
        else "error"
    )
    assert task.status == (
        "completed"
        if record["termination_category"] in ("finish", "timeout")
        else "cancelled"
        if record["termination_category"] == "cancelled"
        else "failed"
    )
    assert record["valid"] is (event in ("timeout", "success", "failure"))
    assert context.cancelled_reason == (
        "task_replaced"
        if event == "replace"
        else "operator"
        if event in ("interrupt", "shutdown", "abort")
        else None
    )
    if event in ("drain", "close", "legacy_close", "daemon", "planner_and_close"):
        assert context.stop_info["cleanup_complete"] is False
    if event == "metrics":
        assert context.stop_info["cleanup_complete"] is True
    if event == "planner_and_close":
        assert "model failed" in context.agent_error
        assert "close failed" in context.agent_error
    assert original_result.stop_info == (
        None if event == "legacy_close" else original_stop
    )
    if event == "legacy_close":
        assert context.stop_info["reason"] == "timeout"
        assert context.stop_info["execution_certain"] is True
        assert original_result.stats == {"stop_info": original_stop}
    assert original_stop["category"] == "timeout"
