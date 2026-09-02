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

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from rpent.dashboard.session import DashboardSessionController
from rpent.dashboard.state import ClaimedTask
from rpent.planner.base import PlannerResult
from rpent.robots import PromptBundle, RunConfig


class ScriptedState:
    def __init__(self, tasks: list[ClaimedTask] | None = None) -> None:
        self.tasks = list(tasks or [])
        self.ready_calls = 0
        self.failed_session: Exception | None = None
        self.completed: list[dict[str, Any]] = []
        self.task_replacement_requested = False

    def shared_services_ready(self) -> None:
        self.ready_calls += 1

    def fail_session(self, error: Exception) -> None:
        self.failed_session = error

    def wait_for_task(self) -> ClaimedTask | None:
        return self.tasks.pop(0) if self.tasks else None

    def complete_task(self, *, state: str, error: Any) -> None:
        self.completed.append({"state": state, "error": error})


class FakeDaemon:
    def __init__(
        self,
        name: str,
        stopped: list[str],
        *,
        stop_error: Exception | None = None,
    ) -> None:
        self.name = name
        self.stopped = stopped
        self.stop_error = stop_error

    def stop(self) -> None:
        self.stopped.append(self.name)
        if self.stop_error is not None:
            raise self.stop_error


def _claimed_task(tmp_path: Path) -> ClaimedTask:
    return ClaimedTask(
        number=1,
        request={"mode": "pick", "seed": 1},
        output_dir=tmp_path / "task",
    )


def test_dashboard_session_shared_start_failure_marks_session_fatal(
    tmp_path: Path,
) -> None:
    state = ScriptedState([_claimed_task(tmp_path)])
    run_calls = 0

    def start_shared():
        raise RuntimeError("shared runtime failed")

    def run_task(claimed: ClaimedTask, shared: dict[str, Any]):
        nonlocal run_calls
        del claimed, shared
        run_calls += 1

    controller = DashboardSessionController(
        state=state,  # type: ignore[arg-type]
        start_shared=start_shared,
        run_task=run_task,
    )

    controller.run()

    assert isinstance(state.failed_session, RuntimeError)
    assert str(state.failed_session) == "shared runtime failed"
    assert state.ready_calls == 0
    assert run_calls == 0
    assert state.completed == []


@pytest.mark.parametrize(
    ("outcome", "expected_state", "expected_error"),
    [
        ("success", "succeeded", None),
        ("returned-error", "failed", "task failed"),
        ("raised-error", "failed", RuntimeError),
        ("replacement", "cancelled", "ignored after replacement"),
    ],
)
def test_dashboard_session_maps_task_outcomes(
    tmp_path: Path,
    outcome: str,
    expected_state: str,
    expected_error: Any,
) -> None:
    claimed = _claimed_task(tmp_path)
    state = ScriptedState([claimed])
    shared = {"model": object()}
    received: list[tuple[ClaimedTask, dict[str, Any]]] = []

    def run_task(task: ClaimedTask, shared_kwargs: dict[str, Any]):
        received.append((task, shared_kwargs))
        if outcome == "returned-error":
            return "task failed"
        if outcome == "raised-error":
            raise RuntimeError("task exploded")
        if outcome == "replacement":
            state.task_replacement_requested = True
            return "ignored after replacement"
        return None

    controller = DashboardSessionController(
        state=state,  # type: ignore[arg-type]
        start_shared=lambda: ([], shared),
        run_task=run_task,
    )

    controller.run()

    assert state.ready_calls == 1
    assert received == [(claimed, shared)]
    assert state.completed[0]["state"] == expected_state
    if expected_error is RuntimeError:
        assert isinstance(state.completed[0]["error"], RuntimeError)
        assert str(state.completed[0]["error"]) == "task exploded"
    else:
        assert state.completed[0]["error"] == expected_error


def test_dashboard_session_stops_shared_daemons_in_reverse_after_cleanup_error(
    caplog: pytest.LogCaptureFixture,
) -> None:
    stopped: list[str] = []
    daemons = [
        FakeDaemon("first", stopped),
        FakeDaemon("second", stopped, stop_error=RuntimeError("stop failed")),
        FakeDaemon("third", stopped),
    ]
    state = ScriptedState()
    controller = DashboardSessionController(
        state=state,  # type: ignore[arg-type]
        start_shared=lambda: (daemons, {}),
        run_task=lambda claimed, shared: None,
    )

    controller.run()

    assert stopped == ["third", "second", "first"]
    assert "shared runtime cleanup failed: stop failed" in caplog.text


@pytest.mark.parametrize("merge_fails", [False, True])
def test_dashboard_exploration_finalizes_memory_and_reports_merge_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    merge_fails: bool,
) -> None:
    from rpent.cli import dashboard as dashboard_cli

    merge_calls: list[dict[str, Any]] = []

    class FakeMemoryManager:
        def merge_memory(self, **kwargs: Any) -> dict[str, int]:
            merge_calls.append(kwargs)
            if merge_fails:
                raise RuntimeError("merge exploded")
            return {"suite": 1}

    class FakeToolkit:
        memory = FakeMemoryManager()

        def solved(self) -> bool:
            return True

        def write_recipe(self, recipe_tag: str) -> str:
            return str(tmp_path / f"{recipe_tag}_recipe.jsonl")

        def close(self) -> None:
            pass

    class FakeState:
        task_replacement_requested = False

        def __init__(self) -> None:
            self.warnings: list[str] = []

        def emit(self, event: Any) -> None:
            del event

        def begin_planner_session(self, **kwargs: Any) -> None:
            del kwargs

        def report_task_warning(self, warning: str) -> None:
            self.warnings.append(warning)

    class FakePlanner:
        def solve(self, **kwargs: Any) -> PlannerResult:
            del kwargs
            return PlannerResult(
                finish_result={"status": "success"},
                messages=[],
                stats={},
            )

    output_dir = tmp_path / "task"
    run_config = RunConfig(
        recipe_tag="libero_s0",
        output_dir=output_dir,
        prompt_vars={},
        task_desc={"robot": "libero"},
    )
    robot_spec = SimpleNamespace(
        parse_config=lambda args: run_config,
        init_runtime=lambda *args: ([], {}),
        prompts=PromptBundle(
            system=lambda variables: "system",
            user=lambda variables: "user",
        ),
    )
    args = SimpleNamespace(
        verbose=False,
        robot_name="libero",
        explore=True,
        auto_merge_memory=True,
        explore_sessions=1,
        explore_attempts_per_session=2,
        planner="api",
        base_url=None,
        model="offline",
        max_tokens=128,
        planner_timeout_s=10,
        reasoning_effort=None,
        claude_code_max_budget_usd=None,
        no_images=False,
        max_turns=2,
    )
    claimed = ClaimedTask(number=1, request={}, output_dir=output_dir)
    state = FakeState()
    monkeypatch.setattr(
        dashboard_cli, "get_toolkit", lambda *args, **kwargs: FakeToolkit()
    )
    monkeypatch.setattr(
        dashboard_cli, "build_planner", lambda *args, **kwargs: FakePlanner()
    )

    error = dashboard_cli._run_dashboard_task(
        args=args,
        robot_spec=robot_spec,
        state=state,
        claimed=claimed,
        shared_primitives_kwargs={},
        unique_components=set(),
        session_root=tmp_path / "session",
    )

    assert error is None
    assert merge_calls == [
        {
            "cell_tag": "libero_s0",
            "run_state_dir": output_dir,
            "solved": True,
        }
    ]
    if merge_fails:
        assert len(state.warnings) == 1
        assert (
            "memory finalization failed: RuntimeError: merge exploded"
            in state.warnings[0]
        )
    else:
        assert state.warnings == []
