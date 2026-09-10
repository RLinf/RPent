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

"""Characterize the terminal planner-session loop before its extraction."""

from __future__ import annotations

import json
import queue
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from rpent.evaluation import RunFinalizationContext
from rpent.planner.base import PlannerResult
from rpent.robots import PromptBundle, RobotSpec, RunConfig


@dataclass
class _Scenario:
    events: list[str] = field(default_factory=list)
    toolkits: list[Any] = field(default_factory=list)
    planner_calls: list[dict[str, Any]] = field(default_factory=list)
    planner_builds: list[dict[str, Any]] = field(default_factory=list)
    finalizations: list[RunFinalizationContext] = field(default_factory=list)
    borrowed_env: object = field(default_factory=object)
    borrowed_vla: object = field(default_factory=object)
    daemon: Any = None
    input_queue: queue.Queue[str | None] | None = None


def _run_scenario(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    planner_results: list[PlannerResult | BaseException],
    solved_sessions: set[int] | None = None,
    explore: bool = True,
    supports_exploration: bool = True,
    finalize_run: bool = False,
    recipe_result: str | BaseException | None = "recipe.jsonl",
    toolkit_failure_session: int | None = None,
    interactive: bool = False,
) -> tuple[int, _Scenario, dict[str, Any]]:
    from rpent.cli import main as cli

    scenario = _Scenario()
    solved_sessions = solved_sessions or set()

    class FakeDaemon:
        stop_count = 0

        def stop(self) -> None:
            self.stop_count += 1
            scenario.events.append("daemon:stop")

    class FakeMemoryManager:
        def merge_memory(self, **kwargs: Any) -> None:
            scenario.events.append(f"memory:merge:{kwargs!r}")

    class FakeToolkit:
        def __init__(self, session_number: int) -> None:
            self.session_number = session_number
            self.memory = FakeMemoryManager()
            self.close_count = 0
            self.solved_calls = 0

        def solved(self) -> bool:
            self.solved_calls += 1
            scenario.events.append(f"toolkit:{self.session_number}:solved")
            return self.session_number in solved_sessions

        def write_recipe(self, recipe_tag: str) -> str | None:
            scenario.events.append(f"toolkit:{self.session_number}:recipe:{recipe_tag}")
            if isinstance(recipe_result, BaseException):
                raise recipe_result
            return recipe_result

        def close(self) -> None:
            self.close_count += 1
            scenario.events.append(f"toolkit:{self.session_number}:close")

    class FakePlanner:
        def __init__(self, planner_number: int) -> None:
            self.planner_number = planner_number

        def solve(self, **kwargs: Any) -> PlannerResult:
            session_number = len(scenario.planner_calls) + 1
            scenario.events.append(f"planner:{self.planner_number}:solve")
            scenario.planner_calls.append(kwargs)
            result = planner_results[session_number - 1]
            if isinstance(result, BaseException):
                raise result
            return result

    def add_cli_args(parser: Any, use_dashboard: bool) -> None:
        del use_dashboard
        parser.add_argument("--explore-sessions", type=int, default=3)
        parser.add_argument("--explore-attempts-per-session", type=int, default=2)
        parser.add_argument("--auto-merge-memory", action="store_true")

    def parse_config(args: Any) -> RunConfig:
        return RunConfig(
            recipe_tag="cell_s7",
            output_dir=Path(args.output_dir),
            prompt_vars={"task": "offline task"},
            task_desc={"task": "offline task", "seed": 7},
        )

    scenario.daemon = FakeDaemon()

    def init_runtime(*args: Any) -> tuple[list[FakeDaemon], dict[str, object]]:
        del args
        scenario.events.append("runtime:init")
        return [scenario.daemon], {
            "env": scenario.borrowed_env,
            "vla": scenario.borrowed_vla,
        }

    def finalize(context: RunFinalizationContext) -> None:
        scenario.events.append("robot:finalize")
        scenario.finalizations.append(context)

    robot_spec = RobotSpec(
        name="testrobot",
        prompts=PromptBundle(
            system=lambda variables: (
                f"system session={variables.get('session_number', 'initial')}"
            ),
            user=lambda variables: str(variables["task"]),
        ),
        add_cli_args=add_cli_args,
        parse_config=parse_config,
        init_runtime=init_runtime,
        finalize_run=finalize if finalize_run else None,
        supports_exploration=supports_exploration,
    )

    def build_planner(*args: Any, **kwargs: Any) -> FakePlanner:
        planner_number = len(scenario.planner_builds) + 1
        scenario.events.append(f"planner:{planner_number}:build")
        scenario.planner_builds.append({"args": args, "kwargs": kwargs})
        return FakePlanner(planner_number)

    def get_toolkit(*args: Any, **kwargs: Any) -> FakeToolkit:
        del args
        session_number = len(scenario.toolkits) + 1
        scenario.events.append(f"toolkit:{session_number}:construct")
        assert kwargs["primitives_kwargs"]["env"] is scenario.borrowed_env
        assert kwargs["primitives_kwargs"]["vla"] is scenario.borrowed_vla
        if session_number == toolkit_failure_session:
            raise RuntimeError("toolkit construction exploded")
        toolkit = FakeToolkit(session_number)
        toolkit.creation_kwargs = kwargs
        scenario.toolkits.append(toolkit)
        return toolkit

    monkeypatch.setattr(cli, "enumerate_robots", lambda: ("testrobot",))
    monkeypatch.setattr(cli, "get_robot_spec", lambda name: robot_spec)
    monkeypatch.setattr(cli, "build_planner", build_planner)
    monkeypatch.setattr(cli, "get_toolkit", get_toolkit)

    if interactive:

        def start_reader(
            supplied_queue: queue.Queue[str | None], **kwargs: Any
        ) -> None:
            assert kwargs["first_prompt_default"] == "offline task\n"
            scenario.input_queue = supplied_queue

        monkeypatch.setattr(cli, "start_interactive_reader", start_reader)

        def start_resolver(supplied_queue: queue.Queue[str | None]) -> Any:
            assert supplied_queue is scenario.input_queue
            return lambda: "interactive task"

        monkeypatch.setattr(
            cli,
            "start_first_prompt_resolver",
            start_resolver,
        )

    argv = [
        "rpent",
        "--robot",
        "testrobot",
        "--planner",
        "codex",
        "--model",
        "test-model",
        "--output-dir",
        str(tmp_path),
        "--memory-profile",
        "local",
        "--max-turns",
        "9",
        "--planner-timeout-s",
        "41",
    ]
    if explore:
        argv.extend(["--explore", "--explore-sessions", str(len(planner_results))])
    if interactive:
        argv.append("--interactive")
    monkeypatch.setattr(sys, "argv", argv)

    return_code = cli.main()
    transcript = json.loads((tmp_path / "transcript_cell_s7.json").read_text())
    return return_code, scenario, transcript


def _planner_result(
    session_number: int,
    *,
    error: str | None = None,
) -> PlannerResult:
    return PlannerResult(
        finish_result={"status": f"session-{session_number}"},
        messages=[{"role": "assistant", "content": f"message-{session_number}"}],
        stats={"session": session_number, "tool_calls": session_number},
        error=error,
    )


def test_non_exploration_forces_one_session_and_root_state_dir(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    return_code, scenario, transcript = _run_scenario(
        monkeypatch,
        tmp_path,
        planner_results=[_planner_result(1), _planner_result(2), _planner_result(3)],
        explore=False,
        supports_exploration=True,
    )

    assert return_code == 0
    assert len(scenario.planner_calls) == 1
    assert len(scenario.toolkits) == 1
    assert scenario.toolkits[0].creation_kwargs["mode"] == "evaluation"
    assert scenario.toolkits[0].creation_kwargs["state_output_dir"] == tmp_path
    assert scenario.toolkits[0].close_count == 1
    assert transcript["messages"] == [{"role": "assistant", "content": "message-1"}]


@pytest.mark.parametrize(
    ("results", "solved_sessions", "expected_sessions", "return_code"),
    [
        ([_planner_result(1)], {1}, 1, 0),
        ([_planner_result(1), _planner_result(2)], {2}, 2, 0),
        ([_planner_result(1), _planner_result(2), _planner_result(3)], set(), 3, 0),
        (
            [
                _planner_result(1, error="Codex SDK timed out after 41s"),
                _planner_result(2),
            ],
            {2},
            2,
            0,
        ),
        ([_planner_result(1, error="API planner timed out after 41s")], set(), 1, 1),
        ([_planner_result(1, error="planner authentication failed")], set(), 1, 1),
    ],
)
def test_exploration_stop_and_timeout_continuation_contracts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    results: list[PlannerResult],
    solved_sessions: set[int],
    expected_sessions: int,
    return_code: int,
) -> None:
    actual_code, scenario, _ = _run_scenario(
        monkeypatch,
        tmp_path,
        planner_results=results,
        solved_sessions=solved_sessions,
    )

    assert actual_code == return_code
    assert len(scenario.planner_calls) == expected_sessions
    assert [toolkit.close_count for toolkit in scenario.toolkits] == [
        1
    ] * expected_sessions
    assert [
        toolkit.creation_kwargs["state_output_dir"] for toolkit in scenario.toolkits
    ] == [
        tmp_path / "sessions" / f"session_{number:03d}"
        for number in range(1, expected_sessions + 1)
    ]
    assert bool([event for event in scenario.events if ":recipe:" in event]) is bool(
        solved_sessions
    )


def test_continuation_order_arguments_handoff_and_last_session_statistics(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    attempts = tmp_path / "attempts"
    attempts.mkdir()
    (attempts / "attempt_02_failed.json").write_text("{}")
    results = [
        _planner_result(1, error="first planner timed out"),
        _planner_result(2, error="second planner timed out"),
        _planner_result(3),
    ]

    return_code, scenario, transcript = _run_scenario(
        monkeypatch,
        tmp_path,
        planner_results=results,
        solved_sessions={3},
    )

    assert return_code == 0
    assert scenario.events.index("planner:1:build") < scenario.events.index(
        "runtime:init"
    )
    for session_number in (2, 3):
        assert scenario.events.index(
            f"planner:{session_number}:build"
        ) < scenario.events.index(f"toolkit:{session_number}:construct")
    assert all(build["args"] == ("codex",) for build in scenario.planner_builds)
    assert all(
        build["kwargs"]["planner_timeout_s"] == 41
        and build["kwargs"]["reasoning_effort"] == "none"
        and build["kwargs"]["model"] == "test-model"
        for build in scenario.planner_builds
    )
    assert "agent 2 of up to 3" in scenario.planner_calls[1]["user_message"]
    assert "attempt_02_failed.json" in scenario.planner_calls[1]["user_message"]
    assert scenario.planner_calls[0]["system_prompt"] == "system session=initial\n"
    assert scenario.planner_calls[1]["system_prompt"] == "system session=2\n"
    assert transcript["messages"] == [
        {"role": "assistant", "content": f"message-{number}"} for number in (1, 2, 3)
    ]
    assert transcript["stats"] == {"session": 3, "tool_calls": 3}
    assert transcript["finish"] == {"status": "session-3"}


def test_recipe_export_success_preserves_recipe_reporting(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    recipe_path = str(tmp_path / "cell_s7_recipe.jsonl")

    return_code, scenario, _ = _run_scenario(
        monkeypatch,
        tmp_path,
        planner_results=[_planner_result(1)],
        solved_sessions={1},
        recipe_result=recipe_path,
    )

    assert return_code == 0
    assert f"recipe: {recipe_path}" in caplog.text
    assert "toolkit:1:recipe:cell_s7" in scenario.events
    assert scenario.toolkits[0].close_count == 1


def test_recipe_export_failure_closes_toolkit_and_preserves_exception_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    return_code, scenario, transcript = _run_scenario(
        monkeypatch,
        tmp_path,
        planner_results=[_planner_result(1)],
        solved_sessions={1},
        recipe_result=OSError("recipe disk full"),
        finalize_run=True,
    )

    assert return_code == 1
    assert scenario.toolkits[0].close_count == 1
    assert scenario.toolkits[0].solved_calls == 2
    assert scenario.finalizations[0].agent_error == "OSError: recipe disk full"
    assert transcript["messages"] == [{"role": "assistant", "content": "message-1"}]
    assert "EXCEPTION in agent loop: OSError: recipe disk full" in caplog.text
    assert "recipe: not written (cell unsolved)" in caplog.text


def test_toolkit_construction_failure_preserves_error_return_and_cleanup(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    return_code, scenario, transcript = _run_scenario(
        monkeypatch,
        tmp_path,
        planner_results=[_planner_result(1)],
        toolkit_failure_session=1,
        finalize_run=True,
    )

    assert return_code == 1
    assert scenario.toolkits == []
    assert scenario.daemon.stop_count == 1
    assert transcript["finish"] is None
    assert transcript["stats"] == {}
    assert transcript["messages"] == []
    assert scenario.finalizations[0].agent_error == (
        "RuntimeError: toolkit construction exploded"
    )
    assert "EXCEPTION in agent loop: RuntimeError: toolkit construction exploded" in (
        caplog.text
    )


def test_planner_invocation_failure_closes_once_and_preserves_error_return(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    return_code, scenario, transcript = _run_scenario(
        monkeypatch,
        tmp_path,
        planner_results=[ValueError("planner invocation exploded")],
        finalize_run=True,
    )

    assert return_code == 1
    assert scenario.toolkits[0].close_count == 1
    assert scenario.toolkits[0].solved_calls == 1
    assert scenario.daemon.stop_count == 1
    assert transcript["finish"] is None
    assert transcript["stats"] == {}
    assert transcript["messages"] == []
    assert scenario.finalizations[0].agent_error == (
        "ValueError: planner invocation exploded"
    )
    assert "EXCEPTION in agent loop: ValueError: planner invocation exploded" in (
        caplog.text
    )


def test_interactive_input_queue_and_opening_prompt_are_forwarded_unchanged(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    return_code, scenario, _ = _run_scenario(
        monkeypatch,
        tmp_path,
        planner_results=[_planner_result(1)],
        explore=False,
        interactive=True,
    )

    assert return_code == 0
    assert scenario.planner_calls[0]["user_message"] == "interactive task"
    assert scenario.planner_calls[0]["input_queue"] is scenario.input_queue
    assert "dashboard_interaction" not in scenario.planner_calls[0]
