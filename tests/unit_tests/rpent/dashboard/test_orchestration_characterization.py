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

"""Characterize Dashboard planner-session orchestration before migration."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from rpent.dashboard.events import StepRecordEvent, UsageEvent
from rpent.dashboard.state import ClaimedTask, DashboardState
from rpent.planner.base import PlannerResult
from rpent.robots import PromptBundle, RunConfig
from rpent.session import EnvState

DASHBOARD_SPEC = {
    "task": {
        "command": "/rpent-task",
        "usage": "/rpent-task <mode> <seed>",
        "fields": (
            {"name": "mode", "suggestions": ("pick", "place")},
            {"name": "seed", "kind": "integer", "minimum": 0},
        ),
        "display": "{mode} / seed {seed}",
        "output_slug": "{mode}_s{seed}",
    },
    "runtime_components": (
        {"name": "model", "label": "MODEL", "scope": "shared"},
        {"name": "env", "label": "ENV", "scope": "unique"},
    ),
    "frame_channels": ({"name": "camera", "label": "camera"},),
}


class TrackingDashboardState(DashboardState):
    def __init__(self, *, output_dir: Path) -> None:
        super().__init__(
            run_id="characterization",
            output_dir=output_dir,
            dashboard_spec=DASHBOARD_SPEC,
        )
        self.begin_calls: list[Path | None] = []
        self.warnings: list[str] = []

    def begin_planner_session(self, *, video_path: Path | None = None) -> None:
        self.begin_calls.append(video_path)
        super().begin_planner_session(video_path=video_path)

    def report_task_warning(self, warning: str) -> None:
        self.warnings.append(warning)
        super().report_task_warning(warning)


class FakeDaemon:
    def __init__(self, name: str, stopped: list[str]) -> None:
        self.name = name
        self.stopped = stopped

    def stop(self) -> None:
        self.stopped.append(self.name)


class FakeMemoryManager:
    def __init__(self, scenario: DashboardScenario, session_number: int) -> None:
        self.scenario = scenario
        self.session_number = session_number

    def merge_memory(self, **kwargs: Any) -> dict[str, int]:
        self.scenario.merge_calls.append(
            {"session_number": self.session_number, **kwargs}
        )
        if self.scenario.merge_error is not None:
            raise self.scenario.merge_error
        return {"merged": 1}


class FakeToolkit:
    def __init__(
        self,
        scenario: DashboardScenario,
        session_number: int,
        kwargs: dict[str, Any],
    ) -> None:
        self.scenario = scenario
        self.session_number = session_number
        self.kwargs = kwargs
        self.memory = FakeMemoryManager(scenario, session_number)
        self.env_state = EnvState(kwargs["state_output_dir"])
        self.close_count = 0
        self.solved_calls = 0
        self.recipe_calls: list[str] = []

    def solved(self) -> bool:
        self.solved_calls += 1
        return self.session_number in self.scenario.solved_sessions

    def write_recipe(self, recipe_tag: str) -> str:
        self.recipe_calls.append(recipe_tag)
        if self.scenario.recipe_error is not None:
            raise self.scenario.recipe_error
        path = self.scenario.output_dir / f"{recipe_tag}_recipe.jsonl"
        self.scenario.recipe_paths.append(str(path))
        return str(path)

    def close(self) -> None:
        self.close_count += 1
        self.scenario.events.append(f"close:{self.session_number}")


class FakePlanner:
    def __init__(self, scenario: DashboardScenario, session_number: int) -> None:
        self.scenario = scenario
        self.session_number = session_number

    def solve(self, **kwargs: Any) -> PlannerResult:
        self.scenario.events.append(f"solve:{self.session_number}")
        self.scenario.planner_calls.append(kwargs)
        assert kwargs["dashboard_interaction"] is self.scenario.state
        toolkit = kwargs["toolkit"]

        with toolkit.env_state.record_step(
            state={"session": self.session_number},
            command={"action": f"act_{self.session_number}"},
            result={"session": self.session_number},
        ):
            toolkit.env_state.save(
                "frame.bin",
                f"frame-{self.session_number}".encode(),
            )
        record = toolkit.env_state.latest_record()
        assert record is not None
        self.scenario.state.emit(
            StepRecordEvent(record, toolkit.env_state, {"camera": "frame.bin"})
        )
        self.scenario.state.emit(
            UsageEvent(
                inp=self.session_number,
                out=self.session_number * 2,
                tool_calls=self.session_number * 3,
            )
        )

        if self.scenario.cancel_after_session == self.session_number:
            self.scenario.state.request_task({"mode": "place", "seed": 2})

        outcome = self.scenario.outcomes[self.session_number - 1]
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


class DashboardScenario:
    def __init__(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        *,
        outcomes: list[PlannerResult | BaseException],
        explore: bool = True,
        solved_sessions: set[int] | None = None,
        auto_merge_memory: bool = False,
        planner: str = "api",
    ) -> None:
        from rpent.cli import dashboard as dashboard_cli

        self.dashboard_cli = dashboard_cli
        self.root = tmp_path
        self.output_dir = tmp_path / "session" / "tasks" / "0001_pick_s1"
        self.outcomes = outcomes
        self.solved_sessions = solved_sessions or set()
        self.cancel_after_session: int | None = None
        self.cancel_during_runtime = False
        self.recipe_error: BaseException | None = None
        self.merge_error: BaseException | None = None
        self.events: list[str] = []
        self.toolkits: list[FakeToolkit] = []
        self.planners: list[FakePlanner] = []
        self.planner_builds: list[dict[str, Any]] = []
        self.planner_calls: list[dict[str, Any]] = []
        self.merge_calls: list[dict[str, Any]] = []
        self.recipe_paths: list[str] = []
        self.runtime_calls: list[tuple[set[str], Path]] = []
        self.finalize_calls: list[Any] = []
        self.stopped: list[str] = []
        self.shared_env = object()
        self.shared_vla = object()
        self.unique_sam = object()

        self.args = SimpleNamespace(
            verbose=False,
            robot_name="robotwin",
            explore=explore,
            auto_merge_memory=auto_merge_memory,
            explore_sessions=len(outcomes),
            explore_attempts_per_session=7,
            planner=planner,
            base_url="http://planner.invalid",
            model="offline",
            max_tokens=128,
            planner_timeout_s=37,
            reasoning_effort="high",
            claude_code_max_budget_usd=4.5,
            no_images=True,
            max_turns=9,
        )
        self.run_config = RunConfig(
            recipe_tag="robotwin_pick_s1",
            output_dir=self.output_dir,
            prompt_vars={"mode": "pick", "seed": 1},
            task_desc={"robot": "robotwin", "mode": "pick", "seed": 1},
        )

        def init_runtime(
            task_args: Any,
            output_dir: Path,
            state: DashboardState,
            components: set[str],
        ) -> tuple[list[FakeDaemon], dict[str, Any]]:
            del task_args
            self.runtime_calls.append((components, output_dir))
            if self.cancel_during_runtime:
                state.request_task({"mode": "place", "seed": 2})
            return (
                [
                    FakeDaemon("unique-first", self.stopped),
                    FakeDaemon("unique-last", self.stopped),
                ],
                {"sam": self.unique_sam},
            )

        self.robot_spec = SimpleNamespace(
            supports_exploration=True,
            parse_config=lambda task_args: RunConfig(
                recipe_tag=self.run_config.recipe_tag,
                output_dir=Path(task_args.output_dir),
                prompt_vars=dict(self.run_config.prompt_vars),
                task_desc=dict(self.run_config.task_desc),
            ),
            init_runtime=init_runtime,
            prompts=PromptBundle(
                system=lambda variables: (
                    f"system-{variables['session_number']}/{variables['session_max']}"
                ),
                user=lambda variables: f"user-{variables['mode']}-{variables['seed']}",
            ),
            finalize_run=lambda context: self.finalize_calls.append(context),
        )

        self.state = TrackingDashboardState(output_dir=tmp_path / "session")
        self.state.shared_services_ready()
        self.state.submit_input("/rpent-task pick 1")
        claimed = self.state.wait_for_task(timeout=0)
        assert claimed is not None
        self.claimed: ClaimedTask = claimed
        assert self.claimed.output_dir == self.output_dir

        def init_output_dir(path: str | Path, *, verbose: bool) -> Path:
            del verbose
            result = Path(path)
            result.mkdir(parents=True, exist_ok=True)
            return result

        def make_toolkit(robot_name: str, **kwargs: Any) -> FakeToolkit:
            del robot_name
            session_number = len(self.toolkits) + 1
            self.events.append(f"toolkit:{session_number}")
            toolkit = FakeToolkit(self, session_number, kwargs)
            self.toolkits.append(toolkit)
            return toolkit

        def make_planner(planner_type: str, **kwargs: Any) -> FakePlanner:
            session_number = len(self.planners) + 1
            self.events.append(f"planner:{session_number}")
            self.planner_builds.append({"planner_type": planner_type, **kwargs})
            planner_instance = FakePlanner(self, session_number)
            self.planners.append(planner_instance)
            return planner_instance

        monkeypatch.setattr(dashboard_cli, "init_output_dir", init_output_dir)
        monkeypatch.setattr(dashboard_cli, "get_toolkit", make_toolkit)
        monkeypatch.setattr(dashboard_cli, "build_planner", make_planner)

    def run(self) -> str | None:
        return self.dashboard_cli._run_dashboard_task(
            args=self.args,
            robot_spec=self.robot_spec,
            state=self.state,
            claimed=self.claimed,
            shared_primitives_kwargs={
                "env": self.shared_env,
                "vla": self.shared_vla,
            },
            unique_components={"sam"},
            session_root=self.root / "session",
        )

    def transcript(self) -> dict[str, Any]:
        path = self.output_dir / f"transcript_{self.run_config.recipe_tag}.json"
        return json.loads(path.read_text())


def _result(session: int, *, error: str | None = None) -> PlannerResult:
    return PlannerResult(
        finish_result={"session": session},
        messages=[{"role": "assistant", "content": f"message-{session}"}],
        stats={
            "total_input_tokens": session,
            "total_output_tokens": session * 2,
            "tool_calls": session * 3,
            "session": session,
        },
        error=error,
    )


def test_dashboard_one_session_preserves_invocation_and_native_success_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenario = DashboardScenario(
        tmp_path,
        monkeypatch,
        outcomes=[_result(1)],
        explore=False,
    )

    error = scenario.run()

    assert error is None
    assert scenario.state.begin_calls == []
    assert scenario.events == ["toolkit:1", "planner:1", "solve:1", "close:1"]
    assert scenario.toolkits[0].solved_calls == 0
    assert scenario.toolkits[0].recipe_calls == []
    assert scenario.planner_calls[0]["dashboard_interaction"] is scenario.state
    assert scenario.planner_calls[0]["max_turns"] == 9
    assert scenario.planner_builds[0]["planner_timeout_s"] == 37
    assert scenario.planner_builds[0]["dashboard_events"] is scenario.state
    assert scenario.finalize_calls == []
    assert scenario.stopped == ["unique-last", "unique-first"]


@pytest.mark.parametrize("solved_session", [1, 2])
def test_dashboard_solved_session_preserves_fresh_ownership_and_accumulation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    solved_session: int,
) -> None:
    scenario = DashboardScenario(
        tmp_path,
        monkeypatch,
        outcomes=[_result(1), _result(2), _result(3)],
        solved_sessions={solved_session},
        auto_merge_memory=True,
    )

    error = scenario.run()

    assert error is None
    assert len(scenario.toolkits) == solved_session
    assert len(scenario.planners) == solved_session
    assert [toolkit.close_count for toolkit in scenario.toolkits] == [
        1
    ] * solved_session
    assert len({id(toolkit) for toolkit in scenario.toolkits}) == solved_session
    assert len({id(toolkit.memory) for toolkit in scenario.toolkits}) == solved_session
    for toolkit in scenario.toolkits:
        primitives = toolkit.kwargs["primitives_kwargs"]
        assert primitives["env"] is scenario.shared_env
        assert primitives["vla"] is scenario.shared_vla
        assert primitives["sam"] is scenario.unique_sam
        assert toolkit.kwargs["attempts_per_session"] == 7
        assert toolkit.kwargs["mode"] == "exploration"
    assert scenario.events == [
        item
        for session in range(1, solved_session + 1)
        for item in (
            f"toolkit:{session}",
            f"planner:{session}",
            f"solve:{session}",
            f"close:{session}",
        )
    ]
    assert scenario.state.begin_calls == [
        scenario.output_dir / "sessions" / f"session_{session:03d}" / "episode.mp4"
        for session in range(1, solved_session + 1)
    ]
    assert [call["system_prompt"] for call in scenario.planner_calls] == [
        f"system-{session}/3\n" for session in range(1, solved_session + 1)
    ]
    if solved_session == 2:
        assert scenario.planner_calls[1]["user_message"].startswith(
            "You are agent 2 of up to 3 on this cell."
        )
    assert scenario.merge_calls == [
        {
            "session_number": solved_session,
            "cell_tag": "robotwin_pick_s1",
            "run_state_dir": scenario.output_dir,
            "solved": True,
        }
    ]
    assert scenario.recipe_paths == [
        str(scenario.output_dir / "robotwin_pick_s1_recipe.jsonl")
    ]
    transcript = scenario.transcript()
    assert transcript["finish"] == {"session": solved_session}
    assert transcript["stats"]["session"] == solved_session
    assert transcript["messages"] == [
        {"role": "assistant", "content": f"message-{session}"}
        for session in range(1, solved_session + 1)
    ]
    expected_total = sum(range(1, solved_session + 1))
    assert scenario.state.run_detail()["usage"] == {
        "in": expected_total,
        "out": expected_total * 2,
        "tool_calls": expected_total * 3,
    }
    assert [item["step"] for item in scenario.state.run_detail()["timeline"]] == list(
        range(solved_session)
    )
    assert scenario.finalize_calls == []


def test_dashboard_clean_unsolved_exhaustion_is_successful_taskrun(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenario = DashboardScenario(
        tmp_path,
        monkeypatch,
        outcomes=[_result(1), _result(2), _result(3)],
        auto_merge_memory=True,
    )

    error = scenario.run()

    assert error is None
    assert len(scenario.toolkits) == 3
    assert scenario.merge_calls == [
        {
            "session_number": 3,
            "cell_tag": "robotwin_pick_s1",
            "run_state_dir": scenario.output_dir,
            "solved": False,
        }
    ]
    scenario.state.complete_task(
        state="succeeded" if not error else "failed", error=error
    )
    assert scenario.state.snapshot()["state"] == "succeeded"
    assert scenario.transcript()["stats"]["session"] == 3


def test_dashboard_intermediate_timeout_continues_and_success_clears_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    scenario = DashboardScenario(
        tmp_path,
        monkeypatch,
        outcomes=[
            _result(1, error="Codex SDK timed out after 37s"),
            _result(2),
        ],
        solved_sessions={2},
    )

    error = scenario.run()

    assert error is None
    assert len(scenario.planners) == 2
    assert "session 1/2 timed out; continuing with a fresh handoff" in caplog.text
    assert scenario.transcript()["messages"] == [
        {"role": "assistant", "content": "message-1"},
        {"role": "assistant", "content": "message-2"},
    ]


@pytest.mark.parametrize(
    ("outcome", "expected_error"),
    [
        (
            _result(1, error="API planner timed out after 37s"),
            "API planner timed out after 37s",
        ),
        (
            _result(1, error="planner authentication failed"),
            "planner authentication failed",
        ),
        (RuntimeError("planner exploded"), "planner exploded"),
    ],
)
def test_dashboard_final_timeout_and_planner_failures_preserve_error_and_close(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    outcome: PlannerResult | BaseException,
    expected_error: str,
) -> None:
    scenario = DashboardScenario(tmp_path, monkeypatch, outcomes=[outcome])

    error = scenario.run()

    assert error == expected_error
    assert scenario.toolkits[0].close_count == 1
    assert scenario.merge_calls == []


@pytest.mark.parametrize("cancel_at", ["before", "between"])
def test_dashboard_replacement_cancellation_stops_at_boundaries_and_suppresses_merge(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cancel_at: str,
) -> None:
    scenario = DashboardScenario(
        tmp_path,
        monkeypatch,
        outcomes=[_result(1), _result(2)],
        auto_merge_memory=True,
    )
    if cancel_at == "before":
        scenario.cancel_during_runtime = True
    else:
        scenario.cancel_after_session = 1

    error = scenario.run()

    assert error is None
    assert len(scenario.toolkits) == (0 if cancel_at == "before" else 1)
    assert len(scenario.planners) == (0 if cancel_at == "before" else 1)
    assert scenario.merge_calls == []
    assert scenario.state.task_replacement_requested is True
    assert scenario.stopped == ["unique-last", "unique-first"]


def test_dashboard_recipe_failure_closes_once_and_skips_merge(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenario = DashboardScenario(
        tmp_path,
        monkeypatch,
        outcomes=[_result(1)],
        solved_sessions={1},
        auto_merge_memory=True,
    )
    scenario.recipe_error = OSError("recipe write failed")

    error = scenario.run()

    assert error == "recipe write failed"
    assert scenario.toolkits[0].close_count == 1
    assert scenario.merge_calls == []


def test_dashboard_merge_failure_remains_warning_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenario = DashboardScenario(
        tmp_path,
        monkeypatch,
        outcomes=[_result(1)],
        auto_merge_memory=True,
    )
    scenario.merge_error = RuntimeError("merge exploded")

    error = scenario.run()

    assert error is None
    assert scenario.state.warnings == [
        "Task succeeded, but memory finalization failed: RuntimeError: merge exploded"
    ]


@pytest.mark.parametrize("planner", ["api", "claude_code", "codex"])
def test_dashboard_forwards_timeout_bound_to_every_planner_backend(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    planner: str,
) -> None:
    scenario = DashboardScenario(
        tmp_path,
        monkeypatch,
        outcomes=[_result(1)],
        planner=planner,
    )

    assert scenario.run() is None

    assert scenario.planner_builds == [
        {
            "planner_type": planner,
            "output_dir": scenario.output_dir,
            "recipe_tag": "robotwin_pick_s1",
            "robot_name": "robotwin",
            "base_url": "http://planner.invalid",
            "model": "offline",
            "max_tokens": 128,
            "planner_timeout_s": 37,
            "reasoning_effort": "high",
            "claude_code_max_budget_usd": 4.5,
            "dashboard_events": scenario.state,
            "no_images": True,
        }
    ]


def test_dashboard_transcript_failure_remains_warning_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    scenario = DashboardScenario(tmp_path, monkeypatch, outcomes=[_result(1)])

    def fail_open(*args: Any, **kwargs: Any) -> None:
        del args, kwargs
        raise OSError("transcript unavailable")

    monkeypatch.setattr(scenario.dashboard_cli, "open", fail_open, raising=False)

    assert scenario.run() is None

    assert "failed to write TaskRun transcript" in caplog.text
    assert "transcript unavailable" in caplog.text
