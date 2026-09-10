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

"""Focused tests for the synchronous planner-session service."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from rpent.orchestration import (
    PlannerSessionContext,
    PlannerSessionRequest,
    PlannerSessionStopReason,
    PreparedPlannerSession,
    SynchronousPlannerSessionService,
    effective_session_count,
)
from rpent.planner.base import PlannerResult


@dataclass
class _Harness:
    results: list[PlannerResult | BaseException]
    solved_sessions: set[int] = field(default_factory=set)
    recipe_result: str | BaseException | None = "recipe.jsonl"
    toolkit_failure_session: int | None = None
    events: list[str] = field(default_factory=list)
    contexts: list[PlannerSessionContext] = field(default_factory=list)
    toolkits: list[Any] = field(default_factory=list)
    timeouts: list[tuple[int, str]] = field(default_factory=list)
    borrowed_env: object = field(default_factory=object)
    borrowed_vla: object = field(default_factory=object)
    cancelled: bool = False

    def prepare(
        self,
        context: PlannerSessionContext,
        first_user_message: str,
    ) -> PreparedPlannerSession:
        self.events.append(f"prepare:{context.session_number}")
        self.contexts.append(context)
        message = (
            first_user_message
            if context.session_number == 1
            else f"handoff-{context.session_number}"
        )
        return PreparedPlannerSession(
            context=context,
            system_prompt=f"system-{context.session_number}",
            user_message=message,
        )

    def create_toolkit(self, context: PlannerSessionContext) -> Any:
        self.events.append(f"construct:{context.session_number}")
        if context.session_number == self.toolkit_failure_session:
            raise RuntimeError("toolkit factory failed")

        harness = self

        class FakeToolkit:
            memory = SimpleNamespace(name=f"memory-{context.session_number}")
            env = harness.borrowed_env
            vla = harness.borrowed_vla
            close_count = 0
            solved_calls = 0

            def close(self) -> None:
                self.close_count += 1
                harness.events.append(f"close:{context.session_number}")

        toolkit = FakeToolkit()
        self.toolkits.append(toolkit)
        return toolkit

    def invoke(self, prepared: PreparedPlannerSession, toolkit: Any) -> PlannerResult:
        number = prepared.context.session_number
        self.events.append(f"invoke:{number}")
        assert toolkit is self.toolkits[-1]
        result = self.results[number - 1]
        if isinstance(result, BaseException):
            raise result
        return result

    def probe_solved(self, toolkit: Any) -> bool:
        toolkit.solved_calls += 1
        number = len(self.toolkits)
        self.events.append(f"solved:{number}")
        return number in self.solved_sessions

    def export_recipe(self, toolkit: Any) -> str | None:
        number = len(self.toolkits)
        self.events.append(f"recipe:{number}")
        if isinstance(self.recipe_result, BaseException):
            raise self.recipe_result
        return self.recipe_result

    def on_timeout(self, context: PlannerSessionContext, error: str) -> None:
        self.timeouts.append((context.session_number, error))

    def cancellation_requested(self) -> bool:
        return self.cancelled

    def service(
        self,
        *,
        probe_solved: bool = True,
        probe_environment_success: bool = False,
    ) -> SynchronousPlannerSessionService:
        return SynchronousPlannerSessionService(
            prepare_session=self.prepare,
            create_toolkit=self.create_toolkit,
            invoke_planner=self.invoke,
            probe_solved=self.probe_solved if probe_solved else None,
            export_recipe=self.export_recipe if probe_solved else None,
            probe_environment_success=(
                self.probe_solved if probe_environment_success else None
            ),
            cancellation_requested=self.cancellation_requested,
            on_intermediate_timeout=self.on_timeout,
        )


def _result(number: int, error: str | None = None) -> PlannerResult:
    return PlannerResult(
        finish_result={"session": number},
        messages=[{"role": "assistant", "content": f"message-{number}"}],
        stats={"session": number},
        error=error,
    )


@pytest.mark.parametrize(
    ("exploration", "requested", "expected"),
    [
        (False, 8, 1),
        (False, None, 1),
        (True, None, 1),
        (True, 0, 1),
        (True, -3, 1),
        (True, 4, 4),
    ],
)
def test_effective_session_count(
    exploration: bool,
    requested: int | None,
    expected: int,
) -> None:
    assert effective_session_count(exploration, requested) == expected


def test_solved_session_owns_fresh_toolkits_but_not_borrowed_clients(
    tmp_path: Path,
) -> None:
    harness = _Harness([_result(1), _result(2)], solved_sessions={2})

    result = harness.service().run(
        PlannerSessionRequest(
            output_dir=tmp_path,
            exploration=True,
            requested_session_count=3,
            first_user_message="task",
        )
    )

    assert result.stop_reason is PlannerSessionStopReason.SOLVED
    assert result.solved is True
    assert result.recipe_path == "recipe.jsonl"
    assert result.error is None
    assert len(result.sessions) == 2
    assert [context.state_output_dir for context in harness.contexts] == [
        tmp_path / "sessions" / "session_001",
        tmp_path / "sessions" / "session_002",
    ]
    assert harness.events == [
        "prepare:1",
        "construct:1",
        "invoke:1",
        "solved:1",
        "close:1",
        "prepare:2",
        "construct:2",
        "invoke:2",
        "solved:2",
        "recipe:2",
        "close:2",
    ]
    assert [toolkit.close_count for toolkit in harness.toolkits] == [1, 1]
    assert all(toolkit.env is harness.borrowed_env for toolkit in harness.toolkits)
    assert all(toolkit.vla is harness.borrowed_vla for toolkit in harness.toolkits)
    assert result.memory_manager is harness.toolkits[-1].memory


def test_clean_exhaustion_accumulates_messages_and_keeps_last_statistics(
    tmp_path: Path,
) -> None:
    harness = _Harness([_result(1), _result(2), _result(3)])

    result = harness.service().run(
        PlannerSessionRequest(
            output_dir=tmp_path,
            exploration=True,
            requested_session_count=3,
            first_user_message="task",
        )
    )

    assert result.stop_reason is PlannerSessionStopReason.EXHAUSTED
    assert result.solved is False
    assert result.finish_result == {"session": 3}
    assert result.stats == {"session": 3}
    assert result.messages == [
        {"role": "assistant", "content": f"message-{number}"} for number in (1, 2, 3)
    ]
    assert [record.stop_reason for record in result.sessions] == [
        None,
        None,
        PlannerSessionStopReason.EXHAUSTED,
    ]


def test_intermediate_timeout_continues_and_final_success_clears_error(
    tmp_path: Path,
) -> None:
    timeout = "Codex SDK timed out after 5s"
    harness = _Harness([_result(1, timeout), _result(2)], solved_sessions={2})

    result = harness.service().run(
        PlannerSessionRequest(
            output_dir=tmp_path,
            exploration=True,
            requested_session_count=2,
            first_user_message="task",
        )
    )

    assert result.stop_reason is PlannerSessionStopReason.SOLVED
    assert result.error is None
    assert harness.timeouts == [(1, timeout)]
    assert len(result.sessions) == 2
    assert result.sessions[0].stop_reason is None


@pytest.mark.parametrize(
    ("error", "reason"),
    [
        ("API planner timed out after 5s", PlannerSessionStopReason.FINAL_TIMEOUT),
        ("planner authentication failed", PlannerSessionStopReason.PLANNER_ERROR),
    ],
)
def test_final_planner_errors_have_typed_stop_reasons(
    tmp_path: Path,
    error: str,
    reason: PlannerSessionStopReason,
) -> None:
    harness = _Harness([_result(1, error)])

    result = harness.service().run(
        PlannerSessionRequest(
            output_dir=tmp_path,
            exploration=True,
            requested_session_count=1,
            first_user_message="task",
        )
    )

    assert result.stop_reason is reason
    assert result.error == error
    assert result.sessions[0].stop_reason is reason
    assert harness.toolkits[0].close_count == 1


def test_toolkit_factory_exception_has_no_owned_toolkit_to_close(
    tmp_path: Path,
) -> None:
    harness = _Harness([_result(1)], toolkit_failure_session=1)

    result = harness.service().run(
        PlannerSessionRequest(
            output_dir=tmp_path,
            exploration=True,
            requested_session_count=1,
            first_user_message="task",
        )
    )

    assert result.stop_reason is PlannerSessionStopReason.EXCEPTION
    assert result.error == "RuntimeError: toolkit factory failed"
    assert result.memory_manager is None
    assert harness.toolkits == []
    assert harness.events == ["prepare:1", "construct:1"]
    assert result.sessions[0].stop_reason is PlannerSessionStopReason.EXCEPTION


@pytest.mark.parametrize(
    ("planner_result", "recipe_result", "expected_error"),
    [
        (
            ValueError("planner exploded"),
            "recipe.jsonl",
            "ValueError: planner exploded",
        ),
        (_result(1), OSError("recipe failed"), "OSError: recipe failed"),
    ],
)
def test_invocation_and_recipe_exceptions_close_the_toolkit_once(
    tmp_path: Path,
    planner_result: PlannerResult | BaseException,
    recipe_result: str | BaseException,
    expected_error: str,
) -> None:
    harness = _Harness(
        [planner_result],
        solved_sessions={1},
        recipe_result=recipe_result,
    )

    result = harness.service().run(
        PlannerSessionRequest(
            output_dir=tmp_path,
            exploration=True,
            requested_session_count=1,
            first_user_message="task",
        )
    )

    assert result.stop_reason is PlannerSessionStopReason.EXCEPTION
    assert result.error == expected_error
    assert harness.toolkits[0].close_count == 1
    assert result.sessions[0].stop_reason is PlannerSessionStopReason.EXCEPTION


def test_environment_success_probe_runs_in_finally_and_overwrites_solved(
    tmp_path: Path,
) -> None:
    harness = _Harness([ValueError("planner exploded")], solved_sessions={1})

    result = harness.service(probe_environment_success=True).run(
        PlannerSessionRequest(
            output_dir=tmp_path,
            exploration=True,
            requested_session_count=1,
            first_user_message="task",
        )
    )

    assert result.stop_reason is PlannerSessionStopReason.EXCEPTION
    assert result.error == "ValueError: planner exploded"
    assert result.environment_success is True
    assert result.solved is True
    assert harness.toolkits[0].solved_calls == 1
    assert harness.toolkits[0].close_count == 1


def test_generic_cancellation_stops_before_constructing_a_toolkit(
    tmp_path: Path,
) -> None:
    harness = _Harness([_result(1)])
    harness.cancelled = True

    result = harness.service().run(
        PlannerSessionRequest(
            output_dir=tmp_path,
            exploration=True,
            requested_session_count=3,
            first_user_message="task",
        )
    )

    assert result.stop_reason is PlannerSessionStopReason.CANCELLED
    assert result.sessions == []
    assert harness.events == []


def test_missing_opening_message_is_clean_zero_session_exhaustion(
    tmp_path: Path,
) -> None:
    harness = _Harness([_result(1)])

    result = harness.service().run(
        PlannerSessionRequest(
            output_dir=tmp_path,
            exploration=False,
            requested_session_count=4,
            first_user_message=None,
        )
    )

    assert result.stop_reason is PlannerSessionStopReason.EXHAUSTED
    assert result.sessions == []
    assert harness.events == []


def test_frontend_can_preserve_its_exception_rendering(tmp_path: Path) -> None:
    harness = _Harness([RuntimeError("planner exploded")])
    service = SynchronousPlannerSessionService(
        prepare_session=harness.prepare,
        create_toolkit=harness.create_toolkit,
        invoke_planner=harness.invoke,
        format_exception=str,
    )

    result = service.run(
        PlannerSessionRequest(
            output_dir=tmp_path,
            exploration=False,
            requested_session_count=1,
            first_user_message="task",
        )
    )

    assert result.stop_reason is PlannerSessionStopReason.EXCEPTION
    assert result.error == "planner exploded"
    assert harness.toolkits[0].close_count == 1
