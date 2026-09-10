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

"""Synchronous orchestration for independent planner sessions."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from rpent.memory import MemoryManager
    from rpent.planner.base import PlannerResult


class PlannerSessionStopReason(str, Enum):
    """Why a synchronous planner-session run stopped."""

    SOLVED = "solved"
    EXHAUSTED = "clean_unsolved_exhaustion"
    FINAL_TIMEOUT = "final_timeout"
    PLANNER_ERROR = "planner_error"
    EXCEPTION = "exception"
    CANCELLED = "cancelled"


@dataclass(frozen=True, slots=True)
class PlannerSessionRequest:
    """Inputs that determine the number and storage roots of planner sessions."""

    output_dir: Path
    exploration: bool
    requested_session_count: int | None
    first_user_message: str | None


@dataclass(frozen=True, slots=True)
class PlannerSessionContext:
    """Stable context selected by the service for one planner session."""

    session_number: int
    session_count: int
    output_dir: Path
    state_output_dir: Path
    exploration: bool


@dataclass(frozen=True, slots=True)
class PreparedPlannerSession:
    """Prompts prepared by a frontend-specific adapter for one session."""

    context: PlannerSessionContext
    system_prompt: str
    user_message: str


@dataclass(frozen=True, slots=True)
class PlannerSessionRecord:
    """Result and stop classification for one attempted planner session."""

    context: PlannerSessionContext
    finish_result: dict[str, Any] | None = None
    messages: list[dict[str, Any]] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    solved: bool = False
    environment_success: bool | None = None
    stop_reason: PlannerSessionStopReason | None = None


@dataclass(frozen=True, slots=True)
class PlannerSessionRunResult:
    """Accumulated terminal-compatible result of all attempted sessions."""

    stop_reason: PlannerSessionStopReason
    sessions: list[PlannerSessionRecord]
    finish_result: dict[str, Any] | None
    messages: list[dict[str, Any]]
    stats: dict[str, Any]
    error: str | None
    solved: bool
    environment_success: bool | None
    recipe_path: str | None
    memory_manager: MemoryManager | None


class PlannerSessionToolkit(Protocol):
    """Owned toolkit surface used directly by the orchestration service."""

    @property
    def memory(self) -> MemoryManager:
        """Return the borrowed memory manager associated with this toolkit."""
        ...

    def close(self) -> None:
        """Release only the resources owned by this toolkit."""
        ...

    def solved(self) -> bool:
        """Return whether the current environment state is solved."""
        ...

    def write_recipe(self, recipe_tag: str) -> str | None:
        """Export the current successful attempt when supported."""
        ...


class PreparePlannerSession(Protocol):
    """Prepare a planner and prompts before the toolkit is constructed."""

    def __call__(
        self,
        context: PlannerSessionContext,
        first_user_message: str,
    ) -> PreparedPlannerSession:
        """Return the prepared invocation for ``context``."""
        ...


class CreatePlannerToolkit(Protocol):
    """Construct a fresh service-owned toolkit for one session."""

    def __call__(self, context: PlannerSessionContext) -> PlannerSessionToolkit:
        """Create the toolkit for ``context``."""
        ...


class InvokePlanner(Protocol):
    """Invoke a prepared planner using a frontend-specific calling convention."""

    def __call__(
        self,
        prepared: PreparedPlannerSession,
        toolkit: PlannerSessionToolkit,
    ) -> PlannerResult:
        """Synchronously run the prepared planner session."""
        ...


def effective_session_count(exploration: bool, requested: int | None) -> int:
    """Return the effective session count used by existing frontends."""
    sessions = max(1, int(requested or 1))
    return sessions if exploration else 1


def _is_timeout(error: str) -> bool:
    return "timed out" in error.lower()


class SynchronousPlannerSessionService:
    """Run independent planner sessions while owning only their toolkits.

    Runtime clients referenced by the toolkit factory, memory managers returned
    by toolkits, and any daemons remain borrowed resources. The service closes
    each toolkit successfully returned by ``create_toolkit`` exactly once.
    """

    def __init__(
        self,
        *,
        prepare_session: PreparePlannerSession,
        create_toolkit: CreatePlannerToolkit,
        invoke_planner: InvokePlanner,
        probe_solved: Callable[[PlannerSessionToolkit], bool] | None = None,
        export_recipe: Callable[[PlannerSessionToolkit], str | None] | None = None,
        probe_environment_success: (
            Callable[[PlannerSessionToolkit], bool] | None
        ) = None,
        cancellation_requested: Callable[[], bool] | None = None,
        on_intermediate_timeout: (
            Callable[[PlannerSessionContext, str], None] | None
        ) = None,
    ) -> None:
        self._prepare_session = prepare_session
        self._create_toolkit = create_toolkit
        self._invoke_planner = invoke_planner
        self._probe_solved = probe_solved
        self._export_recipe = export_recipe
        self._probe_environment_success = probe_environment_success
        self._cancellation_requested = cancellation_requested
        self._on_intermediate_timeout = on_intermediate_timeout

    def _cancelled(self) -> bool:
        return bool(
            self._cancellation_requested is not None and self._cancellation_requested()
        )

    def run(self, request: PlannerSessionRequest) -> PlannerSessionRunResult:
        """Synchronously run planner sessions until one typed stop condition."""
        session_count = effective_session_count(
            request.exploration,
            request.requested_session_count,
        )
        records: list[PlannerSessionRecord] = []
        finish_result: dict[str, Any] | None = None
        messages: list[dict[str, Any]] = []
        stats: dict[str, Any] = {}
        error: str | None = None
        solved = False
        environment_success: bool | None = None
        recipe_path: str | None = ""
        memory_manager: MemoryManager | None = None
        stop_reason = PlannerSessionStopReason.EXHAUSTED

        if request.first_user_message is None:
            return PlannerSessionRunResult(
                stop_reason=stop_reason,
                sessions=records,
                finish_result=finish_result,
                messages=messages,
                stats=stats,
                error=error,
                solved=solved,
                environment_success=environment_success,
                recipe_path=recipe_path,
                memory_manager=memory_manager,
            )

        for session_number in range(1, session_count + 1):
            if self._cancelled():
                stop_reason = PlannerSessionStopReason.CANCELLED
                break

            state_output_dir = request.output_dir
            if request.exploration:
                state_output_dir = (
                    request.output_dir / "sessions" / f"session_{session_number:03d}"
                )
            context = PlannerSessionContext(
                session_number=session_number,
                session_count=session_count,
                output_dir=request.output_dir,
                state_output_dir=state_output_dir,
                exploration=request.exploration,
            )
            session_finish: dict[str, Any] | None = None
            session_messages: list[dict[str, Any]] = []
            session_stats: dict[str, Any] = {}
            session_error: str | None = None
            session_environment_success: bool | None = None

            try:
                prepared = self._prepare_session(
                    context,
                    request.first_user_message,
                )
                toolkit = self._create_toolkit(context)
                try:
                    memory_manager = toolkit.memory
                    planner_result = self._invoke_planner(prepared, toolkit)
                    session_finish = planner_result.finish_result
                    session_messages = planner_result.messages
                    session_stats = planner_result.stats
                    session_error = planner_result.error
                    finish_result = session_finish
                    messages += session_messages
                    stats = session_stats
                    error = session_error
                    if self._probe_solved is not None:
                        solved = self._probe_solved(toolkit)
                        if solved and self._export_recipe is not None:
                            recipe_path = self._export_recipe(toolkit)
                finally:
                    try:
                        if self._probe_environment_success is not None:
                            environment_success = bool(
                                self._probe_environment_success(toolkit)
                            )
                            session_environment_success = environment_success
                            solved = environment_success
                    finally:
                        toolkit.close()
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                stop_reason = PlannerSessionStopReason.EXCEPTION
                records.append(
                    PlannerSessionRecord(
                        context=context,
                        finish_result=session_finish,
                        messages=session_messages,
                        stats=session_stats,
                        error=error,
                        solved=solved,
                        environment_success=session_environment_success,
                        stop_reason=stop_reason,
                    )
                )
                break

            session_stop_reason: PlannerSessionStopReason | None = None
            if solved:
                session_stop_reason = PlannerSessionStopReason.SOLVED
            elif self._cancelled():
                session_stop_reason = PlannerSessionStopReason.CANCELLED
            elif error:
                if (
                    request.exploration
                    and session_number < session_count
                    and _is_timeout(error)
                ):
                    if self._on_intermediate_timeout is not None:
                        self._on_intermediate_timeout(context, error)
                else:
                    session_stop_reason = (
                        PlannerSessionStopReason.FINAL_TIMEOUT
                        if _is_timeout(error)
                        else PlannerSessionStopReason.PLANNER_ERROR
                    )
            elif session_number == session_count:
                session_stop_reason = PlannerSessionStopReason.EXHAUSTED

            records.append(
                PlannerSessionRecord(
                    context=context,
                    finish_result=session_finish,
                    messages=session_messages,
                    stats=session_stats,
                    error=session_error,
                    solved=solved,
                    environment_success=session_environment_success,
                    stop_reason=session_stop_reason,
                )
            )
            if session_stop_reason is not None:
                stop_reason = session_stop_reason
                break

        return PlannerSessionRunResult(
            stop_reason=stop_reason,
            sessions=records,
            finish_result=finish_result,
            messages=messages,
            stats=stats,
            error=error,
            solved=solved,
            environment_success=environment_success,
            recipe_path=recipe_path,
            memory_manager=memory_manager,
        )


__all__ = [
    "CreatePlannerToolkit",
    "InvokePlanner",
    "PlannerSessionContext",
    "PlannerSessionRecord",
    "PlannerSessionRequest",
    "PlannerSessionRunResult",
    "PlannerSessionStopReason",
    "PlannerSessionToolkit",
    "PreparePlannerSession",
    "PreparedPlannerSession",
    "SynchronousPlannerSessionService",
    "effective_session_count",
]
