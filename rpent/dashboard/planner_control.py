"""Backend-neutral control flow for interactive Dashboard planners."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

from rpent.dashboard.interaction import DashboardInteractionPort


class DashboardPlannerControl:
    """Coordinate queued input, interrupts, and task replacement."""

    def __init__(
        self,
        *,
        interaction: DashboardInteractionPort,
        cancel_active_and_wait: Callable[[], None],
        resume_operations: Callable[[], None],
        emit_user: Callable[[str], None],
        emit_initial_user: Callable[[], None],
        defer_message_ack: bool = False,
    ) -> None:
        self._interaction = interaction
        self._cancel_active_and_wait = cancel_active_and_wait
        self._resume_operations = resume_operations
        self._emit_user = emit_user
        self._emit_initial_user = emit_initial_user
        self._defer_message_ack = defer_message_ack
        self._lock = asyncio.Lock()
        self._outstanding_completions = 0

    async def start(self) -> None:
        """Open Dashboard input after the initial backend submission succeeds."""
        self._outstanding_completions = 1
        self._interaction.set_planner_activity(
            "busy",
            accepting_input=not self._interaction.task_replacement_requested,
        )
        self._emit_initial_user()

    async def run(self, driver: Any) -> None:
        """Forward Dashboard commands until the interaction ends."""
        version = self._interaction.interaction_version
        while self._interaction.planner_activity != "ended":
            await self._process(driver)
            version = await asyncio.to_thread(
                self._interaction.wait_for_interaction_change,
                version,
            )

    async def complete(self, driver: Any) -> None:
        """Record one completed backend request and flush queued input."""
        async with self._lock:
            if self._interaction.planner_activity == "ended":
                return
            self._outstanding_completions = max(0, self._outstanding_completions - 1)
            self._interaction.set_planner_activity(
                "busy" if self._outstanding_completions else "idle"
            )
            await self._flush(driver)

    async def tool_completed(self, driver: Any) -> None:
        """Flush input queued while the backend was running a tool."""
        async with self._lock:
            await self._flush(driver)

    def end(self) -> None:
        """Seal the interaction and preserve unfinished messages as unsent."""
        self._interaction.seal_interaction()

    def message_started(self, message_id: str, text: str) -> None:
        """Acknowledge one deferred submission when execution starts."""
        self._interaction.mark_message_sent(message_id)
        self._emit_user(text)

    def message_discarded(self, message_id: str) -> None:
        """Restore one deferred submission that never started."""
        if self._interaction.planner_activity != "ended":
            self._interaction.mark_message_unsent(message_id)

    async def cancel_active_toolkit(self) -> None:
        """Pause Toolkit admission and drain its operations off the event loop."""
        await asyncio.to_thread(self._cancel_active_and_wait)

    async def _process(self, driver: Any) -> None:
        async with self._lock:
            if self._interaction.planner_activity == "ended":
                return
            if self._interaction.task_replacement_requested:
                try:
                    await self.cancel_active_toolkit()
                    await driver.interrupt()
                except Exception as exc:
                    self._interaction.complete_task_replacement(
                        error=f"planner interrupt failed: {_exception_text(exc)}"
                    )
                else:
                    self._interaction.complete_task_replacement()
                return
            if self._interaction.claim_interrupt_request():
                try:
                    await self.cancel_active_toolkit()
                    completed = await driver.interrupt()
                    self._resume_operations()
                    self._outstanding_completions = max(
                        0, self._outstanding_completions - completed
                    )
                except Exception as exc:
                    self._interaction.complete_interrupt(error=_exception_text(exc))
                else:
                    self._interaction.complete_interrupt()
                    self._interaction.set_planner_activity(
                        "busy" if self._outstanding_completions else "idle"
                    )
                    await self._flush(driver)
                return
            if self._interaction.planner_activity == "idle":
                await self._flush(driver)

    async def _flush(self, driver: Any) -> None:
        message = self._interaction.claim_next_pending_message()
        while message is not None and not self._interaction.task_replacement_requested:
            try:
                if self._defer_message_ack:
                    added_completions = await driver.submit_dashboard_message(message)
                else:
                    added_completions = await driver.submit(message.text)
            except Exception as exc:
                self._interaction.mark_message_failed(
                    message.message_id,
                    _exception_text(exc),
                )
            else:
                if not self._defer_message_ack:
                    self.message_started(message.message_id, message.text)
                self._outstanding_completions += added_completions
                self._interaction.set_planner_activity("busy")
            message = self._interaction.claim_next_pending_message()


def _exception_text(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"
