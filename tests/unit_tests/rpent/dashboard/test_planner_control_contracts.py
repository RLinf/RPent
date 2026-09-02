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

import asyncio
from dataclasses import replace

from rpent.dashboard.interaction import (
    DashboardInteractionError,
    DashboardMessage,
    DashboardMessageConflictError,
    InteractionUnavailableError,
    UnknownDashboardMessageError,
)
from rpent.dashboard.planner_control import DashboardPlannerControl


class FakeInteraction:
    """Small in-memory implementation of the planner-facing interaction port."""

    def __init__(self) -> None:
        self.activity = "starting"
        self.accepting_input = False
        self.version = 0
        self.messages: list[DashboardMessage] = []
        self.interrupt_requested = False
        self.interrupt_in_flight = False
        self.interrupt_error: str | None = None
        self.replacement_requested = False
        self.replacement_error: str | None = None
        self.end_on_wait = False

    @property
    def planner_activity(self) -> str:
        return self.activity

    @property
    def interaction_version(self) -> int:
        return self.version

    @property
    def task_replacement_requested(self) -> bool:
        return self.replacement_requested

    def wait_for_interaction_change(
        self,
        since: int,
        timeout: float | None = None,
    ) -> int:
        del since, timeout
        if self.end_on_wait:
            self.seal_interaction()
        return self.version

    def set_planner_activity(
        self,
        activity: str,
        *,
        accepting_input: bool | None = None,
    ) -> None:
        if activity not in {"starting", "idle", "busy", "ended"}:
            raise ValueError(f"unknown planner activity: {activity!r}")
        if self.activity == "ended" and activity != "ended":
            raise InteractionUnavailableError("Dashboard Session has ended")
        if activity == "ended":
            self.seal_interaction()
            return
        changed = self.activity != activity
        self.activity = activity
        if accepting_input is not None:
            changed = changed or self.accepting_input != bool(accepting_input)
            self.accepting_input = accepting_input
        if changed:
            self.version += 1

    def submit(self, message_id: str, text: str) -> DashboardMessage:
        message = DashboardMessage(message_id, text, "pending")
        self.messages.append(message)
        self.version += 1
        return replace(message)

    def claim_next_pending_message(self) -> DashboardMessage | None:
        if self.interrupt_requested or self.replacement_requested:
            return None
        for message in self.messages:
            if message.status == "pending":
                message.status = "sending"
                self.version += 1
                return replace(message)
        return None

    def mark_message_sent(self, message_id: str) -> DashboardMessage:
        return self._transition(message_id, "sent")

    def mark_message_failed(
        self,
        message_id: str,
        error: str,
    ) -> DashboardMessage:
        return self._transition(message_id, "failed", error)

    def mark_message_unsent(self, message_id: str) -> DashboardMessage:
        return self._transition(message_id, "unsent")

    def _transition(
        self,
        message_id: str,
        status: str,
        error: str | None = None,
    ) -> DashboardMessage:
        message = next(
            (item for item in self.messages if item.message_id == message_id),
            None,
        )
        if message is None:
            raise UnknownDashboardMessageError(
                f"unknown Dashboard message: {message_id}"
            )
        if message.status != "sending":
            raise DashboardMessageConflictError(
                f"message is {message.status}, not sending"
            )
        message.status = status
        message.error = error
        self.version += 1
        return replace(message)

    def claim_interrupt_request(self) -> bool:
        if not self.interrupt_requested or self.interrupt_in_flight:
            return False
        self.interrupt_in_flight = True
        self.version += 1
        return True

    def complete_interrupt(self, error: str | None = None) -> None:
        if not self.interrupt_requested or not self.interrupt_in_flight:
            raise DashboardInteractionError("no interrupt request is in flight")
        self.interrupt_requested = False
        self.interrupt_in_flight = False
        self.interrupt_error = error
        self.version += 1

    def seal_interaction(self) -> None:
        self.interrupt_requested = False
        self.interrupt_in_flight = False
        for message in self.messages:
            if message.status in {"pending", "sending"}:
                message.status = "unsent"
                message.error = None
        self.activity = "ended"
        self.accepting_input = False
        self.version += 1

    def complete_task_replacement(self, error: str | None = None) -> None:
        if not self.replacement_requested:
            return
        self.replacement_error = error
        self.seal_interaction()


class FakeDriver:
    def __init__(
        self,
        events: list[str],
        *,
        submit_error: BaseException | None = None,
        interrupt_error: BaseException | None = None,
        interrupt_completions: int = 1,
    ) -> None:
        self.events = events
        self.submit_error = submit_error
        self.interrupt_error = interrupt_error
        self.interrupt_completions = interrupt_completions
        self.submissions: list[str] = []
        self.dashboard_submissions: list[DashboardMessage] = []

    async def submit(self, text: str) -> int:
        self.events.append(f"submit:{text}")
        if self.submit_error is not None:
            raise self.submit_error
        self.submissions.append(text)
        return 1

    async def submit_dashboard_message(self, message: DashboardMessage) -> int:
        self.events.append(f"queue:{message.text}")
        if self.submit_error is not None:
            raise self.submit_error
        self.dashboard_submissions.append(message)
        return 1

    async def interrupt(self) -> int:
        self.events.append("backend-interrupt")
        if self.interrupt_error is not None:
            raise self.interrupt_error
        return self.interrupt_completions


def make_control(
    interaction: FakeInteraction,
    events: list[str],
    *,
    defer_message_ack: bool = False,
    cancel_error: BaseException | None = None,
) -> DashboardPlannerControl:
    def cancel_active_and_wait() -> None:
        events.append("toolkit-cancel")
        if cancel_error is not None:
            raise cancel_error

    return DashboardPlannerControl(
        interaction=interaction,
        cancel_active_and_wait=cancel_active_and_wait,
        emit_user=lambda text: events.append(f"user:{text}"),
        emit_initial_user=lambda: events.append("initial-user"),
        defer_message_ack=defer_message_ack,
    )


def test_start_and_completion_move_between_busy_and_idle() -> None:
    interaction = FakeInteraction()
    events: list[str] = []
    control = make_control(interaction, events)
    driver = FakeDriver(events)

    async def scenario() -> None:
        await control.start()
        assert interaction.activity == "busy"
        assert interaction.accepting_input is True
        await control.complete(driver)

    asyncio.run(scenario())

    assert interaction.activity == "idle"
    assert events == ["initial-user"]


def test_completion_flushes_pending_messages_and_acknowledges_each() -> None:
    interaction = FakeInteraction()
    interaction.submit("one", "first")
    interaction.submit("two", "second")
    events: list[str] = []
    control = make_control(interaction, events)
    driver = FakeDriver(events)

    async def scenario() -> None:
        await control.start()
        await control.complete(driver)

    asyncio.run(scenario())

    assert driver.submissions == ["first", "second"]
    assert [message.status for message in interaction.messages] == ["sent", "sent"]
    assert interaction.activity == "busy"
    assert events == [
        "initial-user",
        "submit:first",
        "user:first",
        "submit:second",
        "user:second",
    ]


def test_failed_submission_is_recorded_and_control_recovers_to_idle() -> None:
    interaction = FakeInteraction()
    interaction.submit("bad", "cannot send")
    events: list[str] = []
    control = make_control(interaction, events)
    driver = FakeDriver(events, submit_error=RuntimeError("backend unavailable"))

    async def scenario() -> None:
        await control.start()
        await control.complete(driver)

    asyncio.run(scenario())

    assert interaction.activity == "idle"
    assert interaction.messages[0].status == "failed"
    assert interaction.messages[0].error == "RuntimeError: backend unavailable"
    assert "user:cannot send" not in events


def test_deferred_ack_restores_a_message_that_never_started() -> None:
    interaction = FakeInteraction()
    interaction.submit("later", "queued for API")
    events: list[str] = []
    control = make_control(interaction, events, defer_message_ack=True)
    driver = FakeDriver(events)

    async def scenario() -> None:
        await control.start()
        await control.complete(driver)
        assert interaction.messages[0].status == "sending"
        control.message_discarded("later")

    asyncio.run(scenario())

    assert interaction.messages[0].status == "unsent"
    assert "user:queued for API" not in events


def test_deferred_ack_emits_only_when_backend_execution_starts() -> None:
    interaction = FakeInteraction()
    interaction.submit("next", "queued for API")
    events: list[str] = []
    control = make_control(interaction, events, defer_message_ack=True)
    driver = FakeDriver(events)

    async def scenario() -> None:
        await control.start()
        await control.complete(driver)
        control.message_started("next", "queued for API")

    asyncio.run(scenario())

    assert interaction.messages[0].status == "sent"
    assert events == ["initial-user", "queue:queued for API", "user:queued for API"]


def test_interrupt_cancels_toolkit_before_backend_and_then_flushes() -> None:
    interaction = FakeInteraction()
    interaction.submit("after", "continue after interrupt")
    interaction.end_on_wait = True
    events: list[str] = []
    control = make_control(interaction, events)
    driver = FakeDriver(events)

    async def scenario() -> None:
        await control.start()
        interaction.interrupt_requested = True
        await control.run(driver)

    asyncio.run(scenario())

    assert events[:2] == ["initial-user", "toolkit-cancel"]
    assert events[2:] == [
        "backend-interrupt",
        "submit:continue after interrupt",
        "user:continue after interrupt",
    ]
    assert interaction.interrupt_error is None


def test_interrupt_failure_is_reported_without_flushing_messages() -> None:
    interaction = FakeInteraction()
    interaction.submit("pending", "keep queued")
    interaction.end_on_wait = True
    events: list[str] = []
    control = make_control(interaction, events)
    driver = FakeDriver(events, interrupt_error=RuntimeError("cannot interrupt"))

    async def scenario() -> None:
        await control.start()
        interaction.interrupt_requested = True
        await control.run(driver)

    asyncio.run(scenario())

    assert events == ["initial-user", "toolkit-cancel", "backend-interrupt"]
    assert interaction.interrupt_error == "RuntimeError: cannot interrupt"
    assert interaction.messages[0].status == "unsent"


def test_replacement_cancels_toolkit_before_backend_and_seals_input() -> None:
    interaction = FakeInteraction()
    interaction.replacement_requested = True
    events: list[str] = []
    control = make_control(interaction, events)
    driver = FakeDriver(events)

    async def scenario() -> None:
        await control.start()
        await control.run(driver)

    asyncio.run(scenario())

    assert events == ["initial-user", "toolkit-cancel", "backend-interrupt"]
    assert interaction.activity == "ended"
    assert interaction.accepting_input is False
    assert interaction.replacement_error is None


def test_replacement_cleanup_failure_is_reported_and_still_seals_input() -> None:
    interaction = FakeInteraction()
    interaction.replacement_requested = True
    events: list[str] = []
    control = make_control(
        interaction,
        events,
        cancel_error=RuntimeError("toolkit stuck"),
    )
    driver = FakeDriver(events)

    async def scenario() -> None:
        await control.start()
        await control.run(driver)

    asyncio.run(scenario())

    assert events == ["initial-user", "toolkit-cancel"]
    assert interaction.activity == "ended"
    assert interaction.replacement_error == (
        "planner interrupt failed: RuntimeError: toolkit stuck"
    )


def test_end_seals_pending_messages_as_unsent() -> None:
    interaction = FakeInteraction()
    interaction.submit("pending", "never submitted")
    events: list[str] = []
    control = make_control(interaction, events)

    control.end()

    assert interaction.activity == "ended"
    assert interaction.messages[0].status == "unsent"
