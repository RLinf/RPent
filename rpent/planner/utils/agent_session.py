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

"""External agent sessions with optional queued terminal or Dashboard input."""

from __future__ import annotations

import asyncio
import contextlib
import json
import queue
import threading
from collections import deque
from collections.abc import Callable
from typing import Any

from rpent.cli.tui import next_user_line
from rpent.dashboard.events import DashboardEventSink, TranscriptEvent
from rpent.dashboard.interaction import DashboardInteractionPort, DashboardMessage
from rpent.dashboard.planner_control import DashboardPlannerControl
from rpent.planner.base import cancel_and_wait
from rpent.planner.utils.agent_runtime import AgentArtifacts, write_jsonl
from rpent.planner.utils.http_mcp_server import HttpMcpServer
from rpent.tools import ToolResult
from rpent.tools.toolkit import Toolkit
from rpent.utils.logging import get_logger

logger = get_logger("agent_session")


class TerminalInteraction:
    """Adapt terminal lines to the existing queued interaction controller.

    All mutations run on the session event loop. The condition only bridges
    the controller's blocking change wait; the input reader owns its queue.
    """

    def __init__(self, input_queue: queue.Queue[str | None]) -> None:
        self._input_queue = input_queue
        self._condition = threading.Condition()
        self._pending: deque[DashboardMessage] = deque()
        self._messages: dict[str, DashboardMessage] = {}
        self.planner_activity = "starting"
        self.interaction_version = 0
        self.task_replacement_requested = False
        self._interrupt_requested = False

    def _changed(self) -> None:
        with self._condition:
            self.interaction_version += 1
            self._condition.notify_all()

    async def read_input(self) -> None:
        """Queue ordinary input; reserve /interrupt for explicit cancellation."""
        while self.planner_activity != "ended":
            line = await asyncio.to_thread(next_user_line, self._input_queue)
            if self.planner_activity == "ended":
                return
            if line is None:
                self.task_replacement_requested = True
                self._changed()
                return
            command, _, remainder = line.partition(" ")
            force = command.lower() == "/interrupt"
            text = remainder.strip() if force else line
            if text:
                message = DashboardMessage(str(len(self._messages)), text, "pending")
                self._messages[message.message_id] = message
                if force:
                    self._pending.appendleft(message)
                else:
                    self._pending.append(message)
                logger.info("[queued] %s", text)
            if force:
                self._interrupt_requested = True
            self._changed()

    def wait_for_interaction_change(
        self, since: int, timeout: float | None = None
    ) -> int:
        with self._condition:
            self._condition.wait_for(
                lambda: (
                    self.interaction_version != since
                    or self.planner_activity == "ended"
                ),
                timeout=timeout,
            )
            return self.interaction_version

    def set_planner_activity(
        self, activity: str, *, accepting_input: bool | None = None
    ) -> None:
        self.planner_activity = activity
        self._changed()

    def claim_next_pending_message(self) -> DashboardMessage | None:
        if (
            not self._pending
            or self.task_replacement_requested
            or self._interrupt_requested
        ):
            return None
        message = self._pending.popleft()
        message.status = "sending"
        return message

    def mark_message_sent(self, message_id: str) -> DashboardMessage:
        message = self._messages[message_id]
        message.status = "sent"
        return message

    def mark_message_failed(self, message_id: str, error: str) -> DashboardMessage:
        message = self._messages[message_id]
        message.status, message.error = "failed", error
        logger.error("Message was not sent: %s", error)
        return message

    def mark_message_unsent(self, message_id: str) -> DashboardMessage:
        message = self._messages[message_id]
        message.status = "unsent"
        return message

    def claim_interrupt_request(self) -> bool:
        requested = self._interrupt_requested
        self._interrupt_requested = False
        return requested

    def complete_interrupt(self, error: str | None = None) -> None:
        if error:
            logger.error("Interrupt failed; messages remain queued: %s", error)
        self._changed()

    def complete_task_replacement(self, error: str | None = None) -> None:
        if error:
            logger.error("Session stop failed: %s", error)
        self.seal_interaction()

    def seal_interaction(self) -> None:
        self.planner_activity = "ended"
        for message in self._messages.values():
            if message.status in {"pending", "sending"}:
                message.status = "unsent"
        self._input_queue.put(None)
        self._changed()


async def run_agent(
    *,
    prompt: str,
    initial_user_text: str,
    toolkit: Toolkit,
    artifacts: AgentArtifacts,
    recorder: Any,
    events: DashboardEventSink,
    create_session: Callable[..., Any],
    timeout_s: float,
    label: str,
    input_queue: queue.Queue[str | None] | None,
    interaction: DashboardInteractionPort | None,
    redact: Callable[[str], str] = str,
    tool_event: Callable[[str, dict[str, Any], ToolResult], dict[str, Any]]
    | None = None,
) -> str | None:
    """Own one agent session, optional input, artifacts, and MCP server.

    The factory receives the MCP URL, shared controller, and event callback.
    Without input, the controller is None and the session runs one turn.
    Its session implements run, submit, interrupt, close, and an error field.
    Toolkit cancellation always precedes backend interruption and shutdown.
    """
    terminal = TerminalInteraction(input_queue) if input_queue is not None else None
    port = terminal if terminal is not None else interaction
    error: str | None = None
    session = None
    reader = None
    loop = asyncio.get_running_loop()
    with (
        artifacts.output_path.open("w") as output,
        artifacts.raw_stream_path.open("w") as raw,
    ):

        def emit_user(text: str, *, initial: bool = False) -> None:
            display = "[initial task instructions submitted]" if initial else text
            output.write(f"\n[user] {display}\n")
            output.flush()
            events.emit(
                TranscriptEvent(
                    {"type": "initial_prompt"}
                    if initial
                    else {"type": "user", "text": text}
                )
            )

        control = (
            DashboardPlannerControl(
                interaction=port,
                cancel_active_and_wait=toolkit.cancel_active_and_wait,
                resume_calls=toolkit.resume_calls,
                emit_user=emit_user,
                emit_initial_user=lambda: emit_user(initial_user_text, initial=True),
                queue_until_complete=True,
            )
            if port is not None
            else None
        )

        def emit_event(event: dict[str, Any]) -> None:
            safe = json.loads(
                redact(json.dumps(event, ensure_ascii=False, default=str))
            )
            write_jsonl(raw, safe)
            if rendered := recorder.observe(safe):
                output.write(rendered)
                output.flush()
                logger.info(rendered.rstrip())
            if control is not None and recorder.finish_result is not None:
                control.end()

        def observe_tool(name, args, result):
            loop.call_soon_threadsafe(emit_event, tool_event(name, args, result))

        server = (
            HttpMcpServer(toolkit, on_tool_result=observe_tool)
            if tool_event is not None
            else HttpMcpServer(toolkit)
        )
        try:
            url = await asyncio.to_thread(server.start)
            session = create_session(url, control, emit_event)
            if terminal is not None:
                reader = asyncio.create_task(terminal.read_input())
            await asyncio.wait_for(session.run(prompt), timeout=timeout_s)
            error = session.error
        except asyncio.TimeoutError:
            error = f"{label} timed out after {timeout_s}s"
        except Exception as exc:
            error = redact(f"{type(exc).__name__}: {exc}")
        finally:
            if control is not None:
                control.end()
            try:
                await cancel_and_wait(toolkit.cancel_active_and_wait)
            except Exception as exc:
                error = error or redact(f"Toolkit cleanup failed: {exc}")
            try:
                if session is not None:
                    await session.close()
                    error = error or session.error
            except Exception as exc:
                error = error or redact(f"Session cleanup failed: {exc}")
            finally:
                await asyncio.to_thread(server.stop)
                if reader is not None:
                    with contextlib.suppress(asyncio.CancelledError):
                        await reader
        if error:
            output.write(f"\n[{artifacts.backend}] {error}\n")
            write_jsonl(raw, {"type": "error", "message": error})
    return error
