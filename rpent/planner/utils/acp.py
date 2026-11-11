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

"""Shared ACP execution for harness-specific planner adapters."""

from __future__ import annotations

import asyncio
import contextlib
import os
import queue
import tempfile
from abc import abstractmethod
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rpent.dashboard.events import (
    DashboardEventSink,
    NullDashboardEventSink,
    TranscriptEvent,
    UsageEvent,
)
from rpent.dashboard.interaction import DashboardInteractionPort, DashboardMessage
from rpent.dashboard.planner_control import PlannerInteractionControl
from rpent.planner.base import Planner, PlannerResult
from rpent.planner.utils.agent_runtime import AgentArtifacts, short_json
from rpent.planner.utils.agent_session import run_agent
from rpent.planner.utils.sdk import int_field, jsonable
from rpent.tools.toolkit import Toolkit


def redact_agent_secrets(text: str) -> str:
    """Redact conventional harness credential variables from emitted text."""
    for name, value in os.environ.items():
        if (
            name.endswith(("_API_KEY", "_PROVIDER_KEY", "_TOKEN", "_SECRET"))
            and len(value) >= 8
        ):
            text = text.replace(value, f"<{name}>")
    return text


# ---------------------------------------------------------------------------
# Shared planner base
# ---------------------------------------------------------------------------


class AcpPlanner(Planner):
    """Base for harness adapters using ACP over stdio and HTTP MCP.

    Subclasses name their backend and prepare an AcpSession in _create_session.
    They own the command, authentication, tool policy, and model-turn limits;
    this base owns interaction, artifacts, and the wall-clock timeout.
    """

    backend: str
    label: str
    _redact = staticmethod(redact_agent_secrets)

    def __init__(
        self,
        *,
        output_dir: str,
        dashboard_events: DashboardEventSink,
        timeout_s: int = 1200,
        output_path: str | Path | None = None,
        model: str | None = None,
        reasoning_effort: str = "none",
    ) -> None:
        """Initialize an ACP backend.

        Args:
            output_dir: Directory containing planner artifacts.
            dashboard_events: Sink for transcript and usage updates.
            timeout_s: Wall-clock limit for the CLI process.
            output_path: Rendered transcript path, or a temporary file if omitted.
            model: Optional model ID interpreted by the harness adapter.
            reasoning_effort: Must be none; configure effort in the harness.
        """
        if reasoning_effort != "none":
            raise ValueError(
                f"{self.backend} does not support --reasoning-effort; use 'none'"
            )
        self._output_dir = Path(output_dir)
        self._timeout_s = timeout_s
        self._output_path = Path(output_path) if output_path else None
        self._model = model
        self._dashboard_events = dashboard_events

    def solve(
        self,
        *,
        system_prompt: str,
        user_message: str,
        toolkit: Toolkit,
        max_turns: int,
        input_queue: queue.Queue[str | None] | None = None,
        dashboard_interaction: DashboardInteractionPort | None = None,
    ) -> PlannerResult:
        """Run an ACP session with optional queued follow-up input."""
        if input_queue is not None and dashboard_interaction is not None:
            raise ValueError(
                "input_queue and dashboard_interaction cannot be used together"
            )
        return asyncio.run(
            self._solve(
                prompt=f"{system_prompt}\n\n{user_message}"
                if system_prompt
                else user_message,
                initial_user_text=user_message,
                toolkit=toolkit,
                max_turns=max_turns,
                input_queue=input_queue,
                interaction=dashboard_interaction,
            )
        )

    async def _solve(
        self,
        *,
        prompt: str,
        initial_user_text: str,
        toolkit: Toolkit,
        max_turns: int,
        input_queue: queue.Queue[str | None] | None,
        interaction: DashboardInteractionPort | None,
    ) -> PlannerResult:
        artifacts = AgentArtifacts(
            self.backend, self._output_path, directory=self._output_dir
        )
        recorder = AcpRecorder(self._dashboard_events, label=self.backend)
        stderr_path = artifacts.sidecar(".stderr.log")
        self._output_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix=f"{self.backend}_", dir=self._output_dir
        ) as cwd:
            root = Path(cwd)

            def create_session(url, control, emit):
                return self._create_session(
                    root,
                    control,
                    emit,
                    stderr_path,
                    mcp_url=url,
                    max_turns=max_turns,
                    tool_names=(tool.name for tool in toolkit.list_tools()),
                )

            error = await run_agent(
                prompt=prompt,
                initial_user_text=initial_user_text,
                toolkit=toolkit,
                artifacts=artifacts,
                recorder=recorder,
                events=self._dashboard_events,
                create_session=create_session,
                timeout_s=self._timeout_s,
                label=self.label,
                input_queue=input_queue,
                interaction=interaction,
                redact=self._redact,
                tool_event=lambda name, args, result: {
                    "type": "rpent_tool",
                    "name": name,
                    "args": args,
                    "result": result.data,
                    "is_error": result.is_error,
                },
            )
        return artifacts.result(
            text="",
            finish_result=recorder.finish_result,
            final_response=recorder.final_response,
            stats=recorder.stats(),
            error=error,
            stderr_path=stderr_path,
        )

    @abstractmethod
    def _create_session(
        self,
        root: Path,
        control: PlannerInteractionControl | None,
        emit: Callable[[dict[str, Any]], Any],
        stderr_path: Path,
        *,
        mcp_url: str | None,
        max_turns: int,
        tool_names: Iterable[str],
    ) -> AcpSession:
        """Prepare a harness session; mcp_url is None during a connection probe."""
        ...

    def probe(self, prompt: str) -> str:
        """Request a minimal reply without mounting RPent tools."""
        recorder = AcpRecorder(NullDashboardEventSink(), label=self.backend)
        with tempfile.TemporaryDirectory(
            prefix=f"rpent_{self.backend}_probe_"
        ) as directory:
            root = Path(directory)

            async def run():
                session = self._create_session(
                    root,
                    None,
                    recorder.observe,
                    root / "stderr.log",
                    mcp_url=None,
                    max_turns=1,
                    tool_names=(),
                )
                try:
                    await asyncio.wait_for(session.run(prompt), timeout=self._timeout_s)
                finally:
                    await session.close()
                if session.error:
                    raise RuntimeError(session.error)

            asyncio.run(run())
        return self._redact(recorder.final_response)


# ---------------------------------------------------------------------------
# ACP session driver
# ---------------------------------------------------------------------------


class AcpSession:
    """Own a persistent ACP session and one cancellable prompt."""

    def __init__(
        self,
        *,
        command: Sequence[str],
        cwd: Path,
        env: dict[str, str],
        mcp_url: str | None = None,
        redact: Callable[[str], str] = redact_agent_secrets,
        model: str | None,
        control: PlannerInteractionControl | None,
        emit_event: Callable[[dict[str, Any]], Any],
        stderr_path: Path,
    ) -> None:
        """Own one agent process, its ACP connection, and at most one active prompt."""
        self._command = command
        self._env = env
        self._mcp_url = mcp_url
        self._redact = redact
        self._cwd = cwd
        self._model = model
        self._control = control
        self._emit = emit_event
        self._stderr_path = stderr_path
        self._transport = None
        self._connection = None
        self._session_id = None
        self._task: asyncio.Task | None = None
        self._done = asyncio.Event()
        self._done.set()
        self._stderr_task: asyncio.Task | None = None
        self._closing = False
        self.error: str | None = None

    async def run(self, prompt: str) -> None:
        from acp import PROTOCOL_VERSION, spawn_agent_process

        command = self._command
        transport = spawn_agent_process(
            self,
            command[0],
            *command[1:],
            cwd=self._cwd,
            env=self._env,
            transport_kwargs={"limit": 10 * 1024 * 1024},
        )
        connection, process = await transport.__aenter__()
        self._transport, self._connection = transport, connection
        self._stderr_task = asyncio.create_task(self._read_stderr(process.stderr))
        initialized = await connection.initialize(protocol_version=PROTOCOL_VERSION)
        if initialized.protocol_version != PROTOCOL_VERSION:
            raise RuntimeError("The agent does not support this ACP protocol version")
        capabilities = initialized.agent_capabilities.mcp_capabilities
        if self._mcp_url and not (capabilities and capabilities.http):
            raise RuntimeError("The ACP agent must support HTTP MCP servers")
        # Authentication remains owned by the harness; do not rewrite its login.
        from acp.schema import HttpMcpServer

        servers = (
            [HttpMcpServer(type="http", name="rpent", url=self._mcp_url, headers=[])]
            if self._mcp_url
            else []
        )
        session = await connection.new_session(cwd=str(self._cwd), mcp_servers=servers)
        self._session_id = session.session_id
        if self._model:
            selector = next(
                (
                    option
                    for option in session.config_options or []
                    if option.category == "model" and option.type == "select"
                ),
                None,
            )
            if selector is None:
                raise ValueError(
                    "The ACP agent has no model selector; configure the model in the harness"
                )
            await connection.set_config_option(
                session_id=self._session_id, config_id=selector.id, value=self._model
            )
        await self.submit(DashboardMessage("initial", prompt, "sending"))
        if self._control is None:
            await asyncio.shield(self._task)
        else:
            await self._control.start()
            await self._control.run(self)

    async def submit(self, message: DashboardMessage) -> int:
        if self._closing or self._session_id is None:
            raise RuntimeError("ACP conversation is closed")
        if not self._done.is_set():
            raise RuntimeError("Cannot submit while an ACP prompt is active")
        self._done = asyncio.Event()
        self._emit({"type": "acp_prompt"})
        self._task = asyncio.create_task(self._consume_prompt(message.text, self._done))
        return 1

    async def _consume_prompt(self, text: str, done: asyncio.Event) -> None:
        from acp import text_block

        try:
            response = await self._connection.prompt(
                session_id=self._session_id,
                prompt=[text_block(text)],
            )
            self._emit(
                {
                    "type": "acp_result",
                    **jsonable(response, by_alias=True),
                }
            )
            done.set()
            if self._closing or self._control is None:
                return
            if response.stop_reason in {"max_turn_requests", "max_tokens", "refusal"}:
                self._control.end()
            else:
                await self._control.complete(self)
        except Exception as exc:
            done.set()
            self.error = self._redact(f"{type(exc).__name__}: {exc}")
            if self._control is not None:
                self._control.end()

    async def interrupt(self) -> int:
        if self._done.is_set():
            return 0
        await self._connection.cancel(session_id=self._session_id)
        try:
            await asyncio.wait_for(self._done.wait(), timeout=15)
        except asyncio.TimeoutError:
            raise RuntimeError(
                "ACP agent did not finish the interrupted prompt"
            ) from None
        return 0

    async def session_update(self, session_id: str, update: Any, **kwargs: Any) -> None:
        self._emit(
            {
                "type": "acp_update",
                "update": jsonable(update, by_alias=True),
            }
        )

    async def request_permission(self, session_id, tool_call, options, **kwargs):
        from acp.schema import DeniedOutcome, RequestPermissionResponse

        # Harness configuration must pre-authorize RPent tools. An ACP tool-call
        # label is not proof that a permission request belongs to our MCP server.
        return RequestPermissionResponse(outcome=DeniedOutcome(outcome="cancelled"))

    async def _read_stderr(self, stream) -> None:
        with self._stderr_path.open("w") as output:
            if stream is not None:
                while line := await stream.readline():
                    output.write(self._redact(line.decode("utf-8", errors="replace")))
                    output.flush()

    async def close(self) -> None:
        self._closing = True
        try:
            if self._task is not None:
                with contextlib.suppress(Exception):
                    await self.interrupt()
                if not self._task.done():
                    self._task.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await self._task
        finally:
            if self._transport is not None:
                await self._transport.__aexit__(None, None, None)
            if self._stderr_task is not None:
                await self._stderr_task


# ---------------------------------------------------------------------------
# Observation layer
# ---------------------------------------------------------------------------


@dataclass
class AcpRecorder:
    """Translate ACP updates and actual MCP results into transcript and stats."""

    dashboard_events: DashboardEventSink
    label: str = "acp"
    tool_calls: int = 0
    finish_result: dict[str, Any] | None = None
    final_response: str = ""
    usage: dict[str, int] = field(
        default_factory=lambda: {
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "total_cached_input_tokens": 0,
        }
    )

    def stats(self) -> dict[str, Any]:
        return {"tool_calls": self.tool_calls, **self.usage}

    def observe(self, event: dict[str, Any]) -> str:
        event_type = str(event.get("type") or "")
        if event_type == "rpent_tool":
            name, arguments, output = event["name"], event["args"], event["result"]
            self.tool_calls += 1
            if (
                name == "finish"
                and not event["is_error"]
                and output.get("_finish") is True
            ):
                self.finish_result = self.finish_result or dict(output)
            result = {
                "status": "error" if event["is_error"] else "success",
                "output": output,
            }
            self.dashboard_events.emit(
                TranscriptEvent({"type": "tool_call", "tool": name, "args": arguments})
            )
            self.dashboard_events.emit(
                TranscriptEvent({"type": "tool_result", "tool": name, "result": result})
            )
            return (
                f"[tool->] {name}: {short_json(arguments)}\n"
                f"[tool<-] {name}: {short_json(result)}\n"
            )
        if event_type == "acp_prompt":
            self.final_response = ""
            return ""
        if event_type == "acp_update":
            update = event["update"]
            kind = update.get("sessionUpdate")
            content = update.get("content") or {}
            if kind == "agent_message_chunk" and content.get("type") == "text":
                text = content["text"]
                self.final_response += text
                self.dashboard_events.emit(
                    TranscriptEvent({"type": "text", "text": text})
                )
                return f"[{self.label}] {text}\n"
            if kind == "agent_thought_chunk" and content.get("type") == "text":
                self.dashboard_events.emit(
                    TranscriptEvent({"type": "thinking", "text": content["text"]})
                )
                return f"[{self.label}-thinking] {content['text']}\n"
            return ""
        if event_type == "acp_result":
            usage = event.get("usage") or {}
            for target, source in (
                ("total_input_tokens", "inputTokens"),
                ("total_output_tokens", "outputTokens"),
                ("total_cached_input_tokens", "cachedReadTokens"),
            ):
                self.usage[target] += int_field(usage, source)
            self.dashboard_events.emit(
                UsageEvent(
                    inp=self.usage["total_input_tokens"],
                    out=self.usage["total_output_tokens"],
                    tool_calls=self.tool_calls,
                )
            )
            return f"[{self.label}-result] {event.get('stopReason', '')}\n"
        return ""
