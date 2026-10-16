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

"""Gemini CLI planner using persistent ACP sessions.

Gemini CLI owns the agent loop. RPent exposes its toolkit through the same
in-process Streamable HTTP MCP bridge used by the Codex planner, then records
the CLI's events into the common planner result. All sessions use the
ACP SDK for queued prompts and acknowledged cancellation, not a model API SDK.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import queue
import shutil
import tempfile
from collections.abc import Iterable
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
from rpent.dashboard.planner_control import DashboardPlannerControl
from rpent.planner.base import Planner, PlannerResult
from rpent.planner.utils.agent_runtime import (
    AgentArtifacts,
    agent_environment,
    short_json,
)
from rpent.planner.utils.agent_session import run_agent
from rpent.planner.utils.sdk import int_field, jsonable
from rpent.tools.toolkit import Toolkit

_MCP_TOOL_PREFIX = "mcp_rpent_"
_SECRET_ENV_NAMES = ("GEMINI_API_KEY",)
# Gemini CLI treats an empty ``tools.core`` list as unset.
_NO_BUILTIN_TOOLS = ("__rpent_no_builtin_tools__",)


# ---------------------------------------------------------------------------
# Public backend
# ---------------------------------------------------------------------------


class GeminiCliPlanner(Planner):
    """Planner that delegates its agent loop to the official Gemini CLI."""

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
        """Initialize the Gemini CLI backend.

        Args:
            output_dir: Directory containing planner artifacts.
            dashboard_events: Sink for transcript and usage updates.
            timeout_s: Wall-clock limit for the CLI process.
            output_path: Rendered transcript path, or a temporary file if omitted.
            model: Optional exact model ID passed to Gemini CLI. When omitted,
                Gemini CLI and its configured upstream choose the model.
            reasoning_effort: Must be ``none`` because Gemini CLI has no matching
                stable command-line option.
        """
        if reasoning_effort != "none":
            raise ValueError(
                "gemini_cli does not support --reasoning-effort; use 'none'"
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
        """Run a Gemini ACP session with optional queued follow-up input."""
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
            "gemini_cli", self._output_path, directory=self._output_dir
        )
        recorder = _Recorder(self._dashboard_events)
        stderr_path = artifacts.sidecar(".stderr.log")
        self._output_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix="gemini_cli_", dir=self._output_dir
        ) as cwd:
            root = Path(cwd)

            def create_session(url, control, emit):
                _write_settings(
                    root,
                    mcp_url=url,
                    max_turns=max_turns,
                    tool_names=(tool.name for tool in toolkit.list_tools()),
                )
                return _GeminiSession(
                    cwd=root,
                    model=self._model,
                    control=control,
                    emit_event=emit,
                    stderr_path=stderr_path,
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
                label="Gemini CLI",
                input_queue=input_queue,
                interaction=interaction,
                redact=_redact_secrets,
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


# ---------------------------------------------------------------------------
# ACP session driver
# ---------------------------------------------------------------------------


class _GeminiSession:
    """Own a persistent Gemini CLI ACP session and one cancellable prompt."""

    def __init__(
        self,
        *,
        cwd: Path,
        model: str | None,
        control: DashboardPlannerControl | None,
        emit_event,
        stderr_path: Path,
    ) -> None:
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

        command = _build_command(model=self._model)
        transport = spawn_agent_process(
            self,
            command[0],
            *command[1:],
            cwd=self._cwd,
            env=_gemini_environment(),
            transport_kwargs={"limit": 10 * 1024 * 1024},
        )
        connection, process = await transport.__aenter__()
        self._transport, self._connection = transport, connection
        self._stderr_task = asyncio.create_task(self._read_stderr(process.stderr))
        await connection.initialize(protocol_version=PROTOCOL_VERSION)
        # new_session uses the CLI's configured authentication; authenticate
        # would select a method and may rewrite the user's CLI settings.
        session = await connection.new_session(cwd=str(self._cwd))
        self._session_id = session.session_id
        await self.submit(DashboardMessage("initial", prompt, "sending"))
        if self._control is None:
            await asyncio.shield(self._task)
        else:
            await self._control.start()
            await self._control.run(self)

    async def submit(self, message: DashboardMessage) -> int:
        if self._closing or self._session_id is None:
            raise RuntimeError("Gemini conversation is closed")
        if not self._done.is_set():
            raise RuntimeError("Cannot submit while a Gemini prompt is active")
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
            self.error = _redact_secrets(f"{type(exc).__name__}: {exc}")
            if self._control is not None:
                self._control.end()

    async def interrupt(self) -> int:
        if self._done.is_set():
            return 0
        await self._connection.cancel(session_id=self._session_id)
        try:
            await asyncio.wait_for(self._done.wait(), timeout=15)
        except asyncio.TimeoutError:
            raise RuntimeError("Gemini did not finish the interrupted prompt") from None
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

        # RPent MCP tools are trusted in the generated settings. Do not grant
        # permissions for any additional tool exposed by a user's configuration.
        return RequestPermissionResponse(outcome=DeniedOutcome(outcome="cancelled"))

    async def _read_stderr(self, stream) -> None:
        with self._stderr_path.open("w") as output:
            if stream is not None:
                while line := await stream.readline():
                    output.write(
                        _redact_secrets(line.decode("utf-8", errors="replace"))
                    )
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
# Connectivity probe
# ---------------------------------------------------------------------------


def run_probe_turn(*, prompt: str, model: str | None, timeout_s: int) -> str:
    """Run a tool-free ACP prompt using the CLI's existing configuration."""
    recorder = _Recorder(NullDashboardEventSink())
    with tempfile.TemporaryDirectory(prefix="rpent_gemini_probe_") as directory:
        root = Path(directory)
        _write_settings(root, mcp_url=None, max_turns=1)
        stderr_path = root / "stderr.log"

        async def run():
            session = _GeminiSession(
                cwd=root,
                model=model,
                control=None,
                emit_event=recorder.observe,
                stderr_path=stderr_path,
            )
            try:
                await asyncio.wait_for(session.run(prompt), timeout=timeout_s)
            except asyncio.TimeoutError:
                raise TimeoutError(
                    f"Gemini CLI did not finish within {timeout_s}s"
                ) from None
            finally:
                await session.close()
            if session.error:
                raise RuntimeError(session.error)

        asyncio.run(run())
    return recorder.final_response


# ---------------------------------------------------------------------------
# Observation layer
# ---------------------------------------------------------------------------


@dataclass
class _Recorder:
    """Translate ACP updates and actual MCP results into transcript and stats."""

    dashboard_events: DashboardEventSink
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
                return f"[gemini] {text}\n"
            if kind == "agent_thought_chunk" and content.get("type") == "text":
                self.dashboard_events.emit(
                    TranscriptEvent({"type": "thinking", "text": content["text"]})
                )
                return f"[gemini-thinking] {content['text']}\n"
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
            return f"[gemini-result] {event.get('stopReason', '')}\n"
        return ""


# ---------------------------------------------------------------------------
# CLI configuration
# ---------------------------------------------------------------------------


def _write_settings(
    root: Path,
    *,
    mcp_url: str | None,
    max_turns: int,
    tool_names: Iterable[str] = (),
) -> Path:
    """Write workspace-only settings without changing the user's Gemini config."""
    allowed_tools = [f"{_MCP_TOOL_PREFIX}{name}" for name in tool_names]
    settings: dict[str, Any] = {
        "context": {
            "fileName": "RPENT_NO_PROJECT_CONTEXT.md",
            "includeDirectoryTree": False,
            "memoryBoundaryMarkers": [],
        },
        "hooksConfig": {"enabled": False},
        "model": {"maxSessionTurns": max(1, int(max_turns))},
        "privacy": {"usageStatisticsEnabled": False},
        "skills": {"enabled": False},
        "tools": {"core": allowed_tools or list(_NO_BUILTIN_TOOLS)},
    }
    if mcp_url:
        settings["mcpServers"] = {
            "rpent": {
                "httpUrl": mcp_url,
                "trust": True,
            }
        }
    settings_dir = root / ".gemini"
    settings_dir.mkdir(parents=True, exist_ok=True)
    path = settings_dir / "settings.json"
    path.write_text(json.dumps(settings, indent=2) + "\n")
    return path


def _build_command(*, model: str | None) -> list[str]:
    binary = os.environ.get("GEMINI_BIN") or shutil.which("gemini")
    if not binary:
        raise FileNotFoundError(
            "Gemini CLI is not installed; install @google/gemini-cli or set GEMINI_BIN"
        )
    command = [
        binary,
        "--acp",
        "--allowed-mcp-server-names",
        "rpent",
        "--extensions",
        "none",
    ]
    if model:
        command.extend(["--model", model])
    return command


def _gemini_environment() -> dict[str, str]:
    return {
        **agent_environment(),
        "GEMINI_CLI_NO_RELAUNCH": "1",
        # Trust the generated workspace so Gemini CLI loads its local settings.
        "GEMINI_CLI_TRUST_WORKSPACE": "true",
        "NO_COLOR": "1",
    }


def _dotenv_value(name: str) -> str | None:
    """Read one Gemini CLI dotenv setting without logging neighboring secrets."""
    path = Path.home() / ".gemini" / ".env"
    try:
        lines = path.read_text().splitlines()
    except OSError:
        return None
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        if key.strip().removeprefix("export ").strip() != name:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        return value or None
    return None


def _redact_secrets(text: str) -> str:
    for name in _SECRET_ENV_NAMES:
        secret = os.environ.get(name) or _dotenv_value(name)
        if secret and len(secret) >= 8:
            text = text.replace(secret, f"<{name}>")
    return text
