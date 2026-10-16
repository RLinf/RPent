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

"""Claude Agent SDK planner.

A thin, SDK-first backend for RPent. ``solve()`` does four things:
prepare output files, bind the in-process tool runtime, drive the SDK
query, and assemble a ``PlannerResult``. Event rendering and stats
collection live in a single observation layer (``_Recorder``) that has
no backend state of its own.
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import json
import queue
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rpent.dashboard.events import (
    DashboardEventSink,
    TranscriptEvent,
    UsageEvent,
)
from rpent.dashboard.interaction import DashboardInteractionPort, DashboardMessage
from rpent.dashboard.planner_control import DashboardPlannerControl
from rpent.planner.base import (
    REASONING_EFFORTS,
    Planner,
    PlannerResult,
    add_mcp_prefix,
    cancel_and_wait,
    execute_tool,
    strip_mcp_prefix,
)
from rpent.planner.utils.agent_runtime import AgentArtifacts, short_json, write_jsonl
from rpent.planner.utils.agent_session import TerminalInteraction
from rpent.planner.utils.http_mcp_server import mcp_result
from rpent.planner.utils.sdk import get_field, jsonable
from rpent.tools import Toolkit, ToolResult
from rpent.utils.config import get_repo_root
from rpent.utils.logging import get_logger, init_output_dir
from rpent.utils.templates import substitute

logger = get_logger("claude")

_MAX_STREAM_BUFFER_BYTES = 8 * 1024 * 1024

# ---------------------------------------------------------------------------
# Public backend
# ---------------------------------------------------------------------------


class ClaudeCodePlanner(Planner):
    """Planner backed by the Claude Agent SDK."""

    def __init__(
        self,
        *,
        output_dir: str,
        dashboard_events: DashboardEventSink,
        repo_root: str | Path | None = None,
        model: str = "sonnet",
        allowed_tools: str = "Bash Read Write Glob Grep",
        timeout_s: int = 600,
        max_budget_usd: float = 10.0,
        extra_dirs: list[str] | None = None,
        output_path: str | Path | None = None,
        reasoning_effort: str = "none",
    ):
        """Initialize the Claude Agent SDK backend."""
        self._output_dir = str(output_dir)
        self._repo_root = str(repo_root) if repo_root else str(get_repo_root())
        self._model = model
        self._allowed_tools = allowed_tools
        self._timeout_s = timeout_s
        self._max_budget_usd = max_budget_usd
        self._extra_dirs = extra_dirs or []
        self._output_path = Path(output_path) if output_path else None
        self._dashboard_events = dashboard_events
        if reasoning_effort not in REASONING_EFFORTS:
            raise ValueError(f"unsupported reasoning effort: {reasoning_effort}")
        self._reasoning_effort = reasoning_effort

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
        """Run a Claude Agent SDK session for the given prompt."""
        if input_queue is not None and dashboard_interaction is not None:
            raise ValueError(
                "input_queue and dashboard_interaction cannot be used together"
            )
        prompt = f"{system_prompt}\n\n{user_message}" if system_prompt else user_message
        return asyncio.run(
            self._solve_async(
                prompt,
                initial_user_text=user_message,
                toolkit=toolkit,
                max_turns=max_turns,
                input_queue=input_queue,
                dashboard_interaction=dashboard_interaction,
            )
        )

    # -- internal lifecycle -------------------------------------------------

    async def _solve_async(
        self,
        prompt: str,
        *,
        initial_user_text: str,
        toolkit: Toolkit,
        max_turns: int,
        input_queue: queue.Queue[str | None] | None = None,
        dashboard_interaction: DashboardInteractionPort | None = None,
    ) -> PlannerResult:
        import claude_agent_sdk

        artifacts = AgentArtifacts("claude_agent_sdk", self._output_path)
        recorder = _Recorder(
            max_turns=max_turns, dashboard_events=self._dashboard_events
        )
        terminal = TerminalInteraction(input_queue) if input_queue is not None else None
        interaction = terminal if terminal is not None else dashboard_interaction
        init_output_dir(self._output_dir)
        error = None
        reader = None
        with (
            artifacts.output_path.open("w") as output,
            artifacts.raw_stream_path.open("w") as raw,
        ):

            def emit(message: Any) -> None:
                write_jsonl(raw, _message_to_json(message))
                if rendered := recorder.observe(message):
                    output.write(rendered)
                    output.flush()
                    logger.info(rendered.rstrip())

            def emit_user(text: str, *, initial: bool = False) -> None:
                display = "[initial task instructions submitted]" if initial else text
                output.write(f"\n[user] {display}\n")
                output.flush()
                self._dashboard_events.emit(
                    TranscriptEvent(
                        {"type": "initial_prompt"}
                        if initial
                        else {"type": "user", "text": text}
                    )
                )

            control = (
                DashboardPlannerControl(
                    interaction=interaction,
                    cancel_active_and_wait=toolkit.cancel_active_and_wait,
                    resume_calls=toolkit.resume_calls,
                    emit_user=emit_user,
                    emit_initial_user=lambda: emit_user(
                        initial_user_text, initial=True
                    ),
                    queue_until_complete=True,
                )
                if interaction is not None
                else None
            )
            try:
                options = self._build_options(
                    claude_agent_sdk,
                    toolkit=toolkit,
                    max_turns=max_turns,
                    on_tool_result=recorder.observe_tool_result,
                )
                driver = _ClaudeSessionDriver(
                    sdk=claude_agent_sdk,
                    options=options,
                    recorder=recorder,
                    emit=emit,
                    toolkit=toolkit,
                    control=control,
                )
                if terminal is not None:
                    reader = asyncio.create_task(terminal.read_input())
                # Human-supervised terminal sessions have no wall-clock cap.
                await asyncio.wait_for(
                    driver.run(prompt),
                    timeout=None if terminal is not None else self._timeout_s,
                )
            except asyncio.TimeoutError:
                error = f"Claude Agent SDK timed out after {self._timeout_s}s"
                write_jsonl(raw, {"type": "timeout", "message": error})
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                write_jsonl(raw, {"type": "error", "message": error})
            finally:
                if control is not None:
                    control.end()
                if reader is not None:
                    await reader
            if error:
                output.write(f"\n[cc-planner] {error}\n")
                logger.info("[cc-planner] %s", error)

        return artifacts.result(
            text="",
            finish_result=recorder.finish_result,
            final_response=None,
            stats=recorder.stats(),
            error=error or recorder.error,
        )

    # -- options + tool bridge ---------------------------------------------

    def _build_options(
        self,
        sdk: Any,
        *,
        toolkit: Toolkit,
        max_turns: int,
        on_tool_result: Callable[[str, ToolResult], None] | None = None,
    ) -> Any:
        allowed = [
            part for part in self._allowed_tools.replace(",", " ").split() if part
        ]
        builtins = [name for name in allowed if "__" not in name]
        allowed.extend(
            add_mcp_prefix(declaration.name) for declaration in toolkit.list_tools()
        )

        thinking = {"type": "disabled"} if self._reasoning_effort == "none" else None
        effort = None if self._reasoning_effort == "none" else self._reasoning_effort
        return sdk.ClaudeAgentOptions(
            cwd=self._repo_root,
            model=self._model,
            max_turns=max_turns,
            max_budget_usd=self._max_budget_usd,
            max_buffer_size=_MAX_STREAM_BUFFER_BYTES,
            tools=builtins or None,
            allowed_tools=list(dict.fromkeys(allowed)),
            mcp_servers={
                "rpent": _build_rpent_server(
                    sdk,
                    toolkit=toolkit,
                    on_tool_result=on_tool_result,
                ),
            },
            add_dirs=[self._output_dir, *self._extra_dirs],
            # Ignore user/project .claude configuration; RPent owns the loop.
            setting_sources=[],
            thinking=thinking,
            effort=effort,
            stderr=lambda line: logger.debug("[claude-sdk] %s", line.rstrip()),
        )


# ---------------------------------------------------------------------------
# SDK session driver
# ---------------------------------------------------------------------------


class _ClaudeSessionDriver:
    """Own one SDK client, queued turns, and its single message consumer."""

    def __init__(
        self,
        *,
        sdk: Any,
        options: Any,
        recorder: "_Recorder",
        emit: Callable[[Any], None],
        toolkit: Toolkit,
        control: DashboardPlannerControl | None = None,
    ) -> None:
        self._sdk = sdk
        self._options = options
        self._recorder = recorder
        self._emit = emit
        self._toolkit = toolkit
        self._control = control
        self._client: Any | None = None
        self._turn_done = asyncio.Event()
        self._turn_done.set()
        self._interrupting = False

    async def run(self, prompt: str) -> None:
        """Run one turn without input, or keep the same client for queued turns."""
        tasks: list[asyncio.Task[Any]] = []
        # Enter and exit the SDK context in the same task: its transport owns
        # an AnyIO task group. Drain tools before closing that transport.
        async with contextlib.AsyncExitStack() as stack:
            try:
                self._client = await stack.enter_async_context(
                    self._sdk.ClaudeSDKClient(options=self._options)
                )
                await self.query(prompt)
                if self._control is not None:
                    await self._control.start()
                tasks.append(asyncio.create_task(self._consume()))
                if self._control is not None:
                    tasks.append(asyncio.create_task(self._control.run(self)))
                done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                for task in done:
                    await task
            finally:
                if self._control is not None:
                    self._control.end()
                try:
                    await cancel_and_wait(self._toolkit.cancel_active_and_wait)
                finally:
                    for task in tasks:
                        task.cancel()
                    await asyncio.gather(*tasks, return_exceptions=True)
                    self._client = None

    async def query(self, text: str) -> None:
        """Submit one user turn after the previous turn has drained."""
        if not self._turn_done.is_set():
            raise RuntimeError("Cannot submit while a Claude turn is active")
        self._turn_done.clear()
        try:
            await self._client.query(text)
        except BaseException:
            self._turn_done.set()
            raise

    async def submit(self, message: DashboardMessage) -> int:
        """Submit queued input as a new Claude query."""
        await self.query(message.text)
        return 1

    async def interrupt(self) -> int:
        """Wait for the interrupted turn's result before accepting new input."""
        if self._turn_done.is_set():
            return 0
        self._interrupting = True
        self._recorder.suppress_next_result_error = True
        try:
            await self._client.interrupt()
            # The SDK acknowledgement can precede old tool request completion.
            await asyncio.wait_for(self._turn_done.wait(), timeout=15)
        except BaseException:
            self._recorder.suppress_next_result_error = False
            raise
        finally:
            self._interrupting = False
        return 1

    async def _consume(self) -> None:
        async for message in self._client.receive_messages():
            self._emit(message)
            completed = _kind(message) == "ResultMessage"
            if completed:
                self._turn_done.set()
            # A confirmed finish ends the task before queued input can flush.
            if self._recorder.finish_result is not None:
                return
            if completed and not self._interrupting:
                if self._control is None:
                    return
                await self._control.complete(self)


# ---------------------------------------------------------------------------
# Observation layer
# ---------------------------------------------------------------------------


@dataclass
class _Recorder:
    """Pure adapter: consume SDK messages, emit text + accumulate stats.

    Holds no backend state. Errors that the SDK itself reports become
    ``recorder.error``; transport-level errors are written beside the transcript.
    """

    max_turns: int
    dashboard_events: DashboardEventSink
    turns: int = 0
    _seen_assistant_ids: set[str] = field(default_factory=set)
    tool_calls: int = 0
    tool_names: dict[str, str] = field(default_factory=dict)
    usage: dict[str, int] = field(
        default_factory=lambda: {
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "total_cache_creation_input_tokens": 0,
            "total_cache_read_input_tokens": 0,
        }
    )
    total_cost_usd: float | None = None
    finish_result: dict[str, Any] | None = None
    error: str | None = None
    #: Set by the interactive loop before a user-initiated ``interrupt`` so the
    #: next result (which the CLI may flag ``is_error``) is not mistaken for a
    #: real failure. Cleared by the next result regardless of its outcome.
    suppress_next_result_error: bool = False

    # -- public ------------------------------------------------------------

    def stats(self) -> dict[str, int | float | None]:
        return {
            "turns_used": self.turns,
            "tool_calls": self.tool_calls,
            "total_cost_usd": self.total_cost_usd,
            **self.usage,
        }

    def observe_tool_result(self, name: str, result: ToolResult) -> None:
        """Accept only the toolkit's authoritative finish verdict."""
        if (
            name == "finish"
            and not result.is_error
            and result.data.get("_finish") is True
            and self.finish_result is None
        ):
            self.finish_result = dict(result.data)

    def observe(self, message: Any) -> str:
        kind = _kind(message)
        if kind == "SystemMessage":
            rendered = self._system(message)
        elif kind == "AssistantMessage":
            rendered = self._assistant(message)
        elif kind == "UserMessage":
            rendered = self._user(message)
        elif kind == "ResultMessage":
            rendered = self._result(message)
        else:
            rendered = ""
        self.dashboard_events.emit(
            UsageEvent(
                inp=self.usage["total_input_tokens"],
                out=self.usage["total_output_tokens"],
                tool_calls=self.tool_calls,
            )
        )
        return rendered

    # -- per-message handlers ---------------------------------------------

    def _system(self, message: Any) -> str:
        subtype = get_field(message, "subtype", "")
        if subtype == "thinking_tokens":
            return ""
        data = get_field(message, "data", {})
        session = data.get("session_id") if isinstance(data, dict) else ""
        return f"[cc-system] subtype={subtype} session={session}\n"

    def _assistant(self, message: Any) -> str:
        self._add_usage(get_field(message, "usage"))
        lines: list[str] = []
        if get_field(message, "parent_tool_use_id") is None:
            # One Claude response may arrive as multiple messages with the same ID.
            assistant_id = get_field(message, "message_id") or get_field(
                message, "uuid"
            )
            if not assistant_id or assistant_id not in self._seen_assistant_ids:
                if assistant_id:
                    self._seen_assistant_ids.add(str(assistant_id))
                self.turns += 1
                lines.append(f"\n[agent] === turn {self.turns}/{self.max_turns} ===\n")
        for block in get_field(message, "content", []) or []:
            block_kind = _kind(block)
            if block_kind == "TextBlock":
                text = str(get_field(block, "text", "")).strip()
                if text:
                    lines.append(f"[claude] {text}\n")
                    self.dashboard_events.emit(
                        TranscriptEvent({"type": "text", "text": text})
                    )
            elif block_kind == "ThinkingBlock":
                thinking = str(get_field(block, "thinking", "")).strip()
                if thinking:
                    lines.append(f"[claude-thinking] {thinking}\n")
                    self.dashboard_events.emit(
                        TranscriptEvent({"type": "thinking", "text": thinking})
                    )
            elif block_kind == "ToolUseBlock":
                tool_id = str(get_field(block, "id", ""))
                name = strip_mcp_prefix(str(get_field(block, "name", "tool")))
                self.tool_names[tool_id] = name
                tool_input = get_field(block, "input", {}) or {}
                lines.append(f"[tool->] {name}: {short_json(tool_input, limit=500)}\n")
                self.dashboard_events.emit(
                    TranscriptEvent(
                        {"type": "tool_call", "tool": name, "args": tool_input}
                    )
                )
            elif block_kind == "ToolResultBlock":
                lines.append(self._tool_result(block))
        if assistant_error := get_field(message, "error"):
            lines.append(f"[cc-assistant-error] {assistant_error}\n")
        return "".join(lines)

    def _user(self, message: Any) -> str:
        tool_use_id = get_field(message, "parent_tool_use_id")
        content = get_field(message, "content", "")
        if tool_use_id:
            return self._tool_result_content(content, tool_use_id=str(tool_use_id))
        if isinstance(content, list):
            return "".join(
                self._tool_result(block)
                for block in content
                if _kind(block) == "ToolResultBlock"
            )
        return ""

    def _tool_result(self, block: Any, *, tool_use_id: str | None = None) -> str:
        return self._tool_result_content(
            get_field(block, "content", ""),
            tool_use_id=tool_use_id or str(get_field(block, "tool_use_id", "")),
            is_error=get_field(block, "is_error"),
        )

    def _tool_result_content(
        self,
        content: Any,
        *,
        tool_use_id: str,
        is_error: Any = None,
    ) -> str:
        self.tool_calls += 1
        name = self.tool_names.get(tool_use_id, "tool_result")
        summary: dict[str, Any] = {"size": _payload_size(content)}
        image_count = _content_image_count(content)
        if image_count:
            summary["images"] = image_count
        if is_error:
            summary["is_error"] = bool(is_error)
        self.dashboard_events.emit(
            TranscriptEvent(
                {
                    "type": "tool_result",
                    "tool": name,
                    "result": {**summary, "is_error": bool(is_error)},
                }
            )
        )
        return f"[tool<-] {name}: {json.dumps(summary, ensure_ascii=False)}\n"

    def _result(self, message: Any) -> str:
        if usage := get_field(message, "usage"):
            self._set_usage(usage)
        if cost := get_field(message, "total_cost_usd"):
            self.total_cost_usd = float(cost)
        suppress = self.suppress_next_result_error
        self.suppress_next_result_error = False
        if get_field(message, "is_error", False) and not suppress:
            self.error = (
                f"Claude Agent SDK result {get_field(message, 'subtype', 'error')}"
            )

        parts = ["[cc-result]", str(get_field(message, "subtype", ""))]
        if duration_ms := get_field(message, "duration_ms"):
            parts.append(f"duration={duration_ms / 1000:.1f}s")
        if self.total_cost_usd is not None:
            parts.append(f"cost=${self.total_cost_usd:.4f}")
        if result := str(get_field(message, "result", "") or ""):
            parts.append(f"result_size={len(result)}")
        usage_line = (
            f"\n[usage] in={self.usage['total_input_tokens']} "
            f"cache_create={self.usage['total_cache_creation_input_tokens']} "
            f"cache_read={self.usage['total_cache_read_input_tokens']} "
            f"out={self.usage['total_output_tokens']} tool_calls={self.tool_calls}"
        )
        return " ".join(p for p in parts if p) + usage_line + "\n"

    # -- usage helpers ----------------------------------------------------

    def _add_usage(self, usage: Any) -> None:
        if not isinstance(usage, dict):
            return
        self.usage["total_input_tokens"] += int(usage.get("input_tokens") or 0)
        self.usage["total_output_tokens"] += int(usage.get("output_tokens") or 0)
        self.usage["total_cache_creation_input_tokens"] += int(
            usage.get("cache_creation_input_tokens") or 0
        )
        self.usage["total_cache_read_input_tokens"] += int(
            usage.get("cache_read_input_tokens") or 0
        )

    def _set_usage(self, usage: Any) -> None:
        if not isinstance(usage, dict):
            return
        self.usage = {
            "total_input_tokens": int(usage.get("input_tokens") or 0),
            "total_output_tokens": int(usage.get("output_tokens") or 0),
            "total_cache_creation_input_tokens": int(
                usage.get("cache_creation_input_tokens") or 0
            ),
            "total_cache_read_input_tokens": int(
                usage.get("cache_read_input_tokens") or 0
            ),
        }


# ---------------------------------------------------------------------------
# Tool bridge (RPent registry -> SDK MCP server)
# ---------------------------------------------------------------------------


def _build_rpent_server(
    sdk: Any,
    *,
    toolkit: Toolkit,
    on_tool_result: Callable[[str, ToolResult], None] | None = None,
) -> Any:
    sdk_tools = []
    for declaration in toolkit.list_tools():
        name = declaration.name
        description = declaration.description
        input_schema = substitute(declaration.input_schema)

        async def run_tool(
            args: dict[str, Any],
            *,
            tool_name: str = name,
        ) -> dict[str, Any]:
            result = await execute_tool(toolkit, tool_name, args or {})
            if on_tool_result is not None:
                on_tool_result(tool_name, result)
            return _tool_result_to_mcp(result)

        run_tool.__name__ = f"rpent_{name}"
        sdk_tools.append(sdk.tool(name, description, input_schema)(run_tool))

    return sdk.create_sdk_mcp_server(name="rpent", version="0.1.0", tools=sdk_tools)


def _tool_result_to_mcp(result: ToolResult) -> dict[str, Any]:
    """Render native tool data and images using the Claude SDK result shape."""
    response: dict[str, Any] = {"content": mcp_result(result)["content"]}
    if result.is_error:
        response["is_error"] = True
    return response


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _kind(value: Any) -> str:
    if isinstance(value, dict):
        return str(value.get("type") or value.get("kind") or "")
    return value.__class__.__name__


def _message_to_json(message: Any) -> dict[str, Any]:
    if isinstance(message, dict) or dataclasses.is_dataclass(message):
        data = message
    elif hasattr(message, "__dict__"):
        data = vars(message)
    else:
        data = {"value": repr(message)}
    return {"type": _kind(message), **jsonable(data)}


def _payload_size(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, str):
        return len(value)
    return len(json.dumps(value, ensure_ascii=False, default=str))


def _content_image_count(value: Any) -> int:
    if isinstance(value, list):
        count = 0
        for item in value:
            if isinstance(item, dict) and item.get("type") == "image":
                count += 1
        return count
    text = str(value)
    return text.count("'type': 'image'") + text.count('"type": "image"')
