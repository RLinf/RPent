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
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rpent.cli.tui import next_user_line
from rpent.dashboard.events import (
    DashboardEventSink,
    TranscriptEvent,
    UsageEvent,
)
from rpent.dashboard.interaction import DashboardInteractionPort
from rpent.dashboard.planner_control import DashboardPlannerControl
from rpent.planner.base import (
    REASONING_EFFORTS,
    PlannerResult,
    add_mcp_prefix,
    strip_mcp_prefix,
)
from rpent.tools.toolkit import Toolkit
from rpent.utils.config import get_repo_root
from rpent.utils.logging import get_logger, init_output_dir

logger = get_logger("claude")

_MAX_STREAM_BUFFER_BYTES = 8 * 1024 * 1024

# ---------------------------------------------------------------------------
# Public backend
# ---------------------------------------------------------------------------


class ClaudeCodePlanner:
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

        sdk = claude_agent_sdk
        if self._output_path is None:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".out", prefix="claude_agent_task_", delete=False
            ) as f:
                output_path = Path(f.name)
        else:
            output_path = self._output_path
            output_path.parent.mkdir(parents=True, exist_ok=True)
        raw_stream_path = output_path.with_suffix(output_path.suffix + ".stream.jsonl")
        recorder = _Recorder(
            max_turns=max_turns,
            dashboard_events=self._dashboard_events,
        )

        init_output_dir(self._output_dir)
        options = self._build_options(sdk, toolkit=toolkit, max_turns=max_turns)

        logger.info("prompt: %d chars", len(prompt))
        logger.info("output_dir: %s", self._output_dir)
        logger.info(
            "invoking Claude Agent SDK model %s (timeout=%ds, budget=$%s)",
            self._model,
            self._timeout_s,
            self._max_budget_usd,
        )

        started = time.time()
        error: str | None = None
        rendered_chunks: list[str] = []
        with open(output_path, "w") as out_f, open(raw_stream_path, "w") as raw_f:

            def _emit(message: Any) -> None:
                _write_jsonl(raw_f, _message_to_json(message))
                if rendered := recorder.observe(message):
                    rendered_chunks.append(rendered)
                    out_f.write(rendered)
                    out_f.flush()
                    logger.info(rendered.rstrip())

            def _emit_user(line: str, *, initial_prompt: bool = False) -> None:
                display_text = (
                    "[initial task instructions submitted]" if initial_prompt else line
                )
                rendered = f"\n[user] {display_text}\n"
                rendered_chunks.append(rendered)
                out_f.write(rendered)
                out_f.flush()
                logger.info(rendered.strip())
                payload: dict[str, Any] = {"type": "user", "text": line}
                if initial_prompt:
                    payload = {"type": "initial_prompt"}
                self._dashboard_events.emit(TranscriptEvent(payload))

            try:
                if input_queue is None:
                    if dashboard_interaction is not None:
                        await asyncio.wait_for(
                            self._run_dashboard_session(
                                sdk,
                                prompt,
                                options,
                                recorder,
                                toolkit=toolkit,
                                dashboard_interaction=dashboard_interaction,
                                initial_user_text=initial_user_text,
                                emit=_emit,
                                emit_user=_emit_user,
                            ),
                            timeout=self._timeout_s,
                        )
                    else:

                        async def consume_stream() -> None:
                            async for message in sdk.query(
                                prompt=prompt, options=options
                            ):
                                _emit(message)
                                if recorder.finish_result is not None:
                                    logger.info(
                                        "FINISH called: %s", recorder.finish_result
                                    )
                                    break

                        await asyncio.wait_for(
                            consume_stream(), timeout=self._timeout_s
                        )
                else:
                    await self._run_interactive(
                        sdk,
                        prompt,
                        options,
                        recorder,
                        input_queue,
                        emit=_emit,
                        emit_user=_emit_user,
                    )
            except asyncio.TimeoutError:
                error = f"Claude Agent SDK timed out after {self._timeout_s}s"
                try:
                    await asyncio.to_thread(toolkit.cancel_active_and_wait)
                except Exception as cancel_error:
                    logger.warning(
                        "failed to cancel toolkit work after Claude timeout: %s",
                        cancel_error,
                    )
                rendered = f"\n[cc-planner] {error}\n"
                rendered_chunks.append(rendered)
                out_f.write(rendered)
                out_f.flush()
                _write_jsonl(raw_f, {"type": "timeout", "message": error})
                logger.info(rendered.rstrip())
            except Exception as e:
                error = f"{type(e).__name__}: {e}"
                rendered = f"\n[cc-planner] {error}\n"
                rendered_chunks.append(rendered)
                out_f.write(rendered)
                out_f.flush()
                _write_jsonl(raw_f, {"type": "error", "message": error})
                logger.info(rendered.rstrip())

        elapsed = time.time() - started
        text = "".join(rendered_chunks) or output_path.read_text(errors="replace")
        error = error or recorder.error

        logger.info("Claude Agent SDK finished in %.1fs", elapsed)
        logger.info("output: %s", output_path)
        logger.info("raw stream: %s", raw_stream_path)

        return PlannerResult(
            finish_result=recorder.finish_result,
            messages=[{"role": "claude_agent_sdk", "content": text}],
            stats={
                "backend": "claude_agent_sdk",
                "elapsed_s": round(elapsed, 1),
                "output_chars": len(text),
                "output_path": str(output_path),
                "raw_stream_path": str(raw_stream_path),
                **recorder.stats(),
            },
            error=error,
        )

    async def _run_interactive(
        self,
        sdk: Any,
        prompt: str,
        options: Any,
        recorder: "_Recorder",
        input_queue,
        *,
        emit,
        emit_user,
    ) -> None:
        """Drive a stateful ``ClaudeSDKClient`` with live steering.

        The opening prompt is sent, then the agent streams autonomously while a
        background pump forwards each user-typed line into the same session via
        ``client.query`` so it steers the agent at its next turn. A quit/EOF
        sentinel (or ``/quit``) interrupts the run; the ``finish`` tool ends it
        normally. Because a human supervises, there is no wall-clock cap here.
        """
        adapter = _TerminalSessionAdapter(input_queue=input_queue, emit_user=emit_user)
        driver = _ClaudeSessionDriver(
            sdk=sdk,
            options=options,
            recorder=recorder,
            emit=emit,
        )
        await driver.run(prompt, adapter)

    async def _run_dashboard_session(
        self,
        sdk: Any,
        prompt: str,
        options: Any,
        recorder: "_Recorder",
        *,
        toolkit: Toolkit,
        dashboard_interaction: DashboardInteractionPort,
        initial_user_text: str,
        emit,
        emit_user,
    ) -> None:
        """Drive the Dashboard-owned long-lived Claude session."""
        adapter = _ClaudeDashboardAdapter(
            interaction=dashboard_interaction,
            cancel_active_and_wait=toolkit.cancel_active_and_wait,
            emit_user=emit_user,
            emit_initial_user=lambda: emit_user(
                initial_user_text,
                initial_prompt=True,
            ),
        )
        driver = _ClaudeSessionDriver(
            sdk=sdk,
            options=options,
            recorder=recorder,
            emit=emit,
        )
        await driver.run(prompt, adapter)

    # -- options + tool bridge ---------------------------------------------

    def _build_options(self, sdk: Any, *, toolkit: Toolkit, max_turns: int) -> Any:
        allowed = [
            part for part in self._allowed_tools.replace(",", " ").split() if part
        ]
        builtins = [name for name in allowed if "__" not in name]
        allowed.extend(
            add_mcp_prefix(str(spec["name"])) for spec in toolkit.get_tools_spec()
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
# Shared long-lived session driver
# ---------------------------------------------------------------------------


class _ClaudeSessionDriver:
    """Own one SDK client and its single message consumer."""

    def __init__(
        self,
        *,
        sdk: Any,
        options: Any,
        recorder: "_Recorder",
        emit: Callable[[Any], None],
    ) -> None:
        self._sdk = sdk
        self._options = options
        self._recorder = recorder
        self._emit = emit
        self._client: Any | None = None

    async def run(self, prompt: str, adapter: Any) -> None:
        """Open one client, submit the first query, and run one input adapter."""
        tasks: list[asyncio.Task[Any]] = []
        async with self._sdk.ClaudeSDKClient(options=self._options) as client:
            self._client = client
            try:
                # Submit before starting the consumer, matching the SDK's
                # streaming-client contract while its transport buffers output.
                await self.query(prompt)
                await adapter.initial_query_succeeded(self)

                consumer = asyncio.create_task(self._consume(adapter))
                command_pump = asyncio.create_task(adapter.run(self))
                tasks = [consumer, command_pump]
                done, _ = await asyncio.wait(
                    tasks,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                for task in done:
                    await task
            finally:
                # Adapter cleanup first wakes terminal queue reads or Dashboard
                # condition waits. Then cancel and drain both long-lived tasks
                # before the SDK client context closes.
                with contextlib.suppress(Exception):
                    await adapter.close()
                for task in tasks:
                    task.cancel()
                for task in tasks:
                    with contextlib.suppress(asyncio.CancelledError, Exception):
                        await task
                self._client = None

    async def query(self, text: str) -> None:
        """Submit one user turn to the owned client."""
        if self._client is None:
            raise RuntimeError("Claude session is not connected")
        await self._client.query(text)

    async def submit(self, text: str) -> int:
        """Submit Dashboard input as a new Claude query."""
        await self.query(text)
        return 1

    async def interrupt(self) -> int:
        """Interrupt the owned client and suppress its expected error result."""
        if self._client is None:
            raise RuntimeError("Claude session is not connected")
        self._recorder.suppress_next_result_error = True
        try:
            await self._client.interrupt()
        except BaseException:
            # A failed interrupt must not hide a later unrelated SDK error.
            self._recorder.suppress_next_result_error = False
            raise
        # Claude emits a ResultMessage for the interrupted query, so the
        # matching completion is accounted for by the normal message path.
        return 0

    async def _consume(self, adapter: Any) -> None:
        if self._client is None:
            raise RuntimeError("Claude session is not connected")
        async for message in self._client.receive_messages():
            self._emit(message)
            # A successful finish tool result owns the boundary: end the
            # session without giving queued Dashboard input a chance to flush.
            if self._recorder.finish_result is not None:
                logger.info("FINISH called: %s", self._recorder.finish_result)
                return
            await adapter.on_message(self, message)


class _TerminalSessionAdapter:
    """Preserve the terminal TUI's interrupt-then-query steering policy."""

    def __init__(self, *, input_queue: Any, emit_user) -> None:
        self._input_queue = input_queue
        self._emit_user = emit_user

    async def initial_query_succeeded(self, driver: _ClaudeSessionDriver) -> None:
        return None

    async def run(self, driver: _ClaudeSessionDriver) -> None:
        while True:
            nxt = await asyncio.to_thread(next_user_line, self._input_queue)
            if nxt is None:
                with contextlib.suppress(Exception):
                    await driver.interrupt()
                return
            self._emit_user(nxt)
            # Keep the current terminal semantics: every steering line
            # interrupts the in-flight turn, then enters the same session.
            with contextlib.suppress(Exception):
                await driver.interrupt()
            with contextlib.suppress(Exception):
                await driver.query(nxt)

    async def on_message(
        self,
        driver: _ClaudeSessionDriver,
        message: Any,
    ) -> None:
        return None

    async def close(self) -> None:
        # Unblock next_user_line() if the consumer (for example finish) won.
        self._input_queue.put(None)


class _ClaudeDashboardAdapter:
    """Translate Claude SDK lifecycle events into shared control boundaries."""

    def __init__(
        self,
        *,
        interaction: DashboardInteractionPort,
        cancel_active_and_wait: Callable[[], None],
        emit_user: Callable[[str], None],
        emit_initial_user: Callable[[], None],
    ) -> None:
        self._control = DashboardPlannerControl(
            interaction=interaction,
            cancel_active_and_wait=cancel_active_and_wait,
            emit_user=emit_user,
            emit_initial_user=emit_initial_user,
        )

    async def initial_query_succeeded(self, driver: _ClaudeSessionDriver) -> None:
        await self._control.start()

    async def run(self, driver: _ClaudeSessionDriver) -> None:
        await self._control.run(driver)

    async def on_message(
        self,
        driver: _ClaudeSessionDriver,
        message: Any,
    ) -> None:
        if _kind(message) == "ResultMessage":
            await self._control.complete(driver)
        elif _has_tool_result(message):
            await self._control.tool_completed(driver)

    async def close(self) -> None:
        try:
            await self._control.cancel_active_toolkit()
        finally:
            self._control.end()


def _has_tool_result(message: Any) -> bool:
    kind = _kind(message)
    if kind == "UserMessage" and _get(message, "parent_tool_use_id"):
        return True
    if kind not in {"AssistantMessage", "UserMessage"}:
        return False
    content = _get(message, "content", [])
    return isinstance(content, list) and any(
        _kind(block) in {"ToolResultBlock", "tool_result"} for block in content
    )


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
    pending_finish: dict[str, dict[str, Any]] = field(default_factory=dict)
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
        subtype = _get(message, "subtype", "")
        if subtype == "thinking_tokens":
            return ""
        data = _get(message, "data", {})
        session = data.get("session_id") if isinstance(data, dict) else ""
        return f"[cc-system] subtype={subtype} session={session}\n"

    def _assistant(self, message: Any) -> str:
        self._add_usage(_get(message, "usage"))
        lines: list[str] = []
        if _get(message, "parent_tool_use_id") is None:
            # One Claude response may arrive as multiple messages with the same ID.
            assistant_id = _get(message, "message_id") or _get(message, "uuid")
            if not assistant_id or assistant_id not in self._seen_assistant_ids:
                if assistant_id:
                    self._seen_assistant_ids.add(str(assistant_id))
                self.turns += 1
                lines.append(f"\n[agent] === turn {self.turns}/{self.max_turns} ===\n")
        for block in _get(message, "content", []) or []:
            block_kind = _kind(block)
            if block_kind == "TextBlock":
                text = str(_get(block, "text", "")).strip()
                if text:
                    lines.append(f"[claude] {text}\n")
                    self.dashboard_events.emit(
                        TranscriptEvent({"type": "text", "text": text})
                    )
            elif block_kind == "ThinkingBlock":
                thinking = str(_get(block, "thinking", "")).strip()
                if thinking:
                    lines.append(f"[claude-thinking] {thinking}\n")
                    self.dashboard_events.emit(
                        TranscriptEvent({"type": "thinking", "text": thinking})
                    )
            elif block_kind == "ToolUseBlock":
                tool_id = str(_get(block, "id", ""))
                name = strip_mcp_prefix(str(_get(block, "name", "tool")))
                self.tool_names[tool_id] = name
                tool_input = _get(block, "input", {}) or {}
                if name == "finish" and isinstance(tool_input, dict):
                    self.pending_finish[tool_id] = dict(tool_input)
                lines.append(f"[tool->] {name}: {_short_json(tool_input, limit=500)}\n")
                self.dashboard_events.emit(
                    TranscriptEvent(
                        {"type": "tool_call", "tool": name, "args": tool_input}
                    )
                )
            elif block_kind == "ToolResultBlock":
                lines.append(self._tool_result(block))
        if assistant_error := _get(message, "error"):
            lines.append(f"[cc-assistant-error] {assistant_error}\n")
        return "".join(lines)

    def _user(self, message: Any) -> str:
        tool_use_id = _get(message, "parent_tool_use_id")
        content = _get(message, "content", "")
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
            _get(block, "content", ""),
            tool_use_id=tool_use_id or str(_get(block, "tool_use_id", "")),
            is_error=_get(block, "is_error"),
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
        # Promote the finish payload once the tool result lands successfully.
        pending = self.pending_finish.pop(tool_use_id, None)
        if pending is not None and not is_error and self.finish_result is None:
            self.finish_result = {"_finish": True, **pending}
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
        if usage := _get(message, "usage"):
            self._set_usage(usage)
        if cost := _get(message, "total_cost_usd"):
            self.total_cost_usd = float(cost)
        suppress = self.suppress_next_result_error
        self.suppress_next_result_error = False
        if _get(message, "is_error", False) and not suppress:
            self.error = f"Claude Agent SDK result {_get(message, 'subtype', 'error')}"

        parts = ["[cc-result]", str(_get(message, "subtype", ""))]
        if duration_ms := _get(message, "duration_ms"):
            parts.append(f"duration={duration_ms / 1000:.1f}s")
        if self.total_cost_usd is not None:
            parts.append(f"cost=${self.total_cost_usd:.4f}")
        if result := str(_get(message, "result", "") or ""):
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


def _build_rpent_server(sdk: Any, *, toolkit: Toolkit) -> Any:
    sdk_tools = []
    tool_execution_lock = asyncio.Lock()
    for spec in toolkit.get_tools_spec():
        name = str(spec["name"])
        description = str(spec.get("description", ""))
        input_schema = spec.get("input_schema", {"type": "object"})

        async def run_tool(
            args: dict[str, Any],
            *,
            tool_name: str = name,
        ) -> dict[str, Any]:
            async with tool_execution_lock:
                result = await asyncio.to_thread(
                    toolkit.execute_tool,
                    tool_name,
                    args or {},
                )
            return _tool_result_to_mcp(result)

        run_tool.__name__ = f"rpent_{name}"
        sdk_tools.append(sdk.tool(name, description, input_schema)(run_tool))

    return sdk.create_sdk_mcp_server(name="rpent", version="0.1.0", tools=sdk_tools)


def _tool_result_to_mcp(tr: Any) -> dict[str, Any]:
    # The toolkit already formatted the result into Anthropic content blocks;
    # translate those into the MCP content shape (text + image).
    blocks = getattr(tr, "content_blocks", None)
    if blocks is None:
        return {"content": [{"type": "text", "text": str(tr)}]}

    content: list[dict[str, Any]] = []
    for block in blocks:
        block_type = _get(block, "type")
        if block_type == "text":
            content.append({"type": "text", "text": _get(block, "text", "")})
        elif block_type == "image":
            src = _get(block, "source", {})
            content.append(
                {
                    "type": "image",
                    "data": _get(src, "data", ""),
                    "mimeType": _get(src, "media_type", "image/png"),
                }
            )

    response: dict[str, Any] = {"content": content}
    # Surface toolkit-level failures as MCP errors. Without this every result
    # looks successful to the SDK, and `_Recorder` promotes a `finish` call to
    # the run's finish_result even when the handler rejected it.
    result_dict = getattr(tr, "result", None)
    if isinstance(result_dict, dict) and result_dict.get("error"):
        response["is_error"] = True
    return response


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _kind(value: Any) -> str:
    if isinstance(value, dict):
        return str(value.get("type") or value.get("kind") or "")
    return value.__class__.__name__


def _get(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(key, default)
    return getattr(value, key, default)


def _message_to_json(message: Any) -> dict[str, Any]:
    if dataclasses.is_dataclass(message):
        data = dataclasses.asdict(message)
    elif hasattr(message, "__dict__"):
        data = vars(message)
    else:
        data = {"value": repr(message)}
    return {"type": _kind(message), **_jsonable(data)}


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, bytes):
        return {"type": "bytes", "size": len(value)}
    return value


def _write_jsonl(file_obj, value: dict[str, Any]) -> None:
    file_obj.write(json.dumps(value, ensure_ascii=False, default=str) + "\n")
    file_obj.flush()


def _short_json(value: Any, *, limit: int) -> str:
    text = json.dumps(value, ensure_ascii=False, default=str)
    if len(text) <= limit:
        return text
    return text[:limit] + f"...(+{len(text) - limit})"


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
