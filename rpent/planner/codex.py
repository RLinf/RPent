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

"""Codex SDK configuration, sessions, and event conversion.

RPent tools are exposed through the shared HTTP MCP server. The agent runtime
owns session cleanup, artifact bookkeeping, and queued interaction; this
adapter owns Codex SDK sessions and event conversion.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import queue
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import openai_codex
from openai_codex.generated.v2_all import ReasoningEffort

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
    strip_mcp_prefix,
)
from rpent.planner.utils.agent_runtime import (
    AgentArtifacts,
    agent_environment,
    short_json,
)
from rpent.planner.utils.agent_session import run_agent
from rpent.planner.utils.sdk import get_field, int_field, json_object, jsonable, unwrap
from rpent.tools.toolkit import Toolkit
from rpent.utils.config import get_repo_root

PROVIDER_ID = "rpent_proxy"
PROVIDER_ENV_KEY = "RPENT_CODEX_PROVIDER_KEY"


# ---------------------------------------------------------------------------
# Public backend
# ---------------------------------------------------------------------------


class CodexPlanner(Planner):
    """Planner backed by the OpenAI Codex Python SDK."""

    def __init__(
        self,
        *,
        output_dir: str,
        dashboard_events: DashboardEventSink,
        repo_root: str | Path | None = None,
        timeout_s: int = 600,
        extra_dirs: list[str] | None = None,
        output_path: str | Path | None = None,
        model: str | None = None,
        reasoning_effort: str = "none",
    ):
        """Initialize the Codex SDK backend."""
        self._output_dir = str(output_dir)
        self._repo_root = str(repo_root) if repo_root else str(get_repo_root())
        self._timeout_s = timeout_s
        self._extra_dirs = extra_dirs or []
        self._output_path = Path(output_path) if output_path else None
        self._model = model or os.environ.get("CODEX_MODEL", None)
        self._base_url = os.environ.get("CODEX_BASE_URL", None)
        self._api_key = os.environ.get("CODEX_API_KEY", None)
        self._service_tier = os.environ.get("CODEX_SERVICE_TIER", None)
        self._dashboard_events = dashboard_events
        if reasoning_effort not in REASONING_EFFORTS:
            raise ValueError(f"unsupported reasoning effort: {reasoning_effort}")
        self._thread_options = {
            "approval_mode": openai_codex.ApprovalMode.deny_all,
            "cwd": self._repo_root,
            "model": self._model,
            "sandbox": openai_codex.Sandbox.full_access,
        }
        if self._service_tier:
            self._thread_options["service_tier"] = self._service_tier
        self._turn_options = {
            **self._thread_options,
            "effort": getattr(ReasoningEffort, reasoning_effort),
        }

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
        """Run one or more Codex SDK turns for the given prompt."""
        if input_queue is not None and dashboard_interaction is not None:
            raise ValueError(
                "input_queue and dashboard_interaction cannot be used together"
            )
        prompt = f"{system_prompt}\n\n{user_message}" if system_prompt else user_message
        return asyncio.run(
            self._solve(
                prompt=prompt,
                initial_user_text=user_message,
                toolkit=toolkit,
                max_turns=max_turns,
                interaction=dashboard_interaction,
                input_queue=input_queue,
            )
        )

    async def _solve(
        self,
        *,
        prompt: str,
        initial_user_text: str,
        toolkit: Toolkit,
        max_turns: int,
        interaction: DashboardInteractionPort | None,
        input_queue: queue.Queue[str | None] | None,
    ) -> PlannerResult:
        """Run one SDK session, optionally accepting queued follow-up input."""
        artifacts = AgentArtifacts("codex_sdk", self._output_path)
        recorder = _Recorder(
            max_turns=max_turns, dashboard_events=self._dashboard_events
        )

        def create_session(url, control, emit):
            return _CodexSession(
                config=self._build_config(url),
                thread_options=self._thread_options,
                turn_options=self._turn_options,
                recorder=recorder,
                emit_event=lambda event: emit(_message_to_json(event)),
                control=control,
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
            label="Codex SDK",
            input_queue=input_queue,
            interaction=interaction,
        )
        last_message_path = artifacts.sidecar(".last")
        if recorder.final_response is not None:
            last_message_path.write_text(recorder.final_response)
        return artifacts.result(
            text="",
            finish_result=recorder.finish_result,
            final_response=recorder.final_response,
            stats=recorder.stats(),
            error=error or recorder.error,
            last_message_path=last_message_path,
        )

    # -- config builder ----------------------------------------------------

    def _build_config(self, mcp_url: str) -> Any:
        return build_codex_config(
            mcp_url=mcp_url,
            base_url=self._base_url,
            api_key=self._api_key,
            cwd=self._repo_root,
        )


class _CodexSession:
    """Own one Codex thread and its current interruptible turn."""

    def __init__(
        self,
        *,
        config: Any,
        thread_options: dict[str, Any],
        turn_options: dict[str, Any],
        recorder: "_Recorder",
        emit_event,
        control: DashboardPlannerControl | None,
    ) -> None:
        self._config = config
        self._thread_options = thread_options
        self._turn_options = turn_options
        self._recorder = recorder
        self._emit_event = emit_event
        self._control = control
        self._codex: Any | None = None
        self._thread: Any | None = None
        self._turn: Any | None = None
        self._turn_done: asyncio.Event | None = None
        self._turn_task: asyncio.Task[Any] | None = None
        self._closing = False
        self.error: str | None = None

    async def run(self, prompt: str) -> None:
        self._codex = openai_codex.AsyncCodex(self._config)
        self._thread = await self._codex.thread_start(**self._thread_options)
        await self._submit_text(prompt)
        if self._control is None:
            await asyncio.shield(self._turn_task)
        else:
            await self._control.start()
            await self._control.run(self)

    async def submit(self, message: DashboardMessage) -> int:
        """Start a queued turn in the same conversation."""
        return await self._submit_text(message.text)

    async def _submit_text(self, text: str) -> int:
        if self._closing or self._thread is None:
            raise RuntimeError("Codex conversation is closed")
        if self._turn is not None:
            raise RuntimeError("Cannot submit while a Codex turn is active")
        self._turn = await self._thread.turn(text, **self._turn_options)
        self._turn_done = asyncio.Event()
        self._turn_task = asyncio.create_task(
            self._consume_turn(self._turn, self._turn_done)
        )
        return 1

    async def interrupt(self) -> int:
        turn = self._turn
        done = self._turn_done
        if turn is None or done is None:
            return 0
        await turn.interrupt()
        try:
            await asyncio.wait_for(done.wait(), timeout=15)
        except asyncio.TimeoutError:
            raise RuntimeError("Codex did not finish the interrupted turn") from None
        # ``_consume_turn`` reports the matching completed turn boundary.
        return 0

    async def close(self) -> None:
        if self._closing:
            return
        self._closing = True
        turn = self._turn
        done = self._turn_done
        if turn is not None and done is not None:
            with contextlib.suppress(Exception):
                await turn.interrupt()
            try:
                await asyncio.wait_for(done.wait(), timeout=15)
            except asyncio.TimeoutError:
                if self._turn_task is not None:
                    self._turn_task.cancel()
        if self._turn_task is not None:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._turn_task
        if self._codex is not None:
            with contextlib.suppress(Exception):
                await self._codex.close()

    async def _consume_turn(self, turn: Any, done: asyncio.Event) -> None:
        limit_reached = False
        try:
            async for event in turn.stream():
                self._emit_event(event)
                method = str(get_field(event, "method", ""))
                payload = get_field(event, "payload")

                if (
                    method != "turn/completed"
                    and not limit_reached
                    and self._recorder.finish_result is None
                    and self._recorder.turns >= self._recorder.max_turns
                ):
                    limit_reached = True
                    try:
                        await turn.interrupt()
                    except openai_codex.JsonRpcError as exc:
                        if exc.message != "no active turn to interrupt":
                            raise
                        # Keep consuming the stream to get the actual turn status.

                if method != "turn/completed":
                    continue
                status = _status(get_field(payload, "turn"))
                self._turn = None
                self._turn_done = None
                done.set()
                if self._closing:
                    return
                if status == "failed":
                    self.error = self._recorder.error or "Codex turn failed"
                if self._control is None:
                    return
                if (
                    self.error is not None
                    or self._recorder.finish_result is not None
                    or self._recorder.turns >= self._recorder.max_turns
                ):
                    self._control.end()
                else:
                    await self._control.complete(self)
                return
            raise RuntimeError("Codex turn stream ended without turn/completed")
        except Exception as exc:
            self._turn = None
            self._turn_done = None
            done.set()
            self.error = f"{type(exc).__name__}: {exc}"
            if not self._closing and self._control is not None:
                self._control.end()


# ---------------------------------------------------------------------------
# Observation layer
# ---------------------------------------------------------------------------


@dataclass
class _Recorder:
    """Pure adapter: consume Codex SDK events, emit text + accumulate stats."""

    max_turns: int
    dashboard_events: DashboardEventSink
    turns: int = 0
    _seen_usage: set[tuple[int, ...]] = field(default_factory=set)
    tool_calls: int = 0
    usage: dict[str, int] = field(
        default_factory=lambda: {
            "total_input_tokens": 0,
            "total_cached_input_tokens": 0,
            "total_output_tokens": 0,
            "total_reasoning_output_tokens": 0,
        }
    )
    final_response: str | None = None
    finish_result: dict[str, Any] | None = None
    error: str | None = None

    def stats(self) -> dict[str, int]:
        return {"turns_used": self.turns, "tool_calls": self.tool_calls, **self.usage}

    def observe(self, event: Any) -> str:
        method = str(get_field(event, "method", ""))
        payload = get_field(event, "payload")

        if method in {"thread/started", "turn/started"}:
            return f"[codex-system] {method}\n"
        if method == "item/completed":
            return self._render_item(get_field(payload, "item"))
        if method == "thread/tokenUsage/updated":
            if self._set_usage(get_field(payload, "token_usage")):
                return f"\n[agent] === turn {self.turns}/{self.max_turns} ===\n"
            return ""
        if method == "turn/completed":
            return self._render_turn_completed(get_field(payload, "turn"))
        if "requestApproval" in method:
            return f"[codex-approval] {method}\n"
        if method in {"error", "fatal"}:
            detail = short_json(jsonable(payload), limit=500)
            retrying = bool(get_field(payload, "will_retry", False))
            if not retrying:
                self.error = detail
            label = (
                "Model connection retrying" if retrying else "Model connection failed"
            )
            self.dashboard_events.emit(
                TranscriptEvent({"type": "text", "text": f"[{label}] {detail}"})
            )
            return f"[codex-error] {detail}\n"
        return ""

    # -- per-item handlers -------------------------------------------------

    def _render_item(self, item: Any) -> str:
        item = unwrap(item)
        item_type = str(get_field(item, "type", ""))

        if item_type == "userMessage":
            text = _extract_text(get_field(item, "content"))
            return f"\n[codex][user] {text}\n" if text else ""

        if item_type in {"hookPrompt", "plan"}:
            return ""

        if item_type == "agentMessage":
            text = str(get_field(item, "text", "")).strip()
            if not text:
                return ""
            self.final_response = text
            self.dashboard_events.emit(TranscriptEvent({"type": "text", "text": text}))
            return f"\n[codex] {text}\n"

        if item_type == "reasoning":
            text = _extract_text(
                get_field(item, "summary") or get_field(item, "content")
            )
            if text:
                self.dashboard_events.emit(
                    TranscriptEvent({"type": "thinking", "text": text})
                )
            return f"[codex-reasoning] {text}\n" if text else ""

        if _is_tool_item(item):
            self.tool_calls += 1
            if item_type in {"mcpToolCall", "dynamicToolCall"}:
                name = strip_mcp_prefix(str(get_field(item, "tool", item_type)))
                self._maybe_capture_finish(name, item)
            elif item_type == "commandExecution":
                name = str(get_field(item, "command", item_type))
            else:
                name = "fileChange"
            payload = _summarise_item(item)
            data = jsonable(item)
            args = data.get("arguments", {}) if isinstance(data, dict) else {}
            self.dashboard_events.emit(
                TranscriptEvent({"type": "tool_call", "tool": name, "args": args})
            )
            self.dashboard_events.emit(
                TranscriptEvent(
                    {"type": "tool_result", "tool": name, "result": payload}
                )
            )
            return f"[tool<-] {name}: {json.dumps(payload, ensure_ascii=False)}\n"

        return ""

    def _render_turn_completed(self, turn: Any) -> str:
        status = str(
            get_field(get_field(turn, "status"), "value", get_field(turn, "status", ""))
        )
        duration_ms = get_field(turn, "duration_ms")
        if error := get_field(turn, "error"):
            self.error = str(get_field(error, "message", str(error)))

        parts = ["[codex-result]", status]
        if duration_ms is not None:
            parts.append(f"duration={float(duration_ms) / 1000:.1f}s")
        usage_line = (
            f"\n[usage] in={self.usage['total_input_tokens']} "
            f"cached={self.usage['total_cached_input_tokens']} "
            f"out={self.usage['total_output_tokens']} "
            f"reasoning={self.usage['total_reasoning_output_tokens']} "
            f"tool_calls={self.tool_calls}"
        )
        return " ".join(p for p in parts if p) + usage_line + "\n"

    # -- helpers -----------------------------------------------------------

    def _set_usage(self, usage: Any) -> bool:
        """Count completed model responses, including reasoning/tool-only ones.

        The SDK updates cumulative token usage after each model response. Text
        and tool items within that response do not consume additional turns.
        Repeated notifications (including context-window-only updates) do not
        count again or overwrite newer usage totals.
        """
        if usage is None:
            return False
        total = get_field(usage, "total", usage)
        updated = {
            "total_input_tokens": int_field(total, "input_tokens"),
            "total_cached_input_tokens": int_field(total, "cached_input_tokens"),
            "total_output_tokens": int_field(total, "output_tokens"),
            "total_reasoning_output_tokens": int_field(
                total, "reasoning_output_tokens"
            ),
        }
        key = tuple(updated.values())
        if not any(key) or key in self._seen_usage:
            return False
        self._seen_usage.add(key)
        self.usage = updated
        self.turns += 1
        self.dashboard_events.emit(
            UsageEvent(
                inp=self.usage["total_input_tokens"],
                out=self.usage["total_output_tokens"],
                tool_calls=self.tool_calls,
            )
        )
        return True

    def _maybe_capture_finish(self, name: str, item: Any) -> None:
        if self.finish_result is not None:
            return
        if name.lower() != "finish":
            return
        status = _status(item)
        if status and status != "completed":
            return
        if get_field(item, "error") not in (None, ""):
            return
        data = jsonable(item)
        if not isinstance(data, dict):
            return
        result = data.get("result")
        if isinstance(result, dict) and ("isError" in result or "content" in result):
            if result.get("isError") is True:
                return
            content = result.get("content")
            if isinstance(content, list):
                for block in content:
                    if not isinstance(block, dict) or block.get("type") != "text":
                        continue
                    payload = json_object(block.get("text"))
                    if isinstance(payload, dict) and payload.get("_finish") is True:
                        self.finish_result = dict(payload)
                        return
            return
        payload = json_object(result)
        if isinstance(result, dict) or payload is not None:
            if payload is not None and payload.get("_finish") is True:
                self.finish_result = dict(payload)
            return
        args = json_object(data.get("arguments"))
        if isinstance(args, dict):
            self.finish_result = {"_finish": True, **args}


# ---------------------------------------------------------------------------
# Codex config overrides
# ---------------------------------------------------------------------------


def build_codex_config(
    *,
    mcp_url: str | None,
    base_url: str | None,
    api_key: str | None,
    cwd: str,
) -> Any:
    """Build the Codex SDK config shared by the planner and the check.

    Args:
        mcp_url: URL of the in-process RPent MCP server, or ``None`` to attach
            no MCP server at all (used by the connectivity probe).
        base_url: Custom Responses-compatible endpoint, or ``None``.
        api_key: Key exported to the child as ``RPENT_CODEX_PROVIDER_KEY``.
        cwd: Working directory handed to the Codex child process.

    Returns:
        The ``openai_codex.CodexConfig`` to open a session with.
    """
    env = agent_environment()
    if api_key:
        env[PROVIDER_ENV_KEY] = api_key
    config_overrides = _codex_mcp_config_overrides(mcp_url=mcp_url, base_url=base_url)
    # Project development instructions and skills are excluded from planner context.
    config_overrides.append("project_doc_max_bytes=0")
    disabled_skills = [
        f"{{ path = {json.dumps(str(path.resolve()), ensure_ascii=False)}, enabled = false }}"
        for path in sorted((Path(cwd) / ".agents" / "skills").glob("*/SKILL.md"))
        if path.is_file()
    ]
    if disabled_skills:
        config_overrides.append(f"skills.config=[{', '.join(disabled_skills)}]")
    kwargs: dict[str, Any] = {
        "config_overrides": tuple(config_overrides),
        "cwd": cwd,
        "env": env,
        # use True to support (namespace tools,
        # web_search, image_generation).
        "experimental_api": True,
    }
    if codex_bin := os.environ.get("CODEX_BIN"):
        kwargs["codex_bin"] = codex_bin
    return openai_codex.CodexConfig(**kwargs)


def run_probe_turn(
    config: Any,
    *,
    prompt: str,
    model: str | None,
    timeout_s: int,
    image_bytes: bytes | None = None,
    output_schema: dict[str, Any] | None = None,
) -> str:
    """Run one tool-free Codex turn and return its final assistant text.

    Connectivity checks and visual grounding use the same asynchronous SDK as
    task sessions. The synchronous boundary is retained for existing callers.
    SDK exceptions propagate unchanged so checks can classify them. The async
    session raises ``TimeoutError`` when the turn exceeds ``timeout_s``.

    Args:
        image_bytes: Optional PNG image supplied to a visual point selector.
        output_schema: Optional structured-output schema for the final response.
        config: Config from :func:`build_probe_config`.
        prompt: The probe prompt to send.
        model: Model id, or ``None`` to use the Codex-configured default.
        timeout_s: Wall-clock cap for the turn.

    Returns:
        The turn's final response text, empty if the model produced none.

    Raises:
        RuntimeError: If the turn reported ``failed`` or produced no result.
    """
    options: dict[str, Any] = {
        "approval_mode": openai_codex.ApprovalMode.deny_all,
        "sandbox": openai_codex.Sandbox.read_only,
    }
    if model:
        options["model"] = model
    if service_tier := os.environ.get("CODEX_SERVICE_TIER", None):
        options["service_tier"] = service_tier

    turn_input = prompt
    turn_options = dict(options)
    if image_bytes is not None:
        import base64

        turn_input = [
            openai_codex.TextInput(prompt),
            openai_codex.ImageInput(
                "data:image/png;base64," + base64.b64encode(image_bytes).decode("ascii")
            ),
        ]
    if output_schema is not None:
        turn_options["output_schema"] = output_schema

    async def run():
        codex = openai_codex.AsyncCodex(config=config)
        turn = None

        async def request():
            nonlocal turn
            thread = await codex.thread_start(**options)
            turn = await thread.turn(turn_input, **turn_options)
            return await turn.run()

        try:
            return await asyncio.wait_for(request(), timeout=timeout_s)
        except asyncio.TimeoutError:
            if turn is not None:
                with contextlib.suppress(Exception):
                    await asyncio.wait_for(turn.interrupt(), timeout=5)
            raise TimeoutError(
                f"the Codex SDK did not finish within {timeout_s}s."
            ) from None
        finally:
            await codex.close()

    result = asyncio.run(run())
    if result is None:
        raise RuntimeError("the Codex SDK returned no turn result")

    status = str(get_field(result, "status", "") or "")
    if status == "failed":
        error = get_field(result, "error")
        raise RuntimeError(f"Codex turn failed: {get_field(error, 'message', error)}")
    return str(get_field(result, "final_response", "") or "")


def build_probe_config(base_url: str | None = None) -> Any:
    """Build a Codex config for the connectivity probe.

    Identical to the planner's config except that no MCP server is attached,
    so the probe starts no in-process HTTP server and exposes no tools.

    Args:
        base_url: Endpoint override; falls back to ``CODEX_BASE_URL``.

    Returns:
        The ``openai_codex.CodexConfig`` for a tool-free probe session.
    """
    return build_codex_config(
        mcp_url=None,
        base_url=base_url or os.environ.get("CODEX_BASE_URL", None),
        api_key=os.environ.get("CODEX_API_KEY", None),
        cwd=str(get_repo_root()),
    )


def _codex_mcp_config_overrides(
    *,
    mcp_url: str | None,
    base_url: str | None,
) -> list[str]:
    config: list[tuple[str, Any]] = []
    if mcp_url:
        config.append(("mcp_servers.rpent.url", mcp_url))
    if base_url:
        normalized = base_url.rstrip("/")
        if not normalized.endswith("/v1"):
            normalized = normalized + "/v1"
        config.extend(
            [
                ("model_provider", PROVIDER_ID),
                (f"model_providers.{PROVIDER_ID}.name", PROVIDER_ID),
                (f"model_providers.{PROVIDER_ID}.base_url", normalized),
                (f"model_providers.{PROVIDER_ID}.wire_api", "responses"),
                (f"model_providers.{PROVIDER_ID}.env_key", PROVIDER_ENV_KEY),
            ]
        )
    model_context_window = os.environ.get("CODEX_MODEL_CONTEXT_WINDOW", None)
    if model_context_window is not None:
        config.append(("model_context_window", int(model_context_window)))
    auto_compact_token_limit = os.environ.get("CODEX_AUTO_COMPACT_TOKEN_LIMIT", None)
    if auto_compact_token_limit is not None:
        config.append(("model_auto_compact_token_limit", int(auto_compact_token_limit)))
    return [f"{key}={json.dumps(value)}" for key, value in config]


# ---------------------------------------------------------------------------
# SDK utilities
# ---------------------------------------------------------------------------


def _message_to_json(message: Any) -> dict[str, Any]:
    return {
        "method": get_field(message, "method", ""),
        "payload": jsonable(get_field(message, "payload")),
    }


def _is_tool_item(item: Any) -> bool:
    return str(get_field(unwrap(item), "type", "")) in {
        "mcpToolCall",
        "dynamicToolCall",
        "commandExecution",
        "fileChange",
    }


def _status(item: Any) -> str:
    value = get_field(item, "status", "")
    return str(getattr(value, "value", value))


def _summarise_item(item: Any) -> dict[str, Any]:
    data = jsonable(item)
    if not isinstance(data, dict):
        return {"size": _payload_size(data)}

    summary: dict[str, Any] = {}
    for key in ("path", "file_path", "filename", "status", "state", "exit_code"):
        value = data.get(key)
        if value not in (None, ""):
            summary[key] = value
    if command := (data.get("command") or data.get("cmd")):
        command_text = str(command)
        if len(command_text) > 200:
            command_text = command_text[:200] + f"...(+{len(command_text) - 200})"
        summary["command"] = command_text
    for key in ("content", "text", "output", "stdout", "stderr", "result"):
        if key in data and data[key] not in (None, ""):
            summary[f"{key}_size"] = _payload_size(data[key])

    if not summary:
        summary["keys"] = sorted(
            key for key in data if key not in {"content", "text", "output"}
        )
    return summary


def _extract_text(value: Any) -> str:
    value = unwrap(value)
    if isinstance(value, str):
        text = value.strip()
        if "data:image" in text or (
            "base64" in text and ("image" in text or "iVBOR" in text)
        ):
            return "<image omitted>"
        return text
    if isinstance(value, list):
        parts = [_extract_text(item) for item in value]
        return "\n".join(part for part in parts if part)
    return ""


def _payload_size(value: Any) -> int:
    return len(str(value or ""))
