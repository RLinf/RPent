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

"""Gemini CLI planner backed by its headless streaming protocol.

Gemini CLI owns the agent loop. RPent exposes its toolkit through the same
in-process Streamable HTTP MCP bridge used by the Codex planner, then records
the CLI's ``stream-json`` events into the common planner result.
"""

from __future__ import annotations

import json
import os
import queue
import shutil
import signal
import subprocess
import tempfile
import threading
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TextIO

from rpent.dashboard.events import (
    DashboardEventSink,
    NullDashboardEventSink,
    TranscriptEvent,
    UsageEvent,
)
from rpent.dashboard.interaction import DashboardInteractionPort
from rpent.planner.base import Planner, PlannerResult
from rpent.planner.utils.http_mcp_server import HttpMcpServer
from rpent.tools.toolkit import Toolkit
from rpent.utils.logging import get_logger

logger = get_logger("gemini_cli")

_LOOPBACK_NO_PROXY_HOSTS = ("127.0.0.1", "localhost")
_MCP_TOOL_PREFIXES = ("mcp_rpent_", "mcp__rpent__")
_SECRET_ENV_NAMES = ("GEMINI_API_KEY",)
# Gemini CLI treats an empty ``tools.core`` list as unset.
_NO_BUILTIN_TOOLS = ("__rpent_no_builtin_tools__",)


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
        """Run one Gemini CLI headless agent session against the RPent toolkit."""
        if input_queue is not None or dashboard_interaction is not None:
            raise ValueError(
                "gemini_cli currently supports non-interactive runs only; "
                "remove --interactive or --dashboard"
            )

        prompt = f"{system_prompt}\n\n{user_message}" if system_prompt else user_message
        output_path, raw_stream_path, stderr_path = self._output_paths()
        recorder = _Recorder(self._dashboard_events)
        state: dict[str, Any] = {}
        error: str | None = None
        started = time.monotonic()

        self._output_dir.mkdir(parents=True, exist_ok=True)
        mcp_server = HttpMcpServer(toolkit)
        try:
            mcp_url = mcp_server.start()
            with tempfile.TemporaryDirectory(
                prefix="gemini_cli_", dir=self._output_dir
            ) as session_dir:
                session_path = Path(session_dir)
                _write_settings(
                    session_path,
                    mcp_url=mcp_url,
                    max_turns=max_turns,
                    tool_names=(tool.name for tool in toolkit.list_tools()),
                )
                command = _build_command(prompt=prompt, model=self._model)
                worker = threading.Thread(
                    target=_run_session,
                    kwargs={
                        "command": command,
                        "cwd": session_path,
                        "recorder": recorder,
                        "output_path": output_path,
                        "raw_stream_path": raw_stream_path,
                        "stderr_path": stderr_path,
                        "state": state,
                    },
                    name="gemini-cli",
                    daemon=True,
                )
                worker.start()
                worker.join(timeout=self._timeout_s)
                if worker.is_alive():
                    error = f"Gemini CLI timed out after {self._timeout_s}s"
                    try:
                        toolkit.cancel_active_and_wait()
                    except Exception as exc:
                        logger.warning(
                            "failed to cancel toolkit work after Gemini timeout: %s",
                            exc,
                        )
                    _terminate_process(state.get("process"))
                    worker.join(timeout=15)
                    _append_text(output_path, f"\n[gemini-cli] {error}\n")
                    _append_jsonl(
                        raw_stream_path, {"type": "timeout", "message": error}
                    )
                elif exc := state.get("error"):
                    error = _redact_secrets(f"{type(exc).__name__}: {exc}")
                    _append_text(output_path, f"\n[gemini-cli] {error}\n")
        finally:
            try:
                toolkit.cancel_active_and_wait()
            except Exception as exc:
                logger.exception("Gemini CLI toolkit cleanup failed")
                error = error or f"Toolkit cleanup failed: {exc}"
            finally:
                mcp_server.stop()

        elapsed = time.monotonic() - started
        text = str(state.get("text") or "")
        if not text and output_path.exists():
            text = output_path.read_text(errors="replace")
        error = error or recorder.error
        return PlannerResult(
            finish_result=recorder.finish_result,
            messages=[{"role": "gemini_cli", "content": text}],
            stats={
                "backend": "gemini_cli",
                "elapsed_s": round(elapsed, 1),
                "output_chars": len(text),
                "output_path": str(output_path),
                "raw_stream_path": str(raw_stream_path),
                "stderr_path": str(stderr_path),
                "last_message_chars": len(recorder.final_response),
                **recorder.stats(),
            },
            error=error,
        )

    def _output_paths(self) -> tuple[Path, Path, Path]:
        if self._output_path is None:
            self._output_dir.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".out",
                prefix="gemini_cli_task_",
                dir=self._output_dir,
                delete=False,
            ) as file_obj:
                output_path = Path(file_obj.name)
        else:
            output_path = self._output_path
            output_path.parent.mkdir(parents=True, exist_ok=True)
        return (
            output_path,
            output_path.with_suffix(output_path.suffix + ".stream.jsonl"),
            output_path.with_suffix(output_path.suffix + ".stderr.log"),
        )


def run_probe_turn(*, prompt: str, model: str | None, timeout_s: int) -> str:
    """Run one tool-free Gemini CLI request and return assistant text."""
    recorder = _Recorder(NullDashboardEventSink())
    with tempfile.TemporaryDirectory(prefix="rpent_gemini_probe_") as session_dir:
        session_path = Path(session_dir)
        _write_settings(session_path, mcp_url=None, max_turns=1)
        command = _build_command(prompt=prompt, model=model)
        process = subprocess.Popen(
            command,
            cwd=session_path,
            env=_gemini_environment(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            start_new_session=True,
        )
        try:
            stdout, stderr = process.communicate(timeout=timeout_s)
        except subprocess.TimeoutExpired as exc:
            _terminate_process(process)
            process.communicate()
            raise TimeoutError(
                f"Gemini CLI did not finish within {timeout_s}s"
            ) from exc

    for line in stdout.splitlines():
        event = _parse_event(line)
        if event is not None:
            recorder.observe(event)
    if process.returncode != 0 or recorder.error:
        detail = _redact_secrets(
            recorder.error
            or _safe_stderr(stderr)
            or (f"Gemini CLI exited with status {process.returncode}")
        )
        raise RuntimeError(detail)
    return recorder.final_response


def has_configured_api_key() -> bool:
    """Return whether Gemini CLI can see ``GEMINI_API_KEY`` without exposing it."""
    return bool(os.environ.get("GEMINI_API_KEY") or _dotenv_value("GEMINI_API_KEY"))


def configured_base_url() -> str | None:
    """Return Gemini CLI's configured custom endpoint, if present."""
    return os.environ.get("GEMINI_BASE_URL") or _dotenv_value("GEMINI_BASE_URL")


def _run_session(
    *,
    command: list[str],
    cwd: Path,
    recorder: "_Recorder",
    output_path: Path,
    raw_stream_path: Path,
    stderr_path: Path,
    state: dict[str, Any],
) -> None:
    """Own the Gemini child and consume its streaming JSON on a worker thread."""
    stderr_chunks: list[str] = []
    rendered_chunks: list[str] = []
    try:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=_gemini_environment(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            start_new_session=True,
        )
        state["process"] = process
        if process.stdout is None or process.stderr is None:
            raise RuntimeError("Gemini CLI pipes were not created")

        stderr_reader = threading.Thread(
            target=_drain_stderr,
            args=(process.stderr, stderr_chunks),
            name="gemini-cli-stderr",
            daemon=True,
        )
        stderr_reader.start()
        with (
            open(output_path, "w") as output_file,
            open(raw_stream_path, "w") as raw_file,
        ):
            for line in process.stdout:
                safe_line = _redact_secrets(line)
                raw_file.write(safe_line)
                raw_file.flush()
                event = _parse_event(line)
                rendered = (
                    recorder.observe(event)
                    if event is not None
                    else f"[gemini-stdout] {safe_line.rstrip()}\n"
                )
                if rendered:
                    rendered_chunks.append(rendered)
                    output_file.write(rendered)
                    output_file.flush()
                    logger.info(rendered.rstrip())

        returncode = process.wait()
        stderr_reader.join(timeout=5)
        stderr_text = "".join(stderr_chunks)
        stderr_path.write_text(_redact_secrets(stderr_text))
        if returncode != 0 and recorder.error is None:
            raise RuntimeError(
                _safe_stderr(stderr_text)
                or f"Gemini CLI exited with status {returncode}"
            )
        state["text"] = "".join(rendered_chunks)
    except Exception as exc:
        state["error"] = exc


@dataclass
class _Recorder:
    """Translate Gemini CLI stream events into RPent transcript and stats."""

    dashboard_events: DashboardEventSink
    tool_calls: int = 0
    finish_result: dict[str, Any] | None = None
    error: str | None = None
    final_response: str = ""
    usage: dict[str, int] = field(
        default_factory=lambda: {
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "total_cached_input_tokens": 0,
        }
    )
    _tool_names: dict[str, str] = field(default_factory=dict)

    def stats(self) -> dict[str, Any]:
        return {"tool_calls": self.tool_calls, **self.usage}

    def observe(self, event: dict[str, Any]) -> str:
        event_type = str(event.get("type") or "")
        if event_type == "init":
            return (
                f"[gemini-system] session={event.get('session_id', '')} "
                f"model={event.get('model', '')}\n"
            )
        if event_type == "message":
            if event.get("role") != "assistant":
                return ""
            text = str(event.get("content") or "")
            if not text:
                return ""
            self.final_response += text
            self.dashboard_events.emit(TranscriptEvent({"type": "text", "text": text}))
            return f"[gemini] {text}\n"
        if event_type == "tool_use":
            name = _bare_tool_name(str(event.get("tool_name") or ""))
            tool_id = str(event.get("tool_id") or "")
            if tool_id:
                self._tool_names[tool_id] = name
            self.tool_calls += 1
            arguments = event.get("parameters")
            if not isinstance(arguments, dict):
                arguments = {}
            self.dashboard_events.emit(
                TranscriptEvent({"type": "tool_call", "tool": name, "args": arguments})
            )
            return f"[tool->] {name}: {_short_json(arguments)}\n"
        if event_type == "tool_result":
            tool_id = str(event.get("tool_id") or "")
            name = self._tool_names.get(tool_id, tool_id or "unknown")
            output = event.get("output")
            status = str(event.get("status") or "")
            if name == "finish" and status == "success":
                self.finish_result = self.finish_result or _find_finish(output)
            result = {
                "status": status,
                "output": _summarize_output(output),
            }
            if event.get("error"):
                result["error"] = event["error"]
            self.dashboard_events.emit(
                TranscriptEvent({"type": "tool_result", "tool": name, "result": result})
            )
            return f"[tool<-] {name}: {_short_json(result)}\n"
        if event_type == "error":
            detail = _redact_secrets(
                str(event.get("message") or event.get("error") or "unknown error")
            )
            if event.get("severity") == "error":
                self.error = detail
            self.dashboard_events.emit(
                TranscriptEvent({"type": "text", "text": f"[Gemini CLI] {detail}"})
            )
            return f"[gemini-error] {detail}\n"
        if event_type == "result":
            self._set_usage(event.get("stats"))
            if event.get("status") == "error":
                error = event.get("error")
                self.error = _redact_secrets(
                    str(error.get("message") if isinstance(error, dict) else error)
                )
            return f"[gemini-result] {event.get('status', '')}\n"
        return ""

    def _set_usage(self, stats: Any) -> None:
        if not isinstance(stats, dict):
            return
        self.tool_calls = max(self.tool_calls, int(stats.get("tool_calls") or 0))
        self.usage = {
            "total_input_tokens": int(stats.get("input_tokens") or 0),
            "total_output_tokens": int(stats.get("output_tokens") or 0),
            "total_cached_input_tokens": int(stats.get("cached") or 0),
        }
        self.dashboard_events.emit(
            UsageEvent(
                inp=self.usage["total_input_tokens"],
                out=self.usage["total_output_tokens"],
                tool_calls=self.tool_calls,
            )
        )


def _write_settings(
    root: Path,
    *,
    mcp_url: str | None,
    max_turns: int,
    tool_names: Iterable[str] = (),
) -> Path:
    """Write workspace-only settings without changing the user's Gemini config."""
    allowed_tools = [f"mcp_rpent_{name}" for name in tool_names]
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
        settings.update(
            {
                "mcp": {"allowed": ["rpent"]},
                "mcpServers": {
                    "rpent": {
                        "httpUrl": mcp_url,
                        "trust": True,
                    }
                },
            }
        )
    settings_dir = root / ".gemini"
    settings_dir.mkdir(parents=True, exist_ok=True)
    path = settings_dir / "settings.json"
    path.write_text(json.dumps(settings, indent=2) + "\n")
    return path


def _build_command(*, prompt: str, model: str | None) -> list[str]:
    binary = os.environ.get("GEMINI_BIN") or shutil.which("gemini")
    if not binary:
        raise FileNotFoundError(
            "Gemini CLI is not installed; install @google/gemini-cli or set GEMINI_BIN"
        )
    command = [
        binary,
        "--prompt",
        prompt,
        "--output-format",
        "stream-json",
        "--approval-mode",
        "yolo",
        "--skip-trust",
        "--allowed-mcp-server-names",
        "rpent",
        "--extensions",
        "none",
    ]
    if model:
        command.extend(["--model", model])
    return command


def _gemini_environment() -> dict[str, str]:
    env = {
        **os.environ,
        "GEMINI_CLI_NO_RELAUNCH": "1",
        # Workspace settings are merged before --skip-trust is processed.
        "GEMINI_CLI_TRUST_WORKSPACE": "true",
        "NO_COLOR": "1",
    }
    entries: list[str] = []
    for key in ("NO_PROXY", "no_proxy"):
        entries.extend(item.strip() for item in env.get(key, "").split(","))
    entries = list(
        dict.fromkeys(item for item in (*entries, *_LOOPBACK_NO_PROXY_HOSTS) if item)
    )
    bypass = ",".join(entries)
    env["NO_PROXY"] = bypass
    env["no_proxy"] = bypass
    return env


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


def _parse_event(line: str) -> dict[str, Any] | None:
    try:
        event = json.loads(line)
    except (TypeError, json.JSONDecodeError):
        return None
    return event if isinstance(event, dict) else None


def _bare_tool_name(name: str) -> str:
    for prefix in _MCP_TOOL_PREFIXES:
        if name.startswith(prefix):
            return name[len(prefix) :]
    return name


def _find_finish(value: Any, *, depth: int = 0) -> dict[str, Any] | None:
    if depth > 6:
        return None
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return None
        return _find_finish(value, depth=depth + 1)
    if isinstance(value, dict):
        if value.get("_finish") is True:
            return dict(value)
        for key in ("content", "text", "output", "result", "data"):
            if key in value and (found := _find_finish(value[key], depth=depth + 1)):
                return found
    elif isinstance(value, list):
        for item in value:
            if found := _find_finish(item, depth=depth + 1):
                return found
    return None


def _summarize_output(value: Any) -> Any:
    if isinstance(value, str) and len(value) > 1000:
        return value[:1000] + f"...(+{len(value) - 1000})"
    return value


def _short_json(value: Any, *, limit: int = 1000) -> str:
    text = json.dumps(value, ensure_ascii=False, default=str)
    if len(text) <= limit:
        return text
    return text[:limit] + f"...(+{len(text) - limit})"


def _drain_stderr(stream: TextIO, chunks: list[str]) -> None:
    for line in stream:
        chunks.append(line)


def _safe_stderr(stderr: str, *, limit: int = 2000) -> str:
    text = _redact_secrets(stderr).strip()
    if len(text) > limit:
        text = text[-limit:]
    return text


def _redact_secrets(text: str) -> str:
    for name in _SECRET_ENV_NAMES:
        secret = os.environ.get(name) or _dotenv_value(name)
        if secret and len(secret) >= 8:
            text = text.replace(secret, f"<{name}>")
    return text


def _terminate_process(process: subprocess.Popen[str] | None) -> None:
    if process is None or process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=5)
    except (OSError, subprocess.TimeoutExpired):
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except OSError:
            pass


def _append_text(path: Path, text: str) -> None:
    with open(path, "a") as file_obj:
        file_obj.write(text)


def _append_jsonl(path: Path, value: dict[str, Any]) -> None:
    with open(path, "a") as file_obj:
        file_obj.write(json.dumps(value, ensure_ascii=False) + "\n")
