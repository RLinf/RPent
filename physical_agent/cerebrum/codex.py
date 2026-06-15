"""Codex CLI cerebrum -- delegates the agent loop to ``codex exec``.

Codex interacts directly with the repository and driver workdir through its
normal local CLI tools.  This backend mirrors ``ClaudeCodeCerebrum``: it sends
one self-contained task prompt to a subprocess and waits for completion.
"""
from __future__ import annotations

import json
import os
import selectors
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable

from physical_agent.cerebrum.base import CerebrumResult
from physical_agent.utils.config import get_repo_root
from physical_agent.utils.logging import get_logger

logger = get_logger("codex")


class CodexCerebrum:
    """Cerebrum backed by the local ``codex exec`` CLI."""

    def __init__(
        self,
        *,
        workdir: str,
        repo_root: str | Path | None = None,
        model: str | None = None,
        timeout_s: int = 600,
        extra_dirs: list[str] | None = None,
        output_path: str | Path | None = None,
        driver_pid: int | None = None,
        enable_mcp: bool = True,
        transport: str = "file",
        transport_host: str = "127.0.0.1",
        transport_port: int = 0,
    ):
        """Initialize the Codex CLI subprocess wrapper."""
        self._workdir = str(workdir)
        self._repo_root = str(repo_root) if repo_root else str(get_repo_root())
        self._model = model
        self._timeout_s = timeout_s
        self._extra_dirs = extra_dirs or []
        self._output_path = Path(output_path) if output_path else None
        self._enable_mcp = enable_mcp
        self._transport = transport
        self._transport_host = transport_host
        self._transport_port = int(transport_port)

    def set_driver_pid(self, pid: int | None) -> None:
        """Compatibility no-op for the runner interface."""
        return None

    def set_driver_process(self, proc: subprocess.Popen | None) -> None:
        """Compatibility no-op for the runner interface."""
        return None

    def set_socket_endpoint(self, host: str, port: int) -> None:
        """Record the driver socket endpoint discovered after startup."""
        self._transport = "socket"
        self._transport_host = host
        self._transport_port = int(port)

    def solve(
        self,
        *,
        system_prompt: str,
        user_message: str,
        tools_spec: list[dict[str, Any]] | None = None,
        tool_handler: Callable[[str, dict[str, Any]], dict[str, Any]] | None = None,
        tool_result_formatter: Callable[[dict[str, Any]], list[dict[str, Any]]] | None = None,
        max_turns: int = 80,
    ) -> CerebrumResult:
        """Run ``codex exec`` with the combined system+user prompt.

        ``tools_spec``, ``tool_handler``, and ``tool_result_formatter`` are
        accepted for protocol compatibility but ignored; Codex uses its own
        local tool loop.
        """
        full_prompt = (
            f"{system_prompt}\n\n{user_message}" if system_prompt else user_message
        )
        prompt_file = None

        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".md", prefix="codex_task_", delete=False
            ) as f:
                f.write(full_prompt)
                prompt_file = f.name

            model_desc = self._model if self._model else "(configured default)"
            logger.info("prompt: %d chars -> %s", len(full_prompt), prompt_file)
            logger.info("workdir: %s", self._workdir)
            logger.info(
                "invoking codex exec --model %s (timeout=%ds)",
                model_desc, self._timeout_s,
            )

            output_path = self._output_path or Path(
                f"/tmp/codex_task_{os.getpid()}_{int(time.time() * 1000)}.out"
            )
            output_path.parent.mkdir(parents=True, exist_ok=True)
            last_message_path = output_path.with_suffix(output_path.suffix + ".last")
            raw_stream_path = output_path.with_suffix(output_path.suffix + ".stream.jsonl")

            cmd = [
                "codex",
                "exec",
                "--json",
                "--cd",
                self._repo_root,
                "--dangerously-bypass-approvals-and-sandbox",
                "--output-last-message",
                str(last_message_path),
            ]
            if self._enable_mcp:
                cmd += _codex_mcp_config_args(
                    workdir=self._workdir,
                    repo_root=self._repo_root,
                    transport=self._transport,
                    transport_host=self._transport_host,
                    transport_port=self._transport_port,
                )
            if self._model:
                cmd += ["--model", self._model]
            for d in [self._workdir, *self._extra_dirs]:
                cmd += ["--add-dir", d]
            cmd.append(full_prompt)

            t0 = time.time()
            timed_out = False
            rendered_chunks: list[str] = []
            renderer = _CodexStreamRenderer(max_turns=max_turns)
            with open(output_path, "w") as out_f, open(raw_stream_path, "w") as raw_f:
                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    cwd=self._repo_root,
                    env={**os.environ},
                    start_new_session=True,
                )

                timed_out = _poll_stdout_until_exit(
                    proc=proc,
                    timeout_s=self._timeout_s,
                    raw_f=raw_f,
                    out_f=out_f,
                    rendered_chunks=rendered_chunks,
                    renderer=renderer,
                    timeout_prefix="[codex-cerebrum]",
                )

            elapsed = time.time() - t0
            stdout_text = "".join(rendered_chunks) or output_path.read_text(errors="replace")
            last_message = (
                last_message_path.read_text(errors="replace")
                if last_message_path.exists()
                else ""
            )
            returncode = proc.returncode
            codex_stats = renderer.stats()

            logger.info(
                "codex exec finished in %.1fs rc=%d", elapsed, returncode,
            )
            logger.info("output: %s", output_path)
            logger.info("raw stream: %s", raw_stream_path)

            error = None
            if timed_out:
                error = f"codex exec timed out after {self._timeout_s}s"
            elif returncode != 0:
                error = f"codex exec exited with rc={returncode}: {stdout_text[-500:]}"

            return CerebrumResult(
                finish_result=None,
                messages=[{"role": "codex_cli", "content": stdout_text}],
                stats={
                    "elapsed_s": round(elapsed, 1),
                    "returncode": returncode,
                    "output_chars": len(stdout_text),
                    "output_path": str(output_path),
                    "raw_stream_path": str(raw_stream_path),
                    "last_message_path": str(last_message_path),
                    "last_message_chars": len(last_message),
                    **codex_stats,
                },
                error=error,
            )
        except subprocess.TimeoutExpired:
            return CerebrumResult(
                error=f"codex exec timed out after {self._timeout_s}s",
                stats={"elapsed_s": self._timeout_s},
            )
        finally:
            if prompt_file is not None:
                try:
                    os.unlink(prompt_file)
                except OSError:
                    pass


def _toml_value(value: Any) -> str:
    return json.dumps(value)


def _codex_mcp_config_args(
    *,
    workdir: str,
    repo_root: str,
    transport: str,
    transport_host: str,
    transport_port: int,
) -> list[str]:
    if transport == "socket" and transport_port <= 0:
        raise RuntimeError("Codex MCP socket transport requires a bound port")

    pythonpath = repo_root
    if os.environ.get("PYTHONPATH"):
        pythonpath = repo_root + os.pathsep + os.environ["PYTHONPATH"]

    server_args = [
        "-m",
        "physical_agent.cerebrum.mcp.mcp",
        "--workdir",
        workdir,
        "--repo-root",
        repo_root,
        "--transport",
        transport,
    ]
    if transport == "socket":
        server_args += [
            "--transport-host",
            transport_host,
            "--transport-port",
            str(transport_port),
        ]

    config: list[tuple[str, Any]] = [
        ("mcp_servers.physical_agent.command", sys.executable),
        ("mcp_servers.physical_agent.args", server_args),
        ("mcp_servers.physical_agent.env.HYBRID_DRIVER_WORKDIR", workdir),
        ("mcp_servers.physical_agent.env.PHYSICAL_AGENT_TRANSPORT", transport),
        ("mcp_servers.physical_agent.env.PHYSICAL_AGENT_TRANSPORT_HOST", transport_host),
        (
            "mcp_servers.physical_agent.env.PHYSICAL_AGENT_TRANSPORT_PORT",
            str(transport_port),
        ),
        ("mcp_servers.physical_agent.env.PYTHONPATH", pythonpath),
    ]

    out: list[str] = []
    for key, value in config:
        out += ["-c", f"{key}={_toml_value(value)}"]
    return out


def _terminate_process_group(proc: subprocess.Popen) -> None:
    """Terminate Codex and tool subprocesses spawned in its process group."""
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    except OSError:
        proc.terminate()
        return

    try:
        proc.wait(timeout=3)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except OSError:
            proc.kill()


def _poll_stdout_until_exit(
    *,
    proc: subprocess.Popen,
    timeout_s: int,
    raw_f,
    out_f,
    rendered_chunks: list[str],
    renderer: "_CodexStreamRenderer",
    timeout_prefix: str,
) -> bool:
    """Poll child stdout in the main thread until exit or timeout."""
    if proc.stdout is None:
        try:
            proc.wait(timeout=timeout_s)
            return False
        except subprocess.TimeoutExpired:
            msg = f"\n{timeout_prefix} TIMEOUT after {timeout_s}s; killing worker.\n"
            _write_rendered(msg, raw_f, out_f, rendered_chunks, "timeout")
            _terminate_process_group(proc)
            proc.wait(timeout=15)
            return True

    selector = selectors.DefaultSelector()
    selector.register(proc.stdout, selectors.EVENT_READ)
    deadline = time.time() + timeout_s
    timed_out = False

    try:
        while True:
            remaining = deadline - time.time()
            if remaining <= 0 and proc.poll() is None:
                timed_out = True
                msg = f"\n{timeout_prefix} TIMEOUT after {timeout_s}s; killing worker.\n"
                _write_rendered(msg, raw_f, out_f, rendered_chunks, "timeout")
                _terminate_process_group(proc)
                proc.wait(timeout=15)

            events = selector.select(timeout=0 if timed_out else min(max(remaining, 0), 0.25))
            for key, _ in events:
                line = key.fileobj.readline()
                if not line:
                    selector.unregister(key.fileobj)
                    break
                raw_f.write(line)
                raw_f.flush()
                rendered = renderer.render(line)
                if rendered:
                    rendered_chunks.append(rendered)
                    out_f.write(rendered)
                    out_f.flush()
                    logger.info(rendered.rstrip())

            if proc.poll() is not None:
                # Drain any buffered lines after process exit.
                while True:
                    line = proc.stdout.readline()
                    if not line:
                        break
                    raw_f.write(line)
                    raw_f.flush()
                    rendered = renderer.render(line)
                    if rendered:
                        rendered_chunks.append(rendered)
                        out_f.write(rendered)
                        out_f.flush()
                        logger.info(rendered.rstrip())
                break
    finally:
        selector.close()
    return timed_out


def _write_rendered(
    msg: str,
    raw_f,
    out_f,
    rendered_chunks: list[str],
    event_type: str,
) -> None:
    out_f.write(msg)
    rendered_chunks.append(msg)
    out_f.flush()
    raw_f.write(json.dumps({"type": event_type, "message": msg}) + "\n")
    raw_f.flush()
    logger.info(msg.rstrip())


class _CodexStreamRenderer:
    def __init__(self, *, max_turns: int):
        self._turn = 0
        self._max_turns = max_turns
        self._tool_calls = 0
        self._total_input_tokens = 0
        self._total_cached_input_tokens = 0
        self._total_output_tokens = 0
        self._total_reasoning_output_tokens = 0

    def stats(self) -> dict[str, int]:
        return {
            "turns_used": self._turn,
            "tool_calls": self._tool_calls,
            "total_input_tokens": self._total_input_tokens,
            "total_cached_input_tokens": self._total_cached_input_tokens,
            "total_output_tokens": self._total_output_tokens,
            "total_reasoning_output_tokens": self._total_reasoning_output_tokens,
        }

    def render(self, line: str) -> str:
        """Convert one Codex JSONL event to a compact readable log line."""
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            return line

        event_type = str(event.get("type") or event.get("event") or "")
        if not event_type:
            return _render_generic_event(event)

        if event_type in {"thread.started", "session_configured", "session.created", "system"}:
            model = event.get("model")
            session_id = event.get("thread_id") or event.get("session_id") or event.get("id")
            parts = ["[codex-system]", event_type]
            if model:
                parts.append(f"model={model}")
            if session_id:
                parts.append(f"session={session_id}")
            return " ".join(parts) + "\n"

        if event_type == "turn.started":
            return ""

        if event_type in {"turn.completed", "turn_complete"}:
            return self._render_turn_completed(event)

        if event_type in {"turn.failed", "turn_failed"}:
            text = _extract_text(event) or json.dumps(event, ensure_ascii=False, default=str)
            return f"[codex-error] {text}\n"

        if event_type == "item.started":
            return _render_codex_item_started(event.get("item", {}))

        if event_type == "item.completed":
            return self._render_codex_item_completed(event.get("item", {}))

        if "reasoning" in event_type:
            text = _extract_text(event)
            return f"[codex-reasoning] {text}\n" if text else ""

        if "message" in event_type or event_type in {"assistant", "agent_message"}:
            text = _extract_text(event)
            return self._render_agent_message(text)

        if "exec" in event_type or "tool" in event_type or "command" in event_type:
            return _render_tool_event(event, event_type)

        if event_type in {"error", "fatal"}:
            text = _extract_text(event) or json.dumps(event, ensure_ascii=False, default=str)
            return f"[codex-error] {text}\n"

        if event_type in {"task_complete", "result", "completed"}:
            text = _extract_text(event)
            return f"[codex-result] {text}\n" if text else f"[codex-result] {event_type}\n"

        return _render_generic_event(event)

    def _render_turn_completed(self, event: dict[str, Any]) -> str:
        usage = event.get("usage")
        if not isinstance(usage, dict):
            return "[usage] in=? out=? tool_calls={}\n".format(self._tool_calls)

        input_tokens = int(usage.get("input_tokens") or 0)
        cached_tokens = int(usage.get("cached_input_tokens") or 0)
        output_tokens = int(usage.get("output_tokens") or 0)
        reasoning_tokens = int(usage.get("reasoning_output_tokens") or 0)
        self._total_input_tokens += input_tokens
        self._total_cached_input_tokens += cached_tokens
        self._total_output_tokens += output_tokens
        self._total_reasoning_output_tokens += reasoning_tokens
        return (
            f"[usage] in={input_tokens} cached={cached_tokens} out={output_tokens} "
            f"reasoning={reasoning_tokens} tool_calls={self._tool_calls}\n"
        )

    def _render_codex_item_completed(self, item: Any) -> str:
        if isinstance(item, dict) and _is_tool_item_type(str(item.get("type") or "")):
            self._tool_calls += 1
        if isinstance(item, dict) and str(item.get("type") or "") == "agent_message":
            return self._render_agent_message(_extract_text(item))
        return _render_codex_item(item)

    def _render_agent_message(self, text: str) -> str:
        if not text:
            return ""
        self._turn += 1
        return f"\n[agent] === turn {self._turn}/{self._max_turns} ===\n[codex] {text}\n"


def _render_stream_line(line: str) -> str:
    return _CodexStreamRenderer(max_turns=80).render(line)


def _is_tool_item_type(item_type: str) -> bool:
    return item_type in {
        "exec_command",
        "command_execution",
        "exec_command_output",
        "tool_call",
        "tool_result",
        "function_call",
        "function_call_output",
        "mcp_tool_call",
        "mcp_tool_call_result",
    }


def _render_tool_event(event: dict[str, Any], event_type: str) -> str:
    name = event.get("name") or event.get("tool") or event.get("command") or event_type
    arrow = "<-" if any(k in event_type for k in ("result", "output", "end", "complete")) else "->"
    payload = _summarise_tool_result(event) if arrow == "<-" else _tool_input_payload(event)
    if isinstance(payload, (dict, list)):
        payload_text = json.dumps(payload, ensure_ascii=False, default=str)
    else:
        payload_text = str(payload)
    payload_text = _omit_image_payload(payload_text.strip())
    if len(payload_text) > 500:
        payload_text = payload_text[:500] + f"...(+{len(payload_text) - 500})"

    if payload_text:
        return f"[tool{arrow}] {name}: {payload_text}\n"
    return f"[tool{arrow}] {name}\n"


def _render_codex_item_started(item: Any) -> str:
    if not isinstance(item, dict):
        return ""
    item_type = str(item.get("type") or "")
    if item_type in {"exec_command", "command_execution"}:
        command = item.get("command") or item.get("cmd") or ""
        payload = {"command": _truncate_text(str(command), 200)} if command else {}
        return f"[tool->] exec_command: {json.dumps(payload, ensure_ascii=False)}\n"
    if item_type in {"tool_call", "function_call", "mcp_tool_call"}:
        name = item.get("name") or item.get("tool_name") or item_type
        payload = _tool_input_payload(item)
        payload_text = json.dumps(payload, ensure_ascii=False, default=str) if payload else "{}"
        if len(payload_text) > 500:
            payload_text = payload_text[:500] + f"...(+{len(payload_text) - 500})"
        return f"[tool->] {name}: {payload_text}\n"
    return ""


def _render_codex_item(item: Any) -> str:
    if not isinstance(item, dict):
        return ""
    item_type = str(item.get("type") or "")
    if item_type == "agent_message":
        text = _extract_text(item)
        return f"[codex] {text}\n" if text else ""
    if item_type in {"reasoning", "reasoning_summary"}:
        text = _extract_text(item)
        return f"[codex-reasoning] {text}\n" if text else ""
    if item_type in {
        "exec_command",
        "command_execution",
        "exec_command_output",
        "tool_call",
        "tool_result",
        "function_call",
        "function_call_output",
        "mcp_tool_call",
        "mcp_tool_call_result",
    }:
        name = _tool_name_for_item(item, item_type)
        payload_text = json.dumps(_summarise_tool_result(item), ensure_ascii=False, default=str)
        if len(payload_text) > 500:
            payload_text = payload_text[:500] + f"...(+{len(payload_text) - 500})"
        return f"[tool<-] {name}: {payload_text}\n"
    return _render_generic_event(item)


def _tool_name_for_item(item: dict[str, Any], item_type: str) -> str:
    if item.get("name"):
        return str(item["name"])
    if item.get("tool_name"):
        return str(item["tool_name"])
    if item_type in {"exec_command", "command_execution", "exec_command_output"}:
        return "exec_command"
    return item_type


def _tool_input_payload(event: dict[str, Any]) -> Any:
    return (
        event.get("arguments")
        or event.get("input")
        or event.get("cmd")
        or event.get("command")
        or ""
    )


def _summarise_tool_result(event: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {}

    path = _first_present(event, "path", "file_path", "filename")
    command = _first_present(event, "command", "cmd")
    status = _first_present(event, "status", "state")
    exit_code = _first_present(event, "exit_code", "returncode", "code")

    nested_input = event.get("input") if isinstance(event.get("input"), dict) else {}
    if path is None and isinstance(nested_input, dict):
        path = _first_present(nested_input, "path", "file_path", "filename")
    if command is None and isinstance(nested_input, dict):
        command = _first_present(nested_input, "command", "cmd")

    if path is not None:
        summary["path"] = str(path)
    if command is not None:
        summary["command"] = _truncate_text(str(command), 200)
    if status is not None:
        summary["status"] = status
    if exit_code is not None:
        summary["exit_code"] = exit_code

    for key in ("content", "text", "output", "aggregated_output", "stdout", "stderr", "result"):
        if key not in event or event[key] in (None, ""):
            continue
        size = _payload_size(event[key])
        if key == "content" and path is not None:
            summary["size"] = size
        elif key == "aggregated_output":
            summary["output_size"] = size
        else:
            summary[f"{key}_size"] = size

    if "size" not in summary and path is not None:
        size = _payload_size(_first_present(event, "content", "text", "output", "result"))
        if size:
            summary["size"] = size

    if not summary:
        keys = sorted(k for k in event.keys() if k not in {"content", "text", "output", "stdout", "stderr", "result"})
        summary["keys"] = keys
    return summary


def _first_present(mapping: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return None


def _payload_size(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, str):
        return len(value)
    try:
        return len(json.dumps(value, ensure_ascii=False, default=str))
    except TypeError:
        return len(str(value))


def _truncate_text(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"...(+{len(text) - limit})"


def _render_generic_event(event: dict[str, Any]) -> str:
    text = _extract_text(event)
    if text:
        return f"[codex] {text}\n"
    return ""


def _extract_text(value: Any) -> str:
    if isinstance(value, str):
        return _omit_image_payload(value.strip())
    if isinstance(value, list):
        parts = [_extract_text(item) for item in value]
        return "\n".join(part for part in parts if part)
    if not isinstance(value, dict):
        return ""

    for key in ("message", "text", "content", "delta", "summary", "result", "output", "item"):
        if key not in value:
            continue
        text = _extract_text(value[key])
        if text:
            return text
    return ""


def _omit_image_payload(text: str) -> str:
    if "data:image" in text or "base64" in text and ("image" in text or "iVBOR" in text):
        return "<image omitted>"
    return text
