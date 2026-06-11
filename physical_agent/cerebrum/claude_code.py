"""Claude Code cerebrum — delegates the agent loop to `claude -p`.

Claude Code interacts directly with the REPL workdir filesystem (Bash,
Read, Write, Grep, Glob).  This cerebrum writes a combined task prompt,
spawns ``claude -p`` with directory access, and waits for completion.
"""
from __future__ import annotations

import json
import os
import selectors
import signal
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Callable

from physical_agent.cerebrum.base import CerebrumResult
from physical_agent.utils.config import get_repo_root
from physical_agent.utils.logging import get_logger

logger = get_logger("claude")


class ClaudeCodeCerebrum:
    """Cerebrum backed by the ``claude`` CLI (Claude Code subscription).

    Constructor parameters
    ----------------------
    workdir:
        REPL working directory (granted to Claude Code via ``--add-dir``).
    repo_root:
        Repository root (used as ``cwd`` for the subprocess so relative
        paths in the prompt resolve correctly).
    model:
        Claude model id passed to ``--model`` (default ``"sonnet"``).
    allowed_tools:
        Space-separated tool list for ``--allowedTools``.
    timeout_s:
        Hard wall-clock cap on the ``claude -p`` subprocess.
    max_budget_usd:
        Passed to ``--max-budget-usd``.
    extra_dirs:
        Additional ``--add-dir`` paths (e.g. the memory snapshot).
    output_path:
        Optional path for full ``claude -p`` stdout/stderr, matching the
        legacy ``claude_<tag>.txt`` artifact.
    """

    def __init__(
        self,
        *,
        workdir: str,
        repo_root: str | Path | None = None,
        model: str = "sonnet",
        allowed_tools: str = "Bash Read Write Glob Grep",
        timeout_s: int = 600,
        max_budget_usd: float = 10.0,
        extra_dirs: list[str] | None = None,
        output_path: str | Path | None = None,
        driver_pid: int | None = None,
    ):
        self._workdir = str(workdir)
        self._repo_root = str(repo_root) if repo_root else str(get_repo_root())
        self._model = model
        self._allowed_tools = allowed_tools
        self._timeout_s = timeout_s
        self._max_budget_usd = max_budget_usd
        self._extra_dirs = extra_dirs or []
        self._output_path = Path(output_path) if output_path else None

    def set_driver_pid(self, pid: int | None) -> None:
        """Compatibility no-op for the runner interface."""
        return None

    def set_driver_process(self, proc: subprocess.Popen | None) -> None:
        """Compatibility no-op for the runner interface."""
        return None

    # ------------------------------------------------------------------
    # Cerebrum protocol
    # ------------------------------------------------------------------

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
        """Run ``claude -p`` with the combined system+user prompt.

        ``tools_spec``, ``tool_handler``, and ``tool_result_formatter`` are
        accepted for protocol compatibility but **ignored** — Claude Code
        uses its own built-in tool set (Bash, Read, Write, Grep, Glob).
        """
        # Build the combined task prompt.  The Claude Code prompt is often a
        # self-contained legacy prompt, so avoid a leading blank when there is
        # no separate system prompt.
        full_prompt = (
            f"{system_prompt}\n\n{user_message}" if system_prompt else user_message
        )

        # Write prompt to a temp file so it can be passed to claude -p.
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".md", prefix="cc_task_", delete=False
        ) as f:
            f.write(full_prompt)
            prompt_file = f.name

        try:
            logger.info("prompt: %d chars -> %s", len(full_prompt), prompt_file)
            logger.info("workdir: %s", self._workdir)
            logger.info(
                "invoking claude -p --model %s (timeout=%ds, budget=$%s)",
                self._model, self._timeout_s, self._max_budget_usd,
            )

            cmd = [
                "claude", "-p",
                full_prompt,               # stdin works too, but explicit is clearer
                "--bare",
                "--model", self._model,
                "--output-format", "stream-json",
                "--verbose",
                "--add-dir", self._workdir,
                "--allowedTools", self._allowed_tools,
                "--max-budget-usd", str(self._max_budget_usd),
            ]
            for d in self._extra_dirs:
                cmd += ["--add-dir", d]

            output_path = self._output_path or Path(prompt_file).with_suffix(".out")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            raw_stream_path = output_path.with_suffix(output_path.suffix + ".stream.jsonl")

            t0 = time.time()
            timed_out = False
            rendered_chunks: list[str] = []
            renderer = _ClaudeCodeStreamRenderer(max_turns=max_turns)
            with open(output_path, "w") as out_f, open(raw_stream_path, "w") as raw_f:
                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    cwd=self._repo_root,
                    env={**os.environ},  # inherit API key / base URL from env
                    start_new_session=True,
                )

                timed_out = _poll_stdout_until_exit(
                    proc=proc,
                    timeout_s=self._timeout_s,
                    raw_f=raw_f,
                    out_f=out_f,
                    rendered_chunks=rendered_chunks,
                    renderer=renderer,
                    timeout_prefix="[cc-cerebrum]",
                )

            elapsed = time.time() - t0
            stdout_text = "".join(rendered_chunks) or output_path.read_text(errors="replace")
            returncode = proc.returncode
            claude_stats = renderer.stats()

            logger.info(
                "claude -p finished in %.1fs rc=%d", elapsed, returncode,
            )
            logger.info("output: %s", output_path)
            logger.info("raw stream: %s", raw_stream_path)

            error = None
            if timed_out:
                error = f"claude -p timed out after {self._timeout_s}s"
            elif returncode != 0:
                error = f"claude -p exited with rc={returncode}: {stdout_text[-500:]}"

            return CerebrumResult(
                finish_result=None,  # Can't easily parse from Claude Code output
                messages=[{"role": "claude_code", "content": stdout_text}],
                stats={
                    "elapsed_s": round(elapsed, 1),
                    "returncode": returncode,
                    "output_chars": len(stdout_text),
                    "output_path": str(output_path),
                    "raw_stream_path": str(raw_stream_path),
                    **claude_stats,
                },
                error=error,
            )
        except subprocess.TimeoutExpired:
            return CerebrumResult(
                error=f"claude -p timed out after {self._timeout_s}s",
                stats={"elapsed_s": self._timeout_s},
            )
        finally:
            # Clean up temp prompt file.
            try:
                os.unlink(prompt_file)
            except OSError:
                pass


def _terminate_process_group(proc: subprocess.Popen) -> None:
    """Terminate Claude and tool subprocesses spawned in its process group."""
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
    renderer: "_ClaudeCodeStreamRenderer",
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


class _ClaudeCodeStreamRenderer:
    def __init__(self, *, max_turns: int):
        self._turn = 0
        self._max_turns = max_turns
        self._tool_calls = 0
        self._tool_uses: dict[str, dict[str, Any]] = {}
        self._seen_usage_message_ids: set[str] = set()
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._total_cache_creation_input_tokens = 0
        self._total_cache_read_input_tokens = 0

    def stats(self) -> dict[str, int]:
        return {
            "turns_used": self._turn,
            "tool_calls": self._tool_calls,
            "total_input_tokens": self._total_input_tokens,
            "total_output_tokens": self._total_output_tokens,
            "total_cache_creation_input_tokens": self._total_cache_creation_input_tokens,
            "total_cache_read_input_tokens": self._total_cache_read_input_tokens,
        }

    def render(self, line: str) -> str:
        """Convert one Claude Code stream-json event to a readable log line."""
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            return line

        event_type = event.get("type")
        if event_type == "system":
            subtype = event.get("subtype")
            session_id = event.get("session_id")
            detail = f" subtype={subtype}" if subtype else ""
            sid = f" session={session_id}" if session_id else ""
            return f"[cc-system]{detail}{sid}\n"

        if event_type == "assistant":
            message = event.get("message", event)
            self._accumulate_usage(message)
            return self._render_assistant_message(message)

        if event_type == "user":
            return self._render_user_event(event)

        if event_type == "result":
            return self._render_result_event(event)

        return ""

    def _accumulate_usage(self, message: dict[str, Any]) -> None:
        msg_id = str(message.get("id") or "")
        if msg_id and msg_id in self._seen_usage_message_ids:
            return
        usage = message.get("usage")
        if not isinstance(usage, dict):
            return
        if msg_id:
            self._seen_usage_message_ids.add(msg_id)
        self._total_input_tokens += int(usage.get("input_tokens") or 0)
        self._total_output_tokens += int(usage.get("output_tokens") or 0)
        self._total_cache_creation_input_tokens += int(
            usage.get("cache_creation_input_tokens") or 0
        )
        self._total_cache_read_input_tokens += int(usage.get("cache_read_input_tokens") or 0)

    def _render_assistant_message(self, message: dict[str, Any]) -> str:
        rendered: list[str] = []
        for block in _iter_content_blocks(message):
            block_type = block.get("type")
            if block_type == "text":
                text = str(block.get("text", "")).strip()
                if text:
                    self._turn += 1
                    rendered.append(
                        f"\n[agent] === turn {self._turn}/{self._max_turns} ===\n"
                        f"[claude] {text}\n"
                    )
            elif block_type == "tool_use":
                tool_id = str(block.get("id") or "")
                name = str(block.get("name") or "tool")
                payload = block.get("input", {})
                if tool_id:
                    self._tool_uses[tool_id] = {"name": name, "input": payload}
                payload_text = json.dumps(payload, ensure_ascii=False, default=str)
                if len(payload_text) > 500:
                    payload_text = payload_text[:500] + f"...(+{len(payload_text) - 500})"
                rendered.append(f"[tool->] {name}: {payload_text}\n")
        return "".join(rendered)

    def _render_user_event(self, event: dict[str, Any]) -> str:
        rendered: list[str] = []
        tool_result_meta = event.get("tool_use_result")
        message = event.get("message", event)
        for block in _iter_content_blocks(message):
            if block.get("type") != "tool_result":
                continue
            self._tool_calls += 1
            tool_use_id = str(block.get("tool_use_id") or "")
            tool = self._tool_uses.get(tool_use_id, {})
            name = str(tool.get("name") or "tool_result")
            summary = _summarise_tool_result(
                content=block.get("content", ""),
                tool_input=tool.get("input"),
                tool_use_result=tool_result_meta,
            )
            rendered.append(f"[tool<-] {name}: {json.dumps(summary, ensure_ascii=False)}\n")
        return "".join(rendered)

    def _render_result_event(self, event: dict[str, Any]) -> str:
        subtype = event.get("subtype")
        duration_ms = event.get("duration_ms")
        cost = event.get("total_cost_usd")
        pieces = ["[cc-result]"]
        if subtype:
            pieces.append(str(subtype))
        if duration_ms is not None:
            pieces.append(f"duration={duration_ms / 1000:.1f}s")
        if cost is not None:
            pieces.append(f"cost=${float(cost):.4f}")
        result = str(event.get("result") or "").strip()
        if result:
            pieces.append(f"result_size={len(result)}")
        usage = (
            f"\n[usage] in={self._total_input_tokens} "
            f"cache_create={self._total_cache_creation_input_tokens} "
            f"cache_read={self._total_cache_read_input_tokens} "
            f"out={self._total_output_tokens} tool_calls={self._tool_calls}"
        )
        return " ".join(pieces) + usage + "\n"


def _render_stream_line(line: str) -> str:
    return _ClaudeCodeStreamRenderer(max_turns=80).render(line)


def _iter_content_blocks(message: dict[str, Any]):
    content = message.get("content")
    if not isinstance(content, list):
        return
    for block in content:
        if isinstance(block, dict):
            yield block


def _summarise_tool_result(
    *,
    content: Any,
    tool_input: Any,
    tool_use_result: Any,
) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    if isinstance(tool_input, dict):
        path = _first_present(tool_input, "file_path", "path", "filename")
        command = _first_present(tool_input, "command", "cmd")
        if path is not None:
            summary["path"] = str(path)
        if command is not None:
            summary["command"] = _truncate_text(str(command), 200)

    if isinstance(tool_use_result, dict):
        for key in ("interrupted", "is_error", "noOutputExpected"):
            if key in tool_use_result:
                summary[key] = tool_use_result[key]
        if "stdout" in tool_use_result:
            summary["stdout_size"] = _payload_size(tool_use_result.get("stdout"))
        if "stderr" in tool_use_result:
            summary["stderr_size"] = _payload_size(tool_use_result.get("stderr"))

    image_count, text_size = _summarise_content_size(content)
    if image_count:
        summary["images"] = image_count
    if text_size:
        summary["size"] = text_size
    if not summary:
        summary["size"] = _payload_size(content)
    return summary


def _summarise_content_size(content: Any) -> tuple[int, int]:
    if isinstance(content, list):
        image_count = 0
        text_size = 0
        for item in content:
            if isinstance(item, dict) and item.get("type") == "image":
                image_count += 1
                continue
            if isinstance(item, dict) and item.get("type") == "text":
                text_size += _payload_size(item.get("text", ""))
            else:
                text_size += _payload_size(item)
        return image_count, text_size

    text = str(content)
    if "'type': 'image'" in text or '"type": "image"' in text:
        return 1, 0
    return 0, _payload_size(content)


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
