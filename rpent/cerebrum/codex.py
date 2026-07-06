"""Codex SDK cerebrum.

Mirror of ``claude_code.py``: a thin, SDK-first backend. ``solve()`` prepares
artifacts, drives one Codex SDK turn, and assembles a ``CerebrumResult``.
RPent tools are exposed via the stdio MCP bridge configured through
``_codex_mcp_config_overrides``; this backend does not register tools in
process. Event rendering and stats live in a single ``_Recorder``.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import openai_codex

from rpent.cerebrum.base import CerebrumResult, strip_mcp_prefix
from rpent.cerebrum.utils.http_mcp_server import HttpMcpServer
from rpent.tools.toolkit import Toolkit
from rpent.utils.config import get_repo_root
from rpent.utils.logging import get_logger

logger = get_logger("codex")

PROVIDER_ID = "rpent_proxy"
PROVIDER_ENV_KEY = "RPENT_CODEX_PROVIDER_KEY"

# ---------------------------------------------------------------------------
# Public backend
# ---------------------------------------------------------------------------


class CodexCerebrum:
    """Cerebrum backed by the OpenAI Codex Python SDK."""

    def __init__(
        self,
        *,
        output_dir: str,
        repo_root: str | Path | None = None,
        timeout_s: int = 600,
        extra_dirs: list[str] | None = None,
        output_path: str | Path | None = None,
    ):
        """Initialize the Codex SDK backend."""
        self._output_dir = str(output_dir)
        self._repo_root = str(repo_root) if repo_root else str(get_repo_root())
        self._timeout_s = timeout_s
        self._extra_dirs = extra_dirs or []
        self._output_path = Path(output_path) if output_path else None
        self._model = os.environ.get("CODEX_MODEL", None)
        self._base_url = os.environ.get("CODEX_BASE_URL", None)
        self._api_key = os.environ.get("CODEX_API_KEY", None)

    def solve(
        self,
        *,
        system_prompt: str,
        user_message: str,
        toolkit: Toolkit,
        max_turns: int,
        input_queue=None,
    ) -> CerebrumResult:
        """Run one Codex SDK turn for the given prompt."""
        # interactive input is unsupported for this backend.
        del input_queue
        prompt = f"{system_prompt}\n\n{user_message}" if system_prompt else user_message
        if self._output_path is None:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".out", prefix="codex_sdk_task_", delete=False
            ) as f:
                output_path = Path(f.name)
        else:
            output_path = self._output_path
            output_path.parent.mkdir(parents=True, exist_ok=True)
        raw_stream_path = output_path.with_suffix(output_path.suffix + ".stream.jsonl")
        last_message_path = output_path.with_suffix(output_path.suffix + ".last")
        recorder = _Recorder(max_turns=max_turns)
        state: dict[str, Any] = {}

        # Start the in-thread MCP HTTP server so Codex can reach the
        # shared toolkit without spawning a subprocess.
        mcp_server = HttpMcpServer(toolkit)
        mcp_url = mcp_server.start()
        logger.info("mcp http endpoint: %s", mcp_url)

        model_desc = self._model or "(configured default)"
        logger.info("prompt: %d chars", len(prompt))
        logger.info("output_dir: %s", self._output_dir)
        logger.info(
            "invoking Codex SDK model %s (timeout=%ds)",
            model_desc,
            self._timeout_s,
        )

        started = time.time()
        worker = threading.Thread(
            target=self._run_session,
            args=(
                prompt,
                output_path,
                raw_stream_path,
                last_message_path,
                recorder,
                state,
                mcp_url,
            ),
            name="codex-sdk",
            daemon=True,
        )
        worker.start()
        try:
            worker.join(timeout=self._timeout_s)

            error: str | None = None
            if worker.is_alive():
                error = f"Codex SDK timed out after {self._timeout_s}s"
                _interrupt(state)
                rendered = f"\n[codex-cerebrum] {error}\n"
                with open(output_path, "a") as out_f:
                    out_f.write(rendered)
                with open(raw_stream_path, "a") as raw_f:
                    _write_jsonl(raw_f, {"type": "timeout", "message": error})
                logger.info(rendered.rstrip())
                worker.join(timeout=15)
            elif "error" in state:
                exc = state["error"]
                error = f"{type(exc).__name__}: {exc}"
                rendered = f"\n[codex-cerebrum] {error}\n"
                with open(output_path, "a") as out_f:
                    out_f.write(rendered)
                with open(raw_stream_path, "a") as raw_f:
                    _write_jsonl(raw_f, {"type": "error", "message": error})
                logger.info(rendered.rstrip())
        finally:
            mcp_server.stop()

        elapsed = time.time() - started
        text = state.get("text", "") or output_path.read_text(errors="replace")
        error = error or recorder.error

        logger.info("Codex SDK finished in %.1fs", elapsed)
        logger.info("output: %s", output_path)
        logger.info("raw stream: %s", raw_stream_path)

        return CerebrumResult(
            finish_result=recorder.finish_result,
            messages=[{"role": "codex_sdk", "content": text}],
            stats={
                "backend": "codex_sdk",
                "elapsed_s": round(elapsed, 1),
                "output_chars": len(text),
                "output_path": str(output_path),
                "raw_stream_path": str(raw_stream_path),
                "last_message_path": str(last_message_path),
                "last_message_chars": len(recorder.final_response or ""),
                **recorder.stats(),
            },
            error=error,
        )

    # -- internal session --------------------------------------------------

    def _run_session(
        self,
        prompt: str,
        output_path: Path,
        raw_stream_path: Path,
        last_message_path: Path,
        recorder: "_Recorder",
        state: dict[str, Any],
        mcp_url: str,
    ) -> None:
        try:
            approval = openai_codex.ApprovalMode.deny_all
            sandbox = openai_codex.Sandbox.full_access
            chunks: list[str] = []
            with openai_codex.Codex(config=self._build_config(mcp_url)) as codex:
                state["codex"] = codex
                thread = codex.thread_start(
                    approval_mode=approval,
                    cwd=self._repo_root,
                    model=self._model,
                    sandbox=sandbox,
                )
                state["thread"] = thread
                turn = thread.turn(
                    prompt,
                    approval_mode=approval,
                    cwd=self._repo_root,
                    model=self._model,
                    sandbox=sandbox,
                )
                state["turn"] = turn

                with (
                    open(output_path, "w") as out_f,
                    open(raw_stream_path, "w") as raw_f,
                ):
                    for event in turn.stream():
                        _write_jsonl(raw_f, _message_to_json(event))
                        if rendered := recorder.observe(event):
                            chunks.append(rendered)
                            out_f.write(rendered)
                            out_f.flush()
                            logger.info(rendered.rstrip())

            state["text"] = "".join(chunks)
            if recorder.final_response is not None:
                last_message_path.write_text(recorder.final_response)
        except Exception as e:
            state["error"] = e

    # -- config builder ----------------------------------------------------

    def _build_config(self, mcp_url: str) -> Any:
        env = {**os.environ}
        if self._api_key:
            env[PROVIDER_ENV_KEY] = self._api_key
        kwargs: dict[str, Any] = {
            "config_overrides": tuple(
                _codex_mcp_config_overrides(
                    mcp_url=mcp_url,
                    base_url=self._base_url,
                )
            ),
            "cwd": self._repo_root,
            "env": env,
            # Disable experimental API features (namespace tools,
            # web_search, image_generation).  This forces the binary to
            # convert namespace MCP tools to function tools internally
            # while preserving its own name→namespace mapping so that
            # ``function_call`` responses can be routed through MCP.
            "experimental_api": False,
        }
        if codex_bin := os.environ.get("CODEX_BIN"):
            kwargs["codex_bin"] = codex_bin
        return openai_codex.CodexConfig(**kwargs)


# ---------------------------------------------------------------------------
# Observation layer
# ---------------------------------------------------------------------------


@dataclass
class _Recorder:
    """Pure adapter: consume Codex SDK events, emit text + accumulate stats."""

    max_turns: int
    turns: int = 0
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
        method = str(_get(event, "method", ""))
        payload = _get(event, "payload")

        if method in {"thread/started", "turn/started"}:
            return f"[codex-system] {method}\n"
        if method == "item/completed":
            return self._render_item(_get(payload, "item"))
        if method == "thread/tokenUsage/updated":
            self._set_usage(_get(payload, "token_usage"))
            return ""
        if method == "turn/completed":
            return self._render_turn_completed(_get(payload, "turn"))
        if "requestApproval" in method:
            return f"[codex-approval] {method}\n"
        if method in {"error", "fatal"}:
            return f"[codex-error] {_short_json(_jsonable(payload), limit=500)}\n"
        return ""

    # -- per-item handlers -------------------------------------------------

    def _render_item(self, item: Any) -> str:
        item = _unwrap(item)
        item_type = str(_get(item, "type", ""))

        if item_type in {"userMessage", "hookPrompt", "plan"}:
            return ""

        if item_type == "agentMessage":
            text = str(_get(item, "text", "")).strip()
            if not text:
                return ""
            self.final_response = text
            self.turns += 1
            return (
                f"\n[agent] === turn {self.turns}/{self.max_turns} ===\n"
                f"[codex] {text}\n"
            )

        if item_type == "reasoning":
            text = _extract_text(_get(item, "summary") or _get(item, "content"))
            return f"[codex-reasoning] {text}\n" if text else ""

        if item_type in {"mcpToolCall", "dynamicToolCall"}:
            self.tool_calls += 1
            name = strip_mcp_prefix(str(_get(item, "tool", item_type)))
            payload = _summarise_item(item)
            self._maybe_capture_finish(name, item)
            return f"[tool<-] {name}: {json.dumps(payload, ensure_ascii=False)}\n"

        if item_type == "commandExecution":
            self.tool_calls += 1
            name = str(_get(item, "command", item_type))
            payload = _summarise_item(item)
            return f"[tool<-] {name}: {json.dumps(payload, ensure_ascii=False)}\n"

        if item_type == "fileChange":
            self.tool_calls += 1
            payload = _summarise_item(item)
            return f"[tool<-] fileChange: {json.dumps(payload, ensure_ascii=False)}\n"

        return ""

    def _render_turn_completed(self, turn: Any) -> str:
        status = str(_get(_get(turn, "status"), "value", _get(turn, "status", "")))
        duration_ms = _get(turn, "duration_ms")
        if error := _get(turn, "error"):
            self.error = str(_get(error, "message", str(error)))

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

    def _set_usage(self, usage: Any) -> None:
        if usage is None:
            return
        self.usage = {
            "total_input_tokens": _int_attr(usage, "input_tokens"),
            "total_cached_input_tokens": _int_attr(usage, "cached_input_tokens"),
            "total_output_tokens": _int_attr(usage, "output_tokens"),
            "total_reasoning_output_tokens": _int_attr(
                usage, "reasoning_output_tokens"
            ),
        }

    def _maybe_capture_finish(self, name: str, item: Any) -> None:
        if self.finish_result is not None:
            return
        if name.lower() != "finish":
            return
        data = _jsonable(item)
        args = data.get("arguments") if isinstance(data, dict) else None
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except Exception:
                args = None
        if isinstance(args, dict):
            self.finish_result = {"_finish": True, **args}


# ---------------------------------------------------------------------------
# Codex config overrides
# ---------------------------------------------------------------------------


def _codex_mcp_config_overrides(
    *,
    mcp_url: str,
    base_url: str | None,
) -> list[str]:
    config: list[tuple[str, Any]] = [
        ("mcp_servers.rpent.url", mcp_url),
    ]
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
    return [f"{key}={json.dumps(value)}" for key, value in config]


# ---------------------------------------------------------------------------
# SDK utilities
# ---------------------------------------------------------------------------


def _interrupt(state: dict[str, Any]) -> None:
    if (turn := state.get("turn")) is not None:
        try:
            turn.interrupt()
        except Exception:
            pass
    if (codex := state.get("codex")) is not None:
        try:
            codex.close()
        except Exception:
            pass


def _write_jsonl(file_obj, value: dict[str, Any]) -> None:
    file_obj.write(json.dumps(value, ensure_ascii=False, default=str) + "\n")
    file_obj.flush()


def _message_to_json(message: Any) -> dict[str, Any]:
    return {
        "method": _get(message, "method", ""),
        "payload": _jsonable(_get(message, "payload")),
    }


def _jsonable(value: Any) -> Any:
    value = _unwrap(value)
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json", exclude_none=True)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, bytes):
        return {"type": "bytes", "size": len(value)}
    return value


def _unwrap(value: Any) -> Any:
    return getattr(value, "root", value)


def _get(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(key, default)
    return getattr(value, key, default)


def _kind(value: Any) -> str:
    value = _unwrap(value)
    if isinstance(value, dict):
        return str(value.get("type") or value.get("kind") or "")
    return value.__class__.__name__


def _summarise_item(item: Any) -> dict[str, Any]:
    data = _jsonable(item)
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
    value = _unwrap(value)
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
    if value is None:
        return 0
    if isinstance(value, str):
        return len(value)
    return len(json.dumps(value, ensure_ascii=False, default=str))


def _short_json(value: Any, *, limit: int) -> str:
    text = json.dumps(value, ensure_ascii=False, default=str)
    if len(text) <= limit:
        return text
    return text[:limit] + f"...(+{len(text) - limit})"


def _int_attr(value: Any, key: str) -> int:
    return int(_get(value, key, 0) or 0)
