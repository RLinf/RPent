"""Codex SDK cerebrum.

Mirror of ``claude_code.py``: a thin, SDK-first backend. ``solve()`` prepares
artifacts, drives one Codex SDK turn, and assembles a ``CerebrumResult``.
PhysicalAgent tools are exposed via the stdio MCP bridge configured through
``_codex_mcp_config_overrides``; this backend does not register tools in
process. Event rendering and stats live in a single ``_Recorder``.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from physical_agent.cerebrum.base import CerebrumResult
from physical_agent.utils.config import get_repo_root
from physical_agent.utils.logging import get_logger

logger = get_logger("codex")


# ---------------------------------------------------------------------------
# Execution policy
# ---------------------------------------------------------------------------

# Approval + sandbox decisions are configured once per backend and reused for
# both ``thread_start`` and the first ``turn``. Tighten ``_SANDBOX`` to
# ``workspace_write`` if we ever stop trusting the agent with arbitrary writes.
_APPROVAL_MODE_NAME = "deny_all"
_SANDBOX_NAME = "full_access"

_FINISH_TOOL_NAME = "finish"


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
        model: str | None = None,
        timeout_s: int = 600,
        extra_dirs: list[str] | None = None,
        output_path: str | Path | None = None,
        enable_mcp: bool = True,
        transport_host: str = "127.0.0.1",
        transport_port: int = 0,
        vla_endpoint: str = "",
        env_name: str = "libero",
        hide_object_coords: bool = False,
        video_path: str = "",
    ):
        """Initialize the Codex SDK backend."""
        self._output_dir = str(output_dir)
        self._repo_root = str(repo_root) if repo_root else str(get_repo_root())
        self._model = model
        self._timeout_s = timeout_s
        self._extra_dirs = extra_dirs or []
        self._output_path = Path(output_path) if output_path else None
        self._enable_mcp = enable_mcp
        self._transport_host = transport_host
        self._transport_port = int(transport_port)
        self._vla_endpoint = vla_endpoint
        self._env_name = env_name
        self._hide_object_coords = bool(hide_object_coords)
        self._video_path = video_path

    def set_socket_endpoint(self, host: str, port: int) -> None:
        """Record the driver socket endpoint discovered after startup."""
        self._transport_host = host
        self._transport_port = int(port)

    def set_vla_endpoint(self, endpoint: str) -> None:
        """Record the vla_server HTTP endpoint discovered after startup."""
        self._vla_endpoint = endpoint

    def solve(
        self,
        *,
        system_prompt: str,
        user_message: str,
        tools_spec: list[dict[str, Any]] | None = None,
        tool_handler: Callable[[str, dict[str, Any]], dict[str, Any]] | None = None,
        tool_result_formatter: Callable[[dict[str, Any]], list[dict[str, Any]]] | None = None,
        max_turns: int,
    ) -> CerebrumResult:
        """Run one Codex SDK turn for the given prompt."""
        del tools_spec, tool_handler, tool_result_formatter
        prompt = f"{system_prompt}\n\n{user_message}" if system_prompt else user_message
        artifacts = _Artifacts.create(self._output_path)
        recorder = _Recorder(max_turns=max_turns)
        state: dict[str, Any] = {}

        model_desc = self._model or "(configured default)"
        logger.info("prompt: %d chars", len(prompt))
        logger.info("output_dir: %s", self._output_dir)
        logger.info(
            "invoking Codex SDK model %s (timeout=%ds)",
            model_desc, self._timeout_s,
        )

        started = time.time()
        worker = threading.Thread(
            target=self._run_session,
            args=(prompt, artifacts, recorder, state),
            name="codex-sdk",
            daemon=True,
        )
        worker.start()
        worker.join(timeout=self._timeout_s)

        error: str | None = None
        if worker.is_alive():
            error = f"Codex SDK timed out after {self._timeout_s}s"
            _interrupt(state)
            _append_event(artifacts, "timeout", error)
            worker.join(timeout=15)
        elif "error" in state:
            exc = state["error"]
            error = f"{type(exc).__name__}: {exc}"
            _append_event(artifacts, "error", error)

        elapsed = time.time() - started
        text = state.get("text", "") or artifacts.output_path.read_text(errors="replace")
        error = error or recorder.error

        logger.info("Codex SDK finished in %.1fs", elapsed)
        logger.info("output: %s", artifacts.output_path)
        logger.info("raw stream: %s", artifacts.raw_path)

        return CerebrumResult(
            finish_result=recorder.finish_result,
            messages=[{"role": "codex_sdk", "content": text}],
            stats={
                "backend": "codex_sdk",
                "elapsed_s": round(elapsed, 1),
                "output_chars": len(text),
                "output_path": str(artifacts.output_path),
                "raw_stream_path": str(artifacts.raw_path),
                "last_message_path": str(artifacts.last_message_path),
                "last_message_chars": len(recorder.final_response or ""),
                **recorder.stats(),
            },
            error=error,
        )

    # -- internal session --------------------------------------------------

    def _run_session(
        self,
        prompt: str,
        artifacts: "_Artifacts",
        recorder: "_Recorder",
        state: dict[str, Any],
    ) -> None:
        try:
            sdk = _import_sdk()
            policy = _resolve_policy(sdk)
            chunks: list[str] = []
            with sdk.Codex(config=self._build_config(sdk)) as codex:
                state["codex"] = codex
                thread = codex.thread_start(
                    approval_mode=policy.approval,
                    cwd=self._repo_root,
                    model=self._model,
                    sandbox=policy.sandbox,
                )
                state["thread"] = thread
                turn = thread.turn(
                    prompt,
                    approval_mode=policy.approval,
                    cwd=self._repo_root,
                    model=self._model,
                    sandbox=policy.sandbox,
                )
                state["turn"] = turn

                with open(artifacts.output_path, "w") as out_f, \
                     open(artifacts.raw_path, "w") as raw_f:
                    for event in turn.stream():
                        _write_jsonl(raw_f, _message_to_json(event))
                        if rendered := recorder.observe(event):
                            chunks.append(rendered)
                            out_f.write(rendered)
                            out_f.flush()
                            logger.info(rendered.rstrip())

            state["text"] = "".join(chunks)
            if recorder.final_response is not None:
                artifacts.last_message_path.write_text(recorder.final_response)
        except BaseException as e:
            state["error"] = e

    # -- config builder ----------------------------------------------------

    def _build_config(
        self,
        sdk: Any,
    ) -> Any:
        overrides: list[str] = []
        if self._enable_mcp:
            overrides.extend(_codex_mcp_config_overrides(
                output_dir=self._output_dir,
                repo_root=self._repo_root,
                transport_host=self._transport_host,
                transport_port=self._transport_port,
                vla_endpoint=self._vla_endpoint,
                env_name=self._env_name,
                hide_object_coords=self._hide_object_coords,
                video_path=self._video_path,
            ))

        kwargs: dict[str, Any] = {
            "config_overrides": tuple(overrides),
            "cwd": self._repo_root,
            "env": {**os.environ},
        }
        if codex_bin := os.environ.get("CODEX_BIN"):
            kwargs["codex_bin"] = codex_bin
        return sdk.CodexConfig(**kwargs)


# ---------------------------------------------------------------------------
# Run artifacts
# ---------------------------------------------------------------------------


@dataclass
class _Artifacts:
    output_path: Path
    raw_path: Path
    last_message_path: Path

    @classmethod
    def create(cls, output_path: Path | None) -> "_Artifacts":
        path = output_path or _tmp_output_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        return cls(
            output_path=path,
            raw_path=path.with_suffix(path.suffix + ".stream.jsonl"),
            last_message_path=path.with_suffix(path.suffix + ".last"),
        )


def _append_event(artifacts: _Artifacts, event_type: str, message: str) -> None:
    rendered = f"\n[codex-cerebrum] {message}\n"
    with open(artifacts.output_path, "a") as out_f:
        out_f.write(rendered)
    with open(artifacts.raw_path, "a") as raw_f:
        _write_jsonl(raw_f, {"type": event_type, "message": message})
    logger.info(rendered.rstrip())


# ---------------------------------------------------------------------------
# Observation layer
# ---------------------------------------------------------------------------


@dataclass
class _Recorder:
    """Pure adapter: consume Codex SDK events, emit text + accumulate stats."""

    max_turns: int
    turns: int = 0
    tool_calls: int = 0
    usage: dict[str, int] = field(default_factory=lambda: {
        "total_input_tokens": 0,
        "total_cached_input_tokens": 0,
        "total_output_tokens": 0,
        "total_reasoning_output_tokens": 0,
    })
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
        item_type = _kind(item)
        item_type_lower = item_type.lower()

        if _is_input_item(item_type_lower):
            return ""

        if "agentmessage" in item_type_lower:
            text = _extract_text(item)
            if not text:
                return ""
            self.final_response = text
            self.turns += 1
            return (
                f"\n[agent] === turn {self.turns}/{self.max_turns} ===\n"
                f"[codex] {text}\n"
            )

        if "reasoning" in item_type_lower:
            text = _extract_text(item)
            return f"[codex-reasoning] {text}\n" if text else ""

        if _is_tool_item(item_type_lower):
            self.tool_calls += 1
            name = _tool_name(item, item_type)
            payload = _summarise_item(item)
            self._maybe_capture_finish(name, item)
            return f"[tool<-] {name}: {json.dumps(payload, ensure_ascii=False)}\n"

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
            "total_reasoning_output_tokens": _int_attr(usage, "reasoning_output_tokens"),
        }

    def _maybe_capture_finish(self, name: str, item: Any) -> None:
        if self.finish_result is not None or _FINISH_TOOL_NAME not in name.lower():
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
# Execution policy resolution
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Policy:
    approval: Any
    sandbox: Any


def _resolve_policy(sdk: Any) -> _Policy:
    return _Policy(
        approval=getattr(sdk.ApprovalMode, _APPROVAL_MODE_NAME),
        sandbox=getattr(sdk.Sandbox, _SANDBOX_NAME),
    )


# ---------------------------------------------------------------------------
# MCP overrides (PhysicalAgent stdio bridge)
# ---------------------------------------------------------------------------


def _codex_mcp_config_overrides(
    *,
    output_dir: str,
    repo_root: str,
    transport_host: str,
    transport_port: int,
    vla_endpoint: str,
    env_name: str,
    hide_object_coords: bool,
    video_path: str,
) -> list[str]:
    if transport_port <= 0:
        raise RuntimeError("Codex SDK MCP socket transport requires a bound port")
    if not vla_endpoint:
        raise RuntimeError("Codex SDK MCP server requires a vla_endpoint")

    pythonpath = repo_root
    if existing := os.environ.get("PYTHONPATH"):
        pythonpath = repo_root + os.pathsep + existing

    server_args = [
        "-m",
        "physical_agent.cerebrum.mcp.mcp",
        "--output-dir",
        output_dir,
        "--repo-root",
        repo_root,
        "--transport-host",
        transport_host,
        "--transport-port",
        str(transport_port),
        "--vla-endpoint",
        vla_endpoint,
        "--env",
        env_name,
    ]
    if hide_object_coords:
        server_args.append("--hide-object-coords")
    if video_path:
        server_args += ["--video-path", video_path]

    config: list[tuple[str, Any]] = [
        ("mcp_servers.physical_agent.command", sys.executable),
        ("mcp_servers.physical_agent.args", server_args),
        ("mcp_servers.physical_agent.env.HYBRID_DRIVER_OUTPUT_DIR", output_dir),
        ("mcp_servers.physical_agent.env.PHYSICAL_AGENT_TRANSPORT_HOST", transport_host),
        ("mcp_servers.physical_agent.env.PHYSICAL_AGENT_TRANSPORT_PORT", str(transport_port)),
        ("mcp_servers.physical_agent.env.PHYSICAL_AGENT_VLA_ENDPOINT", vla_endpoint),
        ("mcp_servers.physical_agent.env.PHYSICAL_AGENT_ENV", env_name),
        ("mcp_servers.physical_agent.env.PYTHONPATH", pythonpath),
    ]
    return [f"{key}={json.dumps(value)}" for key, value in config]


# ---------------------------------------------------------------------------
# SDK utilities
# ---------------------------------------------------------------------------


def _import_sdk() -> Any:
    try:
        import openai_codex
    except ImportError as e:
        raise RuntimeError(
            "openai-codex is required for the codex backend. "
            "Install project dependencies or run `pip install openai-codex`."
        ) from e
    return openai_codex


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


def _tmp_output_path() -> Path:
    return Path(f"/tmp/codex_sdk_task_{os.getpid()}_{int(time.time() * 1000)}.out")


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


def _is_tool_item(item_type_lower: str) -> bool:
    return any(
        marker in item_type_lower
        for marker in ("command", "exec", "tool", "mcp", "filechange", "patch")
    )


def _is_input_item(item_type_lower: str) -> bool:
    return any(
        marker in item_type_lower
        for marker in ("input", "user", "prompt")
    )


def _tool_name(item: Any, item_type: str) -> str:
    return str(
        _get(item, "name")
        or _get(item, "tool_name")
        or _get(item, "command")
        or item_type
    )


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
        summary["command"] = _truncate(str(command), 200)
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
        return _omit_image_payload(value.strip())
    if isinstance(value, list):
        parts = [_extract_text(item) for item in value]
        return "\n".join(part for part in parts if part)
    if isinstance(value, dict):
        for key in ("text", "message", "content", "summary", "result", "output"):
            if key in value:
                text = _extract_text(value[key])
                if text:
                    return text
        return ""
    for key in ("text", "message", "content", "summary", "result", "output"):
        attr = getattr(value, key, None)
        if attr is not None:
            text = _extract_text(attr)
            if text:
                return text
    return ""


def _payload_size(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, str):
        return len(value)
    return len(json.dumps(value, ensure_ascii=False, default=str))


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"...(+{len(text) - limit})"


def _short_json(value: Any, *, limit: int) -> str:
    text = json.dumps(value, ensure_ascii=False, default=str)
    if len(text) <= limit:
        return text
    return text[:limit] + f"...(+{len(text) - limit})"


def _omit_image_payload(text: str) -> str:
    if "data:image" in text or ("base64" in text and ("image" in text or "iVBOR" in text)):
        return "<image omitted>"
    return text


def _int_attr(value: Any, key: str) -> int:
    return int(_get(value, key, 0) or 0)
