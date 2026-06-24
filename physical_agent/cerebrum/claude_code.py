"""Claude Agent SDK cerebrum.

A thin, SDK-first backend for PhysicalAgent. ``solve()`` does four things:
prepare output files, bind the in-process tool runtime, drive the SDK
query, and assemble a ``CerebrumResult``. Event rendering and stats
collection live in a single observation layer (``_Recorder``) that has
no backend state of its own.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import claude_agent_sdk

from physical_agent.cerebrum.base import CerebrumResult
from physical_agent.tools.toolkit import Toolkit
from physical_agent.utils.config import get_repo_root
from physical_agent.utils.logging import get_logger, init_output_dir

logger = get_logger("claude")

# ---------------------------------------------------------------------------
# Public backend
# ---------------------------------------------------------------------------


class ClaudeCodeCerebrum:
    """Cerebrum backed by the Claude Agent SDK."""

    def __init__(
        self,
        *,
        output_dir: str,
        repo_root: str | Path | None = None,
        model: str = "sonnet",
        allowed_tools: str = "Bash Read Write Glob Grep",
        timeout_s: int = 600,
        max_budget_usd: float = 10.0,
        extra_dirs: list[str] | None = None,
        output_path: str | Path | None = None,
        hide_object_coords: bool = False,
        video_path: str = "",
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
        self._hide_object_coords = bool(hide_object_coords)
        self._video_path = video_path

    def solve(
        self,
        *,
        system_prompt: str,
        user_message: str,
        toolkit: Toolkit,
        max_turns: int,
    ) -> CerebrumResult:
        """Run one Claude Agent SDK session for the given prompt."""
        prompt = f"{system_prompt}\n\n{user_message}" if system_prompt else user_message
        return asyncio.run(
            self._solve_async(
                prompt,
                toolkit=toolkit,
                max_turns=max_turns,
            )
        )

    # -- internal lifecycle -------------------------------------------------

    async def _solve_async(
        self,
        prompt: str,
        *,
        toolkit: Toolkit,
        max_turns: int,
    ) -> CerebrumResult:
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
        recorder = _Recorder(max_turns=max_turns)

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
            try:

                async def consume_stream() -> None:
                    async for message in sdk.query(prompt=prompt, options=options):
                        _write_jsonl(raw_f, _message_to_json(message))
                        if rendered := recorder.observe(message):
                            rendered_chunks.append(rendered)
                            out_f.write(rendered)
                            out_f.flush()
                            logger.info(rendered.rstrip())

                await asyncio.wait_for(consume_stream(), timeout=self._timeout_s)
            except asyncio.TimeoutError:
                error = f"Claude Agent SDK timed out after {self._timeout_s}s"
                rendered = f"\n[cc-cerebrum] {error}\n"
                rendered_chunks.append(rendered)
                out_f.write(rendered)
                out_f.flush()
                _write_jsonl(raw_f, {"type": "timeout", "message": error})
                logger.info(rendered.rstrip())
            except Exception as e:
                error = f"{type(e).__name__}: {e}"
                rendered = f"\n[cc-cerebrum] {error}\n"
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

        return CerebrumResult(
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

    # -- options + tool bridge ---------------------------------------------

    def _build_options(self, sdk: Any, *, toolkit: Toolkit, max_turns: int) -> Any:
        allowed = [
            part for part in self._allowed_tools.replace(",", " ").split() if part
        ]
        builtins = [name for name in allowed if "__" not in name]
        allowed.extend(toolkit.allowed_mcp_tool_names)

        return sdk.ClaudeAgentOptions(
            cwd=self._repo_root,
            model=self._model,
            max_turns=max_turns,
            max_budget_usd=self._max_budget_usd,
            tools=builtins or None,
            allowed_tools=list(dict.fromkeys(allowed)),
            mcp_servers={
                "physical_agent": _build_physical_agent_server(
                    sdk,
                    toolkit=toolkit,
                ),
            },
            add_dirs=[self._output_dir, *self._extra_dirs],
            # Ignore user/project .claude configuration; PhysicalAgent owns the loop.
            setting_sources=[],
            stderr=lambda line: logger.debug("[claude-sdk] %s", line.rstrip()),
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
    turns: int = 0
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
            return self._system(message)
        if kind == "AssistantMessage":
            return self._assistant(message)
        if kind == "UserMessage":
            return self._user(message)
        if kind == "ResultMessage":
            return self._result(message)
        return ""

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
        for block in _get(message, "content", []) or []:
            block_kind = _kind(block)
            if block_kind == "TextBlock":
                text = str(_get(block, "text", "")).strip()
                if text:
                    self.turns += 1
                    lines.append(
                        f"\n[agent] === turn {self.turns}/{self.max_turns} ===\n"
                        f"[claude] {text}\n"
                    )
            elif block_kind == "ToolUseBlock":
                tool_id = str(_get(block, "id", ""))
                name = str(_get(block, "name", "tool"))
                self.tool_names[tool_id] = name
                tool_input = _get(block, "input", {}) or {}
                if name in {"finish", "mcp__physical_agent__finish"} and isinstance(
                    tool_input, dict
                ):
                    self.pending_finish[tool_id] = dict(tool_input)
                lines.append(f"[tool->] {name}: {_short_json(tool_input, limit=500)}\n")
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
        if _content_has_image(content):
            summary["images"] = 1
        if is_error:
            summary["is_error"] = bool(is_error)
        # Promote the finish payload once the tool result lands successfully.
        pending = self.pending_finish.pop(tool_use_id, None)
        if pending is not None and not is_error and self.finish_result is None:
            self.finish_result = {"_finish": True, **pending}
        return f"[tool<-] {name}: {json.dumps(summary, ensure_ascii=False)}\n"

    def _result(self, message: Any) -> str:
        if usage := _get(message, "usage"):
            self._set_usage(usage)
        if turns := _get(message, "num_turns"):
            self.turns = int(turns)
        if cost := _get(message, "total_cost_usd"):
            self.total_cost_usd = float(cost)
        if _get(message, "is_error", False):
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
# Tool bridge (PhysicalAgent registry -> SDK MCP server)
# ---------------------------------------------------------------------------


def _build_physical_agent_server(sdk: Any, *, toolkit: Toolkit) -> Any:
    sdk_tools = []
    for spec in toolkit.get_tools_spec():
        name = str(spec["name"])
        description = str(spec.get("description", ""))
        input_schema = spec.get("input_schema", {"type": "object"})

        async def run_tool(
            args: dict[str, Any],
            *,
            tool_name: str = name,
        ) -> dict[str, Any]:
            return _tool_result_to_mcp(toolkit.execute_tool(tool_name, args or {}))

        run_tool.__name__ = f"physical_agent_{name}"
        sdk_tools.append(sdk.tool(name, description, input_schema)(run_tool))

    return sdk.create_sdk_mcp_server(
        name="physical_agent", version="0.1.0", tools=sdk_tools
    )


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


def _content_has_image(value: Any) -> bool:
    if isinstance(value, list):
        return any(
            isinstance(item, dict) and item.get("type") == "image" for item in value
        )
    text = str(value)
    return "'type': 'image'" in text or '"type": "image"' in text
