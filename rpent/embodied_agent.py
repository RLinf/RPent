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

"""Benchmark-neutral entry point for RPent planners and user-owned MCP tools."""

from __future__ import annotations

import asyncio
import inspect
import json
import os
import threading
from collections.abc import Mapping, Sequence
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic_ai import BinaryContent

from rpent.context import ContextDocument, assemble_context, load_skill
from rpent.dashboard.events import NullDashboardEventSink
from rpent.llm import LLMConfig, LLMUsage
from rpent.planner.base import PlannerResult, build_planner
from rpent.session import EnvState
from rpent.tools.toolkit import ToolResult
from rpent.utils.logging import get_logger

logger = get_logger("embodied_agent")


@dataclass(frozen=True, slots=True)
class McpServer:
    """Connection to a user-owned Streamable HTTP or stdio MCP server.

    Exactly one of ``url`` and ``command`` must be set. A stdio server is
    started for each :meth:`EmbodiedAgent.run` and stopped when it returns.
    """

    name: str
    url: str | None = None
    command: str | None = None
    args: tuple[str, ...] = ()
    env: Mapping[str, str] | None = None
    cwd: str | None = None
    headers: Mapping[str, str] | None = None

    def __post_init__(self) -> None:
        if not self.name or not self.name.replace("_", "").isalnum():
            raise ValueError("MCP server name must contain only letters, digits, or _")
        if bool(self.url) == bool(self.command):
            raise ValueError("set exactly one of url and command")
        if self.url and not self.url.startswith(("http://", "https://")):
            raise ValueError("MCP server URL must use http:// or https://")


@dataclass(slots=True)
class _RemoteTool:
    server: str
    name: str
    spec: dict[str, Any]


class _McpToolkit:
    """Adapt MCP discovery and calls to the planner's toolkit contract."""

    def __init__(self, servers: Sequence[McpServer], output_dir: Path):
        self.state = EnvState(output_dir)
        self._servers = tuple(servers)
        self._sessions: dict[str, Any] = {}
        self._tools: dict[str, _RemoteTool] = {}
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._stop: asyncio.Event | None = None
        self._ready = threading.Event()
        self._startup_error: BaseException | None = None
        self._call_lock = threading.Lock()

    async def _serve(self) -> None:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client
        from mcp.client.streamable_http import streamable_http_client

        self._loop = asyncio.get_running_loop()
        self._stop = asyncio.Event()
        try:
            async with AsyncExitStack() as stack:
                for server in self._servers:
                    if server.url:
                        # MCP 1.23 accepted headers on the transport; newer
                        # 1.x releases configure them on an HTTP client.
                        transport_kwargs: dict[str, Any]
                        if (
                            "http_client"
                            in inspect.signature(streamable_http_client).parameters
                        ):
                            import httpx

                            http_client = await stack.enter_async_context(
                                httpx.AsyncClient(
                                    headers=dict(server.headers or {}),
                                    timeout=httpx.Timeout(30, read=300),
                                )
                            )
                            transport_kwargs = {"http_client": http_client}
                        else:
                            transport_kwargs = {"headers": dict(server.headers or {})}
                        read, write, _ = await stack.enter_async_context(
                            streamable_http_client(server.url, **transport_kwargs)
                        )
                    else:
                        params = StdioServerParameters(
                            command=server.command,
                            args=list(server.args),
                            env={**os.environ, **server.env}
                            if server.env is not None
                            else None,
                            cwd=server.cwd,
                        )
                        read, write = await stack.enter_async_context(
                            stdio_client(params)
                        )
                    session = await stack.enter_async_context(
                        ClientSession(read, write)
                    )
                    await session.initialize()
                    self._sessions[server.name] = session
                    for tool in (await session.list_tools()).tools:
                        public_name = f"{server.name}__{tool.name}"
                        if public_name in self._tools or public_name == "finish":
                            raise ValueError(f"duplicate MCP tool: {public_name}")
                        self._tools[public_name] = _RemoteTool(
                            server=server.name,
                            name=tool.name,
                            spec={
                                "name": public_name,
                                "description": tool.description or "",
                                "input_schema": tool.inputSchema,
                            },
                        )
                self._ready.set()
                await self._stop.wait()
        except Exception as exc:
            self._startup_error = exc
            logger.error("MCP connection failed: %s", exc)
            self._ready.set()
        finally:
            self._loop = None

    def start(self) -> None:
        """Connect to every MCP server and discover its tools."""
        if self._thread is not None:
            raise RuntimeError("MCP toolkit is already started")
        self._thread = threading.Thread(
            target=lambda: asyncio.run(self._serve()), name="embodied-mcp", daemon=True
        )
        self._thread.start()
        self._ready.wait()
        if self._startup_error is not None:
            self.close()
            raise RuntimeError("failed to start MCP tools") from self._startup_error

    def get_tools_spec(self) -> list[dict[str, Any]]:
        """Return remote tool schemas and a local completion tool."""
        return [tool.spec for tool in self._tools.values()] + [
            {
                "name": "finish",
                "description": "Report your conclusion. Benchmark success is judged by the environment.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "status": {
                            "type": "string",
                            "enum": ["success", "failure", "stuck"],
                        },
                        "summary": {"type": "string"},
                    },
                    "required": ["status", "summary"],
                    "additionalProperties": False,
                },
            }
        ]

    async def _call(self, tool: _RemoteTool, arguments: dict[str, Any]) -> Any:
        return await self._sessions[tool.server].call_tool(tool.name, arguments)

    def execute_tool(self, name: str, input_dict: dict[str, Any]) -> ToolResult:
        """Call one remote tool while preserving its MCP text and image blocks."""
        if name == "finish":
            status = input_dict.get("status")
            summary = input_dict.get("summary")
            if status not in {"success", "failure", "stuck"} or not isinstance(
                summary, str
            ):
                return ToolResult(
                    name=name, result={"error": "invalid finish arguments"}
                )
            return ToolResult(
                name=name,
                result={"_finish": True, "status": status, "summary": summary},
            )
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult(name=name, result={"error": f"unknown tool: {name}"})
        with self._call_lock:
            if self._loop is None:
                raise RuntimeError("MCP toolkit is closed")
            future = asyncio.run_coroutine_threadsafe(
                self._call(tool, input_dict), self._loop
            )
            try:
                response = future.result()
            except Exception as exc:
                return ToolResult(name=name, result={"error": str(exc)})
        blocks: list[dict[str, Any]] = []
        for item in response.content:
            if item.type == "text":
                blocks.append({"type": "text", "text": item.text})
            elif item.type == "image":
                blocks.append(
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": item.mimeType,
                            "data": item.data,
                        },
                    }
                )
            else:
                blocks.append(
                    {
                        "type": "text",
                        "text": item.model_dump_json(by_alias=True),
                    }
                )
        structured = getattr(response, "structuredContent", None)
        if structured is None:
            text_blocks = [block["text"] for block in blocks if block["type"] == "text"]
            if len(text_blocks) == 1:
                try:
                    structured = json.loads(text_blocks[0])
                except json.JSONDecodeError:
                    structured = {"text": text_blocks[0]}
            else:
                structured = {"text": "\n".join(text_blocks)}
        result_dict = (
            structured if isinstance(structured, dict) else {"value": structured}
        )
        if response.isError:
            result_dict.setdefault("error", "MCP tool failed")
        result = ToolResult(
            name=name,
            result=result_dict,
        )
        result.is_finish = False
        result.content_blocks = blocks or result.content_blocks
        return result

    def cancel_active_and_wait(self) -> None:
        """Wait for an active physical action before releasing its transport."""
        with self._call_lock:
            return None

    def close(self) -> None:
        """Close owned MCP sessions after outstanding calls finish."""
        with self._call_lock:
            if self._loop is not None and self._stop is not None:
                self._loop.call_soon_threadsafe(self._stop.set)
            if self._thread is not None:
                self._thread.join()
            self._thread = None
            self._loop = None


@dataclass(slots=True)
class EmbodiedAgent:
    """Run RPent with benchmark-provided instructions and MCP robot tools.

    Construct once per evaluation configuration, then call :meth:`run` for
    each episode. The benchmark owns reset, success checks, and scoring.
    """

    mcp_servers: Sequence[McpServer]
    output_dir: str | Path
    planner: str = "api"
    model: str | None = None
    base_url: str | None = None
    llm: LLMConfig | None = None
    max_turns: int = 100
    max_tokens: int = 8192
    planner_timeout_s: int | None = None
    reasoning_effort: str = "none"
    _run_lock: threading.Lock = field(
        default_factory=threading.Lock, init=False, repr=False
    )

    def run(
        self,
        task: str,
        *,
        system_prompt: str,
        skills: Sequence[str | Path] = (),
        memory: Sequence[ContextDocument] = (),
        initial_context: Sequence[str | BinaryContent] = (),
        output_dir: str | Path | None = None,
    ) -> PlannerResult:
        """Run one episode, loading each supplied ``SKILL.md`` into context.

        Args:
            task: Episode instruction or benchmark task description.
            system_prompt: Benchmark and robot-specific rules.
            skills: Paths to skill Markdown files. Each file is read fresh for
                this episode, so task-specific files can change between runs.
            memory: Selected, authorized memory excerpts with optional source
                identifiers. Appended to the task as reference context, not
                system instructions. No memory files are read automatically.
            initial_context: Text and images placed after the task in the first
                user message. A stable prefix can be reused by prompt caching.
            output_dir: Optional per-episode artifact directory. Defaults to
                the directory given to the agent constructor.

        Returns:
            RPent's planner result. Its ``finish_result`` is self-reported;
            callers must use the benchmark's own success signal for scoring.
        """
        if self.planner not in {"api", "claude_code", "codex"}:
            raise ValueError(f"unsupported embodied planner: {self.planner}")
        if self.llm is not None and self.planner != "api":
            raise ValueError("llm is supported only by the api planner")
        if initial_context and self.planner != "api":
            raise ValueError("initial_context requires the api planner")
        if self.llm is not None and (
            self.model is not None or self.base_url is not None
        ):
            raise ValueError("pass either llm or model/base_url")
        if not self.mcp_servers:
            raise ValueError("at least one MCP server is required")
        names = [server.name for server in self.mcp_servers]
        if len(names) != len(set(names)):
            raise ValueError("MCP server names must be unique")
        if not task.strip() or not system_prompt.strip():
            raise ValueError("task and system_prompt must be non-empty")
        if self.max_turns < 1:
            raise ValueError("max_turns must be positive")
        context = assemble_context(
            prompt=system_prompt,
            query=task,
            memory=memory,
            skills=[load_skill(path) for path in skills],
            initial_context=initial_context,
        )
        output = Path(output_dir if output_dir is not None else self.output_dir)
        output.mkdir(parents=True, exist_ok=True)
        if not self._run_lock.acquire(blocking=False):
            raise RuntimeError("EmbodiedAgent already has an active episode")
        toolkit = _McpToolkit(self.mcp_servers, output)
        try:
            toolkit.start()
            planner = build_planner(
                self.planner,
                output_dir=output,
                recipe_tag="embodied_agent",
                robot_name="embodied_agent",
                base_url=self.base_url,
                model=self.model,
                llm_config=self.llm,
                max_tokens=self.max_tokens,
                planner_timeout_s=self.planner_timeout_s,
                reasoning_effort=self.reasoning_effort,
                dashboard_events=NullDashboardEventSink(),
            )
            result = planner.solve(
                system_prompt=context.system_prompt,
                user_message=context.user_message,
                toolkit=toolkit,
                max_turns=self.max_turns,
            )
            result.stats["llm_usage"] = LLMUsage.from_planner_stats(
                result.stats
            ).as_dict()
            return result
        finally:
            toolkit.close()
            self._run_lock.release()
