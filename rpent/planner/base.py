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

"""Shared abstract base class for high-level reasoning backends."""

from __future__ import annotations

import asyncio
import os
import queue
from abc import ABC, abstractmethod
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import unquote, urlsplit, urlunsplit

from rpent.dashboard.events import DashboardEventSink
from rpent.dashboard.interaction import DashboardInteractionPort
from rpent.tools.base import ToolResult
from rpent.tools.toolkit import Toolkit
from rpent.utils.config import (
    get_memory_dir,
    get_repo_root,
)

if TYPE_CHECKING:
    from pydantic_ai.models import Model

#: MCP namespace prefix for RPent tools (``mcp__<server>__<tool>``).
#: Toolkits expose plain tool names; planners add/strip this prefix.
MCP_TOOL_PREFIX = "mcp__rpent__"
REASONING_EFFORTS = ("none", "low", "medium", "high", "xhigh")
CODEX_SANDBOX_CHOICES = ("read-only", "workspace-write", "full-access")


def resolve_model(planner: str, model: str | None) -> str | None:
    """Resolve the model using the planner's explicit/environment precedence."""
    return model or (os.environ.get("CODEX_MODEL") if planner == "codex" else None)


def add_mcp_prefix(name: str) -> str:
    """Return the namespaced MCP tool name for a bare tool name."""
    if name.startswith(MCP_TOOL_PREFIX):
        return name
    return f"{MCP_TOOL_PREFIX}{name}"


def strip_mcp_prefix(name: str) -> str:
    """Return the bare tool name, dropping the MCP namespace if present."""
    return name.removeprefix(MCP_TOOL_PREFIX)


def _normalize_google_endpoint(
    model: str, base_url: str | None
) -> tuple[str, str | None]:
    """Split a complete Gemini generateContent URL for the Google SDK."""
    if not base_url or not model.startswith("google:"):
        return model, base_url

    parsed = urlsplit(base_url)
    root_path, marker, endpoint = parsed.path.rpartition("/models/")
    if (
        not marker
        or not root_path.endswith("/v1beta")
        or not endpoint.endswith(":generateContent")
    ):
        return model, base_url

    endpoint_model = endpoint.removesuffix(":generateContent")
    if not endpoint_model or "/" in endpoint_model:
        return model, base_url

    if model.removeprefix("google:") != unquote(endpoint_model):
        raise ValueError("--model does not match the model in the Gemini endpoint URL")

    provider_path = root_path.removesuffix("/v1beta")
    provider_base_url = urlunsplit(parsed._replace(path=provider_path))
    return model, provider_base_url


async def _wait_for_cleanup(worker: asyncio.Future) -> None:
    """Finish cleanup asynchronously even if the caller is cancelled again."""
    cancelled = False
    while not worker.done():
        try:
            await asyncio.shield(worker)
        except asyncio.CancelledError:
            cancelled = True
    worker.result()
    if cancelled:
        raise asyncio.CancelledError


async def cancel_and_wait(cancel: Callable[[], None]) -> None:
    """Drain tools without competing for their potentially saturated worker pool."""
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="tool-control") as pool:
        worker = asyncio.get_running_loop().run_in_executor(pool, cancel)
        await _wait_for_cleanup(worker)


async def execute_tool(toolkit: Toolkit, name: str, arguments: dict) -> ToolResult:
    """Retain tool workers, including queued executor jobs, through cancellation."""
    worker = asyncio.create_task(
        asyncio.to_thread(toolkit.execute_tool, name, arguments)
    )
    try:
        return await asyncio.shield(worker)
    except asyncio.CancelledError:
        try:
            await cancel_and_wait(toolkit.cancel_active_and_wait)
        finally:
            await _wait_for_cleanup(worker)
        raise


class InvalidActionResponse(ValueError):
    """A terminal policy-protocol violation, not a transport or setup failure."""


class PlannerResult:
    """Result returned by a planner invocation.

    ``stop_reason`` optionally describes why planning ended, independently
    of environment success. For example, reaching a request limit can stop
    planning without setting ``error``.
    """

    __slots__ = (
        "finish_result",
        "messages",
        "stats",
        "error",
        "stop_reason",
        "error_kind",
    )

    def __init__(
        self,
        *,
        finish_result: dict | None = None,
        messages: list[dict] | None = None,
        stats: dict | None = None,
        error: str | None = None,
        stop_reason: str | None = None,
        error_kind: str | None = None,
    ):
        """Initialize a serializable planner result."""
        self.finish_result = (
            finish_result  # {"status": "success"/"failure"/"stuck", "summary": "..."}
        )
        self.messages = messages or []  # serialisable conversation transcript
        self.stats = (
            stats or {}
        )  # {"total_input_tokens", "total_output_tokens", "turns_used", "tool_calls"}
        self.error = error  # str | None  — set when the planner raises
        self.stop_reason = stop_reason
        self.error_kind = error_kind


def is_recoverable_agent_error(agent_error: str | None) -> bool:
    """Whether the error allows a fresh exploration session."""
    lowered = (agent_error or "").lower()
    return "timed out" in lowered or "planner response truncated" in lowered


class Planner(ABC):
    """A planner selects or supplies the actions used to solve a task.

    It is given one system prompt, one initial user message, and a set of
    tool definitions.  It returns a ``PlannerResult`` after the task is
    finished or the turn budget is exhausted.
    """

    @abstractmethod
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
        """Run the multi-turn agent loop until completion or budget.

        Args:
            system_prompt: System-level instructions (role, rules, workflow).
            user_message: Initial user message (task description, first steps).
            toolkit: The full :class:`~rpent.tools.toolkit.Toolkit`
                (common + robot tools). Backends derive ``tools_spec`` via
                ``toolkit.list_tools()`` and dispatch calls via
                ``toolkit.execute_tool()``.
            max_turns: Maximum LLM turns before giving up.
            input_queue: Optional queue of user-typed lines for interactive steering.
            dashboard_interaction: Optional Dashboard interaction channel.

        Returns:
            ``PlannerResult`` with finish status, conversation transcript,
            token-usage stats, and optional error string.
        """
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Planner construction
# ---------------------------------------------------------------------------


def build_api_model(
    model: str | None,
    base_url: str | None = None,
    *,
    strict_tool_definitions: bool = True,
    request_trace_dir: str | Path | None = None,
) -> "Model":
    """Resolve the pydantic-ai model used by the API planner.

    This is the single provider-resolution path: both :func:`build_planner`
    and the connectivity check in :mod:`rpent.planner.check` call it, so a
    passing check constructs the same model a real run will.

    The API key is always read from the provider's own env vars (e.g.
    ``ANTHROPIC_API_KEY``, ``GEMINI_API_KEY``, ``OPENAI_API_KEY``). When
    ``base_url`` is given it overrides the provider's default endpoint or base
    URL environment variable.

    Args:
        model: Provider-prefixed model id, e.g. ``anthropic:claude-opus-4-8``.
        base_url: Base URL overriding the provider's own env var.

    Returns:
        The resolved pydantic-ai ``Model``.

    Raises:
        ValueError: If ``model`` is empty or conflicts with the endpoint model.
    """
    if not model:
        raise ValueError(
            "the 'api' planner requires a model id; pass --model with a "
            "provider prefix (e.g. 'anthropic:claude-opus-4-8', "
            "'google:Argon', 'openai:gpt-5.5', "
            "'openai-chat:glm-5.2')."
        )

    if model.startswith("google:"):
        base_url = base_url or os.environ.get("GEMINI_BASE_URL")
    model, base_url = _normalize_google_endpoint(model, base_url)

    import inspect

    from pydantic_ai.models import infer_model
    from pydantic_ai.providers import infer_provider, infer_provider_class

    trace = None
    admission = None
    if os.environ.get("RPENT_EXCLUDE_MODEL_QUEUE") == "1":
        from rpent.planner.admission import AdmissionClock

        admission = AdmissionClock(request_trace_dir)
    if request_trace_dir is not None:
        from rpent.planner.request_trace import RequestTrace, TraceTransport

        trace = RequestTrace(request_trace_dir)

    def _provider_factory(provider_name: str):
        """Build the provider for ``provider_name``."""
        if not base_url and trace is None and admission is None:
            return infer_provider(provider_name)
        provider_cls = infer_provider_class(provider_name)
        params = inspect.signature(provider_cls.__init__).parameters
        kwargs = {}
        if "base_url" in params:
            kwargs["base_url"] = base_url
        if trace is not None or admission is not None:
            import httpx

            if "http_client" not in params:
                raise ValueError(
                    "request recording requires an HTTP-compatible provider"
                )
            transport = None
            if admission is not None:
                from rpent.planner.admission import AdmissionTransport

                transport = AdmissionTransport(admission)
            if trace is not None:
                transport = TraceTransport(trace, wrapped=transport)
            kwargs["http_client"] = httpx.AsyncClient(
                transport=transport, timeout=httpx.Timeout(14400.0)
            )
        return provider_cls(**kwargs)

    resolved = _as_qwen_chat_model(
        infer_model(model, provider_factory=_provider_factory)
    )
    if not strict_tool_definitions:
        resolved = _without_strict_tool_definitions(resolved)
    if trace is not None:
        from rpent.planner.qwen import QwenChatModel

        if not isinstance(resolved, QwenChatModel):
            raise ValueError(
                "native request recording currently requires a Qwen chat model"
            )
        resolved.request_trace = trace
    if admission is not None:
        from rpent.planner.qwen import QwenChatModel

        if not isinstance(resolved, QwenChatModel):
            raise ValueError("Model queue exclusion requires a Qwen chat model")
        resolved.admission_clock = admission
    return resolved


def _as_qwen_chat_model(model: "Model") -> "Model":
    """Serve a Qwen checkpoint through the adapter that keeps its reasoning.

    Args:
        model: The model resolved by :func:`build_api_model`.

    Returns:
        A :class:`~rpent.planner.qwen.QwenChatModel` for a Qwen name behind an
        OpenAI chat endpoint, or ``model`` unchanged.
    """
    from pydantic_ai.models.openai import OpenAIChatModel

    from rpent.planner.qwen import QwenChatModel, is_qwen_model

    if not isinstance(model, OpenAIChatModel) or isinstance(model, QwenChatModel):
        return model
    if not is_qwen_model(model.model_name):
        return model
    return QwenChatModel(
        model.model_name, provider=model.provider, profile=model.profile
    )


def _without_strict_tool_definitions(model: "Model") -> "Model":
    """Rebuild an OpenAI-family model with ``strict`` tool definitions off.

    A server that treats any strict tool as a structural decoding constraint
    applies that constraint to the whole tool set, which corrupts the XML the
    model emits for its calls. The request field is what triggers it, so the
    fix belongs in the model profile rather than in tool schemas.

    Args:
        model: The model resolved by :func:`build_api_model`.

    Returns:
        An equivalent model whose requests omit ``strict``, or ``model``
        unchanged when it has no such request field.
    """
    from pydantic_ai.models.openai import OpenAIChatModel, OpenAIResponsesModel

    if not isinstance(model, (OpenAIChatModel, OpenAIResponsesModel)):
        return model
    profile = {**model.profile, "openai_supports_strict_tool_definition": False}
    return type(model)(model.model_name, provider=model.provider, profile=profile)


def build_planner(
    planner_type: str,
    *,
    output_dir: str | Path,
    recipe_tag: str,
    robot_name: str,
    base_url: str | None = None,
    model: str | None = None,
    memory_dir: str | Path | None = None,
    max_tokens: int = 8192,
    planner_timeout_s: int | None = None,
    reasoning_effort: str = "none",
    codex_sandbox: str = "full-access",
    claude_code_max_budget_usd: float | None = None,
    dashboard_events: DashboardEventSink,
    no_images: bool = False,
    strict_tool_definitions: bool = True,
    interactive: bool = False,
    thinking: bool = True,
    sequential_tools: bool = False,
    context_window: int | None = None,
    compaction_fraction: float = 0.8,
    compaction_max_messages: int = 80,
    compaction_keep_messages: int = 40,
    temperature: float | None = None,
    top_p: float | None = None,
    top_k: int | None = None,
    min_p: float | None = None,
    presence_penalty: float | None = None,
    repetition_penalty: float | None = None,
) -> Planner:
    """Build a planner for the given backend, resolving credentials from env vars.

    ``thinking``, ``sequential_tools``, ``context_window`` and
    ``compaction_fraction`` apply to the ``api`` planner: the last two size its
    message-window compaction to the served model's context.
    """
    # Imports are deferred to avoid a circular import: api_loop / claude_code /
    # codex / gemini_cli all import from this module (PlannerResult).

    if planner_type == "api":
        from rpent.planner.api_loop import ApiAgentLoop, sliding_window_compaction

        recording = os.environ.get("RPENT_RECORD_MODEL_REQUESTS") == "1"
        extra = (
            {"request_trace_dir": Path(output_dir) / "api_trace"} if recording else {}
        )
        api_model = build_api_model(
            model, base_url, strict_tool_definitions=strict_tool_definitions, **extra
        )
        api_timeout_s = planner_timeout_s
        if api_timeout_s is None:
            api_timeout_s = int(os.environ.get("CELL_TIMEOUT_S", "1200"))
        sampling = {
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "min_p": min_p,
            "presence_penalty": presence_penalty,
            "repetition_penalty": repetition_penalty,
        }
        model_settings = None
        if any(value is not None for value in sampling.values()):
            from rpent.planner.qwen import QwenChatModel, qwen_model_settings

            if not isinstance(api_model, QwenChatModel):
                raise ValueError(
                    "Sampling overrides are supported by the Qwen API planner"
                )
            model_settings = qwen_model_settings(
                max_tokens=max_tokens,
                reasoning_effort=reasoning_effort,
                thinking=thinking,
                parallel_tool_calls=not sequential_tools,
                **sampling,
            )
        return ApiAgentLoop(
            model=api_model,
            max_tokens=max_tokens,
            reasoning_effort=reasoning_effort,
            dashboard_events=dashboard_events,
            no_images=no_images,
            timeout_s=api_timeout_s,
            interactive=interactive,
            thinking=thinking,
            sequential_tools=sequential_tools,
            model_settings=model_settings,
            compaction=sliding_window_compaction(
                context_window=context_window,
                max_fraction=compaction_fraction,
                max_messages=compaction_max_messages,
                keep_messages=compaction_keep_messages,
            ),
        )
    if planner_type == "claude_code":
        from rpent.planner.claude_code import ClaudeCodePlanner

        cc_timeout_s = planner_timeout_s
        if cc_timeout_s is None:
            cc_timeout_s = int(os.environ.get("CELL_TIMEOUT_S", "1200"))
        cc_budget = claude_code_max_budget_usd
        if cc_budget is None:
            cc_budget = float(os.environ.get("MAX_BUDGET_USD", "10"))
        return ClaudeCodePlanner(
            output_dir=output_dir,
            repo_root=get_repo_root(),
            model=model or "sonnet",
            timeout_s=cc_timeout_s,
            max_budget_usd=cc_budget,
            extra_dirs=[str(memory_dir or get_memory_dir(robot_name))],
            output_path=Path(output_dir) / f"claude_{recipe_tag}.txt",
            dashboard_events=dashboard_events,
            reasoning_effort=reasoning_effort,
        )
    if planner_type == "codex":
        from rpent.planner.codex import CodexPlanner

        cx_timeout_s = planner_timeout_s
        if cx_timeout_s is None:
            cx_timeout_s = int(
                os.environ.get(
                    "CODEX_TIMEOUT_S",
                    os.environ.get("CELL_TIMEOUT_S", "1200"),
                )
            )
        return CodexPlanner(
            output_dir=output_dir,
            repo_root=get_repo_root(),
            model=model,
            timeout_s=cx_timeout_s,
            extra_dirs=[str(memory_dir or get_memory_dir(robot_name))],
            output_path=Path(output_dir) / f"codex_{recipe_tag}.txt",
            dashboard_events=dashboard_events,
            reasoning_effort=reasoning_effort,
            sandbox=codex_sandbox,
        )
    if planner_type == "gemini_cli":
        from rpent.planner.gemini_cli import GeminiCliPlanner

        gemini_timeout_s = planner_timeout_s
        if gemini_timeout_s is None:
            gemini_timeout_s = int(
                os.environ.get(
                    "GEMINI_CLI_TIMEOUT_S",
                    os.environ.get("CELL_TIMEOUT_S", "1200"),
                )
            )
        return GeminiCliPlanner(
            output_dir=str(output_dir),
            model=model,
            timeout_s=gemini_timeout_s,
            output_path=Path(output_dir) / f"gemini_{recipe_tag}.txt",
            dashboard_events=dashboard_events,
            reasoning_effort=reasoning_effort,
        )
    if planner_type == "flash":
        from rpent.planner.flash import FlashPlanner

        return FlashPlanner(
            recipe_tag=recipe_tag,
            robot_name=robot_name,
        )
    raise ValueError(f"unknown planner_type: {planner_type}")
