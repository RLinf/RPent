"""Shared protocol for high-level reasoning backends."""

from __future__ import annotations

import os
import queue
from pathlib import Path
from typing import Protocol

from rpent.cli.tui import QUIT_TOKENS, START_TOKENS
from rpent.tools.toolkit import Toolkit
from rpent.utils.config import (
    get_memory_dir,
    get_repo_root,
)

#: MCP namespace prefix for RPent tools (``mcp__<server>__<tool>``).
#: Toolkits expose plain tool names; cerebrums add/strip this prefix.
MCP_TOOL_PREFIX = "mcp__rpent__"


def add_mcp_prefix(name: str) -> str:
    """Return the namespaced MCP tool name for a bare tool name."""
    if name.startswith(MCP_TOOL_PREFIX):
        return name
    return f"{MCP_TOOL_PREFIX}{name}"


def strip_mcp_prefix(name: str) -> str:
    """Return the bare tool name, dropping the MCP namespace if present."""
    return name.removeprefix(MCP_TOOL_PREFIX)


def next_user_line(input_queue: "queue.Queue[str | None]") -> str | None:
    """Block for the next actionable user line from an interactive input queue.

    Returns the trimmed line, or ``None`` when the session should end (the queue
    yielded ``None`` or a quit token such as ``/quit``). Empty lines are skipped.
    This is a blocking call; async callers should wrap it with
    :func:`asyncio.to_thread`.
    """
    while True:
        line = input_queue.get()
        if line is None:
            return None
        line = line.strip()
        if line.lower() in QUIT_TOKENS:
            return None
        if line:
            return line


def initial_user_message(
    input_queue: "queue.Queue[str | None]", default_message: str
) -> str | None:
    """Block for the first user turn of an interactive session.
    This is a blocking call; async callers should wrap it with :func:`asyncio.to_thread`.
    """
    while True:
        line = input_queue.get()
        if line is None:
            return None
        line = line.strip()
        if line.lower() in QUIT_TOKENS:
            return None
        if line.lower() in START_TOKENS:
            return default_message
        if line:
            return line


class CerebrumResult:
    """Result returned by a cerebrum invocation."""

    __slots__ = ("finish_result", "messages", "stats", "error")

    def __init__(
        self,
        *,
        finish_result: dict | None = None,
        messages: list[dict] | None = None,
        stats: dict | None = None,
        error: str | None = None,
    ):
        """Initialize a serializable cerebrum result."""
        self.finish_result = (
            finish_result  # {"status": "success"/"failure"/"stuck", "summary": "..."}
        )
        self.messages = messages or []  # serialisable conversation transcript
        self.stats = (
            stats or {}
        )  # {"total_input_tokens", "total_output_tokens", "turns_used", "tool_calls"}
        self.error = error  # str | None  — set when the cerebrum raises


class Cerebrum(Protocol):
    """A cerebrum solves a task by conversing with an LLM/VLM backend.

    It is given one system prompt, one initial user message, and a set of
    tool definitions.  It returns a ``CerebrumResult`` after the task is
    finished or the turn budget is exhausted.
    """

    def solve(
        self,
        *,
        system_prompt: str,
        user_message: str,
        toolkit: Toolkit,
        max_turns: int,
        input_queue: queue.Queue[str | None] | None = None,
    ) -> CerebrumResult:
        """Run the multi-turn agent loop until completion or budget.

        Args:
            system_prompt: System-level instructions (role, rules, workflow).
            user_message: Initial user message (task description, first steps).
            toolkit: The full :class:`~rpent.tools.toolkit.Toolkit`
                (common + env tools). Backends derive ``tools_spec`` via
                ``toolkit.get_tools_spec()`` and dispatch calls via
                ``toolkit.execute_tool()``.
            max_turns: Maximum LLM turns before giving up.
            input_queue: Optional queue of user-typed lines enabling interactive
                mode.

        Returns:
            ``CerebrumResult`` with finish status, conversation transcript,
            token-usage stats, and optional error string.
        """
        ...


# ---------------------------------------------------------------------------
# Cerebrum construction
# ---------------------------------------------------------------------------


def build_cerebrum(
    cerebrum_type: str,
    *,
    output_dir: str | Path,
    recipe_tag: str,
    env_name: str,
    base_url: str | None = None,
    model: str | None = None,
    max_tokens: int = 8192,
    cerebrum_timeout_s: int | None = None,
    claude_code_max_budget_usd: float | None = None,
    dashboard: Any = None,
):
    """Build a cerebrum for the given backend, resolving credentials from env vars."""
    # Imports are deferred to avoid a circular import: api_loop / claude_code /
    # codex all import from this module (CerebrumResult).

    if cerebrum_type == "api":
        if not model:
            raise ValueError(
                "the 'api' cerebrum requires a model id; pass --model with a "
                "provider prefix (e.g. 'anthropic:claude-opus-4-8', "
                "'openai:gpt-5.5', 'openai-chat:glm-5.2')."
            )

        import inspect

        from pydantic_ai.models import infer_model
        from pydantic_ai.providers import infer_provider, infer_provider_class

        from rpent.cerebrum.api_loop import ApiAgentLoop

        def _provider_factory(provider_name: str):
            """Build the provider for ``provider_name``.

            The API key is always read from the provider's own env vars
            (e.g. ``ANTHROPIC_API_KEY``, ``OPENAI_API_KEY``). When
            ``base_url`` is given it overrides the provider's base URL env
            var (e.g. ``ANTHROPIC_BASE_URL`` / ``OPENAI_BASE_URL``).
            """
            if not base_url:
                return infer_provider(provider_name)
            provider_cls = infer_provider_class(provider_name)
            params = inspect.signature(provider_cls.__init__).parameters
            kwargs = {}
            if "base_url" in params:
                kwargs["base_url"] = base_url
            return provider_cls(**kwargs)

        api_model = infer_model(
            model, provider_factory=_provider_factory
        )
        return ApiAgentLoop(model=api_model, max_tokens=max_tokens)
    if cerebrum_type == "claude_code":
        from rpent.cerebrum.claude_code import ClaudeCodeCerebrum

        cc_timeout_s = cerebrum_timeout_s
        if cc_timeout_s is None:
            cc_timeout_s = int(os.environ.get("CELL_TIMEOUT_S", "1200"))
        cc_budget = claude_code_max_budget_usd
        if cc_budget is None:
            cc_budget = float(os.environ.get("MAX_BUDGET_USD", "10"))
        return ClaudeCodeCerebrum(
            output_dir=output_dir,
            repo_root=get_repo_root(),
            model=model or "sonnet",
            timeout_s=cc_timeout_s,
            max_budget_usd=cc_budget,
            extra_dirs=[str(get_memory_dir(env_name))],
            output_path=Path(output_dir) / f"claude_{recipe_tag}.txt",
            dashboard=dashboard,
        )
    if cerebrum_type == "codex":
        from rpent.cerebrum.codex import CodexCerebrum

        cx_timeout_s = cerebrum_timeout_s
        if cx_timeout_s is None:
            cx_timeout_s = int(
                os.environ.get(
                    "CODEX_TIMEOUT_S",
                    os.environ.get("CELL_TIMEOUT_S", "1200"),
                )
            )
        return CodexCerebrum(
            output_dir=output_dir,
            repo_root=get_repo_root(),
            timeout_s=cx_timeout_s,
            extra_dirs=[str(get_memory_dir(env_name))],
            output_path=Path(output_dir) / f"codex_{recipe_tag}.txt",
        )
    raise ValueError(f"unknown cerebrum_type: {cerebrum_type}")
