"""Shared protocol for high-level reasoning backends."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Protocol

from physical_agent.utils.config import (
    get_anthropic_api_key,
    get_anthropic_base_url,
    get_anthropic_model,
    get_memory_dir,
    get_openai_compat_api_key,
    get_openai_compat_base_url,
    get_openai_compat_model,
    get_repo_root,
)
from physical_agent.tools.toolkit import Toolkit


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
        self.finish_result = finish_result  # {"status": "success"/"failure"/"stuck", "summary": "..."}
        self.messages = messages or []       # serialisable conversation transcript
        self.stats = stats or {}             # {"total_input_tokens", "total_output_tokens", "turns_used", "tool_calls"}
        self.error = error                   # str | None  — set when the cerebrum raises


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
    ) -> CerebrumResult:
        """Run the multi-turn agent loop until completion or budget.

        Args:
            system_prompt: System-level instructions (role, rules, workflow).
            user_message: Initial user message (task description, first steps).
            toolkit: The full :class:`~physical_agent.tools.toolkit.Toolkit`
                (common + env tools). Backends derive ``tools_spec`` via
                ``toolkit.get_tools_spec()`` and dispatch calls via
                ``toolkit.execute_tool()``; MCP-based backends also use
                ``toolkit.allowed_mcp_tool_names`` and the driver lifecycle
                hooks.
            max_turns: Maximum LLM turns before giving up.

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
    env_name: str,
    recipe_tag: str,
    api_key: str | None = None,
    base_url: str | None = None,
    model: str | None = None,
    max_tokens: int = 4096,
    perception: bool = False,
    thinking: bool = False,
    claude_code_timeout_s: int | None = None,
    claude_code_max_budget_usd: float | None = None,
    codex_timeout_s: int | None = None,
    transport_host: str = "127.0.0.1",
    transport_port: int = 0,
):
    """Build a cerebrum for the given backend, resolving credentials from env vars."""
    # Imports are deferred to avoid a circular import: api_loop / claude_code /
    # codex all import from this module (CerebrumResult).
    from physical_agent.cerebrum.adapters.anthropic import AnthropicAdapter
    from physical_agent.cerebrum.adapters.openai_compat import (
        OpenAICompatibleAdapter,
    )
    from physical_agent.cerebrum.api_loop import ApiAgentLoop
    from physical_agent.cerebrum.claude_code import ClaudeCodeCerebrum
    from physical_agent.cerebrum.codex import CodexCerebrum

    if cerebrum_type == "anthropic":
        api_key = api_key or get_anthropic_api_key()
        base_url = base_url or get_anthropic_base_url()
        if not api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY env var or --api_key must be set"
            )
        import anthropic
        client = anthropic.Anthropic(
            api_key=api_key,
            max_retries=8,
            timeout=120.0,
            **({"base_url": base_url} if base_url else {}),
        )
        return ApiAgentLoop(
            adapter=AnthropicAdapter(
                client=client,
                model=model or get_anthropic_model(),
                max_tokens=max_tokens,
                thinking=thinking,
            )
        )
    if cerebrum_type == "openai_compat":
        api_key = api_key or get_openai_compat_api_key()
        base_url = base_url or get_openai_compat_base_url()
        if not api_key:
            raise ValueError(
                "OPENAI_COMPAT_API_KEY or OPENAI_API_KEY or --api_key must be set"
            )
        import openai
        client_kwargs = {"api_key": api_key, "max_retries": 0, "timeout": 120.0}
        if base_url:
            client_kwargs["base_url"] = base_url
        client = openai.OpenAI(**client_kwargs)
        return ApiAgentLoop(
            adapter=OpenAICompatibleAdapter(
                client=client,
                model=model or get_openai_compat_model(),
                max_tokens=max_tokens,
                thinking=thinking,
            )
        )
    if cerebrum_type == "claude_code":
        cc_timeout_s = claude_code_timeout_s
        if cc_timeout_s is None:
            cc_timeout_s = int(os.environ.get("CELL_TIMEOUT_S", "1200" if perception else "600"))
        cc_budget = claude_code_max_budget_usd
        if cc_budget is None:
            cc_budget = float(os.environ.get("MAX_BUDGET_USD", "10"))
        return ClaudeCodeCerebrum(
            output_dir=output_dir,
            repo_root=get_repo_root(),
            model=model or "sonnet",
            timeout_s=cc_timeout_s,
            max_budget_usd=cc_budget,
            extra_dirs=[str(get_memory_dir())],
            output_path=Path(output_dir) / f"claude_{recipe_tag}.txt",
            hide_object_coords=perception,
            video_path=str(Path(output_dir) / "episode.mp4"),
        )
    if cerebrum_type == "codex":
        cx_timeout_s = codex_timeout_s
        if cx_timeout_s is None:
            cx_timeout_s = int(os.environ.get(
                "CODEX_TIMEOUT_S",
                os.environ.get("CELL_TIMEOUT_S", "1200" if perception else "600"),
            ))
        return CodexCerebrum(
            output_dir=output_dir,
            repo_root=get_repo_root(),
            model=model,
            timeout_s=cx_timeout_s,
            extra_dirs=[str(get_memory_dir())],
            output_path=Path(output_dir) / f"codex_{recipe_tag}.txt",
            transport_host=transport_host,
            transport_port=transport_port,
            env_name=env_name,
            hide_object_coords=perception,
            video_path=str(Path(output_dir) / "episode.mp4"),
        )
    raise ValueError(f"unknown cerebrum_type: {cerebrum_type}")
