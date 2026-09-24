# Copyright 2026 The RPent Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# https://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS.

"""RoboDojo decision adapter using RPent's MCP-based embodied agent.

The simulator, motion planner, observation annotations, and native scoring stay
with RoboDojo and RoboDawn. This module replaces only RoboDawn's VLM client.
"""

from __future__ import annotations

import json
import os
import re
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from rpent.embodied_agent import EmbodiedAgent, McpServer
from rpent.llm import LLMConfig


class RoboDojoEmbodiedClient:
    """Supply each RoboDawn decision through two user-owned MCP tools.

    ``snapshot`` exposes the current cameras, state, and optional demonstration;
    ``submit_action`` returns a short discrete command sequence. The caller
    executes and scores the commands. Each decision has isolated files and logs.
    """

    def __init__(
        self,
        *,
        model: str,
        base_url: str,
        output_dir: str | Path,
        error_type: type[Exception],
        max_turns: int = 4,
        max_images: int = 24,
        prompt_cache_key: str | None = None,
    ) -> None:
        self.model = model
        self.base_url = base_url
        self.output_dir = Path(output_dir)
        self.error_type = error_type
        self.max_turns = max_turns
        self.max_images = max_images
        self.prompt_cache_key = prompt_cache_key or f"rpent-robodojo-v1:{model}"
        self._decision = 0

    def complete(
        self,
        messages: list[dict[str, Any]],
        max_tokens: int = 8000,
        temperature: float | None = None,
        reasoning_effort: str | None = None,
        deadline: float | None = None,
    ) -> dict[str, Any]:
        """Return RoboDawn's VLM response shape without executing an action."""
        if temperature is not None:
            raise ValueError("RoboDojo embodied adapter requires temperature=None")
        if not messages or messages[0].get("role") != "system":
            raise ValueError("RoboDojo messages must start with a system prompt")
        remaining = None if deadline is None else deadline - time.monotonic()
        if remaining is not None and remaining <= 0:
            raise self.error_type("RoboDojo decision deadline expired")

        self.output_dir.mkdir(parents=True, exist_ok=True)
        while True:
            self._decision += 1
            decision_dir = self.output_dir / f"decision_{self._decision:04d}"
            try:
                decision_dir.mkdir()
                break
            except FileExistsError:
                continue
        started = time.monotonic()
        with tempfile.TemporaryDirectory(prefix="mcp_", dir=decision_dir) as temp:
            snapshot = Path(temp) / "snapshot.json"
            action = Path(temp) / "action.json"
            snapshot.write_text(
                json.dumps(_snapshot_messages(messages[1:], self.max_images)),
                encoding="utf-8",
            )
            agent = EmbodiedAgent(
                mcp_servers=[
                    McpServer(
                        name="dojo",
                        command=sys.executable,
                        args=(
                            "-m",
                            "rpent.benchmarks.robodojo_mcp",
                            str(snapshot),
                            str(action),
                        ),
                        env={"PYTHONPATH": os.environ.get("PYTHONPATH", "")},
                    )
                ],
                output_dir=decision_dir,
                llm=LLMConfig(
                    provider="openai",
                    model=self.model,
                    base_url=self.base_url,
                    openai_format="responses",
                    prompt_cache_key=self.prompt_cache_key,
                    prompt_cache_mode="explicit",
                    image_history_groups=2,
                ),
                max_turns=self.max_turns,
                max_tokens=max_tokens,
                planner_timeout_s=max(1, int(remaining))
                if remaining is not None
                else None,
                reasoning_effort=reasoning_effort or "none",
            )
            result = agent.run(
                "Call dojo__snapshot to inspect the cameras and state. Then call "
                "dojo__submit_action once with your next discrete command sequence. "
                "After submitting, call finish. The simulator executes the submitted commands.",
                system_prompt=_tool_system_prompt(str(messages[0]["content"])),
            )
            if not action.is_file():
                error = _safe_error(result.error)
                raise self.error_type(
                    f"EmbodiedAgent produced no RoboDojo action: {error}",
                    attempts=int(result.stats.get("requests") or 0),
                    latency_s=time.monotonic() - started,
                    status_code=_provider_status_code(result.error),
                )
            payload = json.loads(action.read_text(encoding="utf-8"))
            usage = result.stats.get("llm_usage") or {}
            return {
                "text": json.dumps(payload, ensure_ascii=False),
                "usage": {
                    "prompt_tokens": usage.get("input_tokens", 0),
                    "completion_tokens": usage.get("output_tokens", 0),
                    "prompt_tokens_details": {
                        "cached_tokens": usage.get("cache_read_tokens", 0),
                        "cache_write_tokens": usage.get("cache_write_tokens", 0),
                    },
                },
                "finish_reason": "tool_call",
                "latency_s": time.monotonic() - started,
                "attempts": int(usage.get("requests") or 0),
                "retry_history": [],
            }


def _snapshot_messages(
    messages: list[dict[str, Any]], max_images: int
) -> dict[str, Any]:
    """Bound demo images while always retaining the current three cameras."""
    if max_images < 3:
        raise ValueError("max_images must be at least 3")
    output: list[dict[str, str]] = []
    current = messages[-1] if messages else {}
    prior = messages[:-1]
    current_images = sum(
        part.get("type") == "image_url"
        for part in current.get("content", [])
        if isinstance(part, dict)
    )
    demo_budget = max(0, max_images - current_images)
    for index, message in enumerate([*prior, current]):
        content = message.get("content", "")
        if isinstance(content, str):
            output.append(
                {"type": "text", "text": f"{message.get('role', 'user')}: {content}"}
            )
            continue
        for part in content:
            if part.get("type") == "text":
                prefix = (
                    "Current observation" if index == len(prior) else "Demonstration"
                )
                text = str(part["text"])
                if index == len(prior):
                    text = text.replace(
                        "Reply with the JSON object.",
                        "Submit the next commands with dojo__submit_action.",
                    )
                output.append({"type": "text", "text": f"{prefix}: {text}"})
            elif part.get("type") == "image_url":
                if index < len(prior):
                    if demo_budget <= 0:
                        continue
                    demo_budget -= 1
                url = part.get("image_url", {}).get("url", "")
                if not url.startswith("data:image/jpeg;base64,"):
                    raise ValueError("RoboDojo snapshot requires JPEG data URLs")
                output.append({"type": "image", "data": url.split(",", 1)[1]})
    return {"content": output}


def _tool_system_prompt(prompt: str) -> str:
    """Retain RoboDawn's robot rules while replacing its text-only output format."""
    prefix = prompt.split("\nRESPONSE FORMAT:", 1)[0]
    return (
        prefix
        + "\nTOOL WORKFLOW: Call dojo__snapshot first to view the current cameras, "
        "state, feedback, and demonstration. Then call dojo__submit_action with "
        "scene, progress, memory, plan, and 1-4 discrete commands. The tool "
        "schema replaces the JSON response format. Call finish only after "
        "submit_action. The benchmark ends the episode when its checker records "
        "success; use the done command only when no further action is possible."
    )


def _safe_error(error: str | None) -> str:
    if not error:
        return "no submit_action call"
    # Provider errors occasionally include an echoed URL or bearer header.
    return re.sub(r"(?i)bearer\s+\S+", "Bearer [redacted]", error)[:300]


def _provider_status_code(error: str | None) -> int | None:
    """Preserve HTTP status so RoboDawn can stop on permanent API errors."""
    match = re.search(r"ModelHTTPError: status_code:\s*(\d{3})", error or "")
    return int(match.group(1)) if match else None
