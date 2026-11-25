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

"""Qwen chat-completions adapter: thinking on every turn, tool images kept in the tool loop.

Qwen3.5 and Qwen3.8 are served through SGLang's OpenAI-compatible endpoint and
through RLinf's policy endpoint. Both render the model's own chat template, and
two of its rules decide whether earlier reasoning survives:

* Qwen3.5 keeps an assistant turn's ``<think>`` block only when the turn comes
  after the last ``user`` message whose text is not framed by
  ``<tool_response>`` tags. pydantic-ai delivers a tool's images as a trailing
  ``user`` message, so every image-returning tool would end the tool loop and
  drop all reasoning before it. :class:`QwenChatModel` frames that message.
* ``reasoning_effort: "none"`` makes SGLang render the prompt with
  ``enable_thinking=false``. :func:`qwen_model_settings` never sends it and
  states ``enable_thinking`` explicitly instead.
"""

from __future__ import annotations

import math
import os
from collections.abc import AsyncIterator
from typing import Any

from pydantic_ai.messages import ModelRequest, ToolReturnPart
from pydantic_ai.models.openai import OpenAIChatModel, OpenAIChatModelSettings

#: Qwen's recommended thinking-mode sampling; also what a model's
#: ``generation_config.json`` supplies when it ships one.
SAMPLING = {"temperature": 1.0, "top_p": 0.95, "top_k": 20}
TOOL_RESPONSE_OPEN = "<tool_response>\n"
TOOL_RESPONSE_CLOSE = "\n</tool_response>"


def is_qwen_model(model_name: str) -> bool:
    """Whether a served model name denotes a Qwen checkpoint."""
    return model_name.split("/")[-1].lower().startswith("qwen")


def qwen_model_settings(
    *,
    max_tokens: int,
    reasoning_effort: str = "none",
    thinking: bool = True,
    parallel_tool_calls: bool = False,
    temperature: float | None = None,
    top_p: float | None = None,
    top_k: int | None = None,
    min_p: float | None = None,
    presence_penalty: float | None = None,
    repetition_penalty: float | None = None,
) -> OpenAIChatModelSettings:
    """Request settings for a Qwen chat endpoint.

    ``reasoning_effort`` is sent only as a real tier; ``"none"`` leaves the
    field out because SGLang maps it to ``enable_thinking=false``. Thinking is
    stated explicitly through ``chat_template_kwargs``; ``preserve_thinking``
    is honoured by the Qwen3.8 template and ignored by Qwen3.5.
    Optional ``RPENT_PRESENCE_PENALTY`` and ``RPENT_REPETITION_PENALTY``
    environment overrides affect sampling only; absent overrides preserve defaults.
    """
    from pydantic_ai.profiles.openai import OPENAI_REASONING_EFFORT_MAP

    temperature = SAMPLING["temperature"] if temperature is None else temperature
    top_p = SAMPLING["top_p"] if top_p is None else top_p
    top_k = SAMPLING["top_k"] if top_k is None else top_k
    min_p = 0.0 if min_p is None else min_p
    presence_penalty = (
        float(os.environ.get("RPENT_PRESENCE_PENALTY", "0"))
        if presence_penalty is None
        else presence_penalty
    )
    repetition_penalty = (
        float(os.environ.get("RPENT_REPETITION_PENALTY", "1"))
        if repetition_penalty is None
        else repetition_penalty
    )
    for name, value, valid in (
        ("temperature", temperature, temperature >= 0),
        ("top_p", top_p, 0 < top_p <= 1),
        ("min_p", min_p, 0 <= min_p <= 1),
        ("presence_penalty", presence_penalty, -2 <= presence_penalty <= 2),
        ("repetition_penalty", repetition_penalty, 0 < repetition_penalty <= 2),
    ):
        if not math.isfinite(value) or not valid:
            raise ValueError(f"invalid {name} (RPENT_{name.upper()}): {value}")
    if (
        not isinstance(top_k, int)
        or isinstance(top_k, bool)
        or top_k == 0
        or top_k < -1
    ):
        raise ValueError("top_k must be -1 (disabled) or a positive integer")
    if max_tokens < 1:
        raise ValueError("max_tokens must be positive")
    settings = OpenAIChatModelSettings(
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        presence_penalty=presence_penalty,
        parallel_tool_calls=parallel_tool_calls,
        extra_body={
            "top_k": top_k,
            "min_p": min_p,
            "repetition_penalty": repetition_penalty,
            "chat_template_kwargs": {
                "enable_thinking": thinking,
                "preserve_thinking": True,
            },
        },
    )
    if reasoning_effort == "xhigh":
        # Qwen's tier belongs to its template; older OpenAI endpoint schemas
        # accept "max" instead, which the Qwen template itself rejects.
        settings["extra_body"]["chat_template_kwargs"]["reasoning_effort"] = "xhigh"
    elif reasoning_effort != "none":
        settings["openai_reasoning_effort"] = OPENAI_REASONING_EFFORT_MAP.get(
            reasoning_effort, reasoning_effort
        )
    return settings


def _carries_tool_files(message: ModelRequest) -> bool:
    return any(
        isinstance(part, ToolReturnPart)
        and part.model_response_str_and_user_content()[1]
        for part in message.parts
    )


def _as_tool_response(user_message: dict[str, Any]) -> dict[str, Any]:
    """Frame a user message so the Qwen template reads it as a tool response."""
    content = user_message["content"]
    if isinstance(content, str):
        framed: Any = f"{TOOL_RESPONSE_OPEN}{content}{TOOL_RESPONSE_CLOSE}"
    else:
        framed = [
            {"type": "text", "text": TOOL_RESPONSE_OPEN},
            *content,
            {"type": "text", "text": TOOL_RESPONSE_CLOSE},
        ]
    return {**user_message, "content": framed}


class QwenChatModel(OpenAIChatModel):
    """``OpenAIChatModel`` whose tool-returned images stay inside the tool loop.

    The trailing ``user`` message pydantic-ai builds from a tool's files is
    framed by ``<tool_response>`` tags, so the Qwen3.5 template does not take
    it for a new user query and keeps the reasoning of every earlier turn.
    """

    request_trace = None
    tokenizer_path = None

    async def _completions_create(
        self, messages, stream, model_settings, model_request_parameters
    ):
        """Keep the post-compaction native request beside the exact HTTP payload."""
        trace = self.request_trace
        if trace is None:
            return await super()._completions_create(
                messages, stream, model_settings, model_request_parameters
            )
        from rpent.planner.request_trace import reset_native

        token = trace.native(messages, model_settings, model_request_parameters)
        try:
            return await super()._completions_create(
                messages, stream, model_settings, model_request_parameters
            )
        finally:
            reset_native(token)

    async def _map_user_message(self, message: ModelRequest) -> AsyncIterator[Any]:
        mapped = [item async for item in super()._map_user_message(message)]
        if mapped and mapped[-1].get("role") == "user" and _carries_tool_files(message):
            mapped[-1] = _as_tool_response(mapped[-1])
        for item in mapped:
            yield item

    async def _get_prompt_counter(self):
        """Resolve a local checkpoint explicitly or from a loopback SGLang server."""
        if getattr(self, "_prompt_counter_checked", False):
            return self._prompt_counter
        import asyncio
        from pathlib import Path
        from urllib.parse import urlsplit

        import httpx

        from rpent.planner.token_counting import QwenPromptCounter
        from rpent.utils.logging import get_logger

        configured = self.tokenizer_path or os.environ.get("RPENT_TOKENIZER_PATH")
        path = configured
        backend = os.environ.get("RPENT_TOKEN_COUNT_BACKEND", "auto")
        if backend not in {"auto", "sglang", "chat_template"}:
            raise ValueError(
                "RPENT_TOKEN_COUNT_BACKEND must be auto, sglang or chat_template"
            )
        sglang = backend == "sglang"
        base = urlsplit(str(self.client.base_url))
        if base.hostname in {"127.0.0.1", "localhost", "::1"}:
            try:
                async with httpx.AsyncClient(trust_env=False, timeout=5) as client:
                    result = await client.get(
                        f"{base.scheme}://{base.netloc}/get_model_info"
                    )
                    result.raise_for_status()
                    info = result.json()
                    if backend == "auto":
                        sglang = "tool_call_parser" in info
                    candidate = info.get("tokenizer_path") or info.get("model_path")
                    if path is None and candidate and Path(candidate).is_dir():
                        path = candidate
            except (httpx.HTTPError, ValueError):
                pass
        counter = None
        if path is not None:
            try:
                counter = await asyncio.to_thread(
                    QwenPromptCounter.from_checkpoint, path, sglang=sglang
                )
            except (ImportError, ValueError):
                if configured is not None:
                    raise
        if counter is None:
            get_logger("api").warning(
                "No local Qwen processor: compaction uses an image-aware text "
                "estimate. Set RPENT_TOKENIZER_PATH to the served checkpoint "
                "for exact template and vision token accounting."
            )
        self._prompt_counter, self._prompt_counter_checked = counter, True
        return counter

    async def count_prompt_tokens(self, messages, model_settings, parameters):
        """Count exactly the same chat mapping, tools and template sent on the wire."""
        import asyncio

        from rpent.planner.token_counting import PromptTokens

        settings, parameters = self.prepare_request(model_settings, parameters)
        settings = settings or {}
        counter = await self._get_prompt_counter()
        if counter is None:
            import json

            from rpent.planner.compaction import (
                count_image_tokens_in,
                estimate_request_tokens,
            )

            tools, _ = self._get_tool_choice(settings, parameters)
            return PromptTokens(
                total_tokens=estimate_request_tokens(messages)
                + len(json.dumps(tools)) // 4,
                image_tokens=count_image_tokens_in(messages),
                method="text_heuristic_with_image_geometry",
            )
        mapped = await self._map_messages(messages, parameters, model_settings=settings)
        tools, _ = self._get_tool_choice(settings, parameters)
        kwargs = dict(
            (settings.get("extra_body") or {}).get("chat_template_kwargs") or {}
        )
        effort = self._translate_thinking(settings, parameters)
        if isinstance(effort, str):
            kwargs.setdefault("reasoning_effort", effort)
        # SGLang treats the literal effort "none" as thinking disabled.
        if effort == "none":
            kwargs["enable_thinking"] = False
        return await asyncio.to_thread(
            counter.count, mapped, tools=tools, template_kwargs=kwargs
        )
