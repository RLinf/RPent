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

"""Small provider-independent LLM API built on RPent's Pydantic AI runtime."""

from __future__ import annotations

import threading
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from rpent.llm.retry import RetryLoggingModel, RetryPolicy
from rpent.utils.logging import get_logger

if TYPE_CHECKING:
    from pydantic_ai.models import Model
    from pydantic_ai.usage import RunUsage

ProviderName = Literal["openai", "anthropic"]
OpenAIFormat = Literal["responses", "chat"]
PromptCacheMode = Literal["implicit", "explicit"]

logger = get_logger("llm.client")


@dataclass(frozen=True, slots=True)
class LLMConfig:
    """One OpenAI or Anthropic API endpoint and model.

    ``api_key=None`` lets the provider read ``OPENAI_API_KEY`` or
    ``ANTHROPIC_API_KEY``. OpenAI defaults to the Responses API; set
    ``openai_format="chat"`` for a Chat Completions-compatible endpoint.
    """

    provider: ProviderName
    model: str
    api_key: str | None = None
    base_url: str | None = None
    openai_format: OpenAIFormat | None = None
    prompt_cache_key: str | None = None
    prompt_cache_mode: PromptCacheMode | None = None
    image_history_groups: int | None = None
    retry: RetryPolicy = field(default_factory=RetryPolicy)

    def __post_init__(self) -> None:
        if self.provider not in {"openai", "anthropic"}:
            raise ValueError(f"unsupported LLM provider: {self.provider}")
        if not self.model.strip():
            raise ValueError("LLM model must be non-empty")
        if self.openai_format not in {None, "responses", "chat"}:
            raise ValueError(f"unsupported OpenAI API format: {self.openai_format}")
        if self.provider != "openai" and self.openai_format is not None:
            raise ValueError("openai_format applies only to the OpenAI provider")
        if (
            self.prompt_cache_key is not None or self.prompt_cache_mode is not None
        ) and (self.provider != "openai" or self.openai_format == "chat"):
            raise ValueError("prompt cache settings require OpenAI Responses")
        if self.prompt_cache_key is not None and not self.prompt_cache_key.strip():
            raise ValueError("prompt_cache_key must be non-empty")
        if self.prompt_cache_mode not in (None, "implicit", "explicit"):
            raise ValueError("unsupported prompt cache mode")
        if self.image_history_groups is not None and self.image_history_groups < 1:
            raise ValueError("image_history_groups must be positive")

    def build_model(self) -> Model:
        """Construct the selected Pydantic AI model without changing process env."""
        if self.provider == "anthropic":
            from pydantic_ai.models.anthropic import AnthropicModel
            from pydantic_ai.providers.anthropic import AnthropicProvider

            return AnthropicModel(
                self.model,
                provider=AnthropicProvider(
                    api_key=self.api_key, base_url=self.base_url
                ),
            )

        from pydantic_ai.models.openai import (
            OpenAIChatModel,
            OpenAIResponsesModel,
            OpenAIResponsesModelSettings,
        )
        from pydantic_ai.providers.openai import OpenAIProvider

        provider = OpenAIProvider(api_key=self.api_key, base_url=self.base_url)
        model_cls = (
            OpenAIChatModel if self.openai_format == "chat" else OpenAIResponsesModel
        )
        if model_cls is OpenAIResponsesModel and (
            self.prompt_cache_key is not None or self.prompt_cache_mode is not None
        ):
            settings = OpenAIResponsesModelSettings()
            if self.prompt_cache_key is not None:
                settings["openai_prompt_cache_key"] = self.prompt_cache_key
            if self.prompt_cache_mode is not None:
                settings["openai_prompt_cache_options"] = {
                    "mode": self.prompt_cache_mode
                }
            return model_cls(self.model, provider=provider, settings=settings)
        return model_cls(self.model, provider=provider)


@dataclass(frozen=True, slots=True)
class LLMUsage:
    """Token counts for one request or an accumulated run.

    ``input_tokens`` includes cache reads and writes; ``output_tokens``
    includes reasoning tokens. A zero cache count means the provider did not
    report any cached tokens, which does not prove caching was disabled.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    reasoning_output_tokens: int = 0
    requests: int | None = None
    cost_usd: float | None = None

    @property
    def total_tokens(self) -> int:
        """Return input plus output tokens without double-counting cache."""
        return self.input_tokens + self.output_tokens

    def as_dict(self) -> dict[str, int | float | None]:
        """Return a JSON-serializable usage record."""
        return {**asdict(self), "total_tokens": self.total_tokens}

    @classmethod
    def from_run_usage(cls, usage: RunUsage) -> LLMUsage:
        """Normalize Pydantic AI's accumulated provider usage."""
        return cls(
            input_tokens=int(usage.input_tokens or 0),
            output_tokens=int(usage.output_tokens or 0),
            cache_read_tokens=int(usage.cache_read_tokens or 0),
            cache_write_tokens=int(usage.cache_write_tokens or 0),
            reasoning_output_tokens=int(usage.details.get("reasoning_tokens", 0)),
            requests=int(usage.requests or 0),
            cost_usd=float(usage.cost) if usage.cost is not None else None,
        )

    @classmethod
    def from_planner_stats(cls, stats: Mapping[str, object]) -> LLMUsage:
        """Normalize existing API, Claude SDK, or Codex planner counters."""
        cache_read = int(
            stats.get(
                "cache_read_tokens",
                stats.get(
                    "total_cache_read_input_tokens",
                    stats.get("total_cached_input_tokens", 0),
                ),
            )
            or 0
        )
        cache_write = int(
            stats.get(
                "cache_write_tokens", stats.get("total_cache_creation_input_tokens", 0)
            )
            or 0
        )
        input_tokens = int(stats.get("total_input_tokens", 0) or 0)
        # Claude SDK exposes Anthropic's raw input count, which excludes its
        # separately reported cache reads and writes. API and Codex counts
        # already include cached input.
        if "total_cache_creation_input_tokens" in stats:
            input_tokens += cache_read + cache_write
        requests = stats.get("requests")
        cost = stats.get("total_cost_usd")
        return cls(
            input_tokens=input_tokens,
            output_tokens=int(stats.get("total_output_tokens", 0) or 0),
            cache_read_tokens=cache_read,
            cache_write_tokens=cache_write,
            reasoning_output_tokens=int(
                stats.get("total_reasoning_output_tokens", 0) or 0
            ),
            requests=int(requests) if requests is not None else None,
            cost_usd=float(cost) if cost is not None else None,
        )

    def __add__(self, other: LLMUsage) -> LLMUsage:
        """Combine two complete usage records."""
        return LLMUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cache_read_tokens=self.cache_read_tokens + other.cache_read_tokens,
            cache_write_tokens=self.cache_write_tokens + other.cache_write_tokens,
            reasoning_output_tokens=(
                self.reasoning_output_tokens + other.reasoning_output_tokens
            ),
            requests=(
                self.requests + other.requests
                if self.requests is not None and other.requests is not None
                else None
            ),
            cost_usd=(
                self.cost_usd + other.cost_usd
                if self.cost_usd is not None and other.cost_usd is not None
                else None
            ),
        )


@dataclass(frozen=True, slots=True)
class LLMResponse:
    """Text response and the usage incurred by this call."""

    text: str
    usage: LLMUsage


def build_model_settings(model: Model, max_tokens: int):
    """Carry provider cache settings into direct and agent calls."""
    from pydantic_ai import ModelSettings

    if max_tokens < 1:
        raise ValueError("max_tokens must be positive")
    underlying = model.wrapped if isinstance(model, RetryLoggingModel) else model
    if underlying.system == "anthropic":
        from pydantic_ai.models.anthropic import AnthropicModelSettings

        return AnthropicModelSettings(
            max_tokens=max_tokens,
            anthropic_cache_instructions=True,
            anthropic_cache_tool_definitions=True,
            anthropic_cache_messages=True,
        )
    cache_settings = {
        key: value
        for key in ("openai_prompt_cache_key", "openai_prompt_cache_options")
        if (value := (underlying.settings or {}).get(key)) is not None
    }
    return ModelSettings(max_tokens=max_tokens, **cache_settings)


class LLMClient:
    """Make direct text calls and track both per-call and cumulative usage."""

    def __init__(self, config: LLMConfig, *, log_path: str | Path | None = None):
        self.config = config
        self._model = RetryLoggingModel(
            config.build_model(), policy=config.retry, log_path=log_path
        )
        self._total_usage: LLMUsage | None = None
        self._usage_lock = threading.Lock()

    @property
    def total_usage(self) -> LLMUsage:
        """Return usage across completed and failed calls reported by the SDK."""
        with self._usage_lock:
            return self._total_usage or LLMUsage()

    def _agent(self, system_prompt: str, max_tokens: int):
        from pydantic_ai import Agent

        return Agent(
            self._model,
            instructions=system_prompt or None,
            model_settings=build_model_settings(self._model, max_tokens),
        )

    def _record(self, usage: LLMUsage) -> None:
        with self._usage_lock:
            self._total_usage = (
                usage if self._total_usage is None else self._total_usage + usage
            )

    def _prompt_content(self, prompt: str):
        if self.config.prompt_cache_mode != "explicit":
            return prompt
        from pydantic_ai.messages import CachePoint

        return [prompt, CachePoint()]

    async def generate(
        self, prompt: str, *, system_prompt: str = "", max_tokens: int = 8192
    ) -> LLMResponse:
        """Call the selected model asynchronously and report its token usage."""
        from pydantic_ai.usage import RunUsage

        if not prompt.strip():
            raise ValueError("prompt must be non-empty")
        call_usage = RunUsage()
        try:
            result = await self._agent(system_prompt, max_tokens).run(
                self._prompt_content(prompt), usage=call_usage
            )
        except Exception as exc:
            logger.error(
                "LLM call failed provider=%s model=%s error_type=%s",
                self.config.provider,
                self.config.model,
                type(exc).__name__,
            )
            raise
        finally:
            self._record(LLMUsage.from_run_usage(call_usage))
        return LLMResponse(
            text=str(result.output), usage=LLMUsage.from_run_usage(call_usage)
        )

    def generate_sync(
        self, prompt: str, *, system_prompt: str = "", max_tokens: int = 8192
    ) -> LLMResponse:
        """Call the selected model from synchronous benchmark code."""
        from pydantic_ai.usage import RunUsage

        if not prompt.strip():
            raise ValueError("prompt must be non-empty")
        call_usage = RunUsage()
        try:
            result = self._agent(system_prompt, max_tokens).run_sync(
                self._prompt_content(prompt), usage=call_usage
            )
        except Exception as exc:
            logger.error(
                "LLM call failed provider=%s model=%s error_type=%s",
                self.config.provider,
                self.config.model,
                type(exc).__name__,
            )
            raise
        finally:
            self._record(LLMUsage.from_run_usage(call_usage))
        return LLMResponse(
            text=str(result.output), usage=LLMUsage.from_run_usage(call_usage)
        )
