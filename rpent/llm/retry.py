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

"""Bounded request retries and sanitized error records for LLM providers."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic_ai.exceptions import ModelAPIError, ModelHTTPError
from pydantic_ai.models.wrapper import WrapperModel

from rpent.utils.logging import get_logger

if TYPE_CHECKING:
    from pydantic_ai import ModelSettings, RunContext
    from pydantic_ai.messages import ModelMessage, ModelResponse
    from pydantic_ai.models import Model, ModelRequestParameters, StreamedResponse

logger = get_logger("llm.retry")
_RETRYABLE_HTTP_STATUSES = {408, 409, 425, 429}


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    """Retry transient provider failures; ``max_retries`` excludes the first call."""

    max_retries: int = 2
    initial_delay_s: float = 0.5
    max_delay_s: float = 4.0

    def __post_init__(self) -> None:
        if self.max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        if self.initial_delay_s < 0 or self.max_delay_s < self.initial_delay_s:
            raise ValueError("retry delays must be non-negative and ordered")

    def delay(self, attempt: int) -> float:
        """Return capped exponential backoff after the given failed attempt."""
        return min(self.initial_delay_s * 2 ** (attempt - 1), self.max_delay_s)


def _retryable(exc: Exception) -> bool:
    if isinstance(exc, ModelHTTPError):
        return (
            exc.status_code in _RETRYABLE_HTTP_STATUSES or 500 <= exc.status_code < 600
        )
    return isinstance(exc, (ModelAPIError, ConnectionError, TimeoutError))


class RetryLoggingModel(WrapperModel):
    """Retry a failed model request without replaying completed robot tool calls.

    Error records omit prompts, response bodies, and credentials. A stream is
    retried only before it opens; replaying a partially consumed stream could
    duplicate output or downstream actions.
    """

    def __init__(
        self,
        wrapped: Model,
        *,
        policy: RetryPolicy | None = None,
        log_path: str | Path | None = None,
    ) -> None:
        super().__init__(wrapped)
        self.policy = policy or RetryPolicy()
        self.log_path = Path(log_path) if log_path is not None else None
        # Both SDKs otherwise retry internally before an error reaches this
        # layer, making the configured attempt limit and logs misleading.
        provider = wrapped.provider
        if provider is not None and provider.name in {"openai", "anthropic"}:
            provider.client.max_retries = 0

    def _failed(self, exc: Exception, attempt: int, *, can_retry: bool) -> float:
        delay = self.policy.delay(attempt) if can_retry else 0.0
        record = {
            "time": datetime.now(timezone.utc).isoformat(),
            "provider": self.system,
            "model": self.model_name,
            "error_type": type(exc).__name__,
            "status_code": exc.status_code if isinstance(exc, ModelHTTPError) else None,
            "attempt": attempt,
            "max_attempts": self.policy.max_retries + 1,
            "will_retry": can_retry,
            "retry_delay_s": delay,
        }
        log = logger.warning if can_retry else logger.error
        log("LLM request failed: %s", json.dumps(record, ensure_ascii=False))
        if self.log_path is not None:
            try:
                self.log_path.parent.mkdir(parents=True, exist_ok=True)
                with self.log_path.open("a", encoding="utf-8") as file:
                    file.write(json.dumps(record, ensure_ascii=False) + "\n")
            except OSError as log_error:
                logger.warning("could not write LLM error log: %s", log_error)
        return delay

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        """Retry only the provider request that failed."""
        for attempt in range(1, self.policy.max_retries + 2):
            try:
                return await self.wrapped.request(
                    messages, model_settings, model_request_parameters
                )
            except Exception as exc:
                can_retry = _retryable(exc) and attempt <= self.policy.max_retries
                delay = self._failed(exc, attempt, can_retry=can_retry)
                if not can_retry:
                    raise
                await asyncio.sleep(delay)
        raise AssertionError("unreachable retry state")

    @asynccontextmanager
    async def request_stream(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
        run_context: RunContext[Any] | None = None,
    ) -> AsyncIterator[StreamedResponse]:
        """Retry stream setup; never replay a stream after it has opened."""
        for attempt in range(1, self.policy.max_retries + 2):
            opened = False
            try:
                async with self.wrapped.request_stream(
                    messages, model_settings, model_request_parameters, run_context
                ) as stream:
                    opened = True
                    yield stream
                return
            except Exception as exc:
                can_retry = (
                    not opened
                    and _retryable(exc)
                    and attempt <= self.policy.max_retries
                )
                delay = self._failed(exc, attempt, can_retry=can_retry)
                if not can_retry:
                    raise
                await asyncio.sleep(delay)
        raise AssertionError("unreachable retry state")
