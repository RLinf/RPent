# Copyright 2026 The RPent Authors.
# Licensed under the Apache License, Version 2.0.
"""Sliding-window compaction triggered by the current multimodal request."""

from __future__ import annotations

import time

from pydantic_ai_harness.compaction import SlidingWindowCompaction
from pydantic_ai_harness.compaction._shared import (
    context_for_request,
    record_compaction_reclaim,
    resolve_token_trigger,
)

from rpent.planner.compaction import count_image_tokens_in, estimate_request_tokens
from rpent.planner.token_counting import PromptTokens


class RequestAwareSlidingWindow(SlidingWindowCompaction):
    """Preserve Harness trimming/pair rules; replace its token trigger/accounting.

    Recount after request-view retention and after trimming. Old response usage
    belongs to a different input and cannot measure a rewritten conversation.
    """

    async def _count(self, request_context) -> PromptTokens:
        counter = getattr(request_context.model, "count_prompt_tokens", None)
        if counter is not None:
            return await counter(
                list(request_context.messages),
                request_context.model_settings,
                request_context.model_request_parameters,
            )
        return PromptTokens(
            total_tokens=estimate_request_tokens(
                request_context.messages, self.tokenizer
            ),
            image_tokens=count_image_tokens_in(request_context.messages),
            method="text_heuristic_with_image_geometry",
        )

    def _audit(self, request_context, name: str, value: dict) -> None:
        trace = getattr(request_context.model, "request_trace", None)
        if trace is not None:
            from rpent.planner.request_trace import write_json

            index = trace.native_counter + 1
            write_json(
                trace.directory / "token_counts" / f"{index:06d}-{name}.json", value
            )

    async def before_model_request(self, ctx, request_context):
        """Use real image cost, retaining the configured 80-to-40 message policy."""
        started = time.perf_counter()
        request_ctx = context_for_request(ctx, request_context)
        limit = resolve_token_trigger(
            self.max_tokens,
            self.max_fraction,
            request_ctx.model,
            self.fallback_context_window,
            self.context_window,
        )
        before = await self._count(request_context)
        messages = list(request_context.messages)
        triggered = (
            self.max_messages is not None and len(messages) > self.max_messages
        ) or (limit is not None and before.total_tokens > limit)
        after = before
        if triggered:
            # compact() supplies tool-pair protection, system prompts and the
            # opening user message. It makes no model request.
            request_context.messages = await self.compact(messages, request_ctx)
            after = await self._count(request_context)
            record_compaction_reclaim(
                request_context, before.total_tokens, after.total_tokens
            )
        self._last_count = after
        if triggered and limit is not None and after.total_tokens > limit:
            from rpent.utils.logging import get_logger

            get_logger("api").warning(
                "Retained history is still above compaction threshold: %d > %d "
                "tokens. The configured keep_messages/keep_tokens policy is unchanged.",
                after.total_tokens,
                limit,
            )
        self._audit(
            request_context,
            "input",
            {
                "before": before.to_dict(),
                "after": after.to_dict(),
                "threshold_tokens": limit,
                "triggered": triggered,
                "messages_before": len(messages),
                "messages_after": len(request_context.messages),
                "counting_seconds": time.perf_counter() - started,
                "still_above_threshold": (
                    limit is not None and after.total_tokens > limit
                ),
            },
        )
        return request_context

    async def after_model_request(self, ctx, *, request_context, response):
        """Compare the prediction with provider input usage; never overwrite it."""
        count = getattr(self, "_last_count", None)
        if count is not None:
            actual = response.usage.input_tokens
            if (
                actual
                and count.method.startswith("checkpoint_processor")
                and actual != count.total_tokens
            ):
                from rpent.utils.logging import get_logger

                get_logger("api").warning(
                    "Prompt token count differs from provider usage: predicted=%d "
                    "actual=%d. Check checkpoint, template and preprocessing settings.",
                    count.total_tokens,
                    actual,
                )
            # The native ID is now assigned, unlike before_model_request.
            trace = getattr(request_context.model, "request_trace", None)
            if trace is not None:
                from rpent.planner.request_trace import write_json

                write_json(
                    trace.directory
                    / "token_counts"
                    / f"{trace.native_counter:06d}-usage.json",
                    {
                        "predicted_input_tokens": count.total_tokens,
                        "provider_input_tokens": actual or None,
                        "difference": actual - count.total_tokens if actual else None,
                        "method": count.method,
                    },
                )
        return response
