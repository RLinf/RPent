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

"""Shared stop evidence without importing planner runtimes or toolkits."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

STOP_CATEGORIES = frozenset(
    {"finish", "budget", "timeout", "cancelled", "error", "unknown"}
)


def normalize_stop_info(
    stop_info: Mapping[str, Any] | None,
    *,
    reason: str | None = None,
    error: str | None = None,
    finish_accepted: bool = False,
) -> dict[str, Any]:
    """Preserve a backend's reason alongside a common termination category.

    Args:
        stop_info: Optional backend evidence, including its raw reason/category.
        reason: Raw planner reason when absent from stop_info.
        error: Planner error used only when no reason was supplied.
        finish_accepted: Accepted finish used only when no reason was supplied.

    Returns:
        A copy with reason and category. Unknown reasons remain unknown even
        when a finish or error is present. A category never authorizes recovery;
        callers must also inspect the explicit execution and cleanup evidence.
    """
    normalized = dict(stop_info or {})
    raw_reason = normalized.get("reason")
    if not isinstance(raw_reason, str) or not raw_reason:
        raw_reason = reason if isinstance(reason, str) and reason else None
    if raw_reason is None:
        raw_reason = (
            "planner_error"
            if error is not None
            else "finish"
            if finish_accepted
            else "stopped"
        )
    category = normalized.get("category")
    if not isinstance(category, str) or category not in STOP_CATEGORIES:
        category = {
            "finish": "finish",
            "budget": "budget",
            "request_limit": "budget",
            "timeout": "timeout",
            "interrupted": "cancelled",
            "cancelled": "cancelled",
            "error": "error",
            "planner_error": "error",
            "stopped": "unknown",
        }.get(raw_reason, "unknown")
    normalized.update(reason=raw_reason, category=category)
    return normalized


def resolve_stop_info(
    stop_info: Mapping[str, Any] | None,
    *,
    reason: str | None = None,
    error: str | None = None,
    finish_result: Mapping[str, Any] | None = None,
    cancelled_reason: str | None = None,
) -> dict[str, Any]:
    """Resolve the effective outcome while retaining backend failure reasons.

    Args:
        stop_info: Optional backend reason, category, and boundary evidence.
        reason: Legacy reason used when the structured reason is absent.
        error: Diagnostic; only a classified planner deadline may carry one
            without becoming an error. Provider timeouts must classify as errors.
        finish_result: Accepted finish or operator verdict, independently of
            native environment success.
        cancelled_reason: Actual runner cancellation, such as task replacement.

    Returns:
        A copy of the stop evidence with the effective reason and category.
        Errors take precedence over cancellation and operator verdicts. This
        classification never grants permission to resume execution.
    """
    finish = finish_result or {}
    normalized = normalize_stop_info(
        stop_info,
        reason=reason,
        error=error,
        finish_accepted=finish.get("_finish") is True,
    )
    if normalized["category"] == "error" or (
        error is not None and normalized["category"] != "timeout"
    ):
        normalized["category"] = "error"
        return normalized
    if (
        cancelled_reason is not None
        or finish.get("operator_aborted") is True
        or (
            finish.get("operator_finished") is True
            and finish.get("operator_verdict") == "abort"
        )
    ):
        normalized.update(reason=cancelled_reason or "operator", category="cancelled")
    elif finish.get("operator_finished") is True:
        normalized.update(reason="operator_finished", category="finish")
    return normalized
