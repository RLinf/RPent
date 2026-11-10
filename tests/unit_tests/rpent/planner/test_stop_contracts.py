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

"""Backend reasons remain separate from shared termination categories."""

import pytest

from rpent.planner.base import (
    PlannerCapabilities,
    PlannerResult,
    get_planner_capabilities,
    is_recoverable_planner_result,
    normalize_stop_info,
)


def test_default_capabilities_need_no_optional_backend(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "deepseek_harness", None)
    for name in ("api", "codex", "claude_code", "flash", "future_backend"):
        assert get_planner_capabilities(name) == PlannerCapabilities()


@pytest.mark.parametrize(
    "reason,category",
    [
        ("finish", "finish"),
        ("budget", "budget"),
        ("request_limit", "budget"),
        ("timeout", "timeout"),
        ("interrupted", "cancelled"),
        ("planner_error", "error"),
        ("new_backend_reason", "unknown"),
    ],
)
def test_common_reason_category_does_not_discard_backend_reason(reason, category):
    result = normalize_stop_info(None, reason=reason, finish_accepted=True)
    assert result == {"reason": reason, "category": category}


def test_explicit_backend_category_and_safety_evidence_are_preserved():
    evidence = {
        "reason": "provider_limit",
        "category": "error",
        "execution_certain": True,
        "allow_new_session": False,
    }
    normalized = normalize_stop_info(evidence, reason="finish")
    assert normalized == evidence
    assert normalized is not evidence


@pytest.mark.parametrize("category", ["not_a_category", [], None])
def test_invalid_category_cannot_turn_unknown_reason_into_finish(category):
    normalized = normalize_stop_info(
        {"reason": "future_stop", "category": category},
        error="provider failed",
        finish_accepted=True,
    )
    assert normalized == {"reason": "future_stop", "category": "unknown"}


@pytest.mark.parametrize(
    "error,accepted,expected",
    [(None, True, "finish"), (None, False, "stopped"), ("", True, "planner_error")],
)
def test_legacy_hints_are_used_only_without_a_reason(error, accepted, expected):
    assert (
        normalize_stop_info(None, error=error, finish_accepted=accepted)["reason"]
        == expected
    )


def test_budget_category_alone_does_not_authorize_a_fresh_session():
    result = PlannerResult(stop_info=normalize_stop_info(None, reason="budget"))
    assert not is_recoverable_planner_result(result)


_RECOVERY_EVIDENCE = (
    "execution_certain",
    "allow_new_session",
    "result_delivery_complete",
    "cleanup_complete",
)


@pytest.mark.parametrize("field", _RECOVERY_EVIDENCE)
@pytest.mark.parametrize(
    "value,allowed",
    [(False, False), (True, True), (None, False), (1, False), ("true", False)],
)
def test_recovery_requires_explicit_true_for_all_safety_evidence(field, value, allowed):
    evidence = dict.fromkeys(_RECOVERY_EVIDENCE, True)
    evidence[field] = value
    assert is_recoverable_planner_result(PlannerResult(stop_info=evidence)) is allowed


@pytest.mark.parametrize("field", _RECOVERY_EVIDENCE)
def test_missing_safety_evidence_blocks_recovery_even_with_legacy_timeout(field):
    evidence = dict.fromkeys(_RECOVERY_EVIDENCE, True)
    del evidence[field]
    result = PlannerResult(error="timed out", stop_info=evidence)
    assert not is_recoverable_planner_result(result)


@pytest.mark.parametrize(
    "error,allowed",
    [
        ("timed out", True),
        ("planner response truncated", True),
        ("failed", False),
        (None, False),
    ],
)
def test_planners_without_stop_info_retain_legacy_recovery(error, allowed):
    assert is_recoverable_planner_result(PlannerResult(error=error)) is allowed
