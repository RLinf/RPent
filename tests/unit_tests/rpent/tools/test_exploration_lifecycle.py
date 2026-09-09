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

"""Contracts for shared exploration lifecycle bookkeeping."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from rpent.tools.exploration import (
    ExplorationLifecycle,
    ResetAfterSuccess,
    ResetRefusal,
    records_after_latest_successful_reset,
    refusal_result,
)


def _lifecycle(
    budget: int = 3,
    *,
    mode: str = "exploration",
    reset_after_success: ResetAfterSuccess = ResetAfterSuccess.REFUSE,
) -> ExplorationLifecycle:
    return ExplorationLifecycle(
        mode=mode,
        attempts_per_session=budget,
        adapter_name="test",
        reset_after_success=reset_after_success,
    )


@pytest.mark.parametrize(
    ("budget", "allowed_resets"),
    [(0, 4), (1, 0), (3, 2)],
)
def test_attempt_budgets_and_finish_guard(budget: int, allowed_resets: int) -> None:
    lifecycle = _lifecycle(budget)

    finish_refusal = lifecycle.finish_refusal(solved=False)
    if budget in {0, 1}:
        assert finish_refusal is None
    else:
        assert finish_refusal is not None
        assert (finish_refusal.remaining, finish_refusal.budget) == (2, 3)

    calls = 0
    for expected_attempt in range(2, allowed_resets + 2):
        assert lifecycle.request_reset(solved=False) is None
        assert lifecycle.current_attempt == expected_attempt
        assert lifecycle.reset_in_progress
        calls += 1
        assert lifecycle.perform_reset(lambda: {"ok": True}) == {"ok": True}

    if budget:
        assert (
            lifecycle.request_reset(solved=False) is ResetRefusal.ATTEMPT_BUDGET_SPENT
        )
        assert lifecycle.current_attempt == budget
    assert calls == allowed_resets
    assert lifecycle.finish_refusal(solved=False) is None


def test_reset_attempt_is_incremented_before_operation_and_exception() -> None:
    lifecycle = _lifecycle()
    observed_attempts: list[int] = []

    assert lifecycle.request_reset(solved=False) is None

    def fail() -> dict[str, bool]:
        observed_attempts.append(lifecycle.current_attempt)
        raise RuntimeError("reset failed")

    with pytest.raises(RuntimeError, match="reset failed"):
        lifecycle.perform_reset(fail)

    assert observed_attempts == [2]
    assert lifecycle.current_attempt == 2
    assert lifecycle.reset_failed
    assert not lifecycle.reset_in_progress

    assert lifecycle.request_reset(solved=False) is None
    assert lifecycle.perform_reset(lambda: {"ok": True}) == {"ok": True}
    assert lifecycle.current_attempt == 3
    assert not lifecycle.reset_failed


@pytest.mark.parametrize(
    ("policy", "expected"),
    [
        (ResetAfterSuccess.ALLOW, None),
        (ResetAfterSuccess.REFUSE, ResetRefusal.TASK_SOLVED),
    ],
)
def test_reset_after_success_is_an_explicit_adapter_policy(
    policy: ResetAfterSuccess, expected: ResetRefusal | None
) -> None:
    lifecycle = _lifecycle(reset_after_success=policy)

    assert lifecycle.request_reset(solved=True) is expected
    if expected is None:
        lifecycle.perform_reset(lambda: {})
        assert lifecycle.current_attempt == 2
    else:
        assert lifecycle.current_attempt == 1


def test_evaluation_mode_has_no_finish_guard() -> None:
    lifecycle = _lifecycle(mode="evaluation")

    assert lifecycle.finish_refusal(solved=False) is None


def test_reset_result_and_refusal_envelopes_preserve_adapter_fields() -> None:
    lifecycle = _lifecycle()
    assert lifecycle.request_reset(solved=False) is None
    result = lifecycle.perform_reset(lambda: {"seed": 7, "notice": "native"})

    assert lifecycle.build_reset_result(result, reason="retry") == {
        "seed": 7,
        "notice": "native",
        "attempt": 2,
        "reason": "retry",
    }
    assert refusal_result("reset", "spent") == {
        "error": "reset refused",
        "reason": "spent",
    }


def _record(
    step_idx: int,
    action: str | None = None,
    *,
    error: str | None = None,
) -> SimpleNamespace:
    command = None if action is None else {"action": action}
    result = {} if error is None else {"error": error}
    return SimpleNamespace(step_idx=step_idx, command=command, result=result)


@pytest.mark.parametrize(
    ("records", "expected_steps"),
    [
        ([_record(0, "move_to"), _record(1, "release")], [0, 1]),
        (
            [_record(0, "move_to"), _record(1, "reset"), _record(2, "release")],
            [2],
        ),
        (
            [
                _record(0, "move_to"),
                _record(1, "reset"),
                _record(2, "release"),
                _record(3, "reset", error="failed"),
                _record(4, "move_to"),
                _record(5, "reset"),
                _record(6, "release"),
            ],
            [6],
        ),
    ],
)
def test_records_after_zero_one_or_multiple_successful_resets(
    records: list[SimpleNamespace], expected_steps: list[int]
) -> None:
    assert [
        record.step_idx for record in records_after_latest_successful_reset(records)
    ] == expected_steps


def test_observed_successful_reset_boundary_ignores_failed_resets() -> None:
    lifecycle = _lifecycle()

    lifecycle.observe_record(_record(1, "reset"))
    lifecycle.observe_record(_record(3, "reset", error="failed"))
    lifecycle.observe_record(_record(5, "reset"))

    assert lifecycle.latest_successful_reset_step == 5


def test_successful_operation_boundary_is_not_committed_on_exception() -> None:
    lifecycle = _lifecycle(0)
    assert lifecycle.request_reset(solved=False) is None
    lifecycle.perform_reset(lambda: {}, successful_boundary_after_step=4)
    assert lifecycle.latest_successful_reset_step == 4

    assert lifecycle.request_reset(solved=False) is None

    def fail() -> dict[str, bool]:
        raise RuntimeError("reset failed")

    with pytest.raises(RuntimeError, match="reset failed"):
        lifecycle.perform_reset(fail, successful_boundary_after_step=8)
    assert lifecycle.latest_successful_reset_step == 4


def test_configuration_preserves_mode_errors_and_budget_normalization() -> None:
    lifecycle = _lifecycle(-3)
    assert lifecycle.attempts_per_session == 0

    with pytest.raises(
        ValueError, match="unsupported RoboCasa toolkit mode: 'invalid'"
    ):
        ExplorationLifecycle(
            mode="invalid",
            attempts_per_session=3,
            adapter_name="RoboCasa",
            reset_after_success=ResetAfterSuccess.REFUSE,
        )
