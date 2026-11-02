# Copyright 2026 The RPent Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

"""Actual runtime retry events and metering with deterministic provider failures."""

from .test_deepseek_runtime import call, runtime_state, wait_until
from .test_deepseek_runtime import run_runtime as run_runtime

POLICY = {
    "mode": "normal",
    "maxRetries": 2,
    "retryableCodes": ["RATE_LIMIT"],
    "initialDelayMs": 10,
    "maxDelayMs": 100,
    "jitterRatio": 0,
}
RATE_LIMIT = {"code": "RATE_LIMIT", "message": "deterministic provider rate limit"}


def test_retry_uses_same_root_step_and_preserves_unknown_usage(run_runtime):
    result, toolkit, requests, control = run_runtime(
        [
            {"failure": RATE_LIMIT},
            {"calls": [call("finish")]},
        ],
        max_steps=1,
        retry_policy=POLICY,
    )
    assert result.error is None
    assert len(requests) == 2
    assert result.stats["turns_used"] == 1
    assert toolkit.calls == ["finish"]
    inference = result.stats["inference"]
    assert len(inference["retries_scheduled"]) == 1
    assert len(inference["retries_started"]) == 1
    assert inference["failed_attempts_with_unknown_usage"] == 1
    assert inference["provider_attempts"] is None
    assert result.stats["total_input_tokens"] == 12


def test_retry_attempt_usage_is_not_double_counted_as_accepted_output(run_runtime):
    result, toolkit, requests, control = run_runtime(
        [
            {
                "failure": RATE_LIMIT,
                "usage": {"inputTokens": 3, "outputTokens": 4, "totalTokens": 7},
            },
            {"calls": [call("finish")]},
        ],
        retry_policy=POLICY,
    )
    assert result.error is None
    inference = result.stats["inference"]
    assert inference["failed_attempts"][0]["usage"]["outputTokens"] == 4
    assert inference["failed_attempts_with_unknown_usage"] == 0
    assert result.stats["total_output_tokens"] == 7
    assert len(control["llmRequests"]) == 2


def test_retry_exhaustion_cannot_execute_a_later_action(run_runtime):
    result, toolkit, requests, control = run_runtime(
        [
            {"failure": RATE_LIMIT},
            {"failure": RATE_LIMIT},
            {"failure": RATE_LIMIT},
            {"calls": [call("act")]},
        ],
        retry_policy=POLICY,
    )
    assert result.error
    assert len(requests) == 3
    assert len(result.stats["inference"]["retries_started"]) == 2
    assert not result.stop_info["allow_new_session"]
    assert toolkit.calls == []


def test_cancel_scheduled_retry_does_not_claim_an_attempt_started(run_runtime):
    def script(inputs, requests, root):
        def scheduled():
            paths = list(root.glob("dsh-*/events.jsonl"))
            return paths and '"type": "llm/retry"' in paths[0].read_text()

        wait_until(scheduled)
        inputs.put("/interrupt")
        wait_until(lambda: runtime_state(root).get("paused"))
        inputs.put(None)

    result, toolkit, requests, control = run_runtime(
        [
            {"failure": RATE_LIMIT},
            {"calls": [call("act")]},
        ],
        script=script,
        retry_policy={**POLICY, "initialDelayMs": 2000, "maxDelayMs": 2000},
    )
    assert result.stats["stop_reason"] == "interrupted"
    assert len(requests) == 1
    assert len(result.stats["inference"]["retries_scheduled"]) == 1
    assert result.stats["inference"]["retries_started"] == []
    assert toolkit.calls == []
