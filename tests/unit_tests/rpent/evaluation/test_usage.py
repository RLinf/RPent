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

"""Cross-method coverage, request identity, and SDK-specific token accounting."""

import pytest

from rpent.dashboard.events import NullDashboardEventSink
from rpent.evaluation.usage import TokenLedger
from rpent.planner.claude_code import _Recorder as ClaudeRecorder
from rpent.planner.codex import _Recorder as CodexRecorder
from rpent.planner.gemini_cli import _Recorder as GeminiRecorder


def test_retry_and_unmetered_model_have_distinct_coverage():
    ledger = TokenLedger()
    assert ledger.to_dict() == {"coverage": "unknown", "entries": []}
    row = {
        "module": "planner",
        "model": "a",
        "stage": "planning",
        "input_tokens": 10,
        "output_tokens": 2,
    }
    ledger.record_usage(request_id="1", status="failed", **row)
    ledger.record_usage(request_id="1", status="failed", **row)
    ledger.record_usage(request_id="2", **row)
    with (
        pytest.raises(RuntimeError),
        ledger.unmetered_call(model="wam", module="action_model", stage="execution"),
    ):
        raise RuntimeError("timed out")
    ledger.observe_planner("api", "a", {"total_input_tokens": 20})
    snapshot = ledger.to_dict()
    assert snapshot["coverage"] == "partial"
    a, retry, action = snapshot["entries"]
    assert (a["input_tokens"], retry["output_tokens"], action["status"]) == (
        10,
        2,
        "failed",
    )
    assert action["input_tokens"] is None


def test_replay_has_zero_planner_tokens_and_continuous_actions_are_not_applicable():
    ledger = TokenLedger()
    ledger.observe_planner("flash", None, {})
    ledger.record_usage(
        request_id="1",
        model="script",
        module="action_model",
        stage="execution",
        coverage="not_applicable",
    )
    snap = ledger.to_dict()
    assert snap["coverage"] == "complete"
    assert [e["input_tokens"] for e in snap["entries"]] == [0, None]


def test_codex_thread_snapshot_replay_and_explicit_zero():
    ledger = TokenLedger()
    rec = CodexRecorder(
        max_turns=8, dashboard_events=NullDashboardEventSink(), token_ledger=ledger
    )

    def update(raw):
        rec.observe(
            {
                "method": "thread/tokenUsage/updated",
                "payload": {"token_usage": {"total": raw}},
            }
        )

    update({})
    assert ledger.to_dict()["entries"] == []
    update({"input_tokens": 0, "output_tokens": 0})
    assert ledger.to_dict()["coverage"] == "complete"
    for i in (2, 4, 2, 4):
        update({"input_tokens": i, "output_tokens": i, "cached_input_tokens": 1})
    (entry,) = ledger.to_dict()["entries"]
    assert (entry["input_tokens"], entry["output_tokens"]) == (4, 4)
    assert entry["granularity"] == "session"


def test_claude_authoritative_model_totals_replace_duplicate_assistant_events():
    ledger = TokenLedger()
    rec = ClaudeRecorder(
        max_turns=8, dashboard_events=NullDashboardEventSink(), token_ledger=ledger
    )
    for _ in range(2):
        rec.observe(
            {
                "type": "AssistantMessage",
                "model": "a",
                "message_id": "1",
                "usage": {"input_tokens": 5, "output_tokens": 2},
                "content": [],
            }
        )
    rec.observe(
        {
            "type": "ResultMessage",
            "usage": {"input_tokens": 75, "output_tokens": 19},
            "model_usage": {
                "a": {
                    "inputTokens": 5,
                    "cacheReadInputTokens": 10,
                    "cacheCreationInputTokens": 5,
                    "outputTokens": 9,
                },
                "b": {"inputTokens": 70, "outputTokens": 10},
            },
        }
    )
    rec.flush_pending_usage()
    entries = ledger.to_dict()["entries"]
    assert [(e["model"], e["input_tokens"], e["output_tokens"]) for e in entries] == [
        ("a", 20, 9),
        ("b", 70, 10),
    ]


def test_gemini_counts_each_prompt_once_and_marks_missing_usage():
    ledger = TokenLedger()
    rec = GeminiRecorder(NullDashboardEventSink(), token_ledger=ledger)
    result = {
        "type": "acp_result",
        "usage": {"inputTokens": 12, "outputTokens": 4, "cachedReadTokens": 3},
    }
    for _ in range(2):
        rec.observe({"type": "acp_prompt"})
        rec.observe(result)
        rec.observe(result)
    rec.observe({"type": "acp_prompt"})
    rec.observe({"type": "acp_result"})
    snapshot = ledger.to_dict()
    assert [e["input_tokens"] for e in snapshot["entries"]] == [12, 12, None]
    assert snapshot["coverage"] == "partial"


def test_invalid_source_counters_are_rejected_before_episode_publication():
    ledger = TokenLedger()
    request = {"request_id": "q", "model": "m", "module": "planner", "stage": "online"}
    with pytest.raises(ValueError):
        ledger.record_usage(**request, raw_usage={"invalid": float("nan")})
    with pytest.raises(ValueError):
        ledger.record_usage(**request, input_tokens=-2)
    with pytest.raises(ValueError):
        ledger.record_usage(**request, coverage="complete", output_tokens=0)
    assert ledger.to_dict()["entries"] == []


def test_gemini_preserves_actual_models_and_adds_unattributed_thoughts_once():
    ledger = TokenLedger()
    rec = GeminiRecorder(NullDashboardEventSink(), token_ledger=ledger)
    rec.observe({"type": "acp_prompt"})
    event = {
        "type": "acp_result",
        "usage": {
            "inputTokens": 12,
            "outputTokens": 5,
            "cachedReadTokens": 8,
            "thoughtTokens": 7,
        },
        "_meta": {
            "quota": {
                "model_usage": [
                    {
                        "model": "a",
                        "token_count": {"input_tokens": 7, "output_tokens": 2},
                    },
                    {
                        "model": "b",
                        "token_count": {"input_tokens": 5, "output_tokens": 3},
                    },
                ]
            }
        },
    }
    rec.observe(event)
    rec.observe(event)
    entries = ledger.to_dict()["entries"]
    assert [(r["model"], r["input_tokens"], r["output_tokens"]) for r in entries] == [
        ("a", 7, 2),
        ("b", 5, 3),
        ("unattributed", 0, 7),
    ]
