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

"""Offline lifecycle tests; actual Cordis policy is exercised separately."""

from __future__ import annotations

import hashlib
import json
import math
import sys
import threading
from types import SimpleNamespace

import pytest

from rpent.dashboard.events import NullDashboardEventSink, UsageEvent
from rpent.planner import deepseek_harness as dsh
from rpent.planner.utils.http_mcp_server import mcp_result
from rpent.tools import ToolResult
from rpent.tools.common import CommonTools


class Toolkit:
    def __init__(self):
        self.calls = []
        self.cancelled = False
        self.result = ToolResult(data={"_finish": True, "status": "success"})

    def list_tools(self):
        return (CommonTools.finish,)

    def execute_tool(self, name, arguments):
        self.calls.append(name)
        return self.result

    def cancel_active_and_wait(self):
        self.cancelled = True


def test_native_finish_latches_before_later_call():
    toolkit = Toolkit()
    boundary = dsh._ToolBoundary(toolkit)
    assert boundary.execute_tool("finish", {}).data["_finish"]
    assert boundary.execute_tool("move", {}).is_error
    assert toolkit.calls == ["finish"]
    assert boundary.finish_result == toolkit.result.data


@pytest.mark.parametrize("payload", [{"error": "refused"}, {"status": "success"}])
def test_refused_or_unmarked_finish_does_not_conclude(payload):
    toolkit = Toolkit()
    toolkit.result = ToolResult(data=payload)
    boundary = dsh._ToolBoundary(toolkit)
    boundary.execute_tool("finish", {"status": "success"})
    boundary.execute_tool("move", {})
    assert boundary.finish_result is None
    assert toolkit.calls == ["finish", "move"]


def test_structured_payload_retains_marker_beyond_text_limit():
    result = ToolResult(data={"summary": "x" * 70000, "_finish": True})
    assert mcp_result(result)["content"][0]["text"].endswith("[truncated]")
    assert "structuredContent" not in mcp_result(result)
    assert mcp_result(result, include_structured=True)["structuredContent"]["_finish"]


def test_structured_coordinates_normalize_negative_zero_without_changing_native():
    result = ToolResult(
        data={"result": {"pose": [0.0705, -0.0, 1.0194]}, "note": "literal -0.0"}
    )
    payload = mcp_result(result, include_structured=True)
    structured = payload["structuredContent"]
    text = json.loads(payload["content"][0]["text"])
    assert structured == text == result.data
    assert math.copysign(1, structured["result"]["pose"][1]) == 1
    assert math.copysign(1, text["result"]["pose"][1]) == 1
    assert math.copysign(1, result.data["result"]["pose"][1]) == -1
    assert structured["note"] == "literal -0.0"


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_structured_coordinates_reject_nonfinite_values(value):
    with pytest.raises(ValueError, match="Out of range float"):
        mcp_result(ToolResult(data={"pose": [value]}), include_structured=True)


@pytest.mark.parametrize(
    "failure", [None, "nonfinite", "image", "uncertain", "truncated"]
)
def test_business_refusal_requires_a_complete_canonical_payload(tmp_path, failure):
    toolkit = Toolkit()
    toolkit.result = ToolResult(
        data={"error": "business refusal", "position": [-0.0]},
        images=[b"image"] if failure == "image" else [],
        execution_uncertain=failure == "uncertain",
    )
    if failure == "nonfinite":
        toolkit.result.data["position"] = [float("nan")]
    elif failure == "truncated":
        toolkit.result.data["details"] = "x" * 70000
    path = tmp_path / "ledger.json"
    boundary = dsh._ToolBoundary(toolkit, path)
    result = boundary.execute_tool("act", {})
    ledger = json.loads(path.read_text())
    assert result is toolkit.result
    assert ledger["isError"]
    if failure is None:
        text = mcp_result(result, include_structured=True)["content"][0]["text"]
        assert (
            ledger["canonicalBusinessErrorSha256"]
            == hashlib.sha256(text.encode()).hexdigest()
        )
    else:
        assert ledger["canonicalBusinessErrorSha256"] is None


def test_cancel_prevents_queued_calls():
    toolkit = Toolkit()
    boundary = dsh._ToolBoundary(toolkit)
    boundary.cancel_active_and_wait()
    assert boundary.execute_tool("move", {}).is_error
    assert toolkit.calls == []
    assert toolkit.cancelled


def event(kind, data, seq=1):
    return SimpleNamespace(
        method="session.event",
        payload={"event": {"type": kind, "seq": seq, "data": data}},
    )


def test_events_deduplicate_and_count_steps_not_turns():
    recorder = dsh._Recorder(NullDashboardEventSink())
    recorder.observe(event("turn/start", {}, 1), "receipt")
    recorder.observe(event("step/start", {}, 2), "receipt")
    recorder.observe(event("step/start", {}, 2), "receipt")
    recorder.observe(event("step/start", {}, 3), "receipt")
    assert recorder.steps == 2
    assert recorder.usage == {}


def test_usage_includes_cache_without_double_counting_replayed_events():
    emitted = []
    recorder = dsh._Recorder(SimpleNamespace(emit=emitted.append))
    # Actual openai-codex usage: inputTokens excludes the cache hit.
    sample = event(
        "assistant/message",
        {
            "usage": {
                "inputTokens": 3614,
                "outputTokens": 328,
                "totalTokens": 36454,
                "cacheReadTokens": 32512,
            }
        },
    )
    recorder.observe(sample, "receipt")
    recorder.observe(sample, "receipt")
    assert recorder.usage == {
        "total_uncached_input_tokens": 3614,
        "total_input_tokens": 36126,
        "total_output_tokens": 328,
        "total_cached_input_tokens": 32512,
    }
    assert emitted == [UsageEvent(inp=36126, out=328, tool_calls=0)]

    recorder.observe(
        event(
            "assistant/message",
            {
                "usage": {
                    "inputTokens": 12,
                    "outputTokens": 7,
                    "cacheReadTokens": 3,
                    "cacheWriteTokens": 5,
                    "reasoningTokens": 2,
                }
            },
            seq=2,
        ),
        "receipt",
    )
    assert recorder.usage == {
        "total_uncached_input_tokens": 3626,
        "total_input_tokens": 36146,
        "total_output_tokens": 335,
        "total_cached_input_tokens": 32515,
        "total_cache_creation_input_tokens": 5,
        "total_reasoning_output_tokens": 2,
    }
    assert emitted[-1] == UsageEvent(inp=36146, out=335, tool_calls=0)


@pytest.mark.parametrize(
    "usage",
    [
        None,
        {"inputTokens": 12, "outputTokens": 7},
        {"inputTokens": 12, "outputTokens": 7, "cacheReadTokens": 3},
        {"inputTokens": 12, "outputTokens": 7, "totalTokens": 18},
        {"inputTokens": 12, "outputTokens": 7, "totalTokens": True},
        {
            "inputTokens": 12,
            "outputTokens": 7,
            "cacheReadTokens": 3,
            "cacheWriteTokens": 5,
            "totalTokens": 28,
        },
    ],
)
def test_incomplete_usage_never_becomes_an_exact_cumulative_input_total(usage):
    emitted = []
    recorder = dsh._Recorder(SimpleNamespace(emit=emitted.append))
    known = {"inputTokens": 1, "outputTokens": 2, "totalTokens": 3}
    recorder.observe(event("assistant/message", {"usage": known}), "receipt")
    assert recorder.usage["total_input_tokens"] == 1
    recorder.observe(event("assistant/message", {"usage": usage}, seq=2), "receipt")
    recorder.observe(event("assistant/message", {"usage": known}, seq=3), "receipt")
    assert "total_input_tokens" not in recorder.usage
    assert recorder.usage["total_uncached_input_tokens"] == (2 if usage is None else 14)
    assert emitted == [UsageEvent(inp=1, out=2, tool_calls=0)]


def test_usage_preserves_explicit_zero_and_omits_unknown_cache_buckets():
    recorder = dsh._Recorder(NullDashboardEventSink())
    recorder.observe(
        event("assistant/message", {"usage": {"inputTokens": 0, "outputTokens": 0}}),
        "receipt",
    )
    assert recorder.usage == {
        "total_uncached_input_tokens": 0,
        "total_output_tokens": 0,
    }
    recorder = dsh._Recorder(NullDashboardEventSink())
    recorder.observe(
        event(
            "assistant/message",
            {
                "usage": {
                    "inputTokens": 0,
                    "outputTokens": 0,
                    "cacheReadTokens": 0,
                    "cacheWriteTokens": 0,
                }
            },
        ),
        "receipt",
    )
    assert recorder.usage == {
        "total_uncached_input_tokens": 0,
        "total_input_tokens": 0,
        "total_output_tokens": 0,
        "total_cached_input_tokens": 0,
        "total_cache_creation_input_tokens": 0,
    }


@pytest.fixture
def runtime(monkeypatch, tmp_path):
    state = SimpleNamespace(mode="idle", client=None, server=None, boundary=None)

    class Server:
        def __init__(self, boundary, **kwargs):
            state.server, state.boundary = self, boundary
            self.stopped = False

        def start(self, **kwargs):
            return "http://127.0.0.1:1234/mcp/"

        def stop(self):
            self.stopped = True

    class Client:
        def __init__(self, config):
            self.config, self.closed = config, threading.Event()
            self.index = 0
            state.client = self

        def start(self):
            pass

        def initialize(self, **kwargs):
            if state.mode == "startup_error":
                raise RuntimeError("provider unavailable")

        def close(self):
            self.closed.set()

        def subscribe_session_notifications(self, session_id):
            return self

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def session_prompt(self, *args, **kwargs):
            return "receipt"

        def next(self):
            if state.mode == "timeout":
                self.closed.wait()
                raise RuntimeError("transport closed")
            self.index += 1
            if self.index == 1:
                return event("agent/inbox/spliced", {"inserted": [{"id": "receipt"}]})
            if state.mode == "finish":
                state.boundary.execute_tool("finish", {})
            if state.mode == "budget":
                rows = json.loads(open(self.config.patches[0]).read())
                bridge = rows[-1]["insert"][-1]
                from pathlib import Path

                Path(bridge["config"]["controlPath"]).write_text(
                    '{"stopReason": "budget"}'
                )
            return SimpleNamespace(method="session.status", payload={"status": "idle"})

    monkeypatch.setattr(dsh, "HttpMcpServer", Server)
    monkeypatch.setattr(dsh, "version", lambda _: dsh.SDK_VERSION)
    monkeypatch.setitem(
        sys.modules,
        "deepseek_harness",
        SimpleNamespace(
            HarnessClient=Client, HarnessConfig=lambda **kw: SimpleNamespace(**kw)
        ),
    )
    state.planner = dsh.DeepSeekHarnessPlanner(
        output_dir=tmp_path,
        recipe_tag="test",
        dsh_home=tmp_path / "home",
        model="gpt-6.1-sol",
        reasoning_effort="xhigh",
        timeout_s=2,
        dashboard_events=NullDashboardEventSink(),
    )
    return state


@pytest.mark.parametrize(
    "mode,reason,error",
    [
        ("idle", "missing_finish", True),
        ("finish", "finish", False),
        ("budget", "budget", False),
        ("timeout", "timeout", True),
        ("startup_error", "runtime_error", True),
    ],
)
def test_run_outcomes_and_cleanup(runtime, mode, reason, error):
    runtime.mode = mode
    if mode == "timeout":
        runtime.planner.timeout_s = 0.05
    toolkit = Toolkit()
    result = runtime.planner.solve(
        system_prompt="literal {{braces}}",
        user_message="task",
        toolkit=toolkit,
        max_turns=2,
    )
    assert result.stats["stop_reason"] == reason
    assert bool(result.error) is error
    assert (
        toolkit.cancelled and runtime.client.closed.is_set() and runtime.server.stopped
    )
    assert (result.finish_result is not None) is (mode == "finish")
    assert result.stats["model"] == "gpt-6.1-sol"
    assert result.stats["reasoning_effort"] == "xhigh"


def test_completed_action_without_observed_call_or_result_cannot_handoff(
    runtime, monkeypatch
):
    runtime.mode = "timeout"
    toolkit = Toolkit()
    toolkit.result = ToolResult(data={"acted": True})

    async def stop_before_observing_tool_events(**kwargs):
        # An action completed, but the notification tail was lost as the host
        # closed. An empty set of observed pending IDs alone proves nothing.
        kwargs["boundary"].execute_tool("act", {})
        return "timeout", "execution deadline", 0.05

    monkeypatch.setattr(runtime.planner, "_drive", stop_before_observing_tool_events)
    result = runtime.planner.solve(
        system_prompt="system", user_message="act", toolkit=toolkit, max_turns=2
    )
    assert toolkit.calls == ["act"]
    assert result.stop_info["pending_tool_call_ids"] == []
    assert result.stats["completed_tool_calls"] == 1
    assert result.stats["observed_tool_results"] == 0
    assert result.stop_info["execution_certain"] is True
    assert result.stop_info["result_delivery_complete"] is False
    assert result.stop_info["allow_new_session"] is False


def test_runtime_output_rejection_is_not_delivery_even_with_paired_ids(
    runtime, monkeypatch
):
    runtime.mode = "timeout"
    toolkit = Toolkit()
    toolkit.result = ToolResult(data={"acted": True})

    async def rejected_output(**kwargs):
        kwargs["boundary"].execute_tool("act", {})
        recorder = kwargs["recorder"]
        recorder.observe(event("tool/call", {"callId": "native-act"}, 10), "")
        recorder.observe(
            event(
                "tool/result",
                {
                    "message": {"toolCallId": "native-act", "isError": True},
                    "error": {"name": "ToolOutputError", "code": "INVALID_TOOL_OUTPUT"},
                },
                11,
            ),
            "",
        )
        return "budget", None, 0.05

    monkeypatch.setattr(runtime.planner, "_drive", rejected_output)
    result = runtime.planner.solve(
        system_prompt="system", user_message="act", toolkit=toolkit, max_turns=2
    )
    assert toolkit.calls == ["act"]
    assert (
        result.stats["completed_tool_calls"]
        == result.stats["observed_tool_results"]
        == 1
    )
    assert result.stop_info["pending_tool_call_ids"] == []
    assert result.stats["tool_delivery_errors"][0]["call_id"] == "native-act"
    assert result.error and result.stop_info["reason"] == "tool_transport_error"
    assert result.stop_info["result_delivery_complete"] is False
    assert result.stop_info["allow_new_session"] is False


def test_explicit_login_home_required(monkeypatch):
    monkeypatch.delenv("DSH_HOME", raising=False)
    with pytest.raises(ValueError, match="dsh-home"):
        dsh.resolve_dsh_home(None)


def test_config_rejects_unsupported_effort_before_runtime(tmp_path):
    with pytest.raises(ValueError, match="effort"):
        dsh.DeepSeekHarnessPlanner(
            output_dir=tmp_path,
            recipe_tag="test",
            model="gpt-6.1-sol",
            dsh_home=tmp_path,
            reasoning_effort="none",
            dashboard_events=NullDashboardEventSink(),
        )


def test_relative_paths_resolve_before_child_changes_directory(
    runtime, tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    planner = dsh.DeepSeekHarnessPlanner(
        output_dir="relative-output",
        recipe_tag="paths",
        model="gpt-6.1-sol",
        reasoning_effort="xhigh",
        dsh_home="relative-login",
        repo_root="separate-runtime-directory",
        dashboard_events=NullDashboardEventSink(),
    )
    assert planner.output_dir == tmp_path / "relative-output"
    assert planner.home == tmp_path / "relative-login"
    assert planner.cwd == str(tmp_path / "separate-runtime-directory")


def test_check_uses_selected_home_model_effort(monkeypatch, tmp_path):
    from rpent.planner.check import LlmCheckRequest, check_llm

    calls = []
    monkeypatch.setattr(dsh, "run_probe_turn", lambda **kw: calls.append(kw) or "ok")
    result = check_llm(
        LlmCheckRequest(
            planner="deepseek_harness",
            model="gpt-6.1-sol",
            dsh_home=str(tmp_path),
            reasoning_effort="xhigh",
        )
    )
    assert result.ok
    assert calls[0]["model"] == "gpt-6.1-sol"
    assert calls[0]["reasoning_effort"] == "xhigh"
    assert calls[0]["dsh_home"] == tmp_path


@pytest.mark.skipif(
    sys.platform == "win32", reason="POSIX temporary-directory fallback"
)
def test_attachment_store_avoids_filesystem_without_cross_directory_links(
    monkeypatch, tmp_path
):
    import os
    import shutil

    remote = tmp_path / "remote"
    remote.mkdir()
    monkeypatch.setattr(dsh.tempfile, "gettempdir", lambda: str(remote))
    original_link = os.link

    def link(source, target):
        if str(source).startswith(str(remote)):
            raise PermissionError("Cross-directory links are unsupported")
        original_link(source, target)

    monkeypatch.setattr(dsh.os, "link", link)
    home = dsh._attachment_home()
    try:
        assert home.parent == dsh.Path("/tmp")
        assert list(remote.iterdir()) == []
    finally:
        shutil.rmtree(home)


def test_diagnostic_secrets_are_removed_before_artifacts_and_messages(
    runtime, monkeypatch
):
    """OAuth error bodies must not leak through the raw event artifact."""
    from copy import deepcopy
    from pathlib import Path

    tokens = [f"synthetic-review-token-{index}" for index in range(7)]
    diagnostic = "OpenAI Codex token refresh response missing fields: " + json.dumps(
        {"access_token": tokens[0]}
    )
    escaped_diagnostic = json.dumps(json.dumps({"refresh_token": tokens[1]}))
    notifications = [
        event("agent/inbox/spliced", {"inserted": [{"id": "receipt"}]}, 1),
        event(
            "assistant/message",
            {
                "message": {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": diagnostic},
                        {"type": "text", "text": escaped_diagnostic},
                    ],
                },
            },
            2,
        ),
        event(
            "turn/end",
            {
                "reason": {
                    "kind": "error",
                    "error": {
                        "message": diagnostic,
                        "details": {
                            "accessToken": tokens[2],
                            "refresh_token": tokens[3],
                            "id_token": tokens[4],
                            "api_key": tokens[5],
                            "Authorization": f"Bearer {tokens[6]}",
                            "preserved_context": "request failed",
                        },
                    },
                },
            },
            3,
        ),
        SimpleNamespace(method="session.status", payload={"status": "idle"}),
    ]
    original_payloads = deepcopy(
        [notification.payload for notification in notifications]
    )
    incoming = iter(notifications)
    monkeypatch.setattr(
        sys.modules["deepseek_harness"].HarnessClient,
        "next",
        lambda self: next(incoming),
    )
    runtime.planner.timeout_s = 2
    result = runtime.planner.solve(
        system_prompt="task instructions",
        user_message="task request",
        toolkit=Toolkit(),
        max_turns=2,
    )
    raw = Path(result.stats["raw_stream_path"]).read_text()
    saved_result = (
        Path(result.stats["control_path"]).parent.joinpath("result.json").read_text()
    )
    retained = json.dumps(result.messages)
    for token in tokens:
        assert token not in raw
        assert token not in saved_result
        assert token not in retained
        assert token not in result.error
    assert "request failed" in raw
    assert [notification.payload for notification in notifications] == original_payloads


def test_uncertain_boundary_never_reopens():
    toolkit = Toolkit()
    toolkit.result = ToolResult(data={"acted": True}, execution_uncertain=True)
    boundary = dsh._ToolBoundary(toolkit)
    boundary.execute_tool("move", {})
    boundary.cancel_active_and_wait()
    with pytest.raises(RuntimeError, match="uncertain"):
        boundary.resume_calls()
    assert boundary.execute_tool("move", {}).is_error
    assert toolkit.calls == ["move"]


def test_explicit_stop_contract_overrides_legacy_timeout_text():
    from rpent.planner.base import PlannerResult, is_recoverable_planner_result

    result = PlannerResult(
        error="timed out after an uncertain action",
        stop_info={
            "reason": "timeout",
            "execution_certain": False,
            "allow_new_session": False,
        },
    )
    assert not is_recoverable_planner_result(result)
    result = PlannerResult(
        stop_info={
            "reason": "budget",
            "execution_certain": True,
            "allow_new_session": True,
        }
    )
    assert is_recoverable_planner_result(result)
    assert is_recoverable_planner_result(PlannerResult(error="timed out"))


def test_failed_attempt_usage_remains_separate_and_unknown_when_absent():
    recorder = dsh._Recorder(NullDashboardEventSink())
    attempt = event(
        "assistant/attempt",
        {
            "turn": 1,
            "step": 2,
            "stream": [
                {
                    "type": "chunk",
                    "chunk": {"type": "finish", "reason": {"kind": "error"}},
                },
            ],
        },
    )
    recorder.observe(attempt, "receipt")
    recorder.observe(attempt, "receipt")
    assert len(recorder.failed_attempts) == 1
    assert recorder.failed_attempts[0]["usage"] is None
    assert not recorder.failed_attempts[0]["usage_known"]
    assert recorder.usage == {}


@pytest.mark.parametrize(
    "unresolved",
    [
        None,
        "cleanup_failed",
        "execution_uncertain",
        "pending_result",
        "runtime_error",
        "model_error",
    ],
)
def test_final_budget_reconciles_a_stale_snapshot_without_hiding_errors(
    runtime, monkeypatch, unresolved
):
    """A terminal notification can arrive after the owner read an older snapshot."""
    from pathlib import Path

    runtime.mode = "budget"
    runtime.planner.timeout_s = 2
    toolkit = Toolkit()
    drive = runtime.planner._drive
    read_text = Path.read_text
    client_type = sys.modules["deepseek_harness"].HarnessClient
    next_notification = client_type.next
    terminal_sent = False

    def terminal_before_idle(client):
        nonlocal terminal_sent
        notification = next_notification(client)
        if notification.method == "session.status" and not terminal_sent:
            terminal_sent = True
            return event(
                "turn/end",
                {
                    "reason": {
                        "kind": "error" if unresolved == "model_error" else "blocked"
                    }
                },
                seq=2,
            )
        return notification

    monkeypatch.setattr(client_type, "next", terminal_before_idle)

    async def stale_drive(**kwargs):
        def read_snapshot(path, *args, **options):
            if path == kwargs["control_path"]:
                return "{}"
            return read_text(path, *args, **options)

        with monkeypatch.context() as stale:
            stale.setattr(Path, "read_text", read_snapshot)
            result = await drive(**kwargs)
        assert result[0] == "missing_finish"
        if unresolved == "execution_uncertain":
            kwargs["boundary"].execution_uncertain = True
        elif unresolved == "pending_result":
            kwargs["recorder"].pending_calls.add("unacknowledged-act")
        elif unresolved == "runtime_error":
            kwargs["outcome"]["error"] = RuntimeError("inference failed")
            return "runtime_error", "inference failed", result[2]
        return result

    if unresolved == "cleanup_failed":

        def fail_cleanup():
            raise RuntimeError("cleanup unresolved")

        monkeypatch.setattr(toolkit, "cancel_active_and_wait", fail_cleanup)
    monkeypatch.setattr(runtime.planner, "_drive", stale_drive)
    result = runtime.planner.solve(
        system_prompt="task instructions",
        user_message="task request",
        toolkit=toolkit,
        max_turns=2,
    )
    control_path = Path(result.stats["control_path"])
    assert json.loads(control_path.read_text())["stopReason"] == "budget"
    assert result.finish_result is None
    assert toolkit.calls == []
    assert runtime.client.closed.is_set() and runtime.server.stopped
    if unresolved is None:
        assert result.stats["stop_reason"] == "budget"
        assert result.error is None
        assert result.stop_info["allow_new_session"]
    else:
        expected = (
            "runtime_error" if unresolved == "runtime_error" else "missing_finish"
        )
        assert result.stats["stop_reason"] == expected
        assert result.error
        assert not result.stop_info["allow_new_session"]
        if unresolved == "cleanup_failed":
            assert not result.stop_info["cleanup_complete"]
        elif unresolved == "execution_uncertain":
            assert not result.stop_info["execution_certain"]
        elif unresolved == "pending_result":
            assert not result.stop_info["result_delivery_complete"]
    saved = json.loads(control_path.with_name("result.json").read_text())
    assert saved["error"] == result.error
    assert saved["stop_info"] == result.stop_info


@pytest.mark.parametrize(
    "operator_text,expected_text",
    [
        ("Keep away from the red cup.", "Keep away from the red cup."),
        (
            'Keep away from the red cup. {"access_token":"synthetic-handoff-token"}',
            'Keep away from the red cup. {"access_token":"<redacted>"}',
        ),
    ],
)
def test_final_control_confirms_only_consumed_input_after_interaction_is_sealed(
    runtime, monkeypatch, tmp_path, operator_text, expected_text
):
    """Cleanup may deliver the last receipt after the owner stopped observing."""
    runtime.mode = "timeout"
    captured = {}
    confirmed = []
    closed = False

    def seal():
        nonlocal closed
        closed = True

    def reconcile(message_id):
        assert closed
        confirmed.append(message_id)
        return True

    interaction = SimpleNamespace(
        seal_interaction=seal, reconcile_consumed_message=reconcile
    )

    async def drive(**kwargs):
        captured.update(kwargs)
        kwargs["outcome"]["operator_inputs"] = {
            "current": {"text": operator_text},
            "not-delivered": {"text": "Never consumed"},
            "cancelled": {"text": "Withdrawn input"},
        }
        return "interrupted", None, 0

    client_type = sys.modules["deepseek_harness"].HarnessClient
    close = client_type.close

    def final_receipts(client):
        captured["control_path"].write_text(
            json.dumps(
                {
                    "receipts": {
                        "delivered": {"status": "consumed", "messageId": "current"},
                        "queued": {"status": "queued", "messageId": "not-delivered"},
                        "discarded": {"status": "discarded", "messageId": "cancelled"},
                        "malformed": {"status": "consumed", "messageId": None},
                    }
                }
            )
        )
        captured["incoming"].put(event("user/message", {"id": "delivered"}, seq=99))
        close(client)

    monkeypatch.setattr(runtime.planner, "_drive", drive)
    monkeypatch.setattr(client_type, "close", final_receipts)
    result = runtime.planner.solve(
        system_prompt="instructions",
        user_message="task",
        toolkit=Toolkit(),
        max_turns=2,
        dashboard_interaction=interaction,
    )
    assert confirmed == ["current"]
    assert result.stats["stop_reason"] == "interrupted"
    assert result.error is None
    assert not result.stop_info["allow_new_session"]
    assert result.stats["consumed_operator_inputs"] == [
        {
            "message_id": "current",
            "receipt_id": "delivered",
            "text": expected_text,
        }
    ]
    from rpent.planner.deepseek_context import DeepSeekHandoffContext

    context = DeepSeekHandoffContext(tmp_path)
    context.merge(result.stats, 1)
    assert "synthetic-handoff-token" not in context.path.read_text()
    assert json.dumps(expected_text) in context.append_to("Next session")
    artifacts = list(tmp_path.glob("dsh-*/result.json"))
    assert len(artifacts) == 1
    for artifact in artifacts:
        assert "synthetic-handoff-token" not in artifact.read_text()
