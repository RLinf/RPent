# Copyright 2026 The RPent Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

"""Direct bundled SDK + local MCP contracts, without a Planner or model service."""

from __future__ import annotations

import io
import json
import tempfile
import time
from pathlib import Path

import pytest

pytest.importorskip("deepseek_harness")

from deepseek_harness import HarnessClient, HarnessConfig
from deepseek_harness.errors import JsonRpcError, TransportClosedError
from PIL import Image

from rpent.planner import deepseek_config
from rpent.planner.deepseek_tools import _ToolBoundary
from rpent.planner.utils.http_mcp_server import HttpMcpServer
from rpent.tools import ToolResult, tool
from rpent.utils.logging import init_output_dir

BRIDGE = Path(deepseek_config.__file__).with_name("deepseek_bridge.mjs")
SELECTION = {
    "provider": "openai-codex",
    "model": "gpt-6.1-sol",
    "reasoningEffort": "xhigh",
}
PROMPT = "Literal {{RPent}} system policy."


def call(name):
    return {"name": f"mcp__rpent__{name}", "arguments": {}}


def eventually(read, predicate, timeout=15):
    deadline = time.monotonic() + timeout
    value = None
    while time.monotonic() < deadline:
        value = read()
        if predicate(value):
            return value
        time.sleep(0.02)
    raise AssertionError(f"Runtime did not reach the expected boundary: {value}")


class RecordingToolkit:
    def __init__(self, *, action=None, images=(), refusals=0):
        self.action = action if action is not None else ToolResult(data={"acted": True})
        self.images = list(images)
        self.refusals = refusals
        self.calls = []
        self.cancelled = False

    @tool(readonly=True)
    def observe(self) -> ToolResult:
        """Observe the test scene."""
        return ToolResult(data={"observed": True}, images=self.images)

    @tool
    def act(self) -> ToolResult:
        """Apply one test action."""
        return self.action

    @tool(readonly=True)
    def finish(self) -> ToolResult:
        """Finish after the test condition has been met."""
        if self.refusals:
            self.refusals -= 1
            return ToolResult(data={"error": "not ready", "position": [-0.0]})
        return ToolResult(data={"_finish": True, "status": "success"})

    def list_tools(self):
        return self.observe, self.act, self.finish

    def execute_tool(self, name, arguments):
        self.calls.append(name)
        declaration = next(t for t in self.list_tools() if t.name == name)
        return declaration.handler(
            **declaration.args_schema.model_validate(arguments).model_dump()
        )

    def cancel_active_and_wait(self):
        self.cancelled = True


class BridgeRun:
    def __init__(
        self,
        directory,
        responses,
        *,
        toolkit=None,
        max_steps=10,
        max_tokens=128,
        missing_tool=False,
        expected_model=None,
        model_config=None,
        compact=False,
        auto_compact=False,
        summary_model=None,
    ):
        self.directory = directory
        init_output_dir(directory / "log")
        self.toolkit = toolkit or RecordingToolkit()
        self.boundary = _ToolBoundary(self.toolkit, directory / "ledger.json")
        self.server = HttpMcpServer(self.boundary, include_structured=True)
        self.client = self.subscription = None
        self.process = None
        self.events = []
        self.previous_turns = 0
        self.session_id = "rpent-foundation-contract"
        self.control_path = directory / "control.json"
        self.requests_path = directory / "requests.jsonl"
        self.compaction_path = directory / "compaction.json"
        disabled = (
            "llm-pi-ai",
            "tool-plugin-manager",
            "plugin-manager",
            "hmr",
            "tool-working-directory",
            "tool-bash",
            "tool-pwsh",
            "tool-jobs",
            "tool-fs",
            "tool-fs-search",
            "agent-instructions",
            "skill",
            "skill-filesystem",
            "tool-skill",
            "skill-office",
            "workspace-dependencies",
            "goal",
            "goal-round-driver",
            "command-goal",
            "plan-mode",
            "subagent",
            "subagent-spawn-in-process",
            "subagent-fork-in-process",
            "tool-subagent-control",
            "tool-subagent-list-agents",
            "tool-subagent",
            "tool-subagent-fork",
            "ptc-runtime",
            "workflow-ptc",
            "tool-workflow",
            "tool-todo",
            "tool-goal",
            "tool-web",
            "mcp-resources",
            "session-telemetry-otel",
            "session-title-llm",
        )
        try:
            url = self.server.start()
            patch = [{"id": name, "disabled": True} for name in disabled]
            names = [f"mcp__rpent__{t.name}" for t in self.toolkit.list_tools()]
            if missing_tool:
                names.append("mcp__rpent__missing")
            patch += [
                {"id": "tools", "config": {"mode": "native"}},
                {
                    "id": "agent-loop",
                    "config": {"agents": [], "maxParallelToolCalls": 1},
                },
                {
                    "id": "attachment-local",
                    "config": {"dshHome": str(directory / "images")},
                },
                {
                    "id": "sdk-jsonrpc-server",
                    "inject": [
                        "sdkAppStartup",
                        "loader",
                        "rpentBridgeReady",
                        "rpentTestModelReady",
                    ],
                },
                {
                    "insert": [
                        {
                            "id": "rpent-test-model",
                            "name": str(
                                Path(__file__).parent / "fixtures/deepseek_model.mjs"
                            ),
                            "config": {
                                "responses": responses,
                                "requestsPath": str(self.requests_path),
                                "compactAfterTurn": 1 if compact else None,
                                "compactionPath": str(self.compaction_path),
                                **(model_config or {}),
                            },
                        },
                        {
                            "id": "mcp-rpent",
                            "name": "@deepseek-ai/dsh-mcp-client",
                            "config": {
                                "serverName": "rpent",
                                "transport": "streamable-http",
                                "url": url,
                                "toolCallTimeoutMs": 10000,
                                "failOnStartupError": True,
                                "reconnect": {"enabled": False},
                            },
                        },
                        {
                            "id": "rpent-bridge",
                            "name": str(BRIDGE),
                            "config": {
                                "toolNames": names,
                                "systemPrompt": PROMPT,
                                "maxSteps": max_steps,
                                "controlPath": str(self.control_path),
                                "toolLedgerPath": str(self.boundary.ledger_path),
                                "startupTimeoutMs": 100 if missing_tool else 30000,
                                "expectedSelection": {
                                    **SELECTION,
                                    **(
                                        {"model": expected_model}
                                        if expected_model
                                        else {}
                                    ),
                                },
                            },
                        },
                    ]
                },
            ]
            if compact or auto_compact:
                policy = {
                    "auto": auto_compact,
                    "retainTokens": 0,
                    "headroomTokens": 128,
                    "maxTokens": 128,
                }
                if summary_model:
                    policy.update(
                        summarizationProvider="openai-codex",
                        summarizationModel=summary_model,
                    )
                patch.append({"id": "compaction-basic", "config": policy})
            path = directory / "patch.json"
            path.write_text(json.dumps(patch))
            self.client = HarnessClient(
                HarnessConfig(
                    profile="sdk",
                    dsh_home=str(directory / "home"),
                    cwd=str(directory),
                    patches=(str(path),),
                    initialize_timeout_seconds=15,
                    request_timeout_seconds=15,
                    shutdown_timeout_seconds=1,
                    env={
                        "DSH_TELEMETRY_DISABLED": "1",
                        "NO_PROXY": "127.0.0.1,localhost,::1",
                        "no_proxy": "127.0.0.1,localhost,::1",
                    },
                )
            )
            self.client.start()
            self.process = self.client._proc
            self.client.initialize(
                cwd=str(directory),
                provider=SELECTION["provider"],
                model=SELECTION["model"],
                reasoning_effort="xhigh",
                max_tokens=max_tokens,
            )
            self.subscription = self.client.subscribe_session_notifications(
                self.session_id
            )
        except BaseException:
            self.close()
            raise

    def state(self):
        return json.loads(self.control_path.read_text())

    def requests(self):
        return (
            [json.loads(line) for line in self.requests_path.read_text().splitlines()]
            if self.requests_path.exists()
            else []
        )

    def drain(self):
        self.subscription.drain(
            lambda n: self.events.append({"method": n.method, "payload": n.payload})
        )
        (self.directory / "events.jsonl").write_text(
            "".join(json.dumps(e) + "\n" for e in self.events)
        )
        return [
            e["payload"]["event"] for e in self.events if e["method"] == "session.event"
        ]

    def prompt(self, text="Perform the test task"):
        self.previous_turns = sum(e["type"] == "turn/end" for e in self.drain())
        return self.client.session_prompt(
            self.session_id,
            [{"type": "text", "text": text}],
            notification_subscription=self.subscription,
        )

    def idle(self):
        def completed():
            turns = sum(e["type"] == "turn/end" for e in self.drain())
            state = self.state()
            return state if turns > self.previous_turns and state["quiescent"] else None

        return eventually(completed, bool)

    def close(self):
        try:
            self.boundary.cancel_active_and_wait()
        finally:
            try:
                if self.subscription is not None:
                    self.subscription.close()
                if self.client is not None:
                    self.client.close()
                    assert self.process.poll() is not None
            finally:
                self.server.stop()


@pytest.fixture
def bridge_run():
    active, directories = [], []

    def create(responses, **kwargs):
        directory = tempfile.TemporaryDirectory(prefix="rpent-foundation-", dir="/tmp")
        directories.append(directory)
        run = BridgeRun(Path(directory.name), responses, **kwargs)
        active.append(run)
        return run

    yield create
    for run in reversed(active):
        run.close()
    for directory in directories:
        directory.cleanup()


def test_finish_and_budget_guard_same_batch_and_next_step(bridge_run):
    run = bridge_run([{"calls": [call("act"), call("finish"), call("act")]}])
    run.prompt()
    state = run.idle()
    assert run.toolkit.calls == ["act", "finish"]
    assert state["stopReason"] == "finish" and state["finishResult"]["_finish"]
    assert state["deniedCalls"][0]["name"] == "mcp__rpent__act"
    budget = bridge_run(
        [{"calls": [call("act")]}, {"calls": [call("act")]}], max_steps=1
    )
    budget.prompt()
    assert budget.idle()["stopReason"] == "budget"
    assert len(budget.requests()) == 1 and budget.toolkit.calls == ["act"]


def test_unmarked_prose_and_refused_finish_have_distinct_outcomes(bridge_run):
    prose = bridge_run([{"text": "Task succeeded"}])
    prose.prompt()
    assert prose.idle()["finishResult"] is None and prose.boundary.finish_result is None
    refusal = bridge_run(
        [{"calls": [call("finish")]}, {"calls": [call("act"), call("finish")]}],
        toolkit=RecordingToolkit(refusals=1),
    )
    refusal.prompt()
    assert refusal.idle()["stopReason"] == "finish"
    assert refusal.toolkit.calls == ["finish", "act", "finish"]
    assert "not ready" in json.dumps(refusal.requests()[1])


def test_prompt_allowlist_and_image_geometry_reach_model(bridge_run):
    images = []
    for size in [(256, 256), (1024, 1024)]:
        stream = io.BytesIO()
        Image.new("RGB", size, "blue").save(stream, format="PNG")
        images.append(stream.getvalue())
    run = bridge_run(
        [{"calls": [call("observe")]}, {"calls": [call("finish")]}],
        toolkit=RecordingToolkit(images=images),
    )
    run.prompt()
    state = run.idle()
    requests = run.requests()
    assert requests[0]["messages"][0]["content"] == [{"type": "text", "text": PROMPT}]
    assert {t["name"] for t in requests[0]["tools"]} == set(state["toolNames"])
    model_images = [
        b for m in requests[1]["messages"] for b in m["content"] if b["type"] == "image"
    ]
    assert [
        (b["attachment"]["width"], b["attachment"]["height"]) for b in model_images
    ] == [(256, 256), (1024, 1024)]
    assert len(state["imageChecks"]) == 2
    assert all(
        c["source"]["width"] == c["request"]["width"]
        and c["source"]["height"] == c["request"]["height"]
        for c in state["imageChecks"]
    )
    assert state["observedSelection"]["reasoningEffort"] == "xhigh"


@pytest.mark.parametrize(
    "fault", ["imageStorageFailure", "imageReadFailure", "imageTargetScale"]
)
def test_image_failure_stops_same_batch(bridge_run, fault):
    stream = io.BytesIO()
    Image.new("RGB", (8, 6), "blue").save(stream, format="PNG")
    run = bridge_run(
        [{"calls": [call("observe"), call("act")]}, {"calls": [call("act")]}],
        toolkit=RecordingToolkit(images=[stream.getvalue()]),
        model_config={fault: 0.5 if fault == "imageTargetScale" else True},
    )
    run.prompt()
    assert run.idle()["stopReason"] == "image_pipeline_error"
    assert run.toolkit.calls == ["observe"] and len(run.requests()) == 1


@pytest.mark.parametrize("error", [False, True])
@pytest.mark.parametrize("number", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_payload_is_not_an_acknowledgement(bridge_run, error, number):
    data = {"position": [number], **({"error": "refused"} if error else {})}
    run = bridge_run(
        [{"calls": [call("act"), call("act")]}, {"calls": [call("finish")]}],
        toolkit=RecordingToolkit(action=ToolResult(data=data)),
    )
    run.prompt()
    assert run.idle()["stopReason"] == "tool_transport_error"
    assert run.toolkit.calls == ["act"] and len(run.requests()) == 1


def test_negative_zero_is_delivered_and_uncertainty_cannot_be_hidden(bridge_run):
    zero = bridge_run(
        [{"calls": [call("act")]}, {"calls": [call("finish")]}],
        toolkit=RecordingToolkit(action=ToolResult(data={"pose": [-0.0]})),
    )
    zero.prompt()
    assert zero.idle()["stopReason"] == "finish"
    assert not zero.state().get("toolDeliveryErrors")
    uncertain = bridge_run(
        [{"calls": [call("act"), call("act")]}],
        toolkit=RecordingToolkit(
            action=ToolResult(data={"acted": True}, execution_uncertain=True)
        ),
    )
    uncertain.prompt()
    assert uncertain.idle()["stopReason"] == "tool_transport_error"
    assert uncertain.boundary.execution_uncertain and uncertain.toolkit.calls == ["act"]


@pytest.mark.parametrize("accepted_finish", [False, True])
def test_final_runtime_output_rejection_blocks_later_action(
    bridge_run, monkeypatch, accepted_finish
):
    from rpent.planner.utils import http_mcp_server

    original = http_mcp_server.mcp_result

    def inject(result, **kwargs):
        wire = original(result, **kwargs)
        wire["structuredContent"]["invalid_signed_zero"] = -0.0
        return wire

    monkeypatch.setattr(http_mcp_server, "mcp_result", inject)
    first = "finish" if accepted_finish else "act"
    run = bridge_run([{"calls": [call(first), call("act")]}])
    run.prompt()
    state = run.idle()
    assert state["stopReason"] == "tool_transport_error"
    assert state["toolDeliveryErrors"][0]["code"] == "INVALID_TOOL_OUTPUT"
    assert run.toolkit.calls == [first]
    assert bool(run.boundary.finish_result) == accepted_finish


@pytest.mark.parametrize("fault", ["changed", "lost", "image", "truncated"])
def test_business_refusal_requires_complete_delivery(bridge_run, monkeypatch, fault):
    from rpent.planner.utils import http_mcp_server

    original = http_mcp_server.mcp_result

    def alter(result, **kwargs):
        if fault == "lost":
            raise OSError("Lost the response after the native action")
        wire = original(result, **kwargs)
        if fault == "changed":
            wire["content"][0]["text"] = "Unrelated transport error"
        return wire

    monkeypatch.setattr(http_mcp_server, "mcp_result", alter)
    data = {
        "error": "refused",
        **({"large": "x" * 70000} if fault == "truncated" else {}),
    }
    stream = io.BytesIO()
    Image.new("RGB", (8, 6), "blue").save(stream, format="PNG")
    run = bridge_run(
        [{"calls": [call("act"), call("act")]}],
        toolkit=RecordingToolkit(
            action=ToolResult(
                data=data, images=[stream.getvalue()] if fault == "image" else []
            )
        ),
    )
    run.prompt()
    assert run.idle()["stopReason"] == "tool_transport_error"
    assert run.toolkit.calls == ["act"]


def test_missing_tool_and_model_mismatch_fail_before_inference(bridge_run):
    with pytest.raises(
        (TransportClosedError, TimeoutError),
        match="initialize|startup|closed|timed out",
    ):
        bridge_run([{"calls": [call("act")]}], missing_tool=True)
    run = bridge_run([{"calls": [call("act")]}], expected_model="wrong-model")
    with pytest.raises(JsonRpcError, match="model selection differs"):
        run.prompt()
    eventually(run.state, lambda state: state["stopReason"] == "model_mismatch")
    assert run.requests() == [] and run.toolkit.calls == []


@pytest.mark.parametrize("failure", [None, "summary_error", "wrong_model"])
def test_compaction_preserves_identity_policy_and_blocks_failed_summary(
    bridge_run, failure
):
    run = bridge_run(
        [
            {"text": "First answer", "purpose": "root"},
            {"error": "Summary failed"}
            if failure == "summary_error"
            else {
                "text": "Preserve the original task and completed action record.",
                "purpose": "compaction",
            },
            {"calls": [call("finish")], "purpose": "root"},
        ],
        compact=True,
        summary_model="wrong-model" if failure == "wrong_model" else None,
    )
    run.prompt("Original task history. " * 600)
    eventually(lambda: run.compaction_path.exists(), bool)
    state = run.state()
    if failure:
        assert state["stopReason"] == (
            "model_mismatch" if failure == "wrong_model" else "compaction_error"
        )
        run.prompt("Do not admit another action")
        run.idle()
        assert run.toolkit.calls == []
        assert len(run.requests()) == (1 if failure == "wrong_model" else 2)
    else:
        run.prompt("Finish using the existing record")
        state = run.idle()
        assert [r["purpose"] for r in state["llmRequests"]] == [
            "root",
            "compaction",
            "root",
        ]
        assert state["llmRequests"][1]["selection"]["reasoningEffort"] == "xhigh"
        assert all(r["providerAttempts"] is None for r in state["llmRequests"])
        assert "Preserve the original task" in json.dumps(
            run.requests()[-1]["messages"]
        )
        assert PROMPT in json.dumps(run.requests()[-1]["messages"])


@pytest.mark.parametrize("accepted_finish", [False, True])
def test_lost_native_response_seals_admission_and_retains_finish(
    bridge_run, monkeypatch, accepted_finish
):
    from rpent.planner.utils import http_mcp_server

    def lose_response(result, **kwargs):
        raise OSError("Native result was not delivered")

    monkeypatch.setattr(http_mcp_server, "mcp_result", lose_response)
    first = "finish" if accepted_finish else "act"
    run = bridge_run([{"calls": [call(first), call("act")]}, {"calls": [call("act")]}])
    run.prompt()
    assert run.idle()["stopReason"] == "tool_transport_error"
    assert run.toolkit.calls == [first] and len(run.requests()) == 1
    assert bool(run.boundary.finish_result) == accepted_finish


def test_unknown_tool_has_no_native_effect(bridge_run):
    run = bridge_run(
        [
            {"calls": [{"name": "bash", "arguments": {"command": "touch forbidden"}}]},
            {"calls": [call("finish")]},
        ]
    )
    run.prompt()
    assert run.idle()["stopReason"] == "finish"
    assert run.toolkit.calls == ["finish"]
    assert not (run.directory / "forbidden").exists()


def test_close_during_inference_reaps_owned_runtime_without_actions(bridge_run):
    run = bridge_run([{"hang": True}, {"calls": [call("act")]}])
    run.prompt()
    eventually(run.requests, bool)
    run.close()
    assert run.process.poll() is not None and run.toolkit.cancelled
    assert run.toolkit.calls == [] and len(run.requests()) == 1


def test_retry_preserves_root_step_budget_and_observed_attempts(bridge_run):
    policy = {
        "mode": "normal",
        "maxRetries": 1,
        "retryableCodes": ["RATE_LIMIT"],
        "initialDelayMs": 10,
        "maxDelayMs": 100,
        "jitterRatio": 0,
    }
    run = bridge_run(
        [
            {"failure": {"code": "RATE_LIMIT", "message": "temporary refusal"}},
            {"calls": [call("finish")]},
        ],
        max_steps=1,
        model_config={"retryPolicy": policy},
    )
    run.prompt()
    state = run.idle()
    assert state["stopReason"] == "finish" and state["steps"] == 1
    assert len(run.requests()) == len(state["llmRequests"]) == 2
    assert state["llmRequests"][0]["usage"] is None
    assert all(request["providerAttempts"] is None for request in state["llmRequests"])
    kinds = [event["type"] for event in run.drain()]
    assert kinds.count("llm/retry") == kinds.count("llm/retry-started") == 1
    assert run.toolkit.calls == ["finish"]


def test_automatic_compaction_retains_tool_pairing_without_repeating_action(bridge_run):
    class HistoryToolkit(RecordingToolkit):
        @tool(readonly=True)
        def observe(self) -> ToolResult:
            """Read the completed action journal."""
            return ToolResult(
                data={"history": "The single action has already completed. " * 800}
            )

    run = bridge_run(
        [
            {"calls": [call("act"), call("observe")]},
            {
                "text": "Action recorded.",
                "usage": {"inputTokens": 8100, "outputTokens": 7, "totalTokens": 8107},
            },
            {
                "text": "Act completed and observe confirmed it. Do not repeat act; finish next.",
                "purpose": "compaction",
            },
            {"calls": [call("finish")], "purpose": "root"},
        ],
        toolkit=HistoryToolkit(),
        auto_compact=True,
        max_tokens=8192,
        model_config={"contextWindow": 16384},
    )
    run.prompt()
    run.idle()
    run.prompt("Finish using the recorded action; do not repeat it")
    state = run.idle()
    assert state["stopReason"] == "finish", state
    assert run.toolkit.calls == ["act", "observe", "finish"]
    assert (
        sum(request["purpose"] == "compaction" for request in state["llmRequests"]) == 1
    )
    messages = run.requests()[-1]["messages"]
    calls = {
        block["id"]
        for message in messages
        for block in message["content"]
        if block["type"] == "tool-call"
    }
    results = {
        message.get("toolCallId") or message.get("source", {}).get("callId")
        for message in messages
        if message["role"] == "tool"
    }
    assert calls == results
    assert "Do not repeat act" in json.dumps(messages)
    assert PROMPT in json.dumps(messages)
