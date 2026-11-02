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

"""Actual bundled-runtime host control and compaction, without model services."""

from __future__ import annotations

import base64
import io
import json
import socket
import tempfile
import time
from pathlib import Path

import pytest

pytest.importorskip("deepseek_harness")

from deepseek_harness import HarnessClient, HarnessConfig
from PIL import Image

from rpent.planner import deepseek_harness as planner_module

BRIDGE = Path(planner_module.__file__).with_name("deepseek_bridge.mjs")
SELECTION = {
    "provider": "openai-codex",
    "model": "gpt-6.1-sol",
    "reasoningEffort": "xhigh",
}


def eventually(read, predicate, timeout=12):
    deadline = time.monotonic() + timeout
    value = None
    while time.monotonic() < deadline:
        value = read()
        if predicate(value):
            return value
        time.sleep(0.02)
    raise AssertionError(f"Runtime did not reach the expected boundary: {value}")


class BridgeRun:
    def __init__(
        self,
        directory,
        responses,
        *,
        max_steps=10,
        compact=False,
        summary_model=None,
        auto_compact=False,
    ):
        self.directory = directory
        self.sequence = 0
        self.session_id = "rpent-control-contract"
        self.socket_path = directory / "host.sock"
        self.control_path = directory / "control.json"
        self.requests_path = directory / "requests.jsonl"
        self.compaction_path = directory / "compaction.json"
        fixture = Path(__file__).parent / "fixtures" / "deepseek_model.mjs"
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
        )
        patch = [{"id": name, "disabled": True} for name in disabled]
        patch += [
            {"id": "tools", "config": {"mode": "native"}},
            {"id": "agent-loop", "config": {"agents": [], "maxParallelToolCalls": 1}},
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
                        "name": str(fixture),
                        "config": {
                            "responses": responses,
                            "requestsPath": str(self.requests_path),
                            "compactAfterTurn": 1 if compact else None,
                            "compactionPath": str(self.compaction_path),
                            "contextWindow": 16384 if auto_compact else 500000,
                        },
                    },
                    {
                        "id": "rpent-bridge",
                        "name": str(BRIDGE),
                        "config": {
                            "toolNames": [],
                            "systemPrompt": "Literal {{RPent}} system policy.",
                            "maxSteps": max_steps,
                            "controlPath": str(self.control_path),
                            "expectedSelection": SELECTION,
                            "hostControl": {
                                "runId": "test-run",
                                "socketPath": str(self.socket_path),
                            },
                        },
                    },
                ],
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
        patch_path = directory / "patch.json"
        patch_path.write_text(json.dumps(patch))
        self.client = HarnessClient(
            HarnessConfig(
                profile="sdk",
                dsh_home=str(directory / "home"),
                cwd=str(directory),
                patches=(str(patch_path),),
                initialize_timeout_seconds=15,
                request_timeout_seconds=15,
                shutdown_timeout_seconds=1,
                env={"DSH_TELEMETRY_DISABLED": "1"},
            )
        )
        self.client.start()
        self.process = self.client._proc
        try:
            self.client.initialize(
                cwd=str(directory),
                provider=SELECTION["provider"],
                model=SELECTION["model"],
                reasoning_effort="xhigh",
                max_tokens=128,
            )
        except BaseException:
            self.client.close()
            raise
        self.subscription = self.client.subscribe_session_notifications(self.session_id)

    def state(self):
        return json.loads(self.control_path.read_text())

    def requests(self):
        if not self.requests_path.exists():
            return []
        return [
            json.loads(line) for line in self.requests_path.read_text().splitlines()
        ]

    def prompt(self, text="Initial input"):
        return self.client.session_prompt(
            self.session_id,
            [{"type": "text", "text": text}],
            notification_subscription=self.subscription,
        )

    def command(self, op, **fields):
        self.sequence += 1
        return self.raw({"runId": "test-run", "seq": self.sequence, "op": op, **fields})

    def raw(self, command):
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
            connection.settimeout(18)
            connection.connect(str(self.socket_path))
            connection.sendall(json.dumps(command).encode() + b"\n")
            return json.loads(connection.makefile("rb").readline())

    def idle(self):
        return eventually(self.state, lambda state: state["quiescent"])

    def close(self):
        self.subscription.close()
        self.client.close()
        assert self.process.poll() is not None


@pytest.fixture
def bridge_run():
    active = []
    temporary = []

    def create(responses, **kwargs):
        directory = tempfile.TemporaryDirectory(
            prefix="rpent-bridge-control-", dir="/tmp"
        )
        temporary.append(directory)
        run = BridgeRun(Path(directory.name), responses, **kwargs)
        active.append(run)
        return run

    yield create
    for run in reversed(active):
        run.close()
    for directory in temporary:
        directory.cleanup()


def test_private_socket_idle_submit_and_idempotent_ack(bridge_run):
    run = bridge_run([{"text": "first"}, {"text": "second", "delayMs": 100}])
    run.prompt()
    run.idle()
    assert run.socket_path.stat().st_mode & 0o777 == 0o600
    command = {
        "runId": "test-run",
        "seq": 1,
        "op": "submit",
        "messageId": "second",
        "text": "Continue",
    }
    accepted = run.raw(command)
    assert accepted["ok"] and accepted["newTurn"]
    assert run.raw(command) == accepted
    assert not run.raw({**command, "text": "changed"})["ok"]
    assert not run.raw({**command, "seq": 2, "runId": "wrong"})["ok"]
    state = run.idle()
    assert state["receipts"][accepted["receipt"]]["status"] == "consumed"
    assert state["steps"] == len(run.requests()) == 2
    assert state["hostControl"]["lastCommandSeq"] == 1
    run.sequence = 1
    assert not run.command("withdraw", receipt=accepted["receipt"])["withdrawn"]


def test_busy_steering_is_consumed_at_next_step(bridge_run):
    run = bridge_run([{"text": "first", "delayMs": 300}, {"text": "steered"}])
    run.prompt()
    eventually(run.requests, lambda requests: len(requests) == 1)
    accepted = run.command("submit", messageId="steer", text="Apply next step")
    assert accepted["ok"] and not accepted["newTurn"]
    state = run.idle()
    assert state["receipts"][accepted["receipt"]]["status"] == "consumed"
    assert "Apply next step" in json.dumps(run.requests()[1]["messages"])


def test_pause_discards_pending_input_then_resume_keeps_budget(bridge_run):
    run = bridge_run([{"hang": True}, {"text": "after pause"}], max_steps=2)
    run.prompt()
    eventually(run.requests, lambda requests: len(requests) == 1)
    pending = run.command("submit", messageId="pending", text="Discard this")
    paused = run.command("pause")
    assert paused["ok"] and paused["quiescent"] and paused["state"] == "paused"
    assert paused["discarded"] == [pending["receipt"]]
    assert run.command("resume")["ok"]
    accepted = run.command("submit", messageId="fresh", text="Resume safely")
    assert accepted["ok"]
    assert run.idle()["steps"] == 2
    assert not run.command("submit", messageId="over-budget", text="Too many")["ok"]
    assert run.state()["stopReason"] == "budget"
    assert not run.command("resume")["ok"]
    assert len(run.requests()) == 2


def test_withdraw_only_removes_pending_host_receipt(bridge_run):
    run = bridge_run([{"text": "one", "delayMs": 300}])
    run.prompt()
    eventually(run.requests, lambda requests: len(requests) == 1)
    accepted = run.command("submit", messageId="withdraw", text="Never consume")
    assert run.command("withdraw", receipt=accepted["receipt"])["withdrawn"]
    assert not run.command("withdraw", receipt="foreign")["withdrawn"]
    state = run.idle()
    assert state["receipts"][accepted["receipt"]]["status"] == "discarded"
    assert len(run.requests()) == 1


def test_host_observation_preserves_image_geometry(bridge_run):
    run = bridge_run([{"text": "one"}, {"text": "with observation"}])
    run.prompt()
    run.idle()
    image = io.BytesIO()
    Image.new("RGB", (256, 256), "red").save(image, format="PNG")
    accepted = run.command(
        "submit",
        messageId="observe",
        text="Observe after manual motion",
        observation={
            "content": [
                {"type": "text", "text": "Fresh physical state"},
                {
                    "type": "image",
                    "mimeType": "image/png",
                    "data": base64.b64encode(image.getvalue()).decode(),
                },
            ]
        },
    )
    assert accepted["ok"]
    state = run.idle()
    assert state["imageChecks"][0]["tool"] == "host-observation"
    images = [
        block
        for message in run.requests()[1]["messages"]
        for block in message["content"]
        if block["type"] == "image"
    ]
    assert [
        (block["attachment"]["width"], block["attachment"]["height"])
        for block in images
    ] == [(256, 256)]


def test_failed_host_observation_is_terminal_before_another_model_call(bridge_run):
    run = bridge_run([{"text": "one"}])
    run.prompt()
    run.idle()
    rejected = run.command(
        "submit",
        messageId="bad-observation",
        text="Continue",
        observation={
            "content": [{"type": "image", "mimeType": "image/png", "data": "invalid"}]
        },
    )
    assert not rejected["ok"]
    assert rejected["stopReason"] == "image_pipeline_error"
    assert not run.command("resume")["ok"]
    assert len(run.requests()) == 1


def test_compaction_maintenance_is_not_quiescent_and_cancels_before_ack(bridge_run):
    run = bridge_run(
        [
            {"text": "First answer", "purpose": "root"},
            {"hang": True, "purpose": "compaction"},
        ],
        compact=True,
    )
    run.prompt("Original task details. " * 600)
    eventually(run.requests, lambda requests: len(requests) == 2)
    status = run.command("status")
    assert not status["quiescent"]
    assert status["state"] == "running"
    paused = run.command("pause")
    assert paused["ok"] and paused["quiescent"]
    assert paused["stopReason"] == "compaction_error"
    assert not run.command("resume")["ok"]


def test_real_compaction_uses_exact_effort_and_retains_policy(bridge_run):
    run = bridge_run(
        [
            {"text": "First answer", "purpose": "root"},
            {
                "text": "Remember the task and tool restrictions.",
                "purpose": "compaction",
                "usage": {"inputTokens": 100, "outputTokens": 9, "totalTokens": 109},
            },
            {"text": "After compaction", "purpose": "root"},
        ],
        compact=True,
    )
    run.prompt("Original task details. " * 600)
    eventually(lambda: run.compaction_path.exists(), bool)
    assert json.loads(run.compaction_path.read_text()) == {"compacted": True}
    run.idle()
    assert run.command("submit", messageId="after", text="Continue after summary")["ok"]
    state = run.idle()
    assert state["steps"] == 2
    assert [request["purpose"] for request in state["llmRequests"]] == [
        "root",
        "compaction",
        "root",
    ]
    assert state["llmRequests"][1]["selection"]["reasoningEffort"] == "xhigh"
    assert state["llmRequests"][1]["usage"]["inputTokens"] == 100
    assert all(request["providerAttempts"] is None for request in state["llmRequests"])
    assert any(event["type"] == "compaction/summary" for event in state["compactions"])
    assert "Remember the task and tool restrictions." in json.dumps(
        run.requests()[2]["messages"]
    )
    assert "Literal {{RPent}} system policy." in json.dumps(
        run.requests()[2]["messages"]
    )


@pytest.mark.parametrize("summary_model", [None, "wrong-model"])
def test_compaction_failure_or_wrong_model_cannot_resume(bridge_run, summary_model):
    run = bridge_run(
        [{"text": "First"}, {"error": "Summary failed"}],
        compact=True,
        summary_model=summary_model,
    )
    run.prompt("Important history. " * 600)
    eventually(lambda: run.compaction_path.exists(), bool)
    assert "error" in json.loads(run.compaction_path.read_text())
    state = run.idle()
    assert state["stopReason"] == (
        "model_mismatch" if summary_model else "compaction_error"
    )
    assert not run.command("resume")["ok"]
    assert not run.command("submit", messageId="invalid", text="Continue")["ok"]
    assert len(run.requests()) == (1 if summary_model else 2)


def test_small_context_automatically_compacts_before_continuation(bridge_run):
    run = bridge_run(
        [
            {
                "text": "The red block placement is recorded.",
                "purpose": "root",
                "usage": {
                    "inputTokens": 14000,
                    "outputTokens": 7,
                    "totalTokens": 14007,
                },
            },
            {
                "text": "The target is the blue block; the red block was already placed. Never repeat that completed action.",
                "purpose": "compaction",
            },
            {"text": "Continuing with the blue block.", "purpose": "root"},
        ],
        auto_compact=True,
    )
    run.prompt(
        "Remember: target the blue block. "
        + "Red block was placed and must not move again. " * 1200
    )
    run.idle()
    assert run.command(
        "submit", messageId="pressure", text="Continue, preserving completed actions."
    )["ok"]
    state = run.idle()
    assert [request["purpose"] for request in state["llmRequests"]] == [
        "root",
        "compaction",
        "root",
    ]
    assert state["steps"] == 2
    assert not state["stopReason"]
    continuation = json.dumps(run.requests()[-1]["messages"])
    assert "blue block" in continuation
    assert "red block was already placed" in continuation
    assert "Literal {{RPent}} system policy." in continuation
