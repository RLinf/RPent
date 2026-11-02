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

"""Real SDK/runtime/MCP contracts with an offline deterministic model.

Run with ``pytest tests/e2e_tests/planner/test_deepseek_runtime.py -v`` after
installing ``rpent[test,deepseek-harness]``. No credentials or GPU are used.
"""

from __future__ import annotations

import asyncio
import io
import json
import os
import queue
import signal
import threading
import time
from pathlib import Path

import pytest

pytest.importorskip("deepseek_harness")

from PIL import Image

from rpent.dashboard.events import NullDashboardEventSink
from rpent.memory import MemoryManager
from rpent.planner.deepseek_harness import DeepSeekHarnessPlanner
from rpent.session import EnvState
from rpent.tools import Toolkit, ToolResult, tool
from rpent.utils.logging import init_output_dir


class RecordingToolkit:
    def __init__(self, *, refuse_finish=False, images=None):
        self.calls = []
        self.refuse_finish = refuse_finish
        self.images = images or []
        self.cancelled = False

    @tool(readonly=True)
    def observe(self) -> ToolResult:
        """Read the test scene."""
        return ToolResult(data={"observed": True}, images=self.images)

    @tool()
    def act(self) -> ToolResult:
        """Change the test scene."""
        return ToolResult(data={"acted": True})

    @tool(readonly=True)
    def finish(self) -> ToolResult:
        """Accept completion once the test condition holds."""
        if self.refuse_finish:
            self.refuse_finish = False
            return ToolResult(data={"error": "finish refused: not ready"})
        return ToolResult(data={"_finish": True, "status": "success", "summary": "ok"})

    def list_tools(self):
        return self.observe, self.act, self.finish

    def execute_tool(self, name, arguments):
        self.calls.append(name)
        declaration = next(item for item in self.list_tools() if item.name == name)
        arguments = declaration.args_schema.model_validate(arguments).model_dump()
        return declaration.handler(**arguments)

    def cancel_active_and_wait(self):
        self.cancelled = True

    def resume_calls(self):
        self.cancelled = False


def call(name):
    return {"name": f"mcp__rpent__{name}", "arguments": {}}


@pytest.fixture
def run_runtime(tmp_path, monkeypatch):
    import deepseek_harness

    init_output_dir(tmp_path / "log")
    processes = []
    clients = []
    original_start = deepseek_harness.HarnessClient.start

    def start(client):
        original_start(client)
        clients.append(client)
        processes.append(client._proc)

    monkeypatch.setattr(deepseek_harness.HarnessClient, "start", start)
    write_patch = DeepSeekHarnessPlanner._write_patch
    requests_path = tmp_path / "requests.jsonl"

    def run(
        responses,
        *,
        toolkit=None,
        max_steps=10,
        timeout=15,
        missing_tool=False,
        probe=False,
        expected_model=None,
        interrupt=False,
        image_storage_failure=False,
        image_read_failure=False,
        image_target_scale=1,
        script=None,
        interaction=None,
        retry_policy=None,
        context_window=500000,
        compact_policy=None,
    ):
        def patched(planner, *args):
            path = write_patch(planner, *args)
            rows = json.loads(path.read_text())
            for row in rows:
                if row.get("id") == "llm-pi-ai":
                    row["disabled"] = True
                if row.get("id") == "sdk-jsonrpc-server":
                    row["inject"].append("rpentTestModelReady")
                for inserted in row.get("insert", []):
                    if inserted.get("id") == "rpent-bridge":
                        if missing_tool:
                            inserted["config"]["toolNames"].append(
                                "mcp__rpent__missing"
                            )
                            inserted["config"]["startupTimeoutMs"] = 100
                        if expected_model:
                            inserted["config"]["expectedSelection"]["model"] = (
                                expected_model
                            )
            rows.append(
                {
                    "insert": [
                        {
                            "id": "rpent-test-model",
                            "name": str(
                                Path(__file__).parent
                                / "fixtures"
                                / "deepseek_model.mjs"
                            ),
                            "config": {
                                "responses": responses,
                                "requestsPath": str(requests_path),
                                "imageStorageFailure": image_storage_failure,
                                "imageReadFailure": image_read_failure,
                                "imageTargetScale": image_target_scale,
                                "retryPolicy": retry_policy,
                                "contextWindow": context_window,
                            },
                        }
                    ]
                }
            )
            if compact_policy is not None:
                rows.append({"id": "compaction-basic", "config": compact_policy})
            path.write_text(json.dumps(rows))
            return path

        monkeypatch.setattr(DeepSeekHarnessPlanner, "_write_patch", patched)
        toolkit = toolkit or RecordingToolkit()
        planner = DeepSeekHarnessPlanner(
            output_dir=tmp_path,
            recipe_tag="offline",
            model="gpt-6.1-sol",
            reasoning_effort="xhigh",
            dsh_home=tmp_path / "home",
            timeout_s=timeout,
            dashboard_events=NullDashboardEventSink(),
        )
        cancel_interrupt = threading.Event()
        interrupter = None
        inputs = queue.Queue() if script is not None else None
        script_errors = []
        script_thread = None
        if script is not None:

            def run_script():
                try:
                    script(inputs, requests_path, tmp_path)
                except BaseException as exc:
                    script_errors.append(exc)
                    inputs.put(None)

            script_thread = threading.Thread(target=run_script, daemon=True)
            script_thread.start()
        if interrupt:

            def send_interrupt():
                while not cancel_interrupt.wait(0.02):
                    if requests_path.exists():
                        os.kill(os.getpid(), signal.SIGINT)
                        return

            interrupter = threading.Thread(target=send_interrupt, daemon=True)
            interrupter.start()
        try:
            result = planner._solve(
                "Literal {{unresolved}} RPent prompt.",
                "Run the deterministic test.",
                toolkit,
                max_steps,
                probe=probe,
                input_queue=inputs,
                dashboard_interaction=interaction,
            )
        finally:
            cancel_interrupt.set()
            if interrupter:
                interrupter.join(1)
            if script_thread:
                script_thread.join(10)
                assert not script_thread.is_alive()
                if script_errors:
                    raise script_errors[0]
        assert toolkit.cancelled
        assert processes and all(process.poll() is not None for process in processes)
        if result.error:
            print("Runtime diagnostics:", list(clients[-1]._stderr_lines))
        requests = (
            [json.loads(line) for line in requests_path.read_text().splitlines()]
            if requests_path.exists()
            else []
        )
        control = (
            json.loads(Path(result.stats["control_path"]).read_text())
            if Path(result.stats["control_path"]).exists()
            else {}
        )
        return result, toolkit, requests, control

    return run


def test_finish_blocks_later_calls_in_same_response(run_runtime):
    result, toolkit, requests, control = run_runtime(
        [
            {"calls": [call("finish"), call("act")]},
        ]
    )
    assert result.error is None
    assert result.finish_result["_finish"] is True
    assert toolkit.calls == ["finish"]
    assert len(requests) == 1
    assert control["stopReason"] == "finish"
    assert control["deniedCalls"][0]["name"] == "mcp__rpent__act"
    assert control["selected"]["reasoningEffort"] == "xhigh"
    assert control["observedSelection"] == {
        key: requests[0][key]
        for key in ("provider", "model", "reasoningEffort", "maxTokens")
    }
    assert {item["name"] for item in requests[0]["tools"]} == {
        "mcp__rpent__observe",
        "mcp__rpent__act",
        "mcp__rpent__finish",
    }


def test_step_budget_prevents_extra_model_and_action(run_runtime):
    result, toolkit, requests, control = run_runtime(
        [{"calls": [call("act")]}] * 3,
        max_steps=2,
    )
    assert result.error is None
    assert result.finish_result is None
    assert result.stats["stop_reason"] == "budget"
    assert toolkit.calls == ["act", "act"]
    assert len(requests) == control["steps"] == 2


def test_refused_finish_stays_live_until_accepted(run_runtime):
    result, toolkit, requests, control = run_runtime(
        [
            {"calls": [call("finish")]},
            {"calls": [call("act"), call("finish")]},
        ],
        toolkit=RecordingToolkit(refuse_finish=True),
    )
    assert result.error is None
    assert toolkit.calls == ["finish", "act", "finish"]
    assert len(requests) == 2
    assert control["finishResult"]["_finish"] is True
    assert "finish refused" in json.dumps(requests[1])


def test_text_claim_does_not_become_finish(run_runtime):
    result, toolkit, requests, control = run_runtime([{"text": "Task succeeded."}])
    assert result.error and result.stats["stop_reason"] == "missing_finish"
    assert result.finish_result is None
    assert toolkit.calls == []
    assert len(requests) == 1


def test_literal_prompt_and_image_dimensions_reach_model(run_runtime):
    images = []
    for size, color in [((256, 256), "red"), ((1024, 1024), "blue")]:
        image = io.BytesIO()
        Image.new("RGB", size, color).save(image, format="PNG")
        images.append(image.getvalue())
    result, toolkit, requests, control = run_runtime(
        [
            {"calls": [call("observe")]},
            {"calls": [call("finish")]},
        ],
        toolkit=RecordingToolkit(images=images),
    )
    assert result.error is None
    assert requests[0]["messages"][0]["role"] == "system"
    assert requests[0]["messages"][0]["content"] == [
        {"type": "text", "text": "Literal {{unresolved}} RPent prompt."}
    ]
    model_images = [
        block
        for message in requests[1]["messages"]
        for block in message["content"]
        if block["type"] == "image"
    ]
    assert [
        (block["attachment"]["width"], block["attachment"]["height"])
        for block in model_images
    ] == [(256, 256), (1024, 1024)]
    assert len(control["imageChecks"]) == 2
    assert all(
        check["source"]["width"] == check["request"]["width"]
        and check["source"]["height"] == check["request"]["height"]
        for check in control["imageChecks"]
    )
    assert (Path(result.stats["control_path"]).parent / "attachments").is_dir()


@pytest.mark.parametrize(
    ("kwargs", "error_fragment"),
    [
        ({"image_storage_failure": True}, "projection lost"),
        ({"image_read_failure": True}, "request image read failure"),
        ({"image_target_scale": 0.5}, "target changed the pixel coordinates"),
    ],
)
def test_image_failure_blocks_same_batch_action(run_runtime, kwargs, error_fragment):
    image = io.BytesIO()
    Image.new("RGB", (256, 256), "red").save(image, format="PNG")
    result, toolkit, requests, control = run_runtime(
        [
            {"calls": [call("observe"), call("act")]},
            {"calls": [call("finish")]},
        ],
        toolkit=RecordingToolkit(images=[image.getvalue()]),
        **kwargs,
    )
    assert result.error and result.stats["stop_reason"] == "image_pipeline_error"
    assert toolkit.calls == ["observe"]
    assert len(requests) == 1
    assert error_fragment in control["imageError"]
    assert control["deniedCalls"][0]["name"] == "mcp__rpent__act"


def test_image_normalization_cannot_silently_change_coordinates(run_runtime):
    image = io.BytesIO()
    Image.new("RGB", (3000, 3000), "blue").save(image, format="PNG")
    result, toolkit, requests, control = run_runtime(
        [{"calls": [call("observe"), call("act")]}],
        toolkit=RecordingToolkit(images=[image.getvalue()]),
    )
    assert result.error and result.stats["stop_reason"] == "image_pipeline_error"
    assert toolkit.calls == ["observe"]
    assert len(requests) == 1
    assert "normalization changed the pixel coordinates" in control["imageError"]


def test_model_wait_is_bounded_and_child_reaped(run_runtime):
    result, toolkit, requests, control = run_runtime([{"hang": True}], timeout=5)
    assert result.error and result.stats["stop_reason"] == "timeout"
    assert result.stats["elapsed_s"] < 12
    assert toolkit.calls == []
    assert len(requests) == control["steps"] == 1


def test_missing_mcp_tool_fails_closed_before_model(run_runtime):
    result, toolkit, requests, control = run_runtime(
        [{"calls": [call("act")]}],
        missing_tool=True,
        timeout=5,
    )
    assert result.error
    assert requests == []
    assert toolkit.calls == []
    assert not control.get("ready")


def test_lost_tool_result_stops_before_another_action(run_runtime, monkeypatch):
    import rpent.planner.utils.http_mcp_server as mcp_server

    original = mcp_server.mcp_result

    def lose_result(result, **kwargs):
        if result.data.get("acted"):
            raise RuntimeError("Simulated loss after the action completed")
        return original(result, **kwargs)

    monkeypatch.setattr(mcp_server, "mcp_result", lose_result)
    result, toolkit, requests, control = run_runtime(
        [
            {"calls": [call("act"), call("act")]},
            {"calls": [call("act")]},
        ]
    )
    assert result.error and result.stats["stop_reason"] == "tool_transport_error"
    assert toolkit.calls == ["act"]
    assert len(requests) == 1
    assert control["deniedCalls"][0]["name"] == "mcp__rpent__act"


def test_empty_tool_probe_uses_the_same_runtime(run_runtime):
    class EmptyToolkit(RecordingToolkit):
        def list_tools(self):
            return ()

    result, toolkit, requests, control = run_runtime(
        [{"text": "probe response"}],
        toolkit=EmptyToolkit(),
        probe=True,
    )
    assert result.error is None
    assert result.stats["stop_reason"] == "response"
    assert requests[0].get("tools", []) == []
    assert control["ready"]


def test_handler_exception_cannot_retry_completed_action(run_runtime, tmp_path):
    class UncertainToolkit(Toolkit):
        def __init__(self):
            self.calls = []
            self.cancelled = False
            super().__init__(
                dashboard_events=NullDashboardEventSink(),
                state=EnvState(tmp_path / "state"),
                memory=MemoryManager(tmp_path / "memory"),
            )
            self.add_tool(self.act)

        @tool
        def act(self) -> ToolResult:
            """Perform an action, then lose its RPC result."""
            self.calls.append("act")
            raise OSError("RPC response lost after action execution")

        def get_env_state(self, *, command, result, elapsed_s):
            with self.state.record_step(
                state={"observed": True},
                command=command,
                result=result,
                elapsed_s=elapsed_s,
            ):
                pass
            return ToolResult(data={"observed": True})

        def cancel_active_and_wait(self):
            self.cancelled = True
            super().cancel_active_and_wait()

    result, toolkit, requests, control = run_runtime(
        [{"calls": [call("act"), call("act")]}, {"calls": [call("act")]}],
        toolkit=UncertainToolkit(),
    )
    assert result.error and result.stats["stop_reason"] == "tool_transport_error"
    assert toolkit.calls == ["act"]
    assert len(requests) == 1


def test_accepted_finish_survives_delivery_loss_with_error(run_runtime, monkeypatch):
    import rpent.planner.utils.http_mcp_server as mcp_server

    original = mcp_server.mcp_result

    def lose_finish(result, **kwargs):
        if result.data.get("_finish"):
            raise RuntimeError("Simulated loss after finish was accepted")
        return original(result, **kwargs)

    monkeypatch.setattr(mcp_server, "mcp_result", lose_finish)
    result, toolkit, requests, control = run_runtime(
        [
            {"calls": [call("finish"), call("act")]},
        ]
    )
    assert result.finish_result["_finish"] is True
    assert result.error and result.stats["stop_reason"] == "tool_transport_error"
    assert toolkit.calls == ["finish"]
    assert len(requests) == 1


def test_model_mismatch_stops_before_inference_and_tools(run_runtime):
    result, toolkit, requests, control = run_runtime(
        [{"calls": [call("act")]}],
        expected_model="different-model",
    )
    assert result.error
    assert control["stopReason"] == "model_mismatch"
    assert control["observedSelection"] is None
    assert requests == toolkit.calls == []


@pytest.mark.skipif(os.name == "nt", reason="POSIX SIGINT behavior")
def test_interrupt_drains_tools_and_reaps_runtime(run_runtime):
    result, toolkit, requests, control = run_runtime([{"hang": True}], interrupt=True)
    assert result.error and result.stats["stop_reason"] == "interrupted"
    assert len(requests) == 1
    assert toolkit.calls == []


def test_success_payload_cannot_hide_uncertain_execution(run_runtime):
    class UncertainToolkit(RecordingToolkit):
        @tool
        def act(self) -> ToolResult:
            """Return a payload while declaring the action outcome uncertain."""
            return ToolResult(data={"acted": True}, execution_uncertain=True)

    result, toolkit, requests, control = run_runtime(
        [{"calls": [call("act"), call("act")]}],
        toolkit=UncertainToolkit(),
    )
    assert result.error and result.stats["stop_reason"] == "tool_transport_error"
    assert toolkit.calls == ["act"]
    assert len(requests) == 1


def test_supported_python_value_survives_structured_mcp_result(run_runtime):
    class PathToolkit(RecordingToolkit):
        @tool
        def act(self) -> ToolResult:
            """Return an artifact path accepted by the native result contract."""
            return ToolResult(data={"path": Path("/tmp/test artifact")})

    result, toolkit, requests, control = run_runtime(
        [{"calls": [call("act")]}, {"calls": [call("finish")]}],
        toolkit=PathToolkit(),
    )
    assert result.error is None
    assert toolkit.calls == ["act", "finish"]
    assert "/tmp/test artifact" in json.dumps(requests[1])


def wait_until(predicate, timeout=10):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            value = predicate()
            if value:
                return value
        except (FileNotFoundError, json.JSONDecodeError):
            pass
        time.sleep(0.02)
    raise AssertionError("Runtime did not reach the expected state")


def runtime_state(root):
    paths = list(root.glob("dsh-*/control.json"))
    return json.loads(paths[0].read_text()) if paths else {}


def test_tui_waits_for_input_and_continues_the_same_runtime(run_runtime, caplog):
    def script(inputs, requests, root):
        # Wait for the host to expose idle, not only the child to become idle.
        # An input queued earlier can legitimately skip the waiting display.
        wait_until(lambda: "dsh waiting for input" in caplog.text)
        inputs.put("Continue with the next task.")

    result, toolkit, requests, control = run_runtime(
        [{"text": "Waiting for guidance."}, {"calls": [call("finish")]}],
        script=script,
    )
    assert result.error is None
    assert result.finish_result["_finish"]
    assert len(requests) == control["steps"] == 2
    assert "Continue with the next task." in json.dumps(requests[1]["messages"])
    assert len({request["sessionId"] for request in requests}) == 1
    assert "Waiting for guidance." in caplog.text
    assert "dsh waiting for input" in caplog.text
    operator_inputs = result.stats["consumed_operator_inputs"]
    assert [item["text"] for item in operator_inputs] == [
        "Continue with the next task."
    ]
    assert control["receipts"][operator_inputs[0]["receipt_id"]]["status"] == "consumed"


def test_new_input_rejected_after_native_finish_preserves_completion(
    run_runtime, monkeypatch
):
    from rpent.planner.deepseek_session import HostControl

    request = HostControl.request
    active_root = None
    rejected_after_finish = []
    submit_started = threading.Event()

    class FinishAfterSubmit(RecordingToolkit):
        def execute_tool(self, name, arguments):
            if name == "finish":
                assert submit_started.wait(5), "Host submission did not start"
            return super().execute_tool(name, arguments)

    async def delayed_submit(host, op, **values):
        if op == "submit":
            # Hold only host input submission until the real MCP finish wins.
            # Runtime inference and tool result delivery continue independently.
            submit_started.set()
            await asyncio.to_thread(
                wait_until,
                lambda: runtime_state(active_root).get("stopReason") == "finish",
                timeout=5,
            )
            rejected_after_finish.append(True)
        return await request(host, op, **values)

    monkeypatch.setattr(HostControl, "request", delayed_submit)

    def script(inputs, requests, root):
        nonlocal active_root
        active_root = root
        wait_until(requests.exists)
        inputs.put("This arrives after completion and must remain unconsumed.")

    result, toolkit, requests, control = run_runtime(
        [{"calls": [call("finish")]}], toolkit=FinishAfterSubmit(), script=script
    )
    assert rejected_after_finish == [True]
    assert result.error is None
    assert result.finish_result["_finish"] is True
    assert result.stop_info["reason"] == "finish"
    assert result.stop_info["execution_certain"] is True
    assert result.stop_info["result_delivery_complete"] is True
    assert toolkit.calls == ["finish"]
    assert len(requests) == 1
    assert control["receipts"] == {}
    assert "must remain unconsumed" not in json.dumps(result.messages)
    assert "consumed_operator_inputs" not in result.stats


def test_tui_busy_message_queues_without_interrupting_action(run_runtime):
    def script(inputs, requests, root):
        wait_until(lambda: requests.exists())
        inputs.put("Remember the blue object.")

    result, toolkit, requests, control = run_runtime(
        [{"delayMs": 400, "calls": [call("act")]}, {"calls": [call("finish")]}],
        script=script,
    )
    assert result.error is None
    assert toolkit.calls == ["act", "finish"]
    assert "Remember the blue object." in json.dumps(requests[1]["messages"])
    assert all(item["status"] == "consumed" for item in control["receipts"].values())
    assert [item["text"] for item in result.stats["consumed_operator_inputs"]] == [
        "Remember the blue object."
    ]


def test_tui_explicit_interrupt_pauses_then_continues(run_runtime):
    def script(inputs, requests, root):
        wait_until(lambda: requests.exists())
        inputs.put("/interrupt")
        wait_until(lambda: runtime_state(root).get("paused"))
        inputs.put("Continue after a safe pause.")

    result, toolkit, requests, control = run_runtime(
        [{"hang": True}, {"calls": [call("finish")]}],
        script=script,
    )
    assert result.error is None
    assert result.finish_result["_finish"]
    assert len(requests) == 2
    assert toolkit.calls == ["finish"]


def test_interactive_input_does_not_reset_step_budget(run_runtime):
    def script(inputs, requests, root):
        wait_until(lambda: runtime_state(root).get("quiescent"))
        inputs.put("Another request.")

    result, toolkit, requests, control = run_runtime(
        [{"text": "Waiting."}, {"calls": [call("act")]}],
        script=script,
        max_steps=1,
    )
    assert result.error is None
    assert result.stats["stop_reason"] == "budget"
    assert toolkit.calls == []
    assert len(requests) == 1


def test_tui_quit_cancels_pending_inference(run_runtime):
    def script(inputs, requests, root):
        wait_until(lambda: requests.exists())
        inputs.put("/quit")

    result, toolkit, requests, control = run_runtime(
        [{"hang": True}],
        script=script,
    )
    assert result.stats["stop_reason"] == "interrupted"
    assert result.stop_info["execution_certain"] is True
    assert toolkit.calls == []


def test_idle_user_wait_is_excluded_from_execution_timeout(run_runtime):
    def script(inputs, requests, root):
        wait_until(
            lambda: (
                runtime_state(root).get("quiescent")
                and runtime_state(root).get("steps") == 1
            )
        )
        time.sleep(4)
        inputs.put("Finish now.")

    result, toolkit, requests, control = run_runtime(
        [{"text": "Waiting."}, {"calls": [call("finish")]}],
        script=script,
        timeout=4,
    )
    assert result.error is None
    assert result.stats["elapsed_s"] > 4
    assert result.stats["active_elapsed_s"] < 4
    assert len(requests) == 2


def test_timeout_after_handler_success_cannot_start_a_fresh_session(
    run_runtime, monkeypatch
):
    import rpent.planner.utils.http_mcp_server as mcp_server

    original = mcp_server.mcp_result

    def delay_delivery(result, **kwargs):
        if result.data.get("acted"):
            time.sleep(4)
        return original(result, **kwargs)

    monkeypatch.setattr(mcp_server, "mcp_result", delay_delivery)
    result, toolkit, requests, control = run_runtime(
        [{"calls": [call("act")]}, {"calls": [call("act")]}],
        timeout=3,
    )
    assert result.error
    assert toolkit.calls == ["act"]
    assert not result.stop_info["allow_new_session"]
    assert not result.stop_info["result_delivery_complete"]
    assert result.stop_info["pending_tool_call_ids"]


def test_auto_compaction_preserves_tool_pairing_without_replaying_actions(run_runtime):
    class HistoryToolkit(RecordingToolkit):
        @tool(readonly=True)
        def observe(self) -> ToolResult:
            """Read the recorded scene and prior action journal."""
            return ToolResult(
                data={"history": "The single action has already completed. " * 800}
            )

    def script(inputs, requests, root):
        wait_until(
            lambda: (
                runtime_state(root).get("quiescent")
                and runtime_state(root).get("steps") == 2
            )
        )
        inputs.put(
            "Finish after reconciling the existing action record; do not repeat it."
        )

    result, toolkit, requests, control = run_runtime(
        [
            {"calls": [call("act"), call("observe")]},
            {
                "text": "The action was completed and the journal observed; awaiting input.",
                "usage": {"inputTokens": 8100, "outputTokens": 7, "totalTokens": 8107},
            },
            {
                "text": "The single act call completed successfully and observe confirmed its journal. Do not repeat act; the remaining goal is to finish.",
                "purpose": "compaction",
            },
            {"calls": [call("finish")], "purpose": "root"},
        ],
        toolkit=HistoryToolkit(),
        script=script,
        context_window=16384,
        compact_policy={
            "auto": True,
            "retainTokens": 0,
            "headroomTokens": 128,
            "maxTokens": 128,
        },
    )
    assert result.error is None
    assert toolkit.calls == ["act", "observe", "finish"]
    assert result.stats["inference"]["compactions"] == 1
    assert result.stop_info["result_delivery_complete"]
    final_messages = requests[-1]["messages"]
    paired_calls = {
        block["id"]
        for message in final_messages
        for block in message["content"]
        if block["type"] == "tool-call"
    }
    paired_results = {
        message.get("toolCallId") or message.get("source", {}).get("callId")
        for message in final_messages
        if message["role"] == "tool"
    }
    assert paired_calls == paired_results
    assert "Do not repeat act" in json.dumps(final_messages)
