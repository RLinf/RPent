# Copyright 2026 The RPent Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

"""Real offline runtime with the production Dashboard interaction state."""

from __future__ import annotations

import io
import json
import threading
from types import SimpleNamespace

import pytest
from PIL import Image

from rpent.dashboard.events import RunStartedEvent
from rpent.dashboard.interaction import InteractionUnavailableError
from rpent.dashboard.state import DashboardState
from rpent.tools import ToolResult

from .test_deepseek_runtime import (
    RecordingToolkit,
    call,
    runtime_state,
    wait_until,
)
from .test_deepseek_runtime import (
    run_runtime as run_runtime,
)


class DashboardToolkit(RecordingToolkit):
    def __init__(self):
        super().__init__()
        self.manual_sources = []
        self.refreshes = []
        image = io.BytesIO()
        Image.new("RGB", (32, 24), "blue").save(image, format="PNG")
        self.observation_image = image.getvalue()

    def execute_tool(self, name, arguments, *, source=None):
        if source is not None:
            self.manual_sources.append(source)
        return super().execute_tool(name, arguments)

    def get_env_state(self, *, command, result, elapsed_s):
        self.refreshes.append(command)
        return ToolResult(
            data={
                "fresh_manual_observation": True,
                "actions_completed": len(self.calls),
            },
            images=[self.observation_image],
        )


def ready_state(tmp_path, toolkit):
    state = DashboardState(
        output_dir=tmp_path / "dashboard",
        dashboard_spec={
            "task": {
                "command": "/rpent-task",
                "usage": "/rpent-task <seed>",
                "fields": ({"name": "seed", "kind": "integer", "minimum": 0},),
                "display": "seed {seed}",
                "output_slug": "s{seed}",
            },
            "runtime_components": (),
            "primitives": ("act",),
        },
    )
    state.configure_planner_control(
        refresh_manual_observations=True, withdraw_queued=True
    )
    state.shared_services_ready()
    state.submit_input("/rpent-task 0")
    assert state.wait_for_task(timeout=0) is not None
    state.emit(RunStartedEvent())
    state.bind_toolkit(toolkit)
    return state


def test_busy_dashboard_input_is_consumed_at_next_boundary_without_cancelling(
    run_runtime, tmp_path
):
    toolkit = DashboardToolkit()
    state = ready_state(tmp_path, toolkit)

    def script(inputs, requests, root):
        wait_until(lambda: requests.exists() and state.planner_activity == "busy")
        state.submit_input("Inspect the blue object next.")

    result, toolkit, requests, control = run_runtime(
        [{"delayMs": 500, "calls": [call("act")]}, {"calls": [call("finish")]}],
        toolkit=toolkit,
        interaction=state,
        script=script,
    )
    assert result.error is None
    assert toolkit.calls == ["act", "finish"]
    assert "Inspect the blue object next." in json.dumps(requests[1]["messages"])
    assert state.snapshot()["interaction"]["messages"][0]["status"] == "sent"
    assert all(
        receipt["status"] == "consumed" for receipt in control["receipts"].values()
    )


def test_dashboard_withdraws_an_sdk_queued_message(run_runtime, tmp_path):
    toolkit = DashboardToolkit()
    state = ready_state(tmp_path, toolkit)

    def script(inputs, requests, root):
        wait_until(lambda: requests.exists() and state.planner_activity == "busy")
        message = state.submit_input("Never consume this withdrawn instruction.")
        wait_until(
            lambda: (
                state.snapshot()["interaction"]["messages"][0]["status"] == "sending"
            )
        )
        state.withdraw_message(message.message_id)
        wait_until(
            lambda: (
                state.snapshot()["interaction"]["messages"][0]["status"] == "withdrawn"
            )
        )

    result, toolkit, requests, control = run_runtime(
        [{"delayMs": 1200, "calls": [call("act")]}, {"calls": [call("finish")]}],
        toolkit=toolkit,
        interaction=state,
        script=script,
    )
    assert result.error is None
    assert toolkit.calls == ["act", "finish"]
    assert "Never consume this withdrawn instruction." not in json.dumps(requests)
    assert any(
        receipt["status"] == "discarded" for receipt in control["receipts"].values()
    )
    assert state.snapshot()["interaction"]["messages"][0]["status"] == "withdrawn"
    assert "consumed_operator_inputs" not in result.stats


def test_dashboard_manual_action_requires_pause_and_refreshes_before_resume(
    run_runtime, tmp_path
):
    toolkit = DashboardToolkit()
    state = ready_state(tmp_path, toolkit)

    def script(inputs, requests, root):
        wait_until(lambda: requests.exists() and state.planner_activity == "busy")
        with pytest.raises(InteractionUnavailableError):
            state.execute_primitive("act", {})
        assert state.request_interrupt() == "accepted"
        wait_until(
            lambda: (
                state.planner_activity == "idle" and runtime_state(root).get("paused")
            )
        )
        state.execute_primitive("act", {})
        state.submit_input("Continue using the actual updated scene.")

    result, toolkit, requests, control = run_runtime(
        [{"hang": True}, {"calls": [call("finish")]}],
        toolkit=toolkit,
        interaction=state,
        script=script,
    )
    assert result.error is None
    assert toolkit.calls == ["act", "finish"]
    assert toolkit.manual_sources == ["dashboard"]
    assert toolkit.refreshes == [
        {"action": "dashboard_resume_observation", "source": "dashboard"}
    ]
    assert "fresh_manual_observation" in json.dumps(requests[1]["messages"])
    assert "image" in json.dumps(requests[1]["messages"])
    assert len({request["sessionId"] for request in requests}) == 1
    assert result.stop_info["execution_certain"] is True
    assert result.stop_info["process_interrupted"] is False
    assert [item["text"] for item in result.stats["consumed_operator_inputs"]] == [
        "Continue using the actual updated scene."
    ]


def test_dashboard_task_replacement_stops_old_runtime_before_new_task(
    run_runtime, tmp_path
):
    toolkit = DashboardToolkit()
    state = ready_state(tmp_path, toolkit)

    def script(inputs, requests, root):
        wait_until(lambda: requests.exists() and state.planner_activity == "busy")
        state.submit_input("/rpent-task 1")

    result, toolkit, requests, control = run_runtime(
        [{"hang": True}],
        toolkit=toolkit,
        interaction=state,
        script=script,
    )
    assert result.error is None
    assert result.stop_info["reason"] == "task_replaced"
    assert result.stop_info["allow_new_session"] is False
    assert result.stop_info["process_interrupted"] is False
    assert state.planner_activity == "ended"
    assert state.claim_next_pending_message() is None
    assert toolkit.calls == []
    assert len(requests) == 1
    state.unbind_toolkit(toolkit)
    state.complete_task(state="cancelled")
    replacement = state.wait_for_task(timeout=0)
    assert replacement.number == 2 and replacement.request["seed"] == 1
    assert state.snapshot()["interaction"]["messages"] == []
    assert state.claim_message_withdrawals() == []


def test_busy_process_sigint_exits_dashboard_controller_and_preserves_result(
    run_runtime, tmp_path, monkeypatch
):
    from rpent.cli import dashboard as dashboard_cli
    from rpent.cli.main import _build_argparser
    from rpent.dashboard.session import DashboardSessionController
    from rpent.memory import MemoryManager
    from rpent.robots import PromptBundle, RunConfig

    toolkit = DashboardToolkit()
    toolkit.memory = MemoryManager(tmp_path / "memory")
    closed = []
    toolkit.close = lambda: closed.append("toolkit")
    state = DashboardState(
        output_dir=tmp_path / "dashboard",
        dashboard_spec={
            "task": {
                "command": "/rpent-task",
                "usage": "/rpent-task <seed>",
                "fields": ({"name": "seed", "kind": "integer", "minimum": 0},),
                "display": "seed {seed}",
                "output_slug": "s{seed}",
            },
            "runtime_components": (),
            "primitives": ("act",),
        },
    )
    args = _build_argparser().parse_args(
        ["--robot", "libero", "--dashboard", "--memory-profile", "local"]
    )
    args.robot_name = "libero"
    spec = SimpleNamespace(
        name="libero",
        is_real_robot=False,
        supports_exploration=False,
        prepare_memory=None,
        finalize_run=None,
        parse_config=lambda task_args: RunConfig(
            recipe_tag="sigint",
            output_dir=task_args.output_dir,
            prompt_vars={},
            task_desc={"seed": task_args.seed},
        ),
        init_runtime=lambda *args: ([], {}),
        prompts=PromptBundle(
            system=lambda variables: "system", user=lambda variables: "user"
        ),
    )
    results = []

    class Planner:
        def solve(self, **kwargs):
            result, _, requests, _ = run_runtime(
                [{"hang": True}],
                toolkit=toolkit,
                interaction=kwargs["dashboard_interaction"],
                interrupt=True,
            )
            assert len(requests) == 1
            results.append(result)
            return result

    monkeypatch.setattr(dashboard_cli, "get_toolkit", lambda *a, **kw: toolkit)
    monkeypatch.setattr(dashboard_cli, "build_planner", lambda *a, **kw: Planner())
    monkeypatch.setattr(dashboard_cli, "_preflight_deepseek", lambda args: None)

    def start_shared():
        state.shared_services_ready()
        state.submit_input("/rpent-task 0")
        return [SimpleNamespace(stop=lambda: closed.append("shared"))], {}

    def run_task(claimed, shared):
        return dashboard_cli._run_dashboard_task(
            args=args,
            robot_spec=spec,
            state=state,
            claimed=claimed,
            shared_runtime_kwargs=shared,
            unique_components=set(),
            session_root=tmp_path / "dashboard",
        )

    # Bound regressions where the planner exits but the controller waits forever.
    timed_out = threading.Event()

    def unblock_failure():
        timed_out.set()
        state.request_shutdown()

    watchdog = threading.Timer(25, unblock_failure)
    watchdog.start()
    try:
        DashboardSessionController(
            state=state, start_shared=start_shared, run_task=run_task
        ).run()
    finally:
        watchdog.cancel()
        watchdog.join()

    assert not timed_out.is_set(), state.snapshot()
    assert len(results) == 1
    assert results[0].stop_info["process_interrupted"] is True
    assert results[0].stop_info["cleanup_complete"] is True
    assert results[0].stop_info["allow_new_session"] is False
    assert toolkit.calls == []
    assert closed == ["toolkit", "shared"]
    transcript = next((tmp_path / "dashboard").rglob("transcript_sigint.json"))
    record = json.loads(transcript.read_text())
    assert record["stop_info"]["process_interrupted"] is True
    assert record["stats"]["stop_reason"] == "interrupted"
    assert record["error"] == "DeepSeek Harness interrupted"


def test_uncertain_manual_action_ends_task_without_resuming_model(
    run_runtime, tmp_path
):
    class UncertainManualToolkit(DashboardToolkit):
        def execute_tool(self, name, arguments, *, source=None):
            result = super().execute_tool(name, arguments, source=source)
            if source == "dashboard":
                result.execution_uncertain = True
            return result

    toolkit = UncertainManualToolkit()
    state = ready_state(tmp_path, toolkit)

    def script(inputs, requests, root):
        wait_until(lambda: requests.exists() and state.planner_activity == "busy")
        state.request_interrupt()
        wait_until(
            lambda: (
                state.planner_activity == "idle" and runtime_state(root).get("paused")
            )
        )
        state.execute_primitive("act", {})
        assert state.manual_execution_uncertain
        assert state.claim_next_pending_message() is None

    result, toolkit, requests, control = run_runtime(
        [{"hang": True}],
        toolkit=toolkit,
        interaction=state,
        script=script,
    )
    assert result.error
    assert result.stop_info["execution_certain"] is False
    assert result.stop_info["allow_new_session"] is False
    assert toolkit.calls == ["act"]
    assert len(requests) == 1
