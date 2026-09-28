# Copyright 2026 The RPent Authors.
# Licensed under the Apache License, Version 2.0.

import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from robots.yam.primitives import YamPrimitives
from rpent.tools.toolkit import readonly


def test_operator_ready_and_verdict_are_required(toolkit_factory, receipt):
    toolkit = toolkit_factory()
    result = toolkit.execute_tool("reset", {})
    assert result.result["awaiting_operator"] and toolkit._session_attempt == 0
    receipt("ready")
    toolkit.execute_tool("reset", {})
    assert toolkit._session_attempt == 1
    pending = toolkit.execute_tool("finish", {"status": "success", "summary": "claim"})
    assert pending.result["awaiting_operator"] and not pending.is_finish
    assert not toolkit.solved()
    assert toolkit.exploration_continuation() is None
    assert toolkit.exploration_continuation(explicit=True)


@pytest.mark.parametrize(
    "event,mode,budget,attempt,finished,status",
    [
        ("success", "exploration", 2, 1, True, "success"),
        ("abort", "exploration", 2, 1, True, "failure"),
        ("failure", "exploration", 2, 1, False, "retry"),
        ("failure", "exploration", 2, 2, True, "failure"),
        ("failure", "exploration", 0, 1, True, "failure"),
        ("failure", "evaluation", 2, 1, True, "failure"),
    ],
)
@pytest.mark.parametrize("requested_status", ["success", "failure"])
def test_finish_respects_operator_verdict_and_attempt_budget(
    toolkit_factory,
    ready_client,
    receipt,
    event,
    mode,
    budget,
    attempt,
    finished,
    status,
    requested_status,
):
    toolkit = toolkit_factory(mode=mode)
    toolkit._attempts_per_session = budget
    toolkit._session_attempt = attempt
    receipt(event)
    result = toolkit.execute_tool(
        "finish", {"status": requested_status, "summary": "agent claim"}
    )
    assert result.is_finish is finished and result.result["status"] == status
    assert toolkit.solved() is (event == "success")
    if event != "success":
        assert toolkit.write_recipe("unverified") == ""


def test_pi05_executes_five_steps_per_chunk(ready_client, env):
    calls = []

    def predict(observation):
        calls.append(observation)
        return np.repeat(observation["states"][None, :], 30, axis=0)

    primitives = YamPrimitives(
        env=ready_client,
        model=SimpleNamespace(predict=predict),
        check_cancelled=lambda: None,
    )
    with pytest.raises(ValueError, match="use_length=5"):
        primitives.pi05_act(use_length=30)
    assert not calls and not env._runtime.commands
    result = primitives.pi05_act(chunks=2)
    assert result["completed"] and result["executed_steps"] == 10
    assert len(calls) == 2 and len(env._runtime.commands) == 10


def test_stop_rpc_error_waits_for_active_tool_before_propagating(
    toolkit_factory, ready_client, monkeypatch
):
    toolkit = toolkit_factory()
    entered = threading.Event()
    release = threading.Event()
    stop_attempted = threading.Event()
    cancel_finished = threading.Event()

    @readonly
    def active_tool():
        entered.set()
        assert release.wait(5)
        toolkit.raise_if_cancelled()

    def failed_stop():
        stop_attempted.set()
        raise TimeoutError("stop RPC failed")

    def cancel():
        try:
            toolkit.cancel_active_and_wait()
        finally:
            cancel_finished.set()

    toolkit.add_tool("active_tool", {}, active_tool)
    with monkeypatch.context() as patch, ThreadPoolExecutor(max_workers=2) as pool:
        patch.setattr(ready_client, "request_stop", failed_stop)
        active = pool.submit(toolkit.execute_tool, "active_tool", {})
        try:
            assert entered.wait(5)
            cancellation = pool.submit(cancel)
            assert stop_attempted.wait(5)
            assert not cancel_finished.wait(0.05)
        finally:
            release.set()
        with pytest.raises(TimeoutError, match="stop RPC failed"):
            cancellation.result(timeout=5)
        assert active.result(timeout=5).result["interrupted"] is True


def test_stale_receipt_not_reused_after_reset(ready_client, env, receipt):
    old = env._episode_id
    receipt("success")
    assert ready_client.read_control_state()[1]["episode_status"]["eval_success"]
    receipt("ready")
    ready_client.reset()
    receipt("success", episode_id=old)
    current = ready_client.read_control_state()[1]["episode_status"]
    assert current["episode_id"] != old and not current["eval_success"]
    receipt("abort")
    assert ready_client.read_control_state()[1]["episode_status"]["stop_requested"]


@pytest.mark.parametrize(
    "result,included",
    [
        ({"success": False, "recoverable": True, "executed_steps": 2}, True),
        ({"success": False, "recoverable": True, "executed_steps": 0}, False),
        ({"error": "response lost"}, True),
        ({"error": "plan refused", "motion_refused": True}, False),
    ],
)
def test_recipe_keeps_issued_motion_unless_execution_was_refused(
    toolkit_factory, ready_client, receipt, monkeypatch, result, included
):
    toolkit = toolkit_factory()
    monkeypatch.setattr(
        toolkit._primitives,
        "move_to",
        lambda **kwargs: result,
    )
    toolkit.execute_tool("move_to", {"arm": "right", "xyz": [0, 0, 0]})
    receipt("success")
    toolkit.execute_tool("render", {})
    text = Path(toolkit.write_recipe("solved")).read_text()
    assert ('"action": "move_to"' in text) == included
