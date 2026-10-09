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


@pytest.mark.parametrize(
    "prediction, message",
    [
        (np.zeros((5, 14)), "VLA shape"),
        (np.zeros((30, 13)), "VLA shape"),
        (np.zeros((1, 30, 14)), "VLA shape"),
        (np.full((30, 14), np.nan), "finite"),
        (np.full((30, 14), np.inf), "finite"),
    ],
)
def test_pi05_rejects_invalid_prediction_before_motion(
    ready_client, env, prediction, message
):
    primitives = YamPrimitives(
        env=ready_client,
        model=SimpleNamespace(predict=lambda observation: prediction),
        check_cancelled=lambda: None,
    )
    with pytest.raises(ValueError, match=message):
        primitives.pi05_act()
    assert not env._runtime.commands


def test_pi05_saturates_grippers_without_changing_model_output(ready_client, env):
    prediction = np.repeat(env._runtime.qpos[None, :], 30, axis=0)
    prediction[:, [6, 13]] = [-0.5, 1.5]
    primitives = YamPrimitives(
        env=ready_client,
        model=SimpleNamespace(predict=lambda observation: prediction),
        check_cancelled=lambda: None,
    )
    result = primitives.pi05_act()
    assert result["executed_steps"] == 5 and len(env._runtime.commands) == 5
    assert np.array_equal(np.asarray(env._runtime.commands)[:, [6, 13]], [[0, 1]] * 5)
    assert np.array_equal(prediction[:, [6, 13]], [[-0.5, 1.5]] * 30)


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


@pytest.mark.parametrize("pose_name", ["reset", "park_on_close"])
@pytest.mark.parametrize("new_stop", [False, True])
def test_configured_motion_checks_new_stop_before_next_command(
    env, monkeypatch, pose_name, new_stop
):
    env.observe()
    env._stop_requested.set()  # An existing latch permits operator reset/park.
    target = env._runtime.qpos.copy()
    target[0] += 0.01
    env.config[pose_name] = {
        "enabled": True,
        "left_qpos": target[:7].tolist(),
        "right_qpos": target[7:].tolist(),
        "duration_s": 1.0,
        "max_joint_delta": 0.01,
        "tolerance": 0.01,
        "timeout_s": 2.0,
    }

    def move_to(target, *, cancelled, **kwargs):
        for step in range(3):
            if cancelled():
                env._runtime.hold()
                raise RuntimeError("YAM move cancelled")
            env._runtime.command(target)
            if step == 0 and new_stop:
                env.request_stop()
        return env._runtime.hold()

    monkeypatch.setattr(env._runtime, "move_to", move_to, raising=False)
    operation = env.reset_to_configured_qpos if pose_name == "reset" else env.close
    if new_stop:
        with pytest.raises(RuntimeError, match="cancelled"):
            operation()
        assert len(env._runtime.commands) == 1
        assert "hold" in env._runtime.events and "close" not in env._runtime.events
        assert env.is_started() and env._stop_requested.is_set()
    else:
        operation()
        assert len(env._runtime.commands) == 3


@pytest.mark.parametrize("stop_stage", ["read_target", "after_park"])
def test_shutdown_keeps_output_after_late_stop(env, monkeypatch, stop_stage):
    env.observe()
    env.config["park_on_close"] = {"enabled": True}

    if stop_stage == "read_target":
        target = env._runtime.qpos
        env.config["park_on_close"].update(
            left_qpos=target[:7].tolist(),
            right_qpos=target[7:].tolist(),
            duration_s=1.0,
            max_joint_delta=0.01,
            tolerance=0.01,
            timeout_s=2.0,
        )
        reader = env._read_active_target_locked

        def read_target():
            result = reader()
            env._stop_generation = getattr(env, "_stop_generation", 0) + 1
            return result

        monkeypatch.setattr(env, "_read_active_target_locked", read_target)
        monkeypatch.setattr(
            env._runtime,
            "move_to",
            lambda target, **kwargs: env._runtime.hold(),
            raising=False,
        )
    else:
        monkeypatch.setattr(
            env, "_move_to_configured_qpos", lambda name, **kwargs: env.request_stop()
        )
    with pytest.raises(RuntimeError, match="cancelled"):
        env.close()
    assert env.is_started() and "close" not in env._runtime.events


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
