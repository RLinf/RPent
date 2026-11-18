# Copyright 2026 The RPent Authors.
"""Environment execution receipts, observer delivery and recovery contracts."""

from copy import deepcopy

import numpy as np
import pytest

from robots.robodojo.env_client import (
    ActionRejected,
    ExecutionSyncError,
)
from robots.robodojo.primitives import RoboDojoPrimitives


def action():
    return [0.1] * 12 + [0.25, 0.75]


def feedback(events):
    return [e for e in events if e["kind"] == "action_feedback"]


def test_same_actions_with_and_without_observer_no_duplicate_frames(boundary):
    b = boundary

    def sequence():
        start = len(b.calls)
        b.env.step(action())
        b.env.step_joint_update("left", gripper=0.2)
        b.env.step_joint_update("right", gripper=0.7)
        assert b.calls[start:] == [
            "env.step",
            "env.step_joint_update",
            "env.step_joint_update",
        ]

    sequence()
    baseline = deepcopy(b.applied)
    b.env.reset()
    events = []
    b.env.register_policy_observer(events.append)
    sequence()
    for before, after in zip(baseline, b.applied[3:], strict=True):
        assert before.keys() == after.keys()
        for key in before:
            np.testing.assert_array_equal(before[key], after[key])
    receipts = feedback(events)
    assert [
        (r["step_before"], r["step_after"], r["executed_actions"]) for r in receipts
    ] == [(0, 1, 1), (1, 2, 1), (2, 3, 1)]
    assert [r["observation"]["state"]["counter"] for r in receipts] == [1, 2, 3]
    for _ in range(3):
        b.env.get_obs()
        b.env.require_policy_ready()
    assert len(feedback(events)) == 3
    assert not any(e["kind"] == "reset" for e in events)


def test_validation_and_plan_failure_do_not_damage_history(boundary):
    b = boundary
    events = []
    b.env.register_policy_observer(events.append)
    with pytest.raises(ValueError):
        b.env.step_joint_update("left", gripper=np.nan)
    with pytest.raises(ActionRejected):
        b.env.step_joint_update("left", arm_qpos=[100.0] * 6)
    with pytest.raises(ActionRejected):
        b.env.step([1, 2])
    b.agent.plan_arm_path = lambda *a: {"status": "Failed"}
    primitive = RoboDojoPrimitives(env=b.env, check_cancelled=lambda: None)
    result = primitive.move_to(arm="right", xyz=[0, 0, 1.01]).data
    assert result["stop_reason"] == "plan_failed"
    b.env.require_policy_ready()
    assert not b.applied
    assert not any(e["kind"] == "reset" for e in events)
    assert all(e["executed_actions"] == 0 for e in feedback(events))


@pytest.mark.parametrize("fault", ["timeout", "counter", "missing_obs", "delivery"])
def test_uncertainty_latches_and_preserves_confirmed_progress(boundary, fault):
    b = boundary
    events = []

    def observer(event):
        if (
            fault == "delivery"
            and event["kind"] == "action_feedback"
            and event["step_after"] == 2
        ):
            raise OSError("history delivery failed")
        events.append(event)

    b.env.register_policy_observer(observer)
    original = b.rpc.call

    def call(method, **kwargs):
        result = original(method, **kwargs)
        if method == "env.step_joint_update" and b.native.take_action_cnt[0] == 2:
            if fault == "timeout":
                raise TimeoutError("reply lost after execution")
            if fault == "counter":
                result["step_after"] += 1
            if fault == "missing_obs":
                result.pop("observation")
        return result

    b.rpc.call = call
    primitive = RoboDojoPrimitives(env=b.env, check_cancelled=lambda: None)
    result = primitive.set_gripper(arm="right", val=0.7, steps=4).data
    assert b.native.take_action_cnt[0] == 2
    assert result["executed_actions"] == (
        2 if fault in ("delivery", "missing_obs") else 1
    )
    assert result["episode_status"]["step"] == result["executed_actions"]
    assert result["execution_uncertain"] is (fault not in ("delivery", "missing_obs"))
    assert result["stop_reason"] == (
        "synchronization_lost"
        if fault in ("delivery", "missing_obs")
        else "execution_unknown"
    )
    b.env.resume_execution()  # Resume never repairs lost synchronization.
    for operation in (
        b.env.require_policy_ready,
        lambda: b.env.step_joint_update("right", gripper=0.7),
        lambda: b.env.step(action()),
    ):
        with pytest.raises(ExecutionSyncError):
            operation()
    assert len(b.applied) == 2
    b.rpc.call = original
    b.env.reset()  # Explicit recovery and separate reset notification.
    b.env.require_policy_ready()
    b.env.step(action())
    assert b.native.take_action_cnt[0] == 1
    assert len([e for e in events if e["kind"] == "reset"]) == 1


def test_episode_end_reset_and_cancel_are_distinct(boundary):
    b = boundary
    events = []
    b.native.step_lim = 2
    b.env.register_policy_observer(events.append)
    b.env.step(action())
    b.env.pause_execution()
    with pytest.raises(ExecutionSyncError):
        b.env.step(action())
    b.env.resume_execution()
    b.env.step(action())
    assert b.env.episode_ended
    assert feedback(events)[-1]["truncated"]
    assert [e["kind"] for e in events] == [
        "attached",
        "action_feedback",
        "action_feedback",
    ]
    assert not any(e["kind"] == "reset" for e in events)
    with pytest.raises(ExecutionSyncError):
        b.env.step_joint_update("right", gripper=0.7)
    b.env.reset()
    assert events[-1]["kind"] == "reset"
    b.env.require_policy_ready()


def test_external_actions_are_rejected_at_the_action_boundary(boundary):
    b = boundary
    events = []
    b.env.register_policy_observer(events.append)
    b.facade.step(action())
    b.env.require_policy_ready()
    with pytest.raises(ExecutionSyncError):
        b.env.step_joint_update("right", gripper=0.7)
    assert len(b.applied) == 1


def test_observer_does_not_impose_a_fixed_task_instruction(boundary):
    b = boundary
    events = []
    b.env.register_policy_observer(events.append)
    b.instruction[0] = "Another task"
    b.env.require_policy_ready()
    b.env.step(action())
    assert len(b.applied) == 1
    assert feedback(events)[-1]["observation"]["instruction"] == "Another task"


def test_counter_race_guard_and_all_mutating_entrypoints(boundary):
    b = boundary
    events = []
    b.env.register_policy_observer(events.append)
    with pytest.raises(NotImplementedError):
        b.env.chunk_step([action(), action()])
    with pytest.raises(NotImplementedError):
        b.facade.chunk_step([action()])
    original = b.rpc.call

    def raced(method, **kwargs):
        if method == "env.step_joint_update":
            b.native.take_action_cnt[0] += 1
        return original(method, **kwargs)

    b.rpc.call = raced
    with pytest.raises(ExecutionSyncError):
        b.env.step_joint_update("right", gripper=0.7)
    assert not b.applied  # Stale expected_step was rejected by the server.
    assert not feedback(events)


@pytest.mark.parametrize("kind", ["attached", "closed"])
def test_lifecycle_delivery_failure_blocks_control(boundary, kind):
    b = boundary
    events = []

    def observer(event):
        events.append(event)
        if event["kind"] == kind:
            raise OSError("adapter unavailable")

    if kind == "closed":
        b.env.register_policy_observer(observer)
    with pytest.raises(ExecutionSyncError) as caught:
        if kind == "attached":
            b.env.register_policy_observer(observer)
        else:
            b.env.end_execution()
    assert not caught.value.execution_uncertain
    assert not b.applied
    b.env.resume_execution()
    with pytest.raises(ExecutionSyncError):
        b.env.step_joint_update("right", gripper=0.7)
    if kind == "attached":
        b.env.reset()
        b.env.require_policy_ready()
    else:
        b.env.end_execution()
        assert [e["kind"] for e in events] == ["attached", "closed"]


def test_failed_reset_delivery_never_restores_trust(boundary):
    b = boundary

    def observer(event):
        if event["kind"] == "reset":
            raise OSError("reset frame not accepted")

    b.env.register_policy_observer(observer)
    b.env.step(action())
    with pytest.raises(ExecutionSyncError):
        b.env.reset()
    assert b.native.take_action_cnt[0] == 0
    with pytest.raises(ExecutionSyncError):
        b.env.step(action())
    assert len(b.applied) == 1


def test_reset_recovers_paused_execution(boundary):
    b = boundary
    b.env.register_policy_observer(lambda event: None)
    b.env.pause_execution()
    b.env.reset()
    b.env.require_policy_ready()
    b.env.step_joint_update("right", gripper=0.7)
    assert b.env._history_step == 1


def test_reset_rejects_observation_if_environment_advanced(boundary, monkeypatch):
    b = boundary
    events = []
    b.env.register_policy_observer(events.append)
    events.clear()
    status = b.env.get_status

    def advanced_status():
        b.native.take_action_cnt[0] = 1
        return status()

    monkeypatch.setattr(b.env, "get_status", advanced_status)
    with pytest.raises(ExecutionSyncError, match="advanced after reset"):
        b.env.reset()
    assert not events
    with pytest.raises(RuntimeError, match="history unavailable"):
        b.env.policy_observation()
    with pytest.raises(ExecutionSyncError):
        b.env.step(action())
    assert not b.applied

    monkeypatch.setattr(b.env, "get_status", status)
    b.env.reset()
    b.env.step(action())
    assert [event["kind"] for event in events] == ["reset", "action_feedback"]
    assert feedback(events)[0]["step_before"] == 0
    assert len(b.applied) == 1


def test_attaching_observer_cannot_accept_unacknowledged_actions(boundary, monkeypatch):
    b = boundary
    status = b.env.get_status
    reads = 0
    events = []

    def advance_before_capture():
        nonlocal reads
        reads += 1
        if reads == 2:
            b.native.take_action_cnt[0] += 1
        return status()

    monkeypatch.setattr(b.env, "get_status", advance_before_capture)
    with pytest.raises(ExecutionSyncError, match="advanced"):
        b.env.register_policy_observer(events.append)
    assert not events
    with pytest.raises(ExecutionSyncError):
        b.env.step(action())
    assert not b.applied
