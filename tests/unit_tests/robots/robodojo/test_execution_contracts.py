# Copyright 2026 The RPent Authors.
"""Public cooperation boundary, exercised through the real facade and client."""

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from robots.robodojo.env_client import RoboDojoEnvClient
from robots.robodojo.env_facade import RoboDojoEnvFacade
from robots.robodojo.execution import ActionRejected, ExecutionSyncError
from robots.robodojo.primitives import RoboDojoPrimitives
from tests.unit_tests.robots.robodojo.test_primitive_contracts import make_native
from tests.unit_tests.robots.robodojo.test_vpp2_contracts import observation


@pytest.fixture
def boundary(make_agent, agent_module, monkeypatch):
    native, _ = make_native()
    applied = []
    instruction = ["Move the object."]
    agent = make_agent(native)
    agent.get_status = lambda: {
        "step": native.take_action_cnt[0],
        "step_limit": native.step_lim,
        "success": False,
    }
    agent.get_task_language = lambda: instruction[0]
    agent.get_native_obs = lambda: {
        **observation(native.take_action_cnt[0]),
        "instruction": instruction[0],
        "state": {
            **observation(native.take_action_cnt[0])["state"],
            "counter": native.take_action_cnt[0],
        },
    }
    monkeypatch.setattr(agent_module, "_reward_details", lambda *a: {})

    def apply(action, action_type, **kwargs):
        applied.append(deepcopy(action))
        native.take_action_cnt[0] += 1

    agent.venv.envs[0].apply_action = apply

    def reset():
        native.take_action_cnt[0] = 0
        native.end_flag[0] = False
        return agent.get_native_obs()

    agent.reset_episode = reset
    facade = RoboDojoEnvFacade(agent)
    calls = []

    def call(method, *, args=(), kwargs=None, **options):
        calls.append(method)
        return facade._rpc[method](*args, **(kwargs or {}))

    rpc = SimpleNamespace(call=call)
    client = RoboDojoEnvClient(rpc, expected_meta={})
    return SimpleNamespace(
        env=client,
        native=native,
        agent=agent,
        facade=facade,
        rpc=rpc,
        calls=calls,
        applied=applied,
        instruction=instruction,
    )


def action():
    return [0.1] * 12 + [0.25, 0.75]


def feedback(events):
    return [e for e in events if e["kind"] == "action_feedback"]


def test_same_actions_with_and_without_observer_no_duplicate_frames(boundary):
    b = boundary

    def sequence():
        b.env.step(action())
        b.env.step_joint_update("left", gripper=0.2)
        b.env.hold_step()

    sequence()
    baseline = deepcopy(b.applied)
    b.env.reset()
    events = []
    b.env.register_execution_observer(events.append)
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
    assert [r["source"] for r in receipts] == ["external", "primitive", "hold"]
    assert not any(e["kind"] == "reset" for e in events)


def test_caller_owned_remainder_survives_policy_budget_pause(boundary):
    b = boundary
    events, remainder = [], [action(), action()]

    def observer(event):
        events.append(event)
        if event["kind"] == "actions_invalidated":
            remainder.clear()

    b.env.register_execution_observer(observer)
    b.env.require_policy_ready()
    b.env.step_policy(remainder.pop(0))
    b.env.get_obs()  # No action during the caller's budget pause.
    b.env.require_policy_ready()
    assert len(remainder) == 1
    b.env.step_policy(remainder.pop(0))
    remainder.append(action())
    # Observer must see invalidation before the primitive's native mutation.
    original = b.agent.venv.envs[0].apply_action

    def apply(*args, **kwargs):
        assert remainder == []
        original(*args, **kwargs)

    b.agent.venv.envs[0].apply_action = apply
    b.env.hold_step()
    assert len(feedback(events)) == 3
    assert not any(e["kind"] == "reset" for e in events)


def test_validation_and_plan_failure_do_not_damage_history(boundary):
    b = boundary
    events = []
    b.env.register_execution_observer(events.append)
    with pytest.raises(ValueError):
        b.env.step_joint_update("left", gripper=np.nan)
    with pytest.raises(ActionRejected):
        b.env.step_joint_update("left", arm_qpos=[100.0] * 6)
    with pytest.raises(ActionRejected):
        b.env.step_policy([1, 2])
    b.agent.plan_arm_path = lambda *a: {"status": "Failed"}
    primitive = RoboDojoPrimitives(env=b.env, check_cancelled=lambda: None)
    result = primitive.move_delta(arm="right", dxyz=[0, 0, 0.01]).data
    assert result["stop_reason"] == "plan_failed"
    b.env.require_policy_ready()
    assert not b.applied
    assert not any(e["kind"] in ("reset", "synchronization_lost") for e in events)
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

    b.env.register_execution_observer(observer)
    original = b.rpc.call

    def call(method, **kwargs):
        result = original(method, **kwargs)
        if method == "env.hold_step" and b.native.take_action_cnt[0] == 2:
            if fault == "timeout":
                raise TimeoutError("reply lost after execution")
            if fault == "counter":
                result["step_after"] += 1
            if fault == "missing_obs":
                result.pop("observation")
        return result

    b.rpc.call = call
    primitive = RoboDojoPrimitives(env=b.env, check_cancelled=lambda: None)
    result = primitive.hold(steps=4).data
    assert b.native.take_action_cnt[0] == 2
    assert result["executed_actions"] == (
        2 if fault in ("delivery", "missing_obs") else 1
    )
    assert result["episode_status"]["step"] == result["executed_actions"]
    assert result["execution_uncertain"] is (fault not in ("delivery", "missing_obs"))
    assert result["stop_reason"] == "synchronization_lost"
    b.env.resume_execution()  # Resume never repairs lost synchronization.
    for operation in (
        b.env.require_policy_ready,
        b.env.hold_step,
        lambda: b.env.step(action()),
    ):
        with pytest.raises(ExecutionSyncError):
            operation()
    assert len(b.applied) == 2
    b.rpc.call = original
    b.env.reset()  # Explicit recovery and separate reset notification.
    b.env.require_policy_ready()
    b.env.step_policy(action())
    assert b.native.take_action_cnt[0] == 1
    assert len([e for e in events if e["kind"] == "reset"]) == 1


def test_episode_end_reset_and_cancel_are_distinct(boundary):
    b = boundary
    events = []
    b.native.step_lim = 2
    b.env.register_execution_observer(events.append)
    b.env.step_policy(action())
    b.env.pause_execution()
    with pytest.raises(ExecutionSyncError):
        b.env.step_policy(action())
    b.env.resume_execution()
    b.env.step_policy(action())
    endings = [e for e in events if e["kind"] == "episode_end"]
    assert len(endings) == 1 and endings[0]["truncated"]
    assert not any(e["kind"] == "reset" for e in events)
    with pytest.raises(ExecutionSyncError):
        b.env.hold_step()
    b.env.reset()
    assert events[-1]["kind"] == "reset"
    b.env.require_policy_ready()


@pytest.mark.parametrize("change", ["action", "instruction"])
def test_external_context_changes_block_inference(boundary, change):
    b = boundary
    events = []
    b.env.register_execution_observer(events.append)
    if change == "action":
        b.facade.step(action())
    else:
        b.instruction[0] = "Another task"
    with pytest.raises(ExecutionSyncError):
        b.env.require_policy_ready()
    with pytest.raises(ExecutionSyncError):
        b.env.hold_step()
    assert len(b.applied) == (1 if change == "action" else 0)


def test_counter_race_guard_and_all_mutating_entrypoints(boundary):
    b = boundary
    events = []
    b.env.register_execution_observer(events.append)
    with pytest.raises(NotImplementedError):
        b.env.chunk_step([action(), action()])
    with pytest.raises(NotImplementedError):
        b.facade.chunk_step([action()])
    original = b.rpc.call

    def raced(method, **kwargs):
        if method == "env.hold_step":
            b.native.take_action_cnt[0] += 1
        return original(method, **kwargs)

    b.rpc.call = raced
    with pytest.raises(ExecutionSyncError):
        b.env.hold_step()
    assert not b.applied  # Stale expected_step was rejected by the server.
    assert not feedback(events)


def test_server_terminal_guard_never_updates_controller_targets(boundary):
    b = boundary
    b.native.end_flag[0] = True
    for operation in (
        lambda: b.facade.step(action()),
        b.facade.hold_step,
        lambda: b.facade.step_joint_update("right", gripper=0.5),
    ):
        receipt = operation()
        info = receipt[3] if isinstance(receipt, tuple) else receipt
        assert info["executed_actions"] == 0 and info["terminated"]
    assert not b.applied


def test_observer_cannot_reenter_mutations(boundary):
    b = boundary
    failures = []

    def observer(event):
        if event["kind"] == "action_feedback":
            for operation in (
                b.env.hold_step,
                b.env.reset,
                b.env.pause_execution,
                b.env.end_execution,
            ):
                with pytest.raises(RuntimeError):
                    operation()
                failures.append(operation)

    b.env.register_execution_observer(observer)
    b.env.hold_step()
    assert len(failures) == 4 and len(b.applied) == 1
    b.env.require_policy_ready()


@pytest.mark.parametrize("stop", ["budget", "cancel", "cancel_delivery"])
def test_hold_stops_at_native_budget_or_cancel_with_progress(boundary, stop):
    from rpent.tools.toolkit import ToolCancelled

    b = boundary
    events = []

    def observer(event):
        if stop == "cancel_delivery" and event.get("reason") == "cancelled":
            raise OSError("cancel notification lost")
        events.append(event)

    b.env.register_execution_observer(observer)
    if stop == "budget":
        b.native.step_lim = 3

    def check_cancelled():
        if stop != "budget" and len(b.applied) == 3:
            raise ToolCancelled("stop")

    result = (
        RoboDojoPrimitives(env=b.env, check_cancelled=check_cancelled)
        .hold(steps=7)
        .data
    )
    assert result["executed_actions"] == 3 and not result["completed"]
    assert result["episode_status"]["step"] == 3
    assert len(feedback(events)) == 3
    assert (
        result["stop_reason"]
        == {
            "budget": "budget_exhausted",
            "cancel": "cancelled",
            "cancel_delivery": "synchronization_lost",
        }[stop]
    )
    assert len(b.applied) == 3


def test_takeover_delivery_failure_blocks_before_mutation(boundary):
    b = boundary
    broken = [True]

    def observer(event):
        if broken[0] and event["kind"] == "actions_invalidated":
            raise OSError("adapter unavailable")

    b.env.register_execution_observer(observer)
    with pytest.raises(ExecutionSyncError) as caught:
        b.env.hold_step()
    assert not caught.value.execution_uncertain
    assert not b.applied
    broken[0] = False
    b.env.resume_execution()
    with pytest.raises(ExecutionSyncError):
        b.env.require_policy_ready()
    b.env.reset()
    b.env.require_policy_ready()


def test_failed_reset_delivery_never_restores_trust(boundary):
    b = boundary

    def observer(event):
        if event["kind"] == "reset":
            raise OSError("reset frame not accepted")

    b.env.register_execution_observer(observer)
    b.env.step_policy(action())
    with pytest.raises(ExecutionSyncError):
        b.env.reset()
    assert b.native.take_action_cnt[0] == 0
    with pytest.raises(ExecutionSyncError):
        b.env.step_policy(action())
    assert len(b.applied) == 1


def test_confirmed_mixed_actions_feed_vpp2_once_and_reads_do_not_advance(boundary):
    from tests.unit_tests.robots.robodojo.test_vpp2_contracts import stamps

    b = boundary
    events = []
    b.env.register_execution_observer(events.append)
    primitive = RoboDojoPrimitives(env=b.env, check_cancelled=lambda: None)
    for index in range(3):
        if index == 0:
            b.env.step_policy(action())
        elif index == 1:
            primitive.hold(steps=1)
        else:
            b.env.step_joint_update("left", gripper=0.5)
        assert b.env._history_step == len(feedback(events)) == index + 1
        b.env.get_obs()
        b.env.get_motion_state()
        assert b.native.take_action_cnt[0] == b.env._history_step == index + 1
        assert len(b.env._rgb_history) == index + 2
        assert stamps(b.env.policy_observation())[-1] == index + 1
        assert len(feedback(events)) == index + 1
    assert [(r["step_after"], r["executed_actions"]) for r in feedback(events)] == [
        (1, 1),
        (2, 1),
        (3, 1),
    ]


@pytest.mark.parametrize("fault", ["timeout", "delivery", "reset_delivery"])
def test_vpp2_history_stays_invalid_until_successful_reset(boundary, fault):
    b = boundary
    broken = [True]
    original = b.rpc.call

    def observer(event):
        if (
            broken[0]
            and event["kind"]
            == ("reset" if fault == "reset_delivery" else "action_feedback")
            and fault != "timeout"
        ):
            raise OSError("delivery failed")

    def call(method, **kwargs):
        result = original(method, **kwargs)
        if broken[0] and fault == "timeout" and method == "env.hold_step":
            raise TimeoutError("reply lost")
        return result

    b.env.register_execution_observer(observer)
    b.rpc.call = call
    with pytest.raises(ExecutionSyncError):
        b.env.reset() if fault == "reset_delivery" else b.env.hold_step()
    b.env.get_obs()
    b.env.resume_execution()
    with pytest.raises(RuntimeError, match="history unavailable"):
        b.env.policy_observation()
    broken[0] = False
    b.env.reset()
    assert b.env.policy_observation()["inference_index"] == 0
    primitive = RoboDojoPrimitives(env=b.env, check_cancelled=lambda: None)
    assert primitive.hold(steps=1).data["executed_actions"] == 1
    assert b.env._history_step == 1


def test_reset_recovers_paused_execution(boundary):
    b = boundary
    b.env.register_execution_observer(lambda event: None)
    b.env.pause_execution()
    b.env.reset()
    b.env.require_policy_ready()
    b.env.hold_step()
    assert b.env._history_step == 1
