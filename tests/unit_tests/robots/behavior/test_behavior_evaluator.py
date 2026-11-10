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

"""Official evaluator bridge, policy lifecycle and WebSocket transport."""

from __future__ import annotations

import asyncio
import subprocess
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import msgpack
import numpy as np
import pytest
from websockets.asyncio.client import connect
from websockets.asyncio.server import serve
from websockets.exceptions import ConnectionClosedError

from robots.behavior.eval.evaluator import (
    BridgeCancelled,
    BridgeSessionEnded,
    EvaluatorStepBridge,
    OfficialEvaluatorPolicy,
    Packer,
    _health_check,
    create_official_policy_server,
    unpackb,
)


class ThreadCall:
    def __init__(self, function, *args, **kwargs):
        self.done = threading.Event()
        self.value = None
        self.error = None

        def run():
            try:
                self.value = function(*args, **kwargs)
            except BaseException as error:
                self.error = error
            finally:
                self.done.set()

        self.thread = threading.Thread(target=run, daemon=True)
        self.thread.start()

    def result(self):
        assert self.done.wait(3.0), "bridge worker did not finish"
        self.thread.join()
        if self.error is not None:
            raise self.error
        return self.value


@pytest.fixture
def bridge():
    instance = EvaluatorStepBridge(timeout_s=2.0)
    try:
        yield instance
    finally:
        instance.close()


def test_motion_snapshot_tracks_control_callbacks_not_repeated_reads(bridge):
    obs0 = {"r1::proprio": np.zeros(61), "frame": 0}
    policy0 = ThreadCall(bridge.observe_and_get_action, obs0)
    initial = bridge.get_motion_snapshot()
    assert initial["motion_context"]["initial_bootstrap"] is True
    assert (
        bridge.get_motion_snapshot()["observation_token"]
        == initial["observation_token"]
    )
    initial["observation"]["r1::proprio"][3] = 9
    initial["motion_context"]["current_proprio"][3] = 9
    assert obs0["r1::proprio"][3] == 0
    assert bridge.get_motion_snapshot()["motion_context"]["current_proprio"][3] == 0
    chunk = ThreadCall(
        bridge.chunk_step,
        np.zeros((2, 23)),
        expected_observation_token=initial["observation_token"],
    )
    policy0.result()
    assert bridge.get_motion_snapshot()["motion_context"]["unconfirmed_actions"] == 1
    obs1 = {"r1::proprio": np.full(61, 0.01), "frame": 1}
    policy1 = ThreadCall(bridge.observe_and_get_action, obs1)
    policy1.result()
    obs2 = {"r1::proprio": np.full(61, 0.02), "frame": 2}
    policy2 = ThreadCall(bridge.observe_and_get_action, obs2)
    info = chunk.result()[4]
    snapshot = bridge.get_motion_snapshot()
    assert info["observation_token"] == snapshot["observation_token"]
    assert info["observation_token"]["observation_index"] == 2
    context = info["motion_context"]
    assert context["pair_confirmed"] and not context["initial_bootstrap"]
    assert context["control_dt_s"] == 1 / 30
    assert context["previous_token"]["observation_index"] == 1
    np.testing.assert_array_equal(context["previous_proprio"], obs1["r1::proprio"])
    np.testing.assert_array_equal(context["current_proprio"], obs2["r1::proprio"])
    with pytest.raises(RuntimeError, match="stale_observation"):
        bridge.chunk_step(
            np.zeros((1, 23)), expected_observation_token=initial["observation_token"]
        )
    bridge.close()
    with pytest.raises(BridgeCancelled):
        policy2.result()


def test_same_positions_new_generation_do_not_reuse_planning_token(bridge):
    policy = ThreadCall(bridge.observe_and_get_action, {"r1::proprio": np.zeros(61)})
    old = bridge.get_motion_snapshot()
    bridge.reset()
    with pytest.raises(BridgeCancelled):
        policy.result()
    policy = ThreadCall(bridge.observe_and_get_action, {"r1::proprio": np.zeros(61)})
    current = bridge.get_motion_snapshot()
    assert current["motion_context"]["previous_proprio"] is None
    assert current["motion_context"]["initial_bootstrap"] is True
    with pytest.raises(RuntimeError, match="stale_observation"):
        bridge.chunk_step(
            np.zeros((1, 23)), expected_observation_token=old["observation_token"]
        )
    bridge.close()
    with pytest.raises(BridgeCancelled):
        policy.result()


def test_partial_chunk_motion_history_does_not_confirm_the_unobserved_last_action(
    bridge,
):
    policy0 = ThreadCall(bridge.observe_and_get_action, {"r1::proprio": np.zeros(61)})
    start = bridge.get_motion_snapshot()
    chunk = ThreadCall(
        bridge.chunk_step,
        np.zeros((2, 23)),
        expected_observation_token=start["observation_token"],
    )
    policy0.result()
    policy1 = ThreadCall(bridge.observe_and_get_action, {"r1::proprio": np.ones(61)})
    policy1.result()
    bridge.end_session()
    info = chunk.result()[4]
    assert info["confirmed_actions"] == 1 and info["unconfirmed_actions"] == 1
    assert info["observation_token"]["observation_index"] == 1
    assert info["motion_context"]["current_token"] == info["observation_token"]
    assert info["motion_context"]["unconfirmed_actions"] == 1
    assert info["observation_after_last_issued_action"] is False


def test_each_action_needs_the_next_observation_including_the_last(bridge):
    obs0, obs1, obs2 = ({"frame": i} for i in range(3))
    policy0 = ThreadCall(bridge.observe_and_get_action, obs0)
    assert bridge.get_observation() is obs0
    actions = np.arange(46, dtype=np.float32).reshape(2, 23)
    tool = ThreadCall(bridge.chunk_step, actions, return_all_frames=True)

    np.testing.assert_array_equal(policy0.result(), actions[0])
    assert bridge.stats()["issued"] == 1
    assert bridge.stats()["confirmed"] == 0
    assert not tool.done.wait(0.03)

    policy1 = ThreadCall(bridge.observe_and_get_action, obs1)
    np.testing.assert_array_equal(policy1.result(), actions[1])
    assert bridge.stats()["issued"] == 2
    assert bridge.stats()["confirmed"] == 1
    assert not tool.done.wait(0.03)

    policy2 = ThreadCall(bridge.observe_and_get_action, obs2)
    frames, reward, terminated, truncated, info = tool.result()
    assert frames == [obs1, obs2]
    assert reward is terminated is truncated is None
    assert info["executed_actions"] == 2
    assert info["requested_actions"] == 2
    assert bridge.stats()["unconfirmed"] == 0
    # This request has confirmed the final action but has no new action yet.
    assert not policy2.done.wait(0.03)
    bridge.close()
    with pytest.raises(BridgeCancelled, match="closed"):
        policy2.result()


def test_sequential_chunks_continue_on_the_waiting_evaluator_request(bridge):
    policy0 = ThreadCall(bridge.observe_and_get_action, {"frame": 0})
    first = ThreadCall(bridge.step, np.ones(23))
    np.testing.assert_array_equal(policy0.result(), np.ones(23))

    policy1 = ThreadCall(bridge.observe_and_get_action, {"frame": 1})
    assert first.result()[4]["executed_actions"] == 1
    second = ThreadCall(bridge.chunk_step, np.full((1, 23), 2.0))
    np.testing.assert_array_equal(policy1.result(), np.full(23, 2.0))

    policy2 = ThreadCall(bridge.observe_and_get_action, {"frame": 2})
    assert second.result()[0] == {"frame": 2}
    assert bridge.stats()["issued"] == bridge.stats()["confirmed"] == 2
    bridge.close()
    with pytest.raises(BridgeCancelled):
        policy2.result()


def test_overlapping_chunks_are_rejected_without_discarding_the_first(bridge):
    policy0 = ThreadCall(bridge.observe_and_get_action, {"frame": 0})
    first = ThreadCall(bridge.step, np.zeros(23))
    policy0.result()

    with pytest.raises(RuntimeError, match="outstanding chunk"):
        bridge.chunk_step(np.ones((1, 23)))

    policy1 = ThreadCall(bridge.observe_and_get_action, {"frame": 1})
    assert first.result()[4]["executed_actions"] == 1
    assert bridge.stats()["issued"] == 1
    bridge.close()
    with pytest.raises(BridgeCancelled):
        policy1.result()


def test_reset_keeps_unknown_execution_and_never_reuses_an_old_chunk(bridge):
    old_obs = {"episode": "old", "frame": 0}
    old_policy = ThreadCall(bridge.observe_and_get_action, old_obs)
    old_actions = np.stack([np.full(23, 10.0), np.full(23, 11.0)])
    old_tool = ThreadCall(bridge.chunk_step, old_actions)
    np.testing.assert_array_equal(old_policy.result(), old_actions[0])

    bridge.reset()
    with pytest.raises(BridgeCancelled, match="reset"):
        old_tool.result()
    assert bridge.last_obs is old_obs
    assert bridge.stats()["last_observation_generation"] == 0
    assert bridge.stats()["history"][0]["unconfirmed"] == 1
    assert bridge.stats()["history"][0]["cancel_reason"] == "reset"

    reader = ThreadCall(bridge.get_observation)
    assert not reader.done.wait(0.03)
    new_obs = {"episode": "new", "frame": 0}
    new_policy = ThreadCall(bridge.observe_and_get_action, new_obs)
    assert reader.result() is new_obs
    # The queued old action (11) cannot escape into the new episode.
    assert not new_policy.done.wait(0.03)
    assert bridge.stats()["confirmed"] == 0

    new_tool = ThreadCall(bridge.step, np.full(23, 20.0))
    np.testing.assert_array_equal(new_policy.result(), np.full(23, 20.0))
    new_policy1 = ThreadCall(
        bridge.observe_and_get_action, {"episode": "new", "frame": 1}
    )
    assert new_tool.result()[4]["bridge_generation"] == 1
    assert bridge.stats()["issued"] == 2
    assert bridge.stats()["confirmed"] == 1
    assert bridge.stats()["unconfirmed"] == 1
    bridge.close()
    with pytest.raises(BridgeCancelled):
        new_policy1.result()


def test_reset_cancels_a_waiting_evaluator_request(bridge):
    pending = ThreadCall(bridge.observe_and_get_action, {"episode": "old"})
    bridge.get_observation()
    bridge.reset()
    with pytest.raises(BridgeCancelled, match="reset"):
        pending.result()
    assert bridge.stats()["issued"] == 0


@pytest.mark.parametrize("operation", ["get_observation", "chunk_step", "policy"])
def test_close_wakes_every_kind_of_waiter(bridge, operation):
    if operation == "get_observation":
        pending = ThreadCall(bridge.get_observation)
    elif operation == "chunk_step":
        pending = ThreadCall(bridge.chunk_step, np.zeros((1, 23)))
    else:
        pending = ThreadCall(bridge.observe_and_get_action, {"frame": 0})
    assert not pending.done.wait(0.03)

    bridge.close()
    with pytest.raises(BridgeCancelled, match="closed"):
        pending.result()
    assert bridge.stats()["confirmed"] == 0
    assert bridge.stats()["closed"] is True


def test_timeout_cancels_generation_without_confirming_an_issued_action():
    bridge = EvaluatorStepBridge(timeout_s=0.3)
    try:
        policy = ThreadCall(bridge.observe_and_get_action, {"frame": 0})
        tool = ThreadCall(bridge.step, np.zeros(23))
        policy.result()
        with pytest.raises(BridgeCancelled, match="timed out"):
            tool.result()
        assert bridge.stats()["issued"] == 1
        assert bridge.stats()["confirmed"] == 0
        assert bridge.stats()["unconfirmed"] == 1
        with pytest.raises(BridgeCancelled, match="timed out"):
            bridge.observe_and_get_action({"frame": 1})
        assert bridge.last_obs == {"frame": 0}
        bridge.reset()
        assert "timed out" in bridge.stats()["history"][0]["cancel_reason"]
    finally:
        bridge.close()


def test_concurrent_evaluator_requests_do_not_replace_the_current_observation(bridge):
    observation = {"frame": 0}
    pending = ThreadCall(bridge.observe_and_get_action, observation)
    bridge.get_observation()
    with pytest.raises(RuntimeError, match="serialized evaluator"):
        bridge.observe_and_get_action({"frame": "unexpected"})
    assert bridge.last_obs is observation
    bridge.close()
    with pytest.raises(BridgeCancelled):
        pending.result()


def test_session_end_returns_partial_chunk_without_confirming_last_action(bridge):
    obs0, obs1 = {"frame": 0}, {"frame": 1}
    first_policy = ThreadCall(bridge.observe_and_get_action, obs0)
    tool = ThreadCall(bridge.chunk_step, np.zeros((3, 23)))
    first_policy.result()
    second_policy = ThreadCall(bridge.observe_and_get_action, obs1)
    second_policy.result()

    bridge.end_session()
    observation, reward, terminated, truncated, info = tool.result()
    assert observation is obs1
    assert reward is terminated is truncated is None
    assert info["requested_actions"] == 3
    assert info["issued_actions"] == 2
    assert info["executed_actions"] == info["confirmed_actions"] == 1
    assert info["unconfirmed_actions"] == 1
    assert info["stop_reason"] == "session_ended"
    assert info["session_outcome"] == "unknown"
    assert info["observation_after_last_issued_action"] is False
    assert bridge.get_observation() is obs1
    assert bridge.session_status() == {
        "ended": True,
        "reason": "connection_closed",
        "outcome": "unknown",
        "generation": 0,
    }
    assert bridge.stats()["unconfirmed"] == 1

    # A late tool call reports closure without replaying the queued action.
    late = bridge.step(np.ones(23))
    assert late[4]["issued_actions"] == late[4]["executed_actions"] == 0
    with pytest.raises(BridgeSessionEnded, match="outcome unknown"):
        bridge.observe_and_get_action({"frame": 2})
    assert bridge.last_obs is obs1
    assert bridge.stats()["issued"] == 2


def test_session_end_before_any_confirmation_keeps_frame_list_empty(bridge):
    obs0 = {"frame": 0}
    policy = ThreadCall(bridge.observe_and_get_action, obs0)
    tool = ThreadCall(bridge.chunk_step, np.zeros((1, 23)), return_all_frames=True)
    policy.result()
    bridge.end_session("evaluator_disconnected")

    frames, _, terminated, truncated, info = tool.result()
    assert frames == []
    assert info["last_known_observation_available"] is True
    assert info["observation_after_last_issued_action"] is False
    assert info["issued_actions"] == info["unconfirmed_actions"] == 1
    assert info["executed_actions"] == 0
    assert terminated is truncated is None
    assert bridge.get_observation() is obs0


def test_session_end_without_current_generation_frame_never_returns_old_observation(
    bridge,
):
    old_policy = ThreadCall(bridge.observe_and_get_action, {"episode": "old"})
    bridge.get_observation()
    bridge.reset()
    with pytest.raises(BridgeCancelled, match="reset"):
        old_policy.result()

    reader = ThreadCall(bridge.get_observation)
    tool = ThreadCall(bridge.step, np.zeros(23))
    bridge.end_session()
    for pending in (reader, tool):
        with pytest.raises(BridgeSessionEnded):
            pending.result()
    assert bridge.stats()["last_observation_generation"] == 0
    assert bridge.stats()["issued"] == 0


def test_session_end_releases_evaluator_waiting_for_an_action(bridge):
    pending = ThreadCall(bridge.observe_and_get_action, {"frame": 0})
    bridge.get_observation()
    bridge.end_session()
    with pytest.raises(BridgeSessionEnded):
        pending.result()
    assert bridge.get_observation() == {"frame": 0}


def test_connection_close_after_a_fault_does_not_hide_cancellation(bridge):
    policy = ThreadCall(bridge.observe_and_get_action, {"frame": 0})
    tool = ThreadCall(bridge.step, np.zeros(23))
    policy.result()
    bridge.close()
    bridge.end_session()
    with pytest.raises(BridgeCancelled, match="closed"):
        tool.result()
    assert bridge.session_status()["ended"] is False
    assert bridge.stats()["history"][0]["cancel_reason"] == "closed"
    assert bridge.stats()["unconfirmed"] == 1


def test_new_observations_extend_slow_chunk_wait_but_not_legacy_timeout():
    bridge = EvaluatorStepBridge(
        timeout_s=0.1,
        simulation_timeout_s=0.4,
        action_timeout_s=2.0,
        chunk_timeout_s=2.0,
    )
    try:
        policy = ThreadCall(bridge.observe_and_get_action, {"frame": 0})
        tool = ThreadCall(bridge.chunk_step, np.zeros((3, 23)))
        policy.result()
        for frame in (1, 2):
            time.sleep(0.15)
            policy = ThreadCall(bridge.observe_and_get_action, {"frame": frame})
            policy.result()
        time.sleep(0.15)
        final_policy = ThreadCall(bridge.observe_and_get_action, {"frame": 3})
        result = tool.result()
        assert result[4]["executed_actions"] == 3
        assert result[4]["observation_after_last_issued_action"] is True
        bridge.end_session()
        with pytest.raises(BridgeSessionEnded):
            final_policy.result()
    finally:
        bridge.close()


def test_progress_does_not_extend_absolute_chunk_deadline():
    bridge = EvaluatorStepBridge(
        simulation_timeout_s=0.4,
        action_timeout_s=2.0,
        chunk_timeout_s=0.5,
    )
    try:
        policy = ThreadCall(bridge.observe_and_get_action, {"frame": 0})
        tool = ThreadCall(bridge.chunk_step, np.zeros((5, 23)))
        policy.result()
        for frame in (1, 2):
            time.sleep(0.15)
            policy = ThreadCall(bridge.observe_and_get_action, {"frame": frame})
            policy.result()
        with pytest.raises(BridgeCancelled, match="chunk_step timed out"):
            tool.result()
        assert bridge.stats()["confirmed"] == 2
        assert bridge.stats()["issued"] == 3
        bridge.end_session()
        assert bridge.session_status()["ended"] is False
    finally:
        bridge.close()


def test_no_progress_timeout_is_independent_of_total_chunk_budget():
    bridge = EvaluatorStepBridge(
        simulation_timeout_s=0.15, action_timeout_s=2.0, chunk_timeout_s=2.0
    )
    try:
        policy = ThreadCall(bridge.observe_and_get_action, {"frame": 0})
        tool = ThreadCall(bridge.step, np.zeros(23))
        policy.result()
        with pytest.raises(BridgeCancelled, match="without observation timed out"):
            tool.result()
        assert bridge.stats()["unconfirmed"] == 1
    finally:
        bridge.close()


def test_evaluator_action_wait_has_its_own_timeout():
    bridge = EvaluatorStepBridge(simulation_timeout_s=2.0, action_timeout_s=0.1)
    try:
        with pytest.raises(BridgeCancelled, match="observe_and_get_action timed out"):
            bridge.observe_and_get_action({"frame": 0})
        assert bridge.stats()["issued"] == 0
    finally:
        bridge.close()


@pytest.mark.parametrize(
    "name", ["timeout_s", "simulation_timeout_s", "action_timeout_s", "chunk_timeout_s"]
)
@pytest.mark.parametrize("value", [True, 0, -1, float("inf"), float("nan")])
def test_timeouts_must_be_positive_and_finite(name, value):
    with pytest.raises(ValueError, match=name):
        EvaluatorStepBridge(**{name: value})


def test_official_policy_traces_native_actions_and_unknown_session_end(tmp_path):
    bridge = EvaluatorStepBridge(timeout_s=2)
    resets = []
    policy = OfficialEvaluatorPolicy(
        bridge, task_id=40, reset_policy=lambda: resets.append(1)
    )
    policy.reset()
    assert resets == [1]
    assert bridge.session_status()["generation"] == 1
    actions = np.arange(46, dtype=np.float32).reshape(2, 23)
    with ThreadPoolExecutor(max_workers=2) as threads:
        first = threads.submit(policy.act, {"frame": np.array([0])})
        bridge.get_observation()
        tool = threads.submit(bridge.chunk_step, actions)
        np.testing.assert_array_equal(first.result(timeout=3), actions[0])
        second = threads.submit(policy.act, {"frame": np.array([1])})
        np.testing.assert_array_equal(second.result(timeout=3), actions[1])
        policy.close()
        result = tool.result(timeout=3)
    assert result[4]["executed_actions"] == 1
    assert result[4]["unconfirmed_actions"] == 1
    assert result[4]["stop_reason"] == "session_ended"
    policy.close("repeat_close")
    with pytest.raises(RuntimeError, match="session is closed"):
        policy.act({"frame": 2})


def test_reset_after_observation_cancels_waiters_and_rejects_second_episode(tmp_path):
    bridge = EvaluatorStepBridge(timeout_s=2)
    policy = OfficialEvaluatorPolicy(bridge, task_id=0)
    policy.reset()
    with ThreadPoolExecutor(max_workers=1) as threads:
        pending = threads.submit(policy.act, {"frame": 0})
        bridge.get_observation()
        with pytest.raises(RuntimeError, match="one episode per process"):
            policy.reset()
        with pytest.raises(BridgeCancelled, match="closed"):
            pending.result(timeout=3)
    policy.close()
    assert bridge.stats()["closed"] is True
    assert bridge.session_status()["ended"] is False


def test_failed_model_reset_is_a_fault_not_a_successful_session(tmp_path):
    bridge = EvaluatorStepBridge(timeout_s=2)

    def fail_reset():
        raise RuntimeError("model reset failed")

    policy = OfficialEvaluatorPolicy(bridge, task_id=40, reset_policy=fail_reset)
    with pytest.raises(RuntimeError, match="model reset failed"):
        policy.reset()
    policy.close()
    assert bridge.stats()["closed"] is True
    assert bridge.stats()["session_ended"] is False


def test_connection_close_during_model_reset_does_not_reopen_bridge(tmp_path):
    started = threading.Event()
    release = threading.Event()
    bridge = EvaluatorStepBridge(timeout_s=2)

    def slow_reset():
        started.set()
        assert release.wait(3)

    policy = OfficialEvaluatorPolicy(bridge, task_id=40, reset_policy=slow_reset)
    with ThreadPoolExecutor(max_workers=2) as threads:
        reset = threads.submit(policy.reset)
        try:
            assert started.wait(1)
            close = threads.submit(policy.close)
            close.result(timeout=1)
            assert bridge.session_status()["ended"] is True
        finally:
            release.set()
        with pytest.raises(RuntimeError, match="closed during reset"):
            reset.result(timeout=1)
    assert bridge.session_status()["generation"] == 0
    assert bridge.stats()["session_ended"] is True


def test_policy_rejects_overlapping_requests_without_consuming_observation(tmp_path):
    bridge = EvaluatorStepBridge(timeout_s=2)
    policy = OfficialEvaluatorPolicy(bridge, task_id=40)
    with ThreadPoolExecutor(max_workers=1) as threads:
        pending = threads.submit(policy.act, {"frame": 0})
        bridge.get_observation()
        with pytest.raises(RuntimeError, match="serialized"):
            policy.act({"frame": 1})
        policy.close()
        with pytest.raises(BridgeSessionEnded):
            pending.result(timeout=3)
    assert bridge.get_observation() == {"frame": 0}


@pytest.mark.parametrize("failure", [False, True])
def test_server_connection_exit_releases_waiters_without_inventing_success(
    tmp_path, monkeypatch, failure
):
    class FakeWebsocketPolicyServer:
        def __init__(self, policy, **kwargs):
            self.policy = policy
            self.kwargs = kwargs

        async def _serve_connection(self, websocket):
            if failure:
                raise ConnectionError("connection lost")

    monkeypatch.setattr(
        "robots.behavior.eval.evaluator.WebsocketPolicyServer",
        FakeWebsocketPolicyServer,
    )
    bridge = EvaluatorStepBridge(timeout_s=2)
    policy = OfficialEvaluatorPolicy(bridge, task_id=0)
    server = create_official_policy_server(policy, port=8000, metadata={"test": True})
    assert server.kwargs["metadata"]["task_id"] == 0

    class FakeWebsocket:
        async def wait_closed(self):
            await asyncio.Future()

    if failure:
        with pytest.raises(ConnectionError, match="connection lost"):
            asyncio.run(server._serve_connection(FakeWebsocket()))
    else:
        asyncio.run(server._serve_connection(FakeWebsocket()))
    assert bridge.session_status()["ended"] is True
    assert bridge.session_status()["outcome"] == "unknown"


def test_disconnect_wakes_shielded_policy_thread_before_action_timeout(
    tmp_path, monkeypatch
):
    workers = []

    class FakeWebsocketPolicyServer:
        def __init__(self, policy, **kwargs):
            self.policy = policy

        async def _serve_connection(self, websocket):
            worker = asyncio.create_task(
                asyncio.to_thread(self.policy.act, {"frame": 0})
            )
            workers.append(worker)
            await asyncio.shield(worker)

    monkeypatch.setattr(
        "robots.behavior.eval.evaluator.WebsocketPolicyServer",
        FakeWebsocketPolicyServer,
    )
    bridge = EvaluatorStepBridge(timeout_s=5)
    policy = OfficialEvaluatorPolicy(bridge, task_id=40)
    server = create_official_policy_server(policy, port=8000)

    async def run():
        disconnected = asyncio.Event()

        class FakeWebsocket:
            async def wait_closed(self):
                await disconnected.wait()

        connection = asyncio.create_task(server._serve_connection(FakeWebsocket()))
        await asyncio.to_thread(bridge.get_observation)
        disconnected.set()
        await asyncio.wait_for(connection, timeout=1)
        with pytest.raises(BridgeSessionEnded):
            await asyncio.wait_for(workers[0], timeout=1)

    asyncio.run(run())
    assert bridge.session_status()["ended"] is True
    assert bridge.stats()["issued"] == 0


@pytest.mark.parametrize("task_id", [-1, True, "40"])
def test_policy_requires_explicit_integer_task_identity(task_id):
    with pytest.raises(ValueError, match="task_id"):
        OfficialEvaluatorPolicy(EvaluatorStepBridge(), task_id=task_id)


async def _wait_for_first_observation(bridge):
    # Reset has no protocol acknowledgment. Like the documented service startup,
    # wait for the evaluator's first frame before reading through the Env bridge.
    async with asyncio.timeout(2):
        while bridge.stats()["last_observation_generation"] != 1:
            await asyncio.sleep(0.001)
    return await asyncio.to_thread(bridge.get_observation)


def test_numpy_binary_fields_match_official_wire_format():
    values = np.arange(12, dtype=">f4").reshape(3, 4)
    encoded = Packer().pack({"rgb": values, "count": np.int32(3)})
    raw = msgpack.unpackb(encoded, strict_map_key=False)
    assert raw["rgb"] == {
        b"__ndarray__": True,
        b"data": values.tobytes(),
        b"dtype": ">f4",
        b"shape": [3, 4],
    }
    assert raw["count"] == {b"__npgeneric__": True, b"data": 3, b"dtype": "<i4"}
    decoded = unpackb(encoded, strict_map_key=False)
    np.testing.assert_array_equal(decoded["rgb"], values)
    assert decoded["rgb"].dtype == values.dtype
    assert isinstance(decoded["count"], np.int32)
    with pytest.raises(ValueError, match="Unsupported dtype"):
        Packer().pack(np.array([object()], dtype=object))


def test_real_websocket_reset_action_and_disconnect_preserve_bridge_counts():
    async def run():
        bridge = EvaluatorStepBridge(timeout_s=3)
        resets = []
        policy = OfficialEvaluatorPolicy(
            bridge,
            task_id=40,
            reset_policy=lambda: resets.append(1),
        )
        server = create_official_policy_server(policy, port=0)
        async with serve(
            server._handler, "127.0.0.1", 0, process_request=_health_check
        ) as listener:
            port = listener.sockets[0].getsockname()[1]
            health = await asyncio.to_thread(
                urllib.request.urlopen, f"http://127.0.0.1:{port}/healthz"
            )
            with health:
                assert health.read() == b"OK\n"
            packer = Packer()
            actions = np.arange(46, dtype=np.float32).reshape(2, 23)
            async with connect(f"ws://127.0.0.1:{port}") as client:
                metadata = unpackb(await client.recv())
                assert metadata["task_id"] == 40
                assert metadata["single_session"] is True
                assert metadata["requires_reset"] is True
                assert "chunk_protocol" not in metadata
                await client.send(packer.pack({"reset": True}))
                await client.send(packer.pack({"frame": np.array([0])}))
                await _wait_for_first_observation(bridge)
                tool = asyncio.create_task(
                    asyncio.to_thread(bridge.chunk_step, actions)
                )
                first = unpackb(await client.recv())
                # If reset sent an acknowledgment, this would not be an action.
                np.testing.assert_array_equal(first["action"], actions[0])
                assert first["server_timing"]["total_ms"] >= 0
                async with connect(f"ws://127.0.0.1:{port}") as duplicate:
                    with pytest.raises(ConnectionClosedError) as error:
                        await duplicate.recv()
                    assert error.value.rcvd.code == 1013
                await client.send(packer.pack({"frame": np.array([1])}))
                np.testing.assert_array_equal(
                    unpackb(await client.recv())["action"], actions[1]
                )
            feedback = await asyncio.wait_for(tool, timeout=2)
            assert feedback[4]["executed_actions"] == 1
            assert feedback[4]["unconfirmed_actions"] == 1
            assert feedback[4]["stop_reason"] == "session_ended"
            assert resets == [1]

    asyncio.run(run())


def test_disconnect_during_action_wait_releases_thread_and_never_replays(tmp_path):
    async def run():
        bridge = EvaluatorStepBridge(timeout_s=30)
        policy = OfficialEvaluatorPolicy(bridge, task_id=0)
        server = create_official_policy_server(policy, port=0)
        async with serve(server._handler, "127.0.0.1", 0) as listener:
            port = listener.sockets[0].getsockname()[1]
            async with connect(f"ws://127.0.0.1:{port}") as client:
                await client.recv()
                await client.send(Packer().pack({"reset": True}))
                await client.send(Packer().pack({"frame": 1}))
                await _wait_for_first_observation(bridge)
                assert server._policy_busy.is_set()
            deadline = asyncio.get_running_loop().time() + 2
            while (
                server._policy_busy.is_set()
                and asyncio.get_running_loop().time() < deadline
            ):
                await asyncio.sleep(0.01)
            assert not server._policy_busy.is_set()
            assert bridge.stats()["issued"] == 0
            assert bridge.session_status()["ended"] is True

    asyncio.run(run())


def test_transport_imports_without_private_model_or_simulator_packages():
    code = """
import importlib.abc
import sys
class BlockHeavyImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'behavior_eval', 'rlinf', 'omnigibson', 'openpi'}:
            raise AssertionError('unexpected runtime dependency: ' + fullname)
sys.meta_path.insert(0, BlockHeavyImports())
from robots.behavior.eval.evaluator import WebsocketPolicyServer
server = WebsocketPolicyServer(object(), port=0)
assert server.metadata['requires_reset']
"""
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
