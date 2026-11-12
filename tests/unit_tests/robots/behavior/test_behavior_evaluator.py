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

"""Official WebSocket observations, per-action confirmation and session shutdown."""

from __future__ import annotations

import asyncio
import subprocess
import sys
import threading
import time
import urllib.request
from functools import partial

import msgpack
import numpy as np
import pytest
from websockets.asyncio.client import connect
from websockets.exceptions import ConnectionClosedError

from robots.behavior.eval.evaluator import (
    BridgeCancelled,
    BridgeSessionEnded,
    EvaluatorStepBridge,
    pack_data,
    serve_websocket,
    unpack_data,
)

Packer = partial(msgpack.Packer, default=pack_data)


unpackb = partial(msgpack.unpackb, object_hook=unpack_data)


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
    instance = EvaluatorStepBridge(
        simulation_timeout_s=2.0, action_timeout_s=2.0, chunk_timeout_s=2.0
    )
    try:
        yield instance
    finally:
        instance.stop()


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
    bridge.stop()
    with pytest.raises(BridgeCancelled):
        policy2.result()


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
    bridge.stop()
    with pytest.raises(BridgeCancelled, match="closed"):
        policy2.result()


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

    new_tool = ThreadCall(bridge.chunk_step, np.asarray(np.full(23, 20.0))[None])
    np.testing.assert_array_equal(new_policy.result(), np.full(23, 20.0))
    new_policy1 = ThreadCall(
        bridge.observe_and_get_action, {"episode": "new", "frame": 1}
    )
    assert new_tool.result()[4]["bridge_generation"] == 1
    assert bridge.stats()["issued"] == 2
    assert bridge.stats()["confirmed"] == 1
    assert bridge.stats()["unconfirmed"] == 1
    bridge.stop()
    with pytest.raises(BridgeCancelled):
        new_policy1.result()


@pytest.mark.parametrize("operation", ["get_observation", "chunk_step", "policy"])
def test_close_wakes_every_kind_of_waiter(bridge, operation):
    if operation == "get_observation":
        pending = ThreadCall(bridge.get_observation)
    elif operation == "chunk_step":
        pending = ThreadCall(bridge.chunk_step, np.zeros((1, 23)))
    else:
        pending = ThreadCall(bridge.observe_and_get_action, {"frame": 0})
    assert not pending.done.wait(0.03)

    bridge.stop()
    with pytest.raises(BridgeCancelled, match="closed"):
        pending.result()
    assert bridge.stats()["confirmed"] == 0
    assert bridge.stats()["closed"] is True


def test_session_end_returns_partial_chunk_without_confirming_last_action(bridge):
    obs0, obs1 = {"frame": 0}, {"frame": 1}
    first_policy = ThreadCall(bridge.observe_and_get_action, obs0)
    tool = ThreadCall(bridge.chunk_step, np.zeros((3, 23)))
    first_policy.result()
    second_policy = ThreadCall(bridge.observe_and_get_action, obs1)
    second_policy.result()

    bridge.stop("connection_closed", state="ended")
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
    late = bridge.chunk_step(np.asarray(np.ones(23))[None])
    assert late[4]["issued_actions"] == late[4]["executed_actions"] == 0
    with pytest.raises(BridgeSessionEnded, match="outcome unknown"):
        bridge.observe_and_get_action({"frame": 2})
    assert bridge.last_obs is obs1
    assert bridge.stats()["issued"] == 2


def test_new_observations_refresh_inactivity_timeout_during_slow_chunk():
    bridge = EvaluatorStepBridge(
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
        bridge.stop("connection_closed", state="ended")
        with pytest.raises(BridgeSessionEnded):
            final_policy.result()
    finally:
        bridge.stop()


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
        bridge.stop("connection_closed", state="ended")
        assert bridge.session_status()["ended"] is False
    finally:
        bridge.stop()


def test_connection_close_during_bridge_reset_leaves_session_ended():
    started, release = threading.Event(), threading.Event()

    class SlowBridge(EvaluatorStepBridge):
        def reset(self):
            started.set()
            assert release.wait(3)
            super().reset()

    bridge = SlowBridge(simulation_timeout_s=2, action_timeout_s=2, chunk_timeout_s=2)

    async def run():
        async with serve_websocket(bridge, port=0, task_id=40) as server:
            port = server.sockets[0].getsockname()[1]
            async with connect(f"ws://127.0.0.1:{port}") as client:
                await client.recv()
                await client.send(Packer().pack({"reset": True}))
                try:
                    assert await asyncio.to_thread(started.wait, 1)
                    await client.close()
                finally:
                    release.set()
        assert bridge.session_status()["generation"] == 1
        assert bridge.stats()["session_ended"] is True

    asyncio.run(run())


async def _wait_for_first_observation(bridge):
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
        bridge = EvaluatorStepBridge(
            simulation_timeout_s=3, action_timeout_s=3, chunk_timeout_s=3
        )
        async with serve_websocket(bridge, port=0, task_id=40) as server:
            port = server.sockets[0].getsockname()[1]
            health = await asyncio.to_thread(
                urllib.request.urlopen, f"http://127.0.0.1:{port}/healthz"
            )
            with health:
                assert health.read() == b"OK\n"
            actions = np.arange(46, dtype=np.float32).reshape(2, 23)
            async with connect(f"ws://127.0.0.1:{port}") as client:
                metadata = unpackb(await client.recv())
                assert metadata["task_id"] == 40
                assert metadata["single_session"] is True
                assert metadata["requires_reset"] is True
                assert "chunk_protocol" not in metadata
                await client.send(Packer().pack({"reset": True}))
                await client.send(Packer().pack({"frame": np.array([0])}))
                await _wait_for_first_observation(bridge)
                tool = asyncio.create_task(
                    asyncio.to_thread(bridge.chunk_step, actions)
                )
                first = unpackb(await client.recv())
                np.testing.assert_array_equal(first["action"], actions[0])
                assert first["server_timing"]["total_ms"] >= 0
                async with connect(f"ws://127.0.0.1:{port}") as duplicate:
                    with pytest.raises(ConnectionClosedError) as error:
                        await duplicate.recv()
                    assert error.value.rcvd.code == 1013
                await client.send(Packer().pack({"frame": np.array([1])}))
                np.testing.assert_array_equal(
                    unpackb(await client.recv())["action"], actions[1]
                )
            feedback = await asyncio.wait_for(tool, timeout=2)
            assert feedback[4]["executed_actions"] == 1
            assert feedback[4]["unconfirmed_actions"] == 1
            assert feedback[4]["stop_reason"] == "session_ended"
            assert bridge.session_status()["generation"] == 1

    asyncio.run(run())


def test_disconnect_during_action_wait_releases_thread_and_never_replays():
    bridge = EvaluatorStepBridge(
        simulation_timeout_s=30, action_timeout_s=30, chunk_timeout_s=30
    )

    async def run():
        async with serve_websocket(bridge, port=0, task_id=0) as server:
            port = server.sockets[0].getsockname()[1]
            async with connect(f"ws://127.0.0.1:{port}") as client:
                await client.recv()
                await client.send(Packer().pack({"reset": True}))
                await client.send(Packer().pack({"frame": 1}))
                await _wait_for_first_observation(bridge)
            async with asyncio.timeout(2):
                while not bridge.session_status()["ended"]:
                    await asyncio.sleep(0.001)
            async with connect(f"ws://127.0.0.1:{port}") as client:
                await client.recv()
                await client.send(Packer().pack({"reset": True}))
                with pytest.raises(ConnectionClosedError):
                    await client.recv()
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
from robots.behavior.eval.evaluator import EvaluatorStepBridge, serve_websocket
bridge = EvaluatorStepBridge(simulation_timeout_s=1, action_timeout_s=1, chunk_timeout_s=1)
context = serve_websocket(bridge, port=0, task_id=0)
assert hasattr(context, '__aenter__')
"""
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
