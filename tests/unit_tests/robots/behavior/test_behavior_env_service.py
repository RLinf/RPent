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

"""Env RPC preserves action/pose payloads, execution feedback and clean shutdown."""

from __future__ import annotations

import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import numpy as np
import pytest

from robots.behavior.env_client import BehaviorEnvClient
from robots.behavior.env_server import BehaviorEnvFacade
from robots.behavior.eval.evaluator import EvaluatorStepBridge
from rpent.robots.components.env_client_base import BaseEnvClient
from rpent.utils.rpc import RpcError
from rpent.utils.rpc.http_rpc import HttpRpcClient, HttpRpcServer
from rpent.utils.rpc.socket_rpc import SocketRpcClient, SocketRpcServer
from tests.unit_tests.robots.behavior.conftest import (
    FakeEnvBackend,
    LocalRpc,
)


class TimedLocalRpc(LocalRpc):
    def __init__(self, facade):
        super().__init__(facade)
        self.timeouts = []

    def call(self, method, args=(), kwargs=None, timeout_s=None):
        self.timeouts.append((method, timeout_s))
        return super().call(method, args, kwargs, timeout_s)


def test_service_help_does_not_import_optional_gpu_or_simulator_dependencies():
    code = """
import importlib.abc
import sys
class NoRuntimeImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'behavior_eval', 'rlinf', 'omnigibson'}:
            raise AssertionError('optional import during --help: ' + fullname)
sys.meta_path.insert(0, NoRuntimeImports())
from robots.behavior.env_server import main
main(['--help'])
"""
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert "--policy-port" in result.stdout
    assert "--am-endpoint" not in result.stdout
    assert "--vla-endpoint" not in result.stdout


def test_timeout_overrides_apply_to_actual_calls_without_changing_other_clients():
    behavior_defaults = dict(BehaviorEnvClient._TIMEOUT_S)
    base_defaults = dict(BaseEnvClient._TIMEOUT_S)
    clients = []
    for timeout in (1500.0, 75.0, None):
        rpc = TimedLocalRpc(BehaviorEnvFacade(FakeEnvBackend(), task_id=40))
        client = BehaviorEnvClient(rpc, task_id=40, timeout_s=timeout)
        clients.append((rpc, client, timeout))

    for rpc, client, timeout in clients:
        client.get_observation()
        client.step(np.zeros(23))
        client.chunk_step(np.zeros((2, 23)))
        assert rpc.timeouts == [
            ("env.get_env_meta", timeout or behavior_defaults["default"]),
            ("env.get_observation", timeout or behavior_defaults["default"]),
            ("env.step", timeout or behavior_defaults["env.step"]),
            ("env.chunk_step", timeout or behavior_defaults["env.chunk_step"]),
        ]
    assert BehaviorEnvClient._TIMEOUT_S == behavior_defaults
    assert BaseEnvClient._TIMEOUT_S == base_defaults


@pytest.mark.parametrize("prime_observation", [False, True])
def test_final_chunk_with_no_confirmed_frame_preserves_cached_observation(
    prime_observation,
):
    backend = FakeEnvBackend()
    client = BehaviorEnvClient(
        LocalRpc(BehaviorEnvFacade(backend, task_id=40)), task_id=40
    )
    if prime_observation:
        client.get_observation()
    previous = client.last_obs
    info = {
        "stop_reason": "session_ended",
        "executed_actions": 0,
        "issued_actions": 1,
        "unconfirmed_actions": 1,
        "outcome": "unknown",
    }
    frames = []

    def final_chunk(actions, *, return_all_frames=False):
        assert return_all_frames is True
        return frames, None, None, None, info

    backend.chunk_step = final_chunk
    result = client.chunk_step(np.zeros((1, 23)), return_all_frames=True)
    assert result[0] is frames
    assert result[1:4] == (None, None, None)
    assert result[4] is info
    assert client.last_obs is previous
    assert client.last_info is info
    assert info["unconfirmed_actions"] == 1
    assert "success" not in info


def test_shutdown_rpc_cancels_waiting_chunk_before_taking_dispatch_lock():
    bridge = EvaluatorStepBridge(
        simulation_timeout_s=5, action_timeout_s=5, chunk_timeout_s=5
    )
    facade = BehaviorEnvFacade(bridge, task_id=40, on_shutdown=bridge.stop)
    server = SocketRpcServer(("127.0.0.1", 0), facade._dispatch)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    rpc = SocketRpcClient(*server.server_address)
    try:
        client = BehaviorEnvClient(rpc, task_id=40, timeout_s=5)
        with ThreadPoolExecutor(max_workers=2) as threads:
            policy = threads.submit(bridge.observe_and_get_action, {"frame": 0})
            bridge.get_observation()
            chunk = threads.submit(client.chunk_step, np.zeros((2, 23)))
            policy.result(timeout=2)
            assert not chunk.done()

            assert rpc.call("shutdown", timeout_s=1) == {"ok": True}
            with pytest.raises(RpcError, match="closed"):
                chunk.result(timeout=1)
        assert facade._shutdown_event.is_set()
        assert bridge.stats()["issued"] == bridge.stats()["unconfirmed"] == 1
        assert bridge.stats()["confirmed"] == 0
        assert bridge.session_status()["ended"] is False
    finally:
        bridge.stop()
        rpc.close()
        server.shutdown()
        server.server_close()
        server_thread.join(timeout=2)


def make_env():
    calls = []
    planned = []
    observation = {"fixture_frame": 0}
    info = {
        "issued_actions": 1,
        "confirmed_actions": 0,
        "unconfirmed_actions": 1,
        "executed_actions": 0,
        "stop_reason": "session_ended",
    }

    def execute(value, **kwargs):
        calls.append((value, kwargs))
        return observation, None, None, None, info

    def plan(**kwargs):
        planned.append(kwargs)
        return {"status": "Success", "position": [[0.0] * 7]}

    backend = SimpleNamespace(
        step=execute,
        chunk_step=execute,
        get_observation=lambda: observation,
        session_status=lambda: {"ended": False},
        get_motion_snapshot=lambda: {
            "observation": observation,
            "observation_token": {"generation": 1, "observation_index": 0},
            "motion_context": {"fixture": "public"},
        },
    )
    facade = BehaviorEnvFacade(
        backend,
        task_id=0,
        motion_planner=SimpleNamespace(plan_arm_path=plan),
        motion_model=lambda: SimpleNamespace(
            joint_limits=lambda arm: np.tile([-2.0, 2.0], (7, 1))
        ),
    )
    rpc = LocalRpc(facade)
    client = BehaviorEnvClient(rpc, task_id=0)
    return client, facade, rpc, calls, planned, info


def test_joint_limits_are_lazy_and_cached_without_local_assets(monkeypatch):
    monkeypatch.delenv("OMNIGIBSON_DATA_PATH", raising=False)
    client, _, rpc, calls, planned, _ = make_env()
    assert rpc.calls == ["env.get_env_meta"]
    first = client.arm_joint_limits
    assert client.arm_joint_limits is first
    assert rpc.calls == ["env.get_env_meta", "env.get_arm_joint_limits"]
    assert not calls and not planned
    with pytest.raises(ValueError):
        first["left"][0, 0] = 0


@pytest.fixture(params=["socket", "http"])
def transported_env(request):
    _, facade, _, calls, planned, info = make_env()
    server_type = SocketRpcServer if request.param == "socket" else HttpRpcServer
    server = server_type(("127.0.0.1", 0), facade._dispatch)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    host, port = server.server_address
    rpc = (
        SocketRpcClient(host, port)
        if request.param == "socket"
        else HttpRpcClient(f"http://{host}:{port}")
    )
    try:
        yield BehaviorEnvClient(rpc, task_id=0), calls, planned, info
    finally:
        rpc.close()
        server.shutdown()
        server.server_close()
        worker.join(timeout=2)


def test_joint_limits_cross_existing_transports_without_actions(transported_env):
    client, calls, planned, _ = transported_env
    for arm in ("left", "right"):
        np.testing.assert_array_equal(
            client.arm_joint_limits[arm], np.tile([-2, 2], (7, 1))
        )
    assert not calls and not planned


@pytest.mark.parametrize("method,shape", [("step", (23,)), ("chunk_step", (32, 23))])
@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int64])
def test_action_client_and_facade_preserve_payload_and_unknown_feedback(
    transported_env, method, shape, dtype
):
    client, calls, _, info = transported_env
    actions = np.arange(np.prod(shape), dtype=dtype).reshape(shape)
    if actions.dtype.kind == "f":
        actions += 0.123456789012345
    feedback = getattr(client, method)(actions)
    assert len(calls) == 1
    received = calls[0][0]
    assert received.dtype == actions.dtype
    expected = actions[None] if method == "step" else actions
    assert received.shape == expected.shape
    assert received.tobytes() == expected.tobytes()
    assert feedback[1:4] == (None, None, None)
    assert feedback[4] is client.last_info
    assert feedback[4] == info
    assert feedback[4]["confirmed_actions"] == 0
    assert feedback[4]["unconfirmed_actions"] == 1
    assert "success" not in feedback[4]


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_pose_client_uses_numeric_wire_and_retains_existing_float64_planner_contract(
    transported_env,
    dtype,
):
    client, calls, planned, _ = transported_env
    pose = np.array([0.123456789012345, 0.2, 0.3, 0, 0, 0, 1], dtype=dtype)
    token = {"generation": 1, "observation_index": 0}
    result = client.plan_arm_path("left", pose, expected_observation_token=token)
    assert len(planned) == 1
    assert planned[0]["target_pose"].dtype == np.float64
    np.testing.assert_array_equal(planned[0]["target_pose"], pose.astype(np.float64))
    assert planned[0]["motion_context"] == {"fixture": "public"}
    assert result["observation_token"] == token
    assert calls == []


def test_non_numeric_actions_are_rejected_at_the_rpc_boundary(transported_env):
    client, calls, _, _ = transported_env
    with pytest.raises(RpcError, match="finite real numbers"):
        client._client.call("env.chunk_step", args=(np.zeros((1, 23), dtype=bool),))
    assert calls == []


@pytest.mark.parametrize("method", ["step", "chunk_step", "plan_arm_path"])
def test_transport_failure_is_not_retried(method):
    client, _, rpc, calls, planned, _ = make_env()
    attempts = []

    def broken_call(*args, **kwargs):
        attempts.append((args, kwargs))
        raise ConnectionError("response unavailable")

    rpc.call = broken_call
    with pytest.raises(ConnectionError, match="response unavailable"):
        if method == "plan_arm_path":
            client.plan_arm_path("left", [0, 0, 0, 0, 0, 0, 1])
        else:
            getattr(client, method)(np.zeros(23 if method == "step" else (1, 23)))
    assert len(attempts) == 1
    assert calls == planned == []


@pytest.mark.parametrize("actions", [np.zeros(22), np.full(23, np.nan)])
def test_invalid_action_payload_does_not_reach_the_backend(actions):
    client, _, _, calls, planned, _ = make_env()
    with pytest.raises(ValueError):
        client.step(actions)
    assert calls == planned == []
