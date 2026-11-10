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

"""Env service RPC, numeric wire format and opt-in scenario wiring."""

from __future__ import annotations

import asyncio
import io
import json
import pickle
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from robots.behavior.env_client import (
    BehaviorEnvClient,
    decode_numeric_array,
    encode_numeric_array,
    env_runtime_contract,
)
from robots.behavior.env_server import BehaviorEnvFacade, _parse_args, _serve
from robots.behavior.eval.evaluator import EvaluatorStepBridge
from rpent.robots.components.env_client_base import BaseEnvClient
from rpent.utils.rpc import RpcError
from rpent.utils.rpc.socket_rpc import SocketRpcClient, SocketRpcServer
from tests.e2e_tests.behavior.scenario import BehaviorScenario, runtime_args
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


def service_args(*extra):
    return [
        "--task-id",
        "40",
        "--task-language",
        "make popcorn",
        "--vla-endpoint",
        "socket://127.0.0.1:24002",
        "--port",
        "24000",
        "--policy-port",
        "24001",
        *extra,
    ]


@pytest.mark.parametrize("task_id", [0, 7, 40, 99])
def test_supported_task_parses_without_starting_model_or_environment(task_id):
    args = _parse_args(service_args("--task-id", str(task_id)))
    assert args.task_id == task_id
    assert args.transport == "socket"
    assert args.simulation_timeout_s == 600.0
    assert args.action_timeout_s == 180.0
    assert args.chunk_timeout_s == 1200.0


@pytest.mark.parametrize("option", ["--port", "--policy-port"])
@pytest.mark.parametrize("value", ["0", "-1", "65536"])
def test_invalid_service_port_is_rejected_before_startup(option, value):
    with pytest.raises(SystemExit) as error:
        _parse_args(service_args(f"{option}={value}"))
    assert error.value.code == 2


@pytest.mark.parametrize(
    "extra",
    [
        ["--policy-port", "24000"],
        ["--task-id", "-1"],
        ["--task-id", "100"],
        ["--task-language", " "],
        ["--chunk-timeout-s", "599"],
    ],
)
def test_unsupported_task_or_conflicting_settings_fail_before_startup(extra):
    with pytest.raises(SystemExit) as error:
        _parse_args(service_args(*extra))
    assert error.value.code == 2


@pytest.mark.parametrize(
    "option",
    [
        "--simulation-timeout-s",
        "--action-timeout-s",
        "--chunk-timeout-s",
    ],
)
@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf", "-inf"])
def test_server_timeouts_must_be_positive_and_finite(option, value):
    with pytest.raises(SystemExit) as error:
        _parse_args(service_args(f"{option}={value}"))
    assert error.value.code == 2


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
    assert "--vla-endpoint" in result.stdout


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


@pytest.mark.parametrize(
    "timeout", [True, False, 0, -1, float("nan"), float("inf"), "600"]
)
def test_invalid_client_timeout_never_connects_to_backend(timeout):
    rpc = TimedLocalRpc(BehaviorEnvFacade(FakeEnvBackend(), task_id=40))
    with pytest.raises(ValueError, match="positive and finite"):
        BehaviorEnvClient(rpc, task_id=40, timeout_s=timeout)
    assert rpc.calls == []


@pytest.mark.parametrize("ended", [False, True, None])
def test_session_status_is_passed_through_without_actions_or_task_scoring(ended):
    backend = FakeEnvBackend()
    status = {
        "ended": ended,
        "reason": "connection_closed" if ended else None,
        "outcome": "unknown",
        "generation": 2,
        "unconfirmed_actions": 1,
    }
    backend.session_status = lambda: status
    rpc = LocalRpc(BehaviorEnvFacade(backend, task_id=40))
    client = BehaviorEnvClient(rpc, task_id=40)
    assert client.get_session_status() is status
    assert status["outcome"] == "unknown"
    assert "success" not in status
    assert client.last_obs is None
    assert client.last_info == {}
    assert backend.actions == []
    assert rpc.calls == ["env.get_env_meta", "env.get_session_status"]


def test_backend_without_session_status_reports_unknown_instead_of_success():
    backend = FakeEnvBackend()
    client = BehaviorEnvClient(
        LocalRpc(BehaviorEnvFacade(backend, task_id=0)), task_id=0
    )
    assert client.get_session_status() == {
        "ended": None,
        "reason": None,
        "outcome": "unknown",
    }
    assert backend.actions == []


@pytest.mark.parametrize(
    "status",
    [
        {},
        [],
        {"ended": 0},
        {"ended": 1},
        {"ended": "false"},
    ],
)
def test_malformed_session_status_is_rejected_instead_of_coerced(status):
    backend = FakeEnvBackend()
    backend.session_status = lambda: status
    client = BehaviorEnvClient(
        LocalRpc(BehaviorEnvFacade(backend, task_id=40)), task_id=40
    )
    with pytest.raises(TypeError, match="ended flag"):
        client.get_session_status()


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


def test_empty_chunk_without_session_end_does_not_replace_cached_state():
    backend = FakeEnvBackend()
    client = BehaviorEnvClient(
        LocalRpc(BehaviorEnvFacade(backend, task_id=40)), task_id=40
    )
    original_observation = client.get_observation()
    original_info = client.last_info
    backend.chunk_step = lambda *args, **kwargs: ([], None, None, None, {})
    with pytest.raises(ValueError, match="no observation frames"):
        client.chunk_step(np.zeros((1, 23)), return_all_frames=True)
    assert client.last_obs is original_observation
    assert client.last_info is original_info


def test_shutdown_rpc_cancels_waiting_chunk_before_taking_dispatch_lock():
    bridge = EvaluatorStepBridge(timeout_s=5)
    facade = BehaviorEnvFacade(bridge, task_id=40, on_shutdown=bridge.close)
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
        bridge.close()
        rpc.close()
        server.shutdown()
        server.server_close()
        server_thread.join(timeout=2)


def test_facade_without_owner_callback_does_not_close_borrowed_bridge():
    bridge = EvaluatorStepBridge(timeout_s=2)
    facade = BehaviorEnvFacade(bridge, task_id=40)
    try:
        assert facade._dispatch("shutdown", (), {}) == {"ok": True}
        assert facade._shutdown_event.is_set()
        assert bridge.stats()["closed"] is False
        assert bridge.session_status()["ended"] is False
    finally:
        bridge.close()


def test_service_reports_rpc_cleanup_failure_that_occurs_after_shutdown(
    monkeypatch,
):
    cleanup_allowed = threading.Event()
    server_started = threading.Event()
    closed_clients = []
    cancelled_server = []
    signals = []

    class BridgeWithCleanupGate(EvaluatorStepBridge):
        def close(self):
            super().close()
            cleanup_allowed.set()

    class FailingCleanupFacade(BehaviorEnvFacade):
        def serve(self, **kwargs):
            assert server_started.wait(3)
            self._shutdown_event.set()
            # Delay this failure until _serve has entered its own finally.
            assert cleanup_allowed.wait(3)
            raise RuntimeError("RPC cleanup failed")

    class FakeRpc:
        def close(self):
            closed_clients.append(True)

    class FakePolicyServer:
        async def run(self):
            try:
                server_started.set()
                await asyncio.Future()
            finally:
                cancelled_server.append(True)

    monkeypatch.setattr(
        "robots.behavior.eval.evaluator.EvaluatorStepBridge", BridgeWithCleanupGate
    )
    monkeypatch.setattr(
        "robots.behavior.env_server.BehaviorEnvFacade", FailingCleanupFacade
    )
    monkeypatch.setattr("rpent.utils.rpc.make_rpc_client", lambda endpoint: FakeRpc())
    monkeypatch.setattr(
        "robots.behavior.eval.evaluator.create_official_policy_server",
        lambda *args, **kwargs: FakePolicyServer(),
    )
    args = _parse_args(service_args())

    async def exercise():
        loop = asyncio.get_running_loop()
        monkeypatch.setattr(
            loop,
            "add_signal_handler",
            lambda signum, callback: signals.append(("add", signum)),
        )
        monkeypatch.setattr(
            loop,
            "remove_signal_handler",
            lambda signum: signals.append(("remove", signum)),
        )
        with pytest.raises(RuntimeError, match="Env RPC service failed") as error:
            await _serve(args)
        assert str(error.value.__cause__) == "RPC cleanup failed"

    asyncio.run(exercise())
    assert closed_clients == [True]
    assert cancelled_server == [True]
    assert [signum for action, signum in signals if action == "add"] == [
        signum for action, signum in signals if action == "remove"
    ]


class NoGlobalsUnpickler(pickle.Unpickler):
    """A wire request may contain built-ins, but no imported pickle globals."""

    def find_class(self, module, name):
        raise AssertionError(f"request depends on pickle global {module}.{name}")


def builtin_roundtrip(value):
    return NoGlobalsUnpickler(io.BytesIO(pickle.dumps(value, protocol=4))).load()


class WireRpc:
    def __init__(self, facade):
        self.facade = facade
        self.requests = []

    def call(self, method, args=(), kwargs=None, timeout_s=None):
        request = builtin_roundtrip((method, args, kwargs or {}))
        self.requests.append(request)
        return self.facade._dispatch(*request)


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
        backend, task_id=0, motion_planner=SimpleNamespace(plan_arm_path=plan)
    )
    rpc = WireRpc(facade)
    client = BehaviorEnvClient(rpc, task_id=0)
    return client, facade, rpc, calls, planned, info


@pytest.mark.parametrize(
    "dtype",
    [np.float16, np.float32, np.float64, np.longdouble, np.int32, np.uint64, ">f8"],
)
@pytest.mark.parametrize("layout", ["c", "fortran", "slice"])
def test_numeric_wire_preserves_dtype_shape_and_values_without_pickle_globals(
    dtype, layout
):
    array = np.arange(92).astype(dtype).reshape(4, 23)
    if layout == "fortran":
        array = np.asfortranarray(array)
    elif layout == "slice":
        array = array[::2, ::-1]
    if array.dtype.kind == "f":
        array[0, 0] = -0.0
    restored = decode_numeric_array(builtin_roundtrip(encode_numeric_array(array)))
    assert restored.dtype == array.dtype
    assert restored.shape == array.shape
    np.testing.assert_array_equal(restored, array)
    # float128 padding is not a numeric value and can change during array copies.
    if array.dtype.itemsize <= 8:
        assert restored.tobytes(order="C") == array.tobytes(order="C")


@pytest.mark.parametrize("method,shape", [("step", (23,)), ("chunk_step", (32, 23))])
@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int64])
def test_action_client_and_facade_preserve_payload_and_unknown_feedback(
    method, shape, dtype
):
    client, _, rpc, calls, _, info = make_env()
    actions = np.arange(np.prod(shape), dtype=dtype).reshape(shape)
    feedback = getattr(client, method)(actions)
    assert len(calls) == 1
    received = calls[0][0]
    assert received.dtype == actions.dtype
    assert received.shape == actions.shape
    assert received.tobytes() == actions.tobytes()
    assert [request[0] for request in rpc.requests] == [
        "env.get_env_meta",
        f"env.{method}",
    ]
    assert feedback[1:4] == (None, None, None)
    assert feedback[4] is client.last_info is info
    assert feedback[4]["confirmed_actions"] == 0
    assert feedback[4]["unconfirmed_actions"] == 1
    assert "success" not in feedback[4]


def test_chunk_preserves_frame_option_and_observation_token():
    client, _, _, calls, _, _ = make_env()
    token = {"generation": 1, "observation_index": 0}
    client.chunk_step(
        np.zeros((1, 23), dtype=np.float32),
        return_all_frames=True,
        expected_observation_token=token,
    )
    assert calls[0][1] == {
        "return_all_frames": True,
        "expected_observation_token": token,
    }


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_pose_client_uses_numeric_wire_and_retains_existing_float64_planner_contract(
    dtype,
):
    client, _, rpc, calls, planned, _ = make_env()
    pose = np.array([0.1, 0.2, 0.3, 0, 0, 0, 1], dtype=dtype)
    token = {"generation": 1, "observation_index": 0}
    result = client.plan_arm_path("left", pose, expected_observation_token=token)
    encoded_pose = rpc.requests[-1][2]["target_pose"]
    wire_pose = decode_numeric_array(encoded_pose)
    assert wire_pose.dtype == pose.dtype
    assert wire_pose.tobytes() == pose.tobytes()
    assert len(planned) == 1
    assert planned[0]["target_pose"].dtype == np.float64
    np.testing.assert_array_equal(planned[0]["target_pose"], pose.astype(np.float64))
    assert planned[0]["motion_context"] == {"fixture": "public"}
    assert result["observation_token"] == token
    assert calls == []


def raw_payload(array):
    stream = io.BytesIO()
    np.save(stream, array, allow_pickle=True)
    return {"_behavior_numeric_array": "npy_v1", "payload": stream.getvalue()}


def damaged_payloads():
    valid = encode_numeric_array(np.zeros(23, dtype=np.float32))
    return [
        {},
        {**valid, "_behavior_numeric_array": "npy_v2"},
        {**valid, "extra": 0},
        {"_behavior_numeric_array": "npy_v1"},
        {**valid, "payload": bytearray(valid["payload"])},
        {**valid, "payload": b"not an NPY file"},
        {**valid, "payload": valid["payload"][:12]},
        {**valid, "payload": valid["payload"][:-1]},
        {**valid, "payload": valid["payload"] + b"trailing"},
        raw_payload(np.array([object()], dtype=object)),
        raw_payload(np.zeros(23, dtype=np.complex64)),
    ]


@pytest.mark.parametrize("payload", damaged_payloads())
@pytest.mark.parametrize("method", ["step", "chunk_step", "plan_arm_path"])
def test_malformed_or_unknown_payloads_never_reach_execution_or_planner(
    payload, method
):
    _, facade, _, calls, planned, _ = make_env()
    with pytest.raises(ValueError, match="BEHAVIOR numeric wire"):
        if method == "plan_arm_path":
            facade.plan_arm_path("left", payload)
        else:
            getattr(facade, method)(payload)
    assert calls == planned == []


def test_corrupt_header_size_is_rejected_before_array_allocation(monkeypatch):
    stream = io.BytesIO()
    np.lib.format.write_array_header_1_0(
        stream,
        {
            "descr": "<f8",
            "fortran_order": False,
            "shape": (2**60, 23),
        },
    )

    def forbidden_load(*args, **kwargs):
        raise AssertionError("invalid size reached array allocation")

    monkeypatch.setattr(np, "load", forbidden_load)
    with pytest.raises(ValueError, match="BEHAVIOR numeric wire"):
        decode_numeric_array(
            {"_behavior_numeric_array": "npy_v1", "payload": stream.getvalue()}
        )


@pytest.mark.parametrize(
    "array", [np.zeros(22), np.full(23, np.nan), np.full(23, np.inf)]
)
def test_encoded_action_still_obeys_original_shape_and_finiteness_validation(array):
    _, facade, _, calls, _, _ = make_env()
    with pytest.raises(ValueError, match="BEHAVIOR native actions"):
        facade.step(encode_numeric_array(array))
    assert calls == []


@pytest.mark.parametrize("pose", [np.zeros(6), np.zeros(7), np.full(7, np.nan)])
def test_encoded_pose_still_obeys_original_validation(pose):
    _, facade, _, calls, planned, _ = make_env()
    with pytest.raises(ValueError, match="target_pose"):
        facade.plan_arm_path("left", encode_numeric_array(pose))
    assert calls == planned == []


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


@pytest.mark.parametrize("kind", ["array", "list"])
def test_legacy_direct_facade_requests_remain_usable(kind):
    _, facade, _, calls, planned, _ = make_env()
    action = np.arange(23, dtype=np.float32)
    pose = np.array([0, 0, 0, 0, 0, 0, 1], dtype=np.float64)
    facade.step(action.tolist() if kind == "list" else action)
    facade.plan_arm_path("right", pose.tolist() if kind == "list" else pose)
    np.testing.assert_array_equal(calls[0][0], action)
    np.testing.assert_array_equal(planned[0]["target_pose"], pose)


_CAMERAS = (
    "robot_r1::robot_r1:zed_link:Camera:0::rgb",
    "robot_r1::robot_r1:left_realsense_link:Camera:0::rgb",
    "robot_r1::robot_r1:right_realsense_link:Camera:0::rgb",
)


_SETTINGS = {
    "RPENT_BEHAVIOR_TASK_ID": "7",
    "RPENT_BEHAVIOR_TASK_LANGUAGE": "Pick up the toys.",
    "RPENT_BEHAVIOR_ENV_ENDPOINT": "http://localhost:21000",
    "RPENT_BEHAVIOR_VLA_ENDPOINT": "http://localhost:21001",
}


def _configure(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    for name, value in _SETTINGS.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("RPENT_E2E_OUTPUT_DIR", str(tmp_path / "run"))


class _Transport:
    """In-memory responses only; reject every unrecognized service operation."""

    def __init__(self, *, task_id: int = 7, close_error: bool = False) -> None:
        self.task_id = task_id
        self.close_error = close_error
        self.closed = 0
        self.calls: list[str] = []
        self.chunks: list[np.ndarray] = []
        self.inputs: list[tuple[dict[str, Any], dict[str, Any] | None]] = []
        self.prediction = np.arange(64 * 23, dtype=np.float32).reshape(64, 23)

    def _observation(self) -> dict[str, Any]:
        observation = {
            camera: np.full((1, 6, 8, 3), index + len(self.chunks), np.uint8)
            for index, camera in enumerate(_CAMERAS)
        }
        proprio = np.zeros(61, dtype=np.float32)
        proprio[[23, 48]] = 1
        observation["robot_r1::proprio"] = proprio[None]
        observation["task_id"] = np.array([self.task_id])
        return observation

    def call(self, method: str, args=(), kwargs=None, timeout_s=None) -> Any:
        self.calls.append(method)
        if method == "env.get_env_meta":
            return env_runtime_contract(self.task_id)
        if method == "env.get_task_language":
            return _SETTINGS["RPENT_BEHAVIOR_TASK_LANGUAGE"]
        if method == "env.get_observation":
            return self._observation()
        if method == "env.get_session_status":
            return {"ended": False}
        if method == "vla.predict":
            self.inputs.append(args)
            return self.prediction[None].copy()
        if method == "env.chunk_step":
            actions = decode_numeric_array(args[0])
            self.chunks.append(actions.copy())
            count = len(actions)
            return (
                self._observation(),
                None,
                None,
                None,
                {
                    "executed_actions": count,
                    "issued_actions": count,
                    "confirmed_actions": count,
                    "unconfirmed_actions": 0,
                    "observation_after_last_issued_action": True,
                    "stop_reason": "chunk_completed",
                },
            )
        raise AssertionError(f"unexpected service operation: {method}")

    def close(self) -> None:
        self.closed += 1
        if self.close_error:
            raise RuntimeError("local close failed")


@pytest.mark.parametrize("missing", [*_SETTINGS, "RPENT_E2E_OUTPUT_DIR"])
def test_explicit_runtime_settings_required_before_output_creation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, missing: str
) -> None:
    _configure(monkeypatch, tmp_path)
    monkeypatch.delenv(missing)
    with pytest.raises(RuntimeError, match=missing):
        runtime_args()
    assert not (tmp_path / "run").exists()


def test_one_toolkit_chunk_uses_actual_prediction_and_only_closes_transports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _configure(monkeypatch, tmp_path)
    monkeypatch.setattr("rpent.utils.templates.get_output_dir", lambda: tmp_path)
    env, vla = _Transport(), _Transport()
    transports = {
        _SETTINGS["RPENT_BEHAVIOR_ENV_ENDPOINT"]: env,
        _SETTINGS["RPENT_BEHAVIOR_VLA_ENDPOINT"]: vla,
    }
    monkeypatch.setattr(
        "tests.e2e_tests.behavior.scenario.make_rpc_client", transports.__getitem__
    )
    scenario = BehaviorScenario(runtime_args())
    try:
        _, observation = scenario.environment
        assert observation["proprio"]["shape"] == [61]
        assert len(observation["cameras"]) == 3
        assert scenario.vla["prediction"]["shape"] == [64, 23]
        assert env.chunks == []
        result = scenario.policy_chain
        assert result["official_task_success"] is None
        assert result["cli_run"] is result["vlm_planner_assessed"] is False
        assert result["execution"]["confirmed_actions"] == 32
        assert result["finish"]["status"] == "blocked"
        assert len(env.chunks) == 1
        assert len(vla.inputs) == 2  # Component request, then one tool request.
        np.testing.assert_array_equal(env.chunks[0], vla.prediction[:32])
        assert all(
            obs["task_descriptions"] == ["Pick up the toys."] for obs, _ in vla.inputs
        )
        for name in ("chain-prediction.npy", "chain-submitted-actions.npy"):
            assert (scenario.output_dir / name).is_file()
        saved = json.loads((scenario.output_dir / "policy-chain.json").read_text())
        assert saved == result
    finally:
        scenario.close()
    assert env.closed == vla.closed == 1
    scenario.close()
    assert env.closed == vla.closed == 1
    assert not any(
        "reset" in call or "shutdown" in call for call in env.calls + vla.calls
    )


def test_task_mismatch_preserves_transport_for_local_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _configure(monkeypatch, tmp_path)
    env = _Transport(task_id=40)
    monkeypatch.setattr(
        "tests.e2e_tests.behavior.scenario.make_rpc_client", lambda _: env
    )
    scenario = BehaviorScenario(runtime_args())
    try:
        with pytest.raises(AssertionError, match="env_meta mismatch"):
            _ = scenario.environment
    finally:
        scenario.close()
    assert env.calls == ["env.get_env_meta"]
    assert env.closed == 1


def test_close_failure_still_closes_other_local_connections(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _configure(monkeypatch, tmp_path)
    env, vla = _Transport(), _Transport(close_error=True)
    transports = iter((env, vla))
    monkeypatch.setattr(
        "tests.e2e_tests.behavior.scenario.make_rpc_client", lambda _: next(transports)
    )
    scenario = BehaviorScenario(runtime_args())
    _ = scenario.env, scenario.model
    with pytest.raises(RuntimeError, match="local close failed"):
        scenario.close()
    assert env.closed == vla.closed == 1
    scenario.close()
    assert env.closed == vla.closed == 1
