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

"""Cross-host tool declarations, exact artifacts and uncertain-action cleanup."""

from __future__ import annotations

import argparse
import base64
import contextlib
import hashlib
import json
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from robots.rl import remote, tools
from robots.rl.remote_manager import BackendManager, validate_request
from robots.rl.units import CENTIMETERS
from rpent.dashboard.events import NullDashboardEventSink
from rpent.memory import MemoryManager
from rpent.utils.rpc import RpcError, RpcFacade
from rpent.utils.rpc.http_rpc import HttpRpcClient, HttpRpcServer
from tests.unit_tests.robots.rl.test_libero_rl_contracts import (
    FakeEnv,
    execute_with_si_values,
    make_toolkit,
)


@pytest.mark.parametrize("environment", ["libero", "robotwin", "robocasa", "robodojo"])
def test_episode_runtime_request_is_accepted_by_its_remote_manager(
    tmp_path, environment
):
    from robots.rl.robot_spec import get_robot_spec

    parser = argparse.ArgumentParser()
    spec = get_robot_spec()
    spec.add_cli_args(parser, False)
    task_args = (
        ["--suite", "libero_object", "--task", "9", "--libero-type", "pro"]
        if environment == "libero"
        else ["--task-name", "native_task"]
    )
    args = parser.parse_args(
        [
            "--simulator",
            environment,
            "--remote-toolkit-endpoint",
            "http://localhost:28490",
            "--cuda-device",
            "0",
            "--seed",
            "1",
            *task_args,
        ]
    )
    args.output_dir = tmp_path
    _, runtime = spec.init_runtime(
        args, tmp_path, NullDashboardEventSink(), {"env", "sam3"}
    )
    request = runtime["remote_request"]
    validate_request(request, {0})
    assert ("layout" in request["task"]) == (environment == "robodojo")


class Bridge(RpcFacade):
    """Exercise the production HTTP boundary with an existing local toolkit."""

    def __init__(self, worker):
        super().__init__()
        self.worker = worker
        self.calls = []
        self.released = False
        self.mismatch = None
        self.fail_after_action = False
        self.fail_cancel = False
        self._rpc.update(
            {
                "backend.open": self.open,
                "backend.call": self.call,
                "backend.renew": lambda session: {},
                "backend.release": self.release,
            }
        )
        self._readonly_methods.update(self._rpc)

    def open(self, request, owner_id=None):
        self.owner_id = owner_id
        result = {
            "session": "owned-session",
            "owner_id": owner_id,
            **self.worker.describe(),
        }
        if self.mismatch is not None:
            result[self.mismatch] = "different"
        return result

    def call(self, session, method, arguments):
        self.calls.append(method)
        if self.fail_cancel and method == "rl.cancel":
            raise RuntimeError("connection lost during cancellation")
        result = self.worker._dispatch(method, (), arguments)
        if self.fail_after_action and method == "rl.execute":
            raise RuntimeError("response lost after action")
        return result

    def release(self, session):
        self.released = True
        self.worker.close()
        return {}


@pytest.fixture
def connected(tmp_path, monkeypatch, request):
    # The worker is an in-process fake, including its Linux host identity.
    # Keep the HTTP contract test runnable on non-Linux development machines.
    read_text = Path.read_text

    def host_identity(path, *args, **kwargs):
        if path == Path("/proc/sys/kernel/random/boot_id"):
            return "d831e11e-b157-46fb-a341-b276fc7d3e46\n"
        return read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", host_identity)
    monkeypatch.setattr(tools, "IMAGE_SIZE", 32)
    from rpent.utils import templates

    monkeypatch.setattr(
        templates, "default_variables", lambda: {"output_dir": "/offline/output"}
    )
    options = dict(getattr(request, "param", {}))
    if options.get("calibrated", False):
        from tests.unit_tests.robots.rl.test_coordinates import CalibratedEnv

        env = CalibratedEnv([1, -2, 0.5], 0.4, success_below_z=1.45)
    else:
        env = FakeEnv(success_below_z=0.95)
    options.pop("calibrated", None)
    toolkit = make_toolkit(tmp_path / "backend", "adaptive", env=env, **options)
    release = contextlib.ExitStack()
    release.callback(toolkit.close)
    task = {
        "environment": "libero",
        "suite": "libero_object",
        "task": 0,
        "seed": 1,
        **options,
    }
    request = {"task": task, "gpu": 0, "max_episode_steps": 10000}
    config = {**request, "output_dir": str(tmp_path / "backend/run")}
    worker = remote.ToolkitWorker(toolkit, release, config)
    bridge = Bridge(worker)
    server = HttpRpcServer(("127.0.0.1", 0), bridge._dispatch)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    client_config = SimpleNamespace(output_dir=tmp_path / "frontend")

    def connect(owner_id=None):
        return remote.RemoteToolkit(
            owner_id=owner_id,
            endpoint=f"http://127.0.0.1:{server.server_address[1]}",
            request=request,
            config=client_config,
            dashboard_events=NullDashboardEventSink(),
            memory=MemoryManager(tmp_path / "frontend-memory"),
        )

    try:
        yield SimpleNamespace(
            connect=connect,
            worker=worker,
            bridge=bridge,
            toolkit=toolkit,
            env=env,
            output=client_config.output_dir,
        )
    finally:
        release.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)
        assert not thread.is_alive()


def test_remote_preserves_native_tools_observations_and_original_bytes(connected):
    client = connected.connect()
    assert remote.declarations(client) == remote.declarations(connected.toolkit)
    assert client.initial_observation().data["state"]["step"] == 0
    from tests.unit_tests.robots.rl.test_libero_rl_contracts import execute_item

    invalid = client.execute_tool(
        "execute_actions", {"actions": [{"action": "move_to", "xyz": "bad"}]}
    )
    assert invalid.is_error and connected.bridge.calls == ["rl.observe"]
    moved = execute_item(client, "move_to", {"xyz": [0.1, -0.2, 0.9]})
    # Native success terminates this fixture before the requested pose settles.
    assert moved.data["status"] == "partial"
    assert moved.data["reason"] == "terminated"
    assert client.solved() is True
    viewed = execute_with_si_values(client, "view_env", {})
    assert len(viewed.images) == 2
    assert client.state.records() == connected.toolkit.state.records()
    client.close()
    client.close()
    assert connected.bridge.released
    backend = connected.worker.exchange.output
    for path in backend.rglob("*"):
        if path.is_file():
            relative = path.relative_to(backend)
            assert (connected.output / relative).read_bytes() == path.read_bytes()
    assert (connected.output / "episode.mp4").is_file()
    assert (
        json.loads((connected.output / "rollout.json").read_text())["images_sent"] == 2
    )


@pytest.mark.parametrize(
    "connected",
    [
        {
            "max_actions_per_call": 3,
            "reasoning_history": 3,
            "image_history": 1,
        }
    ],
    indirect=True,
)
def test_remote_batch_has_identical_contract_and_never_executes_invalid_list(connected):
    client = connected.connect()
    try:
        assert remote.declarations(client) == remote.declarations(connected.toolkit)
        assert {tool.name for tool in client.list_tools()} == {
            "execute_actions",
            "view_env",
            "segment",
            "back_project",
        }
        assert client.reasoning_history == 3 and client.image_history == 1
        invalid = execute_with_si_values(
            client,
            "execute_actions",
            {
                "actions": [
                    {"action": "set_gripper", "opening": 0},
                    {"action": "move_to", "xyz": [1, 2]},
                ]
            },
        )
        assert invalid.is_error
        assert connected.bridge.calls == ["rl.observe"]
        assert connected.env.actions == []
        moved = execute_with_si_values(
            client,
            "execute_actions",
            {
                "actions": [
                    {"action": "move_to", "xyz": [0.1, -0.2, 0.9]},
                    {"action": "set_gripper", "opening": 1},
                ]
            },
        )
        assert moved.data["executed_count"] == 1
        assert moved.data["stop_reason"] == "terminated"
        assert client.workflow_stats["atomic_actions"] == 1
        assert client.state.records() == connected.toolkit.state.records()
    finally:
        client.close()


def test_response_loss_cannot_become_a_tool_error_and_replay_the_action(connected):
    client = connected.connect()
    connected.bridge.fail_after_action = True
    with pytest.raises(RuntimeError, match="outcome is uncertain"):
        execute_with_si_values(
            client,
            "execute_actions",
            {"actions": [{"action": "move_to", "xyz": [0.1, -0.2, 0.9]}]},
        )
    count = len(connected.env.actions)
    with pytest.raises(RuntimeError, match="outcome is uncertain"):
        execute_with_si_values(client, "move_to", {"xyz": [0.1, -0.2, 1.0]})
    assert len(connected.env.actions) == count
    assert connected.bridge.calls.count("rl.execute") == 1
    client.close()
    assert connected.bridge.released


def test_lease_is_running_before_initial_artifact_transfer(connected, monkeypatch):
    lease_started = threading.Event()

    def renew(client):
        lease_started.set()
        client._stopped.wait()

    original_observe = connected.worker.observe

    def observe():
        assert lease_started.wait(timeout=2), "initial transfer began without a lease"
        return original_observe()

    monkeypatch.setattr(remote.RemoteToolkit, "_renew", renew)
    connected.worker._rpc["rl.observe"] = observe
    assert "files" not in connected.worker.describe()
    client = connected.connect()
    try:
        assert client.state.latest_step == 0
    finally:
        client.close()


def test_close_releases_owned_worker_even_if_cancellation_connection_fails(connected):
    client = connected.connect()
    connected.bridge.fail_cancel = True
    with pytest.raises(RuntimeError, match="cancellation"):
        client.close()
    assert connected.bridge.released
    client.close()


@pytest.mark.parametrize(
    "field", ["source_digest", "task", "tools", "max_episode_steps", "episode_options"]
)
def test_incompatible_backend_is_rejected_and_released(connected, field):
    connected.bridge.mismatch = field
    with pytest.raises(ValueError):
        connected.connect()
    assert connected.bridge.released


def test_worker_rejects_replayed_out_of_order_and_closed_calls(connected):
    worker = connected.worker
    worker.describe()
    with pytest.raises(ValueError):
        worker.execute(1, "view_env", {})
    worker.execute(0, "view_env", {})
    with pytest.raises(ValueError):
        worker.execute(0, "view_env", {})
    worker.finish()
    with pytest.raises(ValueError):
        worker.execute(1, "view_env", {})


def test_worker_finishes_without_querying_the_closed_simulator(connected, monkeypatch):
    simulator_open = True
    terminal = {"status": "completed", "success": False, "truncated": True}

    def completion_result():
        if not simulator_open:
            raise ConnectionRefusedError("native simulator has closed")
        return terminal

    def close_simulator():
        nonlocal simulator_open
        simulator_open = False

    monkeypatch.setattr(connected.toolkit, "completion_result", completion_result)
    connected.worker.release.callback(close_simulator)
    client = connected.connect()
    client.close()
    assert connected.bridge.released and not simulator_open
    assert client.completion_result() == terminal
    assert client.solved() is False
    assert (connected.output / "rollout.json").is_file()
    assert connected.worker.finish()["completion_result"] == terminal


@pytest.mark.parametrize(
    "path", ["../outside.png", "/outside.png", "nested/../../outside.png"]
)
def test_artifact_delta_rejects_paths_outside_the_owned_directory(tmp_path, path):
    item = {
        "path": path,
        "bytes": base64.b64encode(b"image").decode(),
        "sha256": hashlib.sha256(b"image").hexdigest(),
    }
    with pytest.raises(ValueError, match="artifact"):
        remote.ArtifactExchange.receive(tmp_path / "output", [item])
    assert not (tmp_path / "outside.png").exists()


def test_artifact_corruption_is_detected_before_any_delta_file_is_written(tmp_path):
    item = {
        "path": "frame.png",
        "bytes": base64.b64encode(b"image").decode(),
        "sha256": hashlib.sha256(b"image").hexdigest(),
    }
    bad = {**item, "path": "states.json", "sha256": "0" * 64}
    with pytest.raises(ValueError, match="checksum"):
        remote.ArtifactExchange.receive(tmp_path, [item, bad])
    assert not (tmp_path / "frame.png").exists()


@pytest.mark.parametrize(
    "changes",
    [
        {"gpu": 8},
        {"max_episode_steps": 0},
        {"task": {"environment": "robotwin", "task_name": "../task", "seed": 1}},
        {
            "task": {
                "environment": "libero",
                "suite": "libero_10",
                "task": 0,
                "seed": 1,
                "workflow": "zap_turn",
            }
        },
    ],
)
def test_manager_rejects_tasks_outside_its_native_resource_contract(changes):
    request = {
        "task": {"environment": "robotwin", "task_name": "adjust_bottle", "seed": 1},
        "gpu": 0,
        "max_episode_steps": 10000,
        "episode_contract_version": 3,
        **changes,
    }
    with pytest.raises(ValueError):
        validate_request(request, {0, 1})


@pytest.mark.parametrize(
    "bad",
    [
        {"action_mode": "batched"},
        {"max_actions_per_call": True},
        {"max_actions_per_call": 0},
        {"image_size": 999},
        {"reasoning_history": -2},
        {"perception_mode": "batch"},
    ],
)
def test_manager_validates_workflow_options_before_starting_a_worker(bad):
    request = {
        "task": {
            "environment": "libero",
            "suite": "libero_goal",
            "task": 0,
            "seed": 0,
            **bad,
        },
        "gpu": 0,
        "max_episode_steps": 10000,
        "episode_contract_version": 3,
    }
    with pytest.raises(ValueError):
        validate_request(request, {0})


def test_manager_capacity_includes_reserved_initializations_and_cleanup(tmp_path):
    manager = BackendManager(tmp_path, {0: 1}, 1, "http://127.0.0.1:1")
    stopped = []
    manager._workers["existing"] = {
        "gpu": 0,
        "rpc": None,
        "ready": False,
        "closing": False,
        "daemon": SimpleNamespace(stop=lambda: stopped.append(True)),
    }
    request = {
        "task": {"environment": "robotwin", "task_name": "adjust_bottle", "seed": 1},
        "gpu": 0,
        "max_episode_steps": 10000,
        "episode_contract_version": 3,
    }
    try:
        with pytest.raises(RuntimeError, match="slots are full"):
            manager.open(request)
        assert manager.status() == {"reserved": 1, "ready": 0, "per_gpu": {0: 1}}
    finally:
        manager.close()
    assert stopped == [True]
    assert manager.status()["reserved"] == 0


def _owned_request():
    return {
        "task": {"environment": "robotwin", "task_name": "adjust_bottle", "seed": 1},
        "gpu": 0,
        "max_episode_steps": 10000,
        "episode_contract_version": 3,
    }


def test_cancel_owner_before_open_rejects_late_arrival(tmp_path):
    manager = BackendManager(tmp_path, {0: 2}, 1, "http://127.0.0.1:1")
    owner = "a" * 32
    try:
        receipt = manager.cancel_owners([owner])
        assert receipt["cleanup_complete"] and receipt["remaining"] == {}
        with pytest.raises(RuntimeError, match="cancelled"):
            manager.open(_owned_request(), owner_id=owner)
        assert manager.status()["reserved"] == 0
        assert list((tmp_path / "cancellations").glob("*.json"))
    finally:
        manager.close()


def test_cancel_owner_during_cold_start_never_spawns_it(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor

    from robots.rl import remote_manager

    manager = BackendManager(tmp_path, {0: 2}, 1, "http://127.0.0.1:1")
    owner = "b" * 32
    constructed = []
    monkeypatch.setattr(
        remote_manager, "ProcessDaemon", lambda **kwargs: constructed.append(kwargs)
    )
    manager._cold[0].acquire()
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(manager.open, _owned_request(), owner)
            for _ in range(500):
                if manager.status()["reserved"] == 1:
                    break
                threading.Event().wait(0.001)
            assert manager.status()["reserved"] == 1
            receipt = manager.cancel_owners([owner])
            assert receipt["cleanup_complete"]
            assert manager.status()["reserved"] == 0
            manager._cold[0].release()
            with pytest.raises(RuntimeError, match="was cancelled"):
                future.result(timeout=3)
        # The wrapper can be constructed after the semaphore wakes, but start()
        # is never reached for the removed reservation.
        assert len(constructed) <= 1
    finally:
        manager.close()


def test_cancel_owner_stops_only_its_initializing_worker(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor

    from robots.rl import remote_manager

    manager = BackendManager(tmp_path, {0: 3}, 2, "http://127.0.0.1:1")
    owner, foreign = "c" * 32, "d" * 32
    waiting, stopped = threading.Event(), threading.Event()
    foreign_stopped = []
    manager._workers["foreign"] = {
        "gpu": 0,
        "rpc": None,
        "ready": False,
        "closing": False,
        "owner_id": foreign,
        "output": tmp_path / "foreign",
        "daemon": SimpleNamespace(stop=lambda: foreign_stopped.append(True)),
    }
    manager._owners[foreign] = "foreign"

    class Daemon:
        def __init__(self, **kwargs):
            self.name = kwargs["name"]

        def start(self):
            pass

        def stop(self):
            stopped.set()

    def endpoint(daemon, timeout_s):
        waiting.set()
        assert stopped.wait(timeout=3)
        raise RuntimeError("owned daemon stopped during initialization")

    monkeypatch.setattr(remote_manager, "ProcessDaemon", Daemon)
    monkeypatch.setattr(remote_manager, "wait_for_endpoint", endpoint)
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(manager.open, _owned_request(), owner)
            assert waiting.wait(timeout=3)
            receipt = manager.cancel_owners([owner])
            with pytest.raises(RuntimeError, match="stopped"):
                future.result(timeout=3)
        assert receipt["cleanup_complete"] and receipt["remaining"] == {}
        assert list(manager._workers) == ["foreign"]
        assert not foreign_stopped
    finally:
        manager.close()
    assert foreign_stopped == [True]


def test_cancel_owner_does_not_free_capacity_until_process_stop_finishes(tmp_path):
    from concurrent.futures import ThreadPoolExecutor

    manager = BackendManager(tmp_path, {0: 1}, 1, "http://127.0.0.1:1")
    owner = "e" * 32
    started, finish = threading.Event(), threading.Event()

    def stop():
        started.set()
        assert finish.wait(timeout=3)

    manager._workers["owned"] = {
        "gpu": 0,
        "rpc": None,
        "ready": False,
        "closing": False,
        "owner_id": owner,
        "output": tmp_path / "owned",
        "daemon": SimpleNamespace(stop=stop),
    }
    manager._owners[owner] = "owned"
    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            cancel = executor.submit(manager.cancel_owners, [owner])
            assert started.wait(timeout=3)
            assert manager.status()["reserved"] == 1
            with pytest.raises(RuntimeError, match="slots are full"):
                manager.open(_owned_request())
            duplicate = executor.submit(manager.cancel_owners, [owner])
            finish.set()
            assert cancel.result(timeout=3)["cleanup_complete"]
            assert duplicate.result(timeout=3)["cleanup_complete"]
        assert manager.status()["reserved"] == 0
    finally:
        finish.set()
        manager.close()


@pytest.mark.parametrize("owner", ["", "../x", "x" * 32, "a" * 31, None, 3])
def test_cancel_owner_validates_the_complete_list_before_tombstoning(tmp_path, owner):
    manager = BackendManager(tmp_path, {0: 1}, 1, "http://127.0.0.1:1")
    try:
        with pytest.raises(ValueError, match="UUID"):
            manager.cancel_owners(["a" * 32, owner])
        assert manager._cancelled_owners == set()
    finally:
        manager.close()


def test_remote_attempt_owner_is_sent_before_open_and_echoed(connected, monkeypatch):
    owner = "f" * 32
    monkeypatch.setenv("RPENT_REMOTE_OWNER_ID", owner)
    client = connected.connect()
    try:
        assert connected.bridge.owner_id == owner
    finally:
        client.close()


def test_remote_cancel_owners_uses_rpc_and_retains_tombstones(tmp_path):
    manager = BackendManager(tmp_path, {0: 2}, 1, "http://127.0.0.1:1")
    server = HttpRpcServer(("127.0.0.1", 0), manager._dispatch)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        endpoint = f"http://127.0.0.1:{server.server_address[1]}"
        receipt = remote.cancel_remote_owners(endpoint, ["a" * 32, "b" * 32])
        assert receipt["cleanup_complete"] and receipt["remaining"] == {}
        assert manager._cancelled_owners == {"a" * 32, "b" * 32}
        assert manager.status()["reserved"] == 0
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)
        manager.close()


def test_remote_explicit_attempt_owner_is_independent_of_process_environment(
    connected, monkeypatch
):
    monkeypatch.setenv("RPENT_REMOTE_OWNER_ID", "a" * 32)
    client = connected.connect(owner_id="b" * 32)
    try:
        assert connected.bridge.owner_id == "b" * 32
    finally:
        client.close()


def test_cancel_owners_runs_bounded_parallel_cleanup(tmp_path):
    from concurrent.futures import ThreadPoolExecutor

    manager = BackendManager(tmp_path, {0: 20}, 1, "http://127.0.0.1:1")
    gate, sixteen = threading.Event(), threading.Event()
    lock = threading.Lock()
    counts = {"started": 0, "active": 0, "peak": 0}

    def stop():
        with lock:
            counts["started"] += 1
            counts["active"] += 1
            counts["peak"] = max(counts["peak"], counts["active"])
            if counts["active"] == 16:
                sixteen.set()
        assert gate.wait(timeout=3)
        with lock:
            counts["active"] -= 1

    owners = [f"{i:032x}" for i in range(20)]
    for owner in owners:
        manager._workers[owner] = {
            "gpu": 0,
            "rpc": None,
            "ready": False,
            "closing": False,
            "owner_id": owner,
            "output": tmp_path / owner,
            "daemon": SimpleNamespace(stop=stop),
        }
        manager._owners[owner] = owner
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(manager.cancel_owners, owners)
            assert sixteen.wait(timeout=3)
            assert counts["started"] == counts["peak"] == 16
            gate.set()
            assert future.result(timeout=3)["cleanup_complete"]
        assert counts["started"] == 20 and counts["peak"] == 16
        assert manager.status()["reserved"] == 0
    finally:
        gate.set()
        manager.close()


def test_cancel_owner_cleanup_failure_keeps_evidence_and_cleans_other_owners(tmp_path):
    manager = BackendManager(tmp_path, {0: 2}, 1, "http://127.0.0.1:1")
    owners = ["a" * 32, "b" * 32]
    stopped = []

    def bad_stop():
        raise RuntimeError("owned process still alive")

    for owner in owners:
        manager._workers[owner] = {
            "gpu": 0,
            "rpc": None,
            "ready": False,
            "closing": False,
            "owner_id": owner,
            "output": tmp_path / owner,
            "daemon": SimpleNamespace(
                stop=bad_stop if owner == owners[0] else lambda: stopped.append(True)
            ),
        }
        manager._owners[owner] = owner
    try:
        with pytest.raises(RuntimeError, match="cleanup failed"):
            manager.cancel_owners(owners)
        assert stopped == [True]
        assert manager.status()["reserved"] == 1
        receipts = list((tmp_path / "cancellations").glob("*.json"))
        receipt = json.loads(receipts[0].read_text())
        assert not receipt["cleanup_complete"]
        assert owners[0] in receipt["errors"] and owners[0] in receipt["remaining"]
    finally:
        manager._workers.clear()
        manager._owners.clear()
        manager.close()


def test_cancel_tombstone_blocks_cold_start_before_queued_release_runs(
    tmp_path, monkeypatch
):
    from concurrent.futures import ThreadPoolExecutor

    from robots.rl import remote_manager

    manager = BackendManager(tmp_path, {0: 1}, 1, "http://127.0.0.1:1")
    owner = "f" * 32
    spawned = []

    class Daemon:
        def __init__(self, **kwargs):
            pass

        def start(self):
            spawned.append(True)

    monkeypatch.setattr(remote_manager, "ProcessDaemon", Daemon)
    manager._cold[0].acquire()
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(manager.open, _owned_request(), owner)
            for _ in range(500):
                if manager.status()["reserved"]:
                    break
                threading.Event().wait(0.001)
            assert manager.status()["reserved"] == 1
            # The bounded release executor may still be busy with earlier owners.
            with manager._lock:
                manager._cancelled_owners.add(owner)
            manager._cold[0].release()
            with pytest.raises(RuntimeError, match="cancelled"):
                future.result(timeout=3)
            assert not spawned
            assert manager.status()["reserved"] == 0
    finally:
        manager.close()


@contextlib.contextmanager
def _manager_rpc_proxy(tmp_path, handler):
    """Use the actual manager and two HTTP boundaries without a simulator."""
    from types import SimpleNamespace

    owner = "a" * 32
    origin = RpcFacade()
    origin._rpc.update(
        dict.fromkeys(("rl.observe", "rl.execute", "rl.close", "rl.cancel"), handler)
    )
    origin._readonly_methods.update(origin._rpc)
    manager = BackendManager(tmp_path, {0: 32}, 1, "http://127.0.0.1:1")
    servers = []
    threads = []
    clients = []
    try:
        for dispatch in (origin._dispatch, manager._dispatch):
            server = HttpRpcServer(("127.0.0.1", 0), dispatch)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            servers.append(server)
            threads.append(thread)
            clients.append(
                HttpRpcClient(f"http://127.0.0.1:{server.server_address[1]}")
            )
        manager._workers["owned"] = {
            "gpu": 0,
            "last_seen": 0.0,
            "daemon": None,
            "rpc": clients[0],
            "ready": True,
            "output": tmp_path / "owned",
            "owner_id": owner,
            "released": threading.Event(),
            "closing": False,
        }
        manager._owners[owner] = "owned"
        yield SimpleNamespace(
            manager=manager,
            client=clients[1],
            upstream=clients[0],
            origin=origin,
        )
    finally:
        for client in clients:
            client.close()
        for server in reversed(servers):
            server.shutdown()
            server.server_close()
        for thread in threads:
            thread.join(timeout=3)
            assert not thread.is_alive()
        manager.close()


def test_manager_http_forwards_exact_envelope_and_preserves_direct_calls(tmp_path):
    import numpy as np

    payload = {
        "data": {"pose": np.arange(6, dtype=np.float32).reshape(2, 3)},
        "images": [base64.b64encode(b"image-bytes").decode()],
        "files": [
            {
                "path": "world.npz",
                "bytes": base64.b64encode(b"geometry-bytes").decode(),
                "sha256": hashlib.sha256(b"geometry-bytes").hexdigest(),
            }
        ],
    }
    with _manager_rpc_proxy(tmp_path, lambda: payload) as proxy:
        direct = proxy.manager.call("owned", "rl.observe", {})
        np.testing.assert_array_equal(direct["data"]["pose"], payload["data"]["pose"])

        def decoded_call_would_duplicate_work(*args, **kwargs):
            raise AssertionError("manager must not decode its upstream HTTP response")

        proxy.upstream.call = decoded_call_would_duplicate_work
        expected = proxy.upstream.forward("rl.observe").body
        arguments = {
            "session": "owned",
            "method": "rl.observe",
            "arguments": {},
        }
        actual = proxy.client.forward("backend.call", kwargs=arguments).body
        assert actual == expected
        assert hashlib.sha256(actual).digest() == hashlib.sha256(expected).digest()
        decoded = proxy.client.call("backend.call", kwargs=arguments)
        np.testing.assert_array_equal(decoded["data"]["pose"], payload["data"]["pose"])
        assert decoded["data"]["pose"].dtype == np.float32
        assert decoded["images"] == payload["images"]
        assert decoded["files"] == payload["files"]


def test_manager_http_forwards_remote_errors_without_success_wrapping(tmp_path):
    def fail():
        raise RuntimeError("owned camera failed")

    with _manager_rpc_proxy(tmp_path, fail) as proxy:
        with pytest.raises(RpcError, match="owned camera failed") as info:
            proxy.client.call(
                "backend.call",
                kwargs={"session": "owned", "method": "rl.observe", "arguments": {}},
            )
        assert info.value.method == "backend.call"
        assert "RuntimeError: owned camera failed" in info.value.server_traceback
        assert proxy.client.call("backend.status")["reserved"] == 1


def test_manager_http_rejects_disallowed_method_before_forwarding(tmp_path):
    calls = []
    with _manager_rpc_proxy(tmp_path, lambda: calls.append(True)) as proxy:
        with pytest.raises(RpcError, match="unsupported toolkit operation"):
            proxy.client.call(
                "backend.call",
                kwargs={"session": "owned", "method": "shutdown", "arguments": {}},
            )
        assert calls == []
        assert not proxy.origin._shutdown_event.is_set()


def test_manager_control_calls_respond_while_bulk_replies_are_pending(tmp_path):
    from concurrent.futures import ThreadPoolExecutor

    count = 32
    entered = threading.Barrier(count + 1)
    release = threading.Event()
    blob = base64.b64encode(b"body" * (256 * 1024)).decode()

    def bulk():
        entered.wait(timeout=5)
        assert release.wait(timeout=5)
        return {"files": [{"bytes": blob}], "native_success": False}

    with _manager_rpc_proxy(tmp_path, bulk) as proxy:

        def no_decoded_call(*args, **kwargs):
            raise AssertionError("manager must use opaque upstream forwarding")

        proxy.upstream.call = no_decoded_call
        with ThreadPoolExecutor(max_workers=count) as executor:
            futures = [
                executor.submit(
                    proxy.client.call,
                    "backend.call",
                    kwargs={
                        "session": "owned",
                        "method": "rl.observe",
                        "arguments": {},
                    },
                    timeout_s=10,
                )
                for _ in range(count)
            ]
            try:
                entered.wait(timeout=5)
                assert proxy.client.call("backend.status", timeout_s=1)["ready"] == 1
                assert proxy.client.call(
                    "backend.renew", kwargs={"session": "owned"}, timeout_s=1
                ) == {"renewed": True}
            finally:
                release.set()
            for future in futures:
                result = future.result(timeout=10)
                assert result["files"][0]["bytes"] == blob
                assert result["native_success"] is False


@pytest.mark.parametrize(
    "connected",
    [{"calibrated": True}],
    indirect=True,
)
def test_remote_general_converts_once_and_carries_native_completion(connected):
    import numpy as np

    client = connected.connect()
    np.testing.assert_allclose(
        client.initial_observation().data["state"]["pose"]["xyz"],
        CENTIMETERS.payload(connected.toolkit._robot_state())["pose"]["xyz"],
    )
    scale = 100
    measured = client.execute_tool("segment", {"prompt": "cube", "step": 0})
    suffix = "cm"
    assert measured.data[f"top_z_{suffix}"] == pytest.approx(scale, abs=0.01)
    moved = client.execute_tool(
        "execute_actions",
        {
            "actions": [
                {"action": "move_to", "xyz": [v * scale for v in [0.1, -0.2, 0.9]]}
            ]
        },
    )
    np.testing.assert_allclose(
        moved.data["results"][0]["pose"]["xyz"],
        np.asarray(connected.toolkit._robot_state()["pose"]["xyz"]) * scale,
    )
    client.close()


@pytest.mark.parametrize("connected", [{"calibrated": True}], indirect=True)
def test_remote_general_accepts_measured_floor_calibration(connected):
    from dataclasses import replace

    primitive = connected.toolkit._primitives
    primitive.coordinates = replace(
        primitive.coordinates, table_height_m=None, floor_height_m=-0.025
    )
    client = connected.connect()
    coordinates = connected.worker.describe()["coordinates"]
    assert coordinates["support_surface"] == "floor"
    assert coordinates["table_height_above_base_m"] is None
    assert coordinates["floor_height_above_base_m"] == pytest.approx(-0.025)
    client.close()


def test_router_reports_real_readiness_and_releases_only_cohort_owners():
    from robots.rl.router import ToolkitRouter

    rows = {"unrelated": True}
    backend = RpcFacade()

    def open_worker(request, owner_id):
        rows[owner_id] = False
        return {"session": owner_id}

    def status(owner_ids=None):
        selected = {
            owner: ready
            for owner, ready in rows.items()
            if owner_ids is None or owner in owner_ids
        }
        return {"reserved": len(selected), "ready": sum(selected.values())}

    def cancel_owners(owner_ids):
        for owner in owner_ids:
            rows.pop(owner, None)
        return {"cleanup_complete": True, "remaining": {}}

    backend._rpc.update(
        {
            "backend.open": open_worker,
            "backend.status": status,
            "backend.cancel_owners": cancel_owners,
        }
    )
    with HttpRpcServer(("127.0.0.1", 0), backend._dispatch) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        router = ToolkitRouter([(f"http://127.0.0.1:{server.server_port}", 2)], 2)
        try:
            assert router.acquire_group("first:0") == "admitted"
            assert router.acquire_group("second:0") == "waiting"
            router.open({"gpu": None}, "1" * 32, "first:0")
            router.open({"gpu": None}, "2" * 32, "first:0")
            assert router.status("first")["ready"] == 0
            rows["1" * 32] = rows["2" * 32] = True
            assert router.status("first")["ready"] == 2
            assert router.status("second")["reserved"] == 0
            router.finish_group("first:0")
            assert router.acquire_group("first:0") == "closed"
            assert router.acquire_group("second:0") == "admitted"
            owner = "3" * 32
            router.cancel_owners([owner])
            with pytest.raises(ValueError, match="new attempt owner"):
                router.open({"gpu": None}, owner, "second:0")
            assert rows == {"unrelated": True}
        finally:
            router.close()
            server.shutdown()
            thread.join()
