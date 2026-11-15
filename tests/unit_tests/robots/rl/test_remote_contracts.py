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

import base64
import contextlib
import hashlib
import json
import threading
from types import SimpleNamespace

import pytest

from robots.rl import remote, tools
from robots.rl.remote_manager import BackendManager, validate_request
from rpent.dashboard.events import NullDashboardEventSink
from rpent.memory import MemoryManager
from rpent.utils.rpc import RpcFacade
from rpent.utils.rpc.http_rpc import HttpRpcServer
from tests.unit_tests.robots.rl.test_libero_rl_contracts import FakeEnv, make_toolkit


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
    monkeypatch.setattr(tools, "IMAGE_SIZE", 32)
    from rpent.utils import templates

    monkeypatch.setattr(
        templates, "default_variables", lambda: {"output_dir": "/offline/output"}
    )
    env = FakeEnv(success_below_z=0.95)
    options = getattr(request, "param", {})
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
    invalid = client.execute_tool("move_to", {"xyz": "bad"})
    assert invalid.is_error and connected.bridge.calls == ["rl.observe"]
    moved = client.execute_tool("move_to", {"xyz": [0.1, -0.2, 0.9]})
    # Native success terminates this fixture before the requested pose settles.
    assert moved.data["status"] == "partial"
    assert moved.data["reason"] == "terminated"
    assert client.solved() is True
    viewed = client.execute_tool("view_env", {})
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
            "action_mode": "batch",
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
            "finish",
        }
        assert client.reasoning_history == 3 and client.image_history == 1
        invalid = client.execute_tool(
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
        moved = client.execute_tool(
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
        client.execute_tool("move_to", {"xyz": [0.1, -0.2, 0.9]})
    count = len(connected.env.actions)
    with pytest.raises(RuntimeError, match="outcome is uncertain"):
        client.execute_tool("move_to", {"xyz": [0.1, -0.2, 1.0]})
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
    "field", ["source_digest", "task", "tools", "max_episode_steps", "workflow_options"]
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
    }
    with pytest.raises(ValueError):
        validate_request(request, {0})


def test_manager_capacity_includes_reserved_initializations_and_cleanup(tmp_path):
    manager = BackendManager(tmp_path, {0: 1}, 1, "http://127.0.0.1:1")
    stopped = []
    manager._workers["existing"] = {
        "gpu": 0,
        "rpc": None,
        "daemon": SimpleNamespace(stop=lambda: stopped.append(True)),
    }
    request = {
        "task": {"environment": "robotwin", "task_name": "adjust_bottle", "seed": 1},
        "gpu": 0,
        "max_episode_steps": 10000,
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
            with pytest.raises(KeyError):
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
