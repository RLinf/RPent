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

    def open(self, request):
        result = {"session": "owned-session", **self.worker.describe()}
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
def connected(tmp_path, monkeypatch):
    monkeypatch.setattr(tools, "IMAGE_SIZE", 32)
    from rpent.utils import templates

    monkeypatch.setattr(
        templates, "default_variables", lambda: {"output_dir": "/offline/output"}
    )
    env = FakeEnv(success_below_z=0.95)
    toolkit = make_toolkit(tmp_path / "backend", "adaptive", env=env)
    release = contextlib.ExitStack()
    release.callback(toolkit.close)
    task = {"environment": "libero", "suite": "libero_object", "task": 0, "seed": 1}
    request = {"task": task, "gpu": 0, "max_episode_steps": 10000}
    config = {**request, "output_dir": str(tmp_path / "backend/run")}
    worker = remote.ToolkitWorker(toolkit, release, config)
    bridge = Bridge(worker)
    server = HttpRpcServer(("127.0.0.1", 0), bridge._dispatch)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    client_config = SimpleNamespace(output_dir=tmp_path / "frontend")

    def connect():
        return remote.RemoteToolkit(
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
    "field", ["source_digest", "task", "tools", "max_episode_steps"]
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
