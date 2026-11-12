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

"""Motor lifecycle and HTTP contracts using public RLinf part fakes."""

import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager

import numpy as np
import pytest

from robots.lekiwi.env_client import LeKiwiEnvClient
from robots.lekiwi.env_server import LeKiwiEnvFacade
from rpent.utils.rpc import RpcError, SocketRpcServer, make_rpc_client
from rpent.utils.rpc.http_rpc import HttpRpcServer


def wait_for(predicate, timeout=3):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    pytest.fail("condition not reached")


@contextmanager
def rpc_server(backend, *, camera_only=False, transport="http"):
    facade = LeKiwiEnvFacade(backend, camera_only=camera_only)
    server_type = HttpRpcServer if transport == "http" else SocketRpcServer
    server = server_type(("127.0.0.1", 0), facade._dispatch)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    endpoint = f"{transport}://127.0.0.1:{server.server_address[1]}"
    rpc = make_rpc_client(endpoint)
    try:
        yield rpc, endpoint
    finally:
        rpc.close()
        server.shutdown()
        server.server_close()
        thread.join(2)


def test_server_lifecycle(robot_parts, monkeypatch):
    from robots.lekiwi.env_server import main

    monkeypatch.setattr(sys, "argv", ["lekiwi-server"])
    events = []

    class Backend:
        def __init__(self, cfg):
            events.append("start")

        def close_env(self):
            events.append("close")

    class Facade:
        def __init__(self, backend):
            pass

        def serve(self, **kwargs):
            events.append("serve")

    assert main(backend_class=Backend, facade_class=Facade) == 0
    assert events == ["start", "serve", "close"]


def test_drive_holds_velocity_continuously_then_returns_stopped(robot_parts):
    backend, robot = robot_parts.build()
    base = robot.base
    stops = base.stops
    with ThreadPoolExecutor() as pool:
        pending = pool.submit(backend.drive, [0.2, 0, 0], 0.3)
        wait_for(lambda: len(base.commands) >= 3)
        assert not pending.done()
        assert base.stops == stops
        assert all(np.array_equal(v, [0.2, 0, 0]) for _, v in base.commands)
        result = pending.result(timeout=2)
    assert result["status"] == "succeeded", result
    assert result["elapsed_s"] >= 0.3
    assert result["stop_confirmed"]
    assert base.stops == stops + 1
    assert not base.velocity.any()


@pytest.mark.parametrize("angle", [-5, 5])
def test_rotation_uses_measured_feedback(robot_parts, angle):
    backend, robot = robot_parts.build()
    result = backend.rotate(angle)
    assert result["status"] == "succeeded", result
    assert result["stop_confirmed"]
    assert abs(result["remaining_angle_deg"]) <= 2
    assert all(not v[:2].any() for _, v in robot.base.commands)


@pytest.mark.parametrize(
    "method,args",
    [
        ("drive", ([0.1, 0, 0], 0)),
        ("drive", ([0.1, 0, 0], 11)),
        ("drive", ([float("nan"), 0, 0], 1)),
        ("drive", ([[1, 2, 3]], 1)),
        ("rotate", (361,)),
        ("rotate", (float("inf"),)),
    ],
)
def test_invalid_commands_never_reach_hardware(robot_parts, method, args):
    backend, robot = robot_parts.build()
    with pytest.raises(ValueError):
        getattr(backend, method)(*args)
    assert not robot.base.commands


def test_drive_clips_configured_velocity(robot_parts):
    backend, robot = robot_parts.build()
    result = backend.drive([9, -9, 100], 0.05)
    assert result["commanded_velocity"] == pytest.approx([0.3, -0.3, 60])
    assert all(v.shape == (3,) for _, v in robot.base.commands)


@pytest.mark.parametrize("transport", ["http", "socket"])
def test_client_can_drive_without_reset_and_cancel_over_rpc(robot_parts, transport):
    backend, robot = robot_parts.build()
    with rpc_server(backend, transport=transport) as (rpc, endpoint):
        client = LeKiwiEnvClient(rpc, reset_on_connect=False)
        operator = make_rpc_client(endpoint)
        try:
            with ThreadPoolExecutor() as pool:
                pending = pool.submit(client.drive, [0.1, 0, 0], 5)
                wait_for(lambda: robot.base.commands)
                assert operator.call("env.stop")["ok"]
                command_count = len(robot.base.commands)
                result = pending.result(timeout=3)
                assert len(robot.base.commands) == command_count
            assert result["status"] == "cancelled"
            assert result["stop_confirmed"]
            for obsolete in ("move_to", "hold", "step", "get_motion", "set_velocity"):
                with pytest.raises(RpcError, match="unknown RPC"):
                    rpc.call(f"env.{obsolete}")
        finally:
            operator.close()


def test_concurrent_motion_is_rejected_without_queuing(robot_parts):
    backend, robot = robot_parts.build()
    with ThreadPoolExecutor() as pool:
        pending = pool.submit(backend.drive, [0.1, 0, 0], 5)
        try:
            wait_for(lambda: robot.base.commands)
            with pytest.raises(RuntimeError, match="already active"):
                backend.rotate(30)
            with pytest.raises(RuntimeError, match="already active"):
                backend.drive([0.1, 0, 0], 1)
        finally:
            backend.stop()
        assert pending.result(timeout=2)["status"] == "cancelled"
    assert backend.drive([0, 0, 0], 0.02)["status"] == "succeeded"


@pytest.mark.parametrize("method", ["get_observation", "send_action", "reset"])
def test_fault_is_latched_until_verified_reset(robot_parts, monkeypatch, method):
    backend, robot = robot_parts.build()

    def fail(*args, **kwargs):
        raise OSError("hardware unavailable")

    with monkeypatch.context() as patch:
        patch.setattr(robot.base, method, fail)
        result = backend.drive([0.1, 0, 0], 0.05)
        assert result["status"] == "failed"
        with pytest.raises(RuntimeError, match="reset"):
            backend.rotate(5)
    assert backend.reset()["ok"]
    assert backend.drive([0, 0, 0], 0.02)["status"] == "succeeded"


@pytest.mark.parametrize("reason", ["stale", "invalid", "timeout"])
def test_feedback_failures_and_execution_timeout_stop_motion(
    robot_parts, monkeypatch, reason
):
    import robots.lekiwi.env_server as server

    backend, robot = robot_parts.build()
    if reason == "stale":

        def read():
            time.sleep(0.3)
            return {"velocity": np.zeros(3)}

        monkeypatch.setattr(robot.base, "get_observation", read)
    elif reason == "invalid":
        monkeypatch.setattr(
            robot.base, "get_observation", lambda: {"velocity": [float("nan"), 0, 0]}
        )
    elif reason == "timeout":
        monkeypatch.setattr(server, "MOTION_TIMEOUT_S", 0.15)
        monkeypatch.setattr(
            robot.base, "get_observation", lambda: {"velocity": np.zeros(3)}
        )
    result = backend.rotate(90)
    assert result["status"] == "failed", result
    assert result["error"]
    assert not robot.base.velocity.any()


def test_stop_allows_bounded_deceleration(robot_parts, monkeypatch):
    backend, robot = robot_parts.build()
    original_reset = robot.base.reset

    def decelerate():
        threading.Timer(0.08, original_reset).start()

    monkeypatch.setattr(robot.base, "reset", decelerate)
    result = backend.drive([0.1, 0, 0], 0.05)
    assert result["status"] == "succeeded", result
    assert result["stop_confirmed"]


def test_unconfirmed_stop_requires_reset(robot_parts, monkeypatch):
    backend, robot = robot_parts.build()
    with monkeypatch.context() as patch:
        patch.setattr(robot.base, "reset", lambda: None)
        result = backend.drive([0.1, 0, 0], 0.05)
        assert result["status"] == "failed"
        assert not result["stop_confirmed"]
    with pytest.raises(RuntimeError, match="reset"):
        backend.drive([0.1, 0, 0], 1)
    assert backend.reset()["ok"]


@pytest.mark.parametrize("method", ["reset", "close_env"])
def test_reset_and_shutdown_stop_and_drain_motion(robot_parts, method):
    backend, robot = robot_parts.build()
    with ThreadPoolExecutor() as pool:
        pending = pool.submit(backend.drive, [0.1, 0, 0], 5)
        wait_for(lambda: robot.base.commands)
        getattr(backend, method)()
        assert pending.result(timeout=2)["status"] == "cancelled"
    assert not robot.base.velocity.any()
    if method == "close_env":
        assert not robot.connected
        with pytest.raises(RuntimeError, match="unavailable"):
            backend.drive([0.1, 0, 0], 1)


@pytest.mark.parametrize("transport", ["http", "socket"])
def test_camera_wait_does_not_block_motion_or_stop(robot_parts, monkeypatch, transport):
    backend, robot = robot_parts.build()
    reading, release = threading.Event(), threading.Event()

    def frames(**kwargs):
        reading.set()
        assert release.wait(4)
        return np.zeros((48, 64, 3), dtype=np.uint8)

    monkeypatch.setattr(robot.cameras["front"], "get_frame", frames)
    with rpc_server(backend, transport=transport) as (rpc, endpoint):
        camera_rpc = make_rpc_client(endpoint)
        operator = make_rpc_client(endpoint)
        try:
            with ThreadPoolExecutor() as pool:
                camera = pool.submit(camera_rpc.call, "env.get_observation")
                try:
                    assert reading.wait(1)
                    motion = pool.submit(
                        rpc.call,
                        "env.drive",
                        kwargs={"velocity": [0.1, 0, 0], "duration_s": 2},
                    )
                    wait_for(lambda: robot.base.commands)
                    assert operator.call("env.stop", timeout_s=0.5)["ok"]
                    assert motion.result(timeout=1)["status"] == "cancelled"
                    assert not camera.done()
                finally:
                    release.set()
                assert camera.result()["main_images"].shape == (128, 128, 3)
        finally:
            camera_rpc.close()
            operator.close()


def test_client_timeout_leaves_only_the_bounded_action_running(robot_parts):
    backend, robot = robot_parts.build()
    stops = robot.base.stops
    with rpc_server(backend) as (rpc, _):
        with pytest.raises(RpcError, match="timed out"):
            rpc.call(
                "env.drive",
                kwargs={"velocity": [0.1, 0, 0], "duration_s": 0.3},
                timeout_s=0.05,
            )
        assert robot.base.velocity.any()
        wait_for(lambda: robot.base.stops > stops)
        assert not robot.base.velocity.any()
        count = len(robot.base.commands)
        time.sleep(0.1)
        assert len(robot.base.commands) == count


def test_camera_only_has_no_motor_routes_or_connection(robot_parts):
    from robots.lekiwi.env_server import LeKiwiCameraBackend

    backend = LeKiwiCameraBackend(robot_parts.cfg)
    try:
        with rpc_server(backend, camera_only=True) as (rpc, _):
            client = LeKiwiEnvClient(rpc, reset_on_connect=False)
            assert client.meta["camera_only"]
            assert client.get_observation()["main_images"].shape == (48, 64, 3)
            for method in (
                "drive",
                "rotate",
                "stop",
                "reset",
                "get_motion",
                "set_velocity",
                "get_robot_state",
            ):
                with pytest.raises(RpcError, match="unknown RPC"):
                    rpc.call(f"env.{method}")
    finally:
        backend.close_env()
    assert not robot_parts.instances
    assert not backend._camera.connected


@pytest.mark.parametrize("failure", ["connect", "ready", "serve"])
def test_camera_only_failure_releases_camera(robot_parts, monkeypatch, failure):
    from robots.lekiwi.env_server import main

    instances = []
    original = robot_parts.Camera.of

    def create(info):
        camera = original(info)
        instances.append(camera)
        return camera

    monkeypatch.setattr(robot_parts.Camera, "of", create)

    def fail(*args, **kwargs):
        raise OSError("failed")

    if failure == "connect":
        monkeypatch.setattr(robot_parts.Camera, "connect", fail)
    elif failure == "ready":
        monkeypatch.setattr(robot_parts.Camera, "is_ready", lambda *a, **k: False)
    else:
        monkeypatch.setattr(LeKiwiEnvFacade, "serve", fail)
    monkeypatch.setattr(sys, "argv", ["server", "--camera-only"])
    with pytest.raises((RuntimeError, OSError)):
        main()
    assert instances[0].releases == 1
    assert not robot_parts.instances
