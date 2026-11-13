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

"""Offline test for the LeKiwi RPC facade/client round-trip over HTTP."""

from __future__ import annotations

import sys
import threading
import time
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from robots.lekiwi.env_client import LeKiwiEnvClient
from robots.lekiwi.env_server import LeKiwiEnvFacade
from rpent.utils.rpc.http_rpc import HttpRpcClient, HttpRpcServer


class FakeBackend:
    def get_env_meta(self):
        return {"ok": True, "action_dim": 3}

    def reset(self):
        return {"ok": True, "states": np.zeros(3, dtype=np.float32)}

    def get_robot_state(self):
        return {"velocity": [0.0, 0.0, 0.0]}

    def get_observation(self):
        return {"states": np.zeros(3, dtype=np.float32)}

    def get_camera_meta(self):
        return {"cameras": {}}

    def drive(self, velocity, *, duration_s):
        return {
            "ok": True,
            "requested_velocity": np.asarray(velocity),
            "duration_s": duration_s,
        }

    def hold(self, velocity, *, ttl_s):
        return {
            "ok": True,
            "commanded_velocity": np.asarray(velocity),
            "ttl_s": ttl_s,
        }

    def stop(self):
        return {"ok": True}


def test_lekiwi_client_and_facade_round_trip_numpy_over_http():
    backend = FakeBackend()
    # This transport test exercises manual driving; other Env routes are stubs.
    for name in (
        "get_stereo_frame",
        "get_navigation_state",
        "navigate_to",
        "list_map_nodes",
        "inspect_map_node",
        "view_map",
        "recording_start",
        "recording_state",
        "recording_stop",
    ):
        setattr(backend, name, lambda **kwargs: {})
    facade = LeKiwiEnvFacade(backend)
    server = HttpRpcServer(("127.0.0.1", 0), facade._dispatch)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    try:
        client = LeKiwiEnvClient(HttpRpcClient(f"http://{host}:{port}"))
        result = client.drive([0.1, 0.0, 5.0], duration_s=0.5)
        held = client.hold([0.2, 0.0, 0.0], ttl_s=0.75)
        stopped = client.stop()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    assert result["ok"] is True
    np.testing.assert_allclose(result["requested_velocity"], [0.1, 0.0, 5.0])
    assert result["duration_s"] == 0.5
    np.testing.assert_allclose(held["commanded_velocity"], [0.2, 0.0, 0.0])
    assert held["ttl_s"] == 0.75
    assert stopped["ok"] is True


def test_local_worker_info_keeps_every_part_local(
    fake_rlinf_realworld_modules, monkeypatch
):
    # No Ray on the robot: every node rank stays None so RLinf's parts open
    # their devices in this process instead of on a scheduler worker.
    from robots.lekiwi.env_server import _local_worker_info
    from robots.lekiwi.runtime_config import load_runtime_config

    class Recorder:
        def __init__(self, **kwargs: Any) -> None:
            self.__dict__.update(kwargs)

    discovery = types.ModuleType("rlinf.robotics.discovery")
    discovery.RobotInfo = Recorder
    monkeypatch.setitem(sys.modules, "rlinf.robotics.discovery", discovery)
    scheduler = types.ModuleType("rlinf.scheduler")
    scheduler.WorkerInfo = Recorder
    scheduler.WorkerAddress = lambda _name, _rank: None
    scheduler.AcceleratorType = SimpleNamespace(NO_ACCEL="NO_ACCEL")
    monkeypatch.setitem(sys.modules, "rlinf.scheduler", scheduler)

    example = Path(__file__).parents[4] / "robots/lekiwi/config/example.yaml"
    info = _local_worker_info(load_runtime_config(example).rlinf)

    assert info.cluster_node_rank is None
    assert info.hardware_infos[0].config.node_rank is None


class _FakeInnerEnv:
    """One env whose step paces like the real control period."""

    def __init__(self) -> None:
        import gymnasium as gym

        self.action_space = gym.spaces.Box(
            -np.ones(3, dtype=np.float32), np.ones(3, dtype=np.float32)
        )
        self.config = type("C", (), {"step_frequency": 50.0, "is_dummy": False})
        self.unwrapped = self
        self.stepped: list[tuple[int, ...]] = []
        self.stops = 0
        self.moving_steps = 0
        self.velocity = np.zeros(3)
        self.sample = {}

    def stop(self) -> None:
        self.stops += 1
        self.velocity = np.zeros(3)

    def get_velocity_sample(self):
        return dict(self.sample)

    def step(self, action):
        self.stepped.append(tuple(np.asarray(action).shape))
        self.velocity = np.asarray(action)
        self.moving_steps += int(self.velocity.any())
        self.sample = {
            "velocity": self.velocity.tolist(),
            "sampled_at": time.time(),
            "sampled_monotonic": time.monotonic(),
        }
        time.sleep(1.0 / self.config.step_frequency)
        return (
            {"states": np.zeros((1, 3), dtype=np.float32)},
            0.0,
            False,
            False,
            {},
        )


def _fake_backend(monkeypatch):
    """Build a backend over an in-process fake env and return both."""
    import gymnasium as gym

    inner = _FakeInnerEnv()

    class FakeVector:
        # Mirrors NoAutoResetSyncVectorEnv: a batched (1, 3) action space that
        # slices the batch off before stepping each env.
        def __init__(self) -> None:
            self.envs = [inner]
            self._batched = gym.spaces.Box(
                -np.ones((1, 3), dtype=np.float32), np.ones((1, 3), dtype=np.float32)
            )

        @property
        def action_space(self):
            return self._batched

        @property
        def single_action_space(self):
            return inner.action_space

        def call(self, _attr: str, _name: str) -> list[Any]:
            return [inner.config]

        def step(self, actions):
            return (inner.step(actions[0])[0], 0.0, False, False, {})

    class FakeRealWorldEnv:
        def __init__(self, _cfg: Any, **_rest: Any) -> None:
            self._vector = FakeVector()

        @property
        def env(self):
            return self._vector

        @property
        def action_space(self):
            return self._vector.action_space

        def step(self, actions):
            return self._vector.step(actions)

        def reset(self):
            return {"states": np.zeros((1, 3))}, {}

        def close(self) -> None:
            return None

    real_env_module = types.ModuleType("rlinf.envs.real.env")
    real_env_module.RealWorldEnv = FakeRealWorldEnv
    monkeypatch.setitem(sys.modules, "rlinf.envs.real.env", real_env_module)

    import robots.lekiwi.env_server as env_server

    monkeypatch.setattr(env_server, "_local_worker_info", lambda _cfg: None)
    cfg = type("Cfg", (), {"env": type("Env", (), {"eval": None})()})()
    backend = env_server.LeKiwiEnvBackend(
        cfg=cfg,
        controller_config={
            "drive_max_duration_s": 2.0,
            "hold_max_ttl_s": 1.0,
        },
        control_settings={
            "max_linear": 1.0,
            "max_angular": np.radians(1.0),
            "command_timeout": 0.3,
        },
        simulated=True,
    )
    backend.control.start()
    return backend, inner


def test_backend_drive_commands_unbatched_velocity(
    fake_rlinf_realworld_modules, monkeypatch
):
    # Regression: the backend clipped a commanded (3,) velocity against the
    # vector env's batched (1, 3) action space, which broadcast the action to
    # (1, 3) and the base rejected it ("Expected velocity shape (3,)").
    # drive() must send each env a plain (3,) action.
    backend, inner = _fake_backend(monkeypatch)
    try:
        result = backend.drive([0.5, 0.0, 0.0], duration_s=0.1)
    finally:
        backend.close_env()

    assert result["ok"] is True
    assert inner.stepped, "drive() must step the env at least once"
    assert all(shape == (3,) for shape in inner.stepped), inner.stepped
    np.testing.assert_allclose(backend.velocity_limits, [1.0, 1.0, 1.0])


def test_backend_hold_keeps_stepping_until_the_command_expires(
    fake_rlinf_realworld_modules, monkeypatch
):
    # A hold is driven by the backend's control thread, not by the RPC, so
    # the base moves continuously for as long as the command is refreshed.
    backend, inner = _fake_backend(monkeypatch)
    try:
        result = backend.hold([0.5, 0.0, 0.0], ttl_s=0.5)
        assert result["commanded_velocity"] == pytest.approx([0.5, 0.0, 0.0])

        time.sleep(0.2)
        assert len(inner.stepped) >= 3, inner.stepped
        assert all(shape == (3,) for shape in inner.stepped)

        # The control thread already read the sensors in the step it took, so
        # the frame poll serves that copy instead of touching the bus again.
        observation = backend.get_observation()
        assert observation["steps"] >= 1
        np.testing.assert_allclose(observation["states"], np.zeros(3))

        # Idle steps continue sampling feedback, but no motion may be renewed.
        time.sleep(0.4)
        driven = inner.moving_steps
        time.sleep(0.2)
        assert inner.moving_steps == driven, "an expired hold must stop motion"
    finally:
        backend.close_env()


def test_backend_stop_ends_a_hold_before_its_ttl(
    fake_rlinf_realworld_modules, monkeypatch
):
    backend, inner = _fake_backend(monkeypatch)
    try:
        backend.hold([0.5, 0.0, 0.0], ttl_s=1.0)
        time.sleep(0.15)
        backend.stop()
        settled = inner.moving_steps
        time.sleep(0.15)
        assert inner.moving_steps == settled
        assert inner.stops >= 1
    finally:
        backend.close_env()


def test_navigation_mode_uses_steps_and_rejects_old_commands(
    fake_rlinf_realworld_modules, monkeypatch
):
    backend, inner = _fake_backend(monkeypatch)
    try:
        backend.hold([0.1, 0, 0], ttl_s=1)
        backend.control.switch_mode("navigation")
        with pytest.raises(RuntimeError, match="inactive mode"):
            backend.hold([0.1, 0, 0])
        generation = backend.control.navigation_info()["generation"]
        backend.control.arm_navigation(backend.control.stop_revision)
        backend.control.set_velocity(
            generation, [0, 0.1, 0], time.time() + backend.control.command_timeout
        )
        time.sleep(0.05)
        assert inner.velocity[1] > 0
        backend.control.switch_mode("stopping")
        backend.control.switch_mode("manual")
        with pytest.raises(RuntimeError, match="inactive mode"):
            backend.control.set_velocity(generation, [0.1, 0, 0], time.time() + 1)
        assert backend.drive([0.1, 0, 0], duration_s=0.05)["ok"]
    finally:
        backend.close_env()


def test_backend_stop_cancels_goal_without_shutting_down_service(
    fake_rlinf_realworld_modules, monkeypatch
):
    from unittest.mock import Mock

    from robots.lekiwi.navigation.runtime import NavigationRuntime

    backend, inner = _fake_backend(monkeypatch)
    runtime = NavigationRuntime({}, backend.control)
    client, daemon = Mock(), Mock()
    runtime.client, runtime.daemon = client, daemon
    backend.runtime = runtime
    stops = inner.stops

    def cancel(method, **kwargs):
        assert method == "navigation.cancel"
        assert inner.stops == stops + 1
        assert backend.control._target == [0, 0, 0]
        return {"state": "cancelled"}

    client.call.side_effect = cancel
    try:
        assert backend.stop()["ok"]
        client.call.assert_called_once_with("navigation.cancel", timeout_s=5)
        daemon.stop.assert_not_called()
        assert inner.stops == stops + 1
    finally:
        backend.close_env()
    daemon.stop.assert_called_once()
    client.close.assert_called_once()


def test_stereo_rpc_recovers_after_a_missing_frame():
    import queue
    from unittest.mock import Mock

    from robots.lekiwi.env_server import LeKiwiEnvBackend

    stamp = time.monotonic()
    frame = np.full((2, 4, 3), 42, np.uint8)
    camera = Mock()
    camera.get_frame.side_effect = [queue.Empty, (stamp, frame)]
    backend = LeKiwiEnvBackend.__new__(LeKiwiEnvBackend)
    backend.camera = camera
    facade = LeKiwiEnvFacade(backend)
    server = HttpRpcServer(("127.0.0.1", 0), facade._dispatch)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    client = HttpRpcClient(f"http://127.0.0.1:{server.server_port}")
    try:
        assert client.call("env.get_stereo_frame", timeout_s=3) is None
        recovered = client.call("env.get_stereo_frame", timeout_s=3)
        assert recovered["source_stamp"] == stamp
        np.testing.assert_array_equal(recovered["frame"], frame)
        assert recovered["clock_domain"] == "host_monotonic"
        assert camera.get_frame.call_count == 2
        camera.get_frame.assert_called_with(
            timeout=0.5, max_age=0.5, return_timestamp=True
        )
    finally:
        client.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_runtime_cli_defaults_and_path_overrides(monkeypatch, tmp_path):
    import sys

    from robots.lekiwi.env_server import _parse_args
    from robots.lekiwi.runtime_config import LEKIWI_ROOT, runtime_settings_from_args

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["env_server", "--robot-config", "robot.yaml"])
    defaults = runtime_settings_from_args(_parse_args())
    assert defaults["mode"] == "capture"
    assert defaults["ros_domain_id"] == 79
    assert defaults["map_yaml"] == str(tmp_path / "map.yaml")
    assert defaults["ros_python"] is None
    assert defaults["recording"] == {
        "root": str(LEKIWI_ROOT / "local/data/recordings"),
        "sample_fps": 5,
        "maximum_recording_seconds": 600,
    }

    interpreter = tmp_path / "ros-python"
    interpreter.symlink_to(sys.executable)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "env_server",
            "--robot-config",
            "robot.yaml",
            "--mode",
            "navigation",
            "--ros-domain-id",
            "8",
            "--map",
            "maps/room.yaml",
            "--ros-python",
            "ros-python",
            "--recording-dir",
            "captures",
            "--recording-fps",
            "2",
            "--capture-seconds",
            "40",
        ],
    )
    settings = runtime_settings_from_args(_parse_args())
    assert settings["mode"] == "navigation"
    assert settings["ros_domain_id"] == 8
    assert settings["map_yaml"] == str(tmp_path / "maps/room.yaml")
    assert settings["ros_python"] == str(interpreter)
    assert settings["recording"] == {
        "root": str(tmp_path / "captures"),
        "sample_fps": 2,
        "maximum_recording_seconds": 40,
    }


def test_stop_interrupts_drive_even_while_rpc_write_lock_is_held(
    fake_rlinf_realworld_modules, monkeypatch
):
    from concurrent.futures import ThreadPoolExecutor

    backend, inner = _fake_backend(monkeypatch)
    facade = LeKiwiEnvFacade(backend)
    submitted = threading.Event()
    submit = backend.control.submit

    def record_submit(*args, **kwargs):
        result = submit(*args, **kwargs)
        submitted.set()
        return result

    monkeypatch.setattr(backend.control, "submit", record_submit)
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            drive = pool.submit(
                facade._dispatch,
                "env.drive",
                (),
                {"velocity": [0.1, 0, 0], "duration_s": 2.0},
            )
            assert submitted.wait(1)
            with facade._dispatch_lock.write():
                stop = pool.submit(facade._dispatch, "env.stop", (), {})
                assert stop.result(timeout=1)["ok"]
            assert drive.result(timeout=1)["cancelled"]
            np.testing.assert_array_equal(inner.velocity, np.zeros(3))
    finally:
        backend.close_env()
