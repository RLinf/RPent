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

"""Opt-in CPU checks against the companion RLinf camera implementation."""

import os
import queue
import threading
import time
from pathlib import Path
from types import MethodType, SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from robots.lekiwi.env_server import LeKiwiEnvFacade
from rpent.utils.rpc.http_rpc import HttpRpcClient, HttpRpcServer

pytestmark = pytest.mark.skipif(
    os.environ.get("RPENT_RUN_LEKIWI_INTEGRATION") != "1",
    reason="set RPENT_RUN_LEKIWI_INTEGRATION=1 with companion RLinf installed",
)


@pytest.fixture
def camera():
    from rlinf.robotics.parts.cameras.base import BaseCamera

    # Exercise the real RLinf queue reader without starting its hardware thread.
    camera = SimpleNamespace(
        _frame_capturing_start=True,
        _latest_frame=None,
        _frame_queue=queue.Queue(),
        name="front",
        reopen=Mock(),
    )
    camera.get_frame = MethodType(BaseCamera.get_frame, camera)
    return camera


def test_runtime_declares_local_rlinf_hardware(monkeypatch):
    from rlinf.envs.real.lekiwi.base import LeKiwiEnvConfig
    from rlinf.robotics.parts.cameras import Camera

    from robots.lekiwi import runtime_config
    from robots.lekiwi.env_server import _local_worker_info
    from robots.lekiwi.rpent_env import RPentLeKiwiEnv

    example = Path(__file__).parents[3] / "robots/lekiwi/config/example.yaml"
    monkeypatch.setattr(runtime_config, "get_robot_config_path", lambda: example)
    runtime = runtime_config.load_runtime_config(example)
    worker = _local_worker_info(runtime.rlinf)
    hardware = worker.hardware_infos[0].config
    assert worker.cluster_node_rank is None
    assert hardware.node_rank is None
    assert hardware.serial_port == "/dev/ttyACM0"
    env = RPentLeKiwiEnv.__new__(RPentLeKiwiEnv)
    env.hardware = hardware
    env.config = LeKiwiEnvConfig(**runtime.rlinf.env.eval.override_cfg)
    [info] = env._camera_infos()
    assert info.camera_type == "uvc"
    assert info.resolution == (2560, 720)
    assert Camera.backend(info.camera_type) is not None


def test_environment_step_preserves_velocity_before_stop():
    import gymnasium as gym

    from robots.lekiwi.rpent_env import RPentLeKiwiEnv

    velocity = np.array([0.05, 0.0, 2.0], dtype=np.float32)
    stopped = False

    def stop():
        nonlocal stopped
        stopped = True

    def read_state():
        return {"base": {"velocity": np.zeros(3) if stopped else velocity.copy()}}

    env = RPentLeKiwiEnv.__new__(RPentLeKiwiEnv)
    env.config = SimpleNamespace(is_dummy=False, step_frequency=1000, max_num_steps=10)
    env.action_space = gym.spaces.Box(-10.0, 10.0, (3,), dtype=np.float32)
    env.observation_space = gym.spaces.Dict(
        {"state": gym.spaces.Dict({"velocity": env.action_space})}
    )
    env._base_observation_space = env.observation_space
    env._num_steps = 0
    env.robot = SimpleNamespace(send_action=Mock())
    env._state_parts = SimpleNamespace(get_observation=read_state)
    env._base = SimpleNamespace(reset=stop)

    observation, _, _, _, _ = env.step(np.zeros(3, dtype=np.float32))

    assert stopped
    np.testing.assert_array_equal(observation["state"]["velocity"], velocity)
    np.testing.assert_array_equal(env.get_velocity_sample()["velocity"], velocity)
    resting, _ = env.reset()
    np.testing.assert_array_equal(resting["state"]["velocity"], np.zeros(3))
    np.testing.assert_array_equal(env.get_velocity_sample()["velocity"], np.zeros(3))


def test_stereo_rpc_recovers_after_missing_and_stale_frames(camera):
    from robots.lekiwi.env_server import LeKiwiEnvBackend

    backend = LeKiwiEnvBackend.__new__(LeKiwiEnvBackend)
    backend.camera = camera
    facade = LeKiwiEnvFacade(backend)
    server = HttpRpcServer(("127.0.0.1", 0), facade._dispatch)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    client = HttpRpcClient(f"http://127.0.0.1:{server.server_port}")
    try:
        assert client.call("env.get_stereo_frame", timeout_s=3) is None
        camera.reopen.assert_called_once()
        frame = np.full((2, 4, 3), 42, np.uint8)
        stale = time.monotonic() - 2
        camera._latest_frame = (stale, frame)
        camera._frame_queue.put((stale, frame))
        assert client.call("env.get_stereo_frame", timeout_s=3) is None

        camera._frame_queue.put((time.monotonic() - 0.1, frame))
        before = time.monotonic()
        recovered = client.call("env.get_stereo_frame", timeout_s=3)
        assert before <= recovered["source_stamp"] <= time.monotonic()
        np.testing.assert_array_equal(recovered["frame"], frame)
        assert recovered["clock_domain"] == "host_monotonic"
        assert recovered["timestamp_kind"] == "env_read_complete"
    finally:
        client.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


@pytest.mark.parametrize("eye_shape", [(2, 2), (2, 4), (4, 2)])
def test_observation_reads_left_rgb_with_rlinf_camera(camera, eye_shape):
    import gymnasium as gym

    from robots.lekiwi.rpent_env import RPentLeKiwiEnv

    height, width = eye_shape
    stereo = np.full((height, width * 2, 3), 255, np.uint8)
    stereo[:, :width] = np.arange(height * width * 3, dtype=np.uint8).reshape(
        height, width, 3
    )
    camera._latest_frame = (time.monotonic(), stereo)
    env = RPentLeKiwiEnv.__new__(RPentLeKiwiEnv)
    env.config = SimpleNamespace(camera_max_age=0.5)
    env._cameras = [camera]
    env._camera_metadata = {}
    env.observation_space = gym.spaces.Dict(
        {
            "frames": gym.spaces.Dict(
                {"front": gym.spaces.Box(0, 255, (2, 2, 3), np.uint8)}
            )
        }
    )

    before = time.monotonic()
    frames = env._get_camera_frames()

    top, left = (height - 2) // 2, (width - 2) // 2
    np.testing.assert_array_equal(
        frames["front"], stereo[top : top + 2, left : left + 2, ::-1]
    )
    metadata = env._camera_metadata["front"]
    assert before <= metadata["source_stamp"] <= time.monotonic()
    assert metadata["timestamp_kind"] == "env_read_complete"
