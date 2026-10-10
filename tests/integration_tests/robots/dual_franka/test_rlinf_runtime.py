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

"""Official RLinf contracts, with only hardware readings/transport replaced.

Run with RPENT_RUN_RLINF_CONTRACT=1 after installing the official Franka extra.
This suite never starts Ray or connects to robot/camera hardware.
"""

import dataclasses
import os
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("RPENT_RUN_RLINF_CONTRACT") != "1",
    reason="requires the installed official RLinf and explicit opt-in",
)


@pytest.fixture(autouse=True)
def no_hardware_or_ray(monkeypatch):
    import ray
    from rlinf.envs.real.env import RealWorldEnv
    from rlinf.envs.real.franka.dual_base import DualFrankaEnv
    from rlinf.robotics.parts.base import Connection

    def forbidden(*args, **kwargs):
        pytest.fail("offline contract attempted hardware or Ray startup")

    monkeypatch.setattr(ray, "init", forbidden)
    monkeypatch.setattr(RealWorldEnv, "__init__", forbidden)
    monkeypatch.setattr(RealWorldEnv, "realworld_setup", forbidden)
    monkeypatch.setattr(Connection, "connect", forbidden)
    monkeypatch.setattr(DualFrankaEnv, "_setup_hardware", forbidden)
    monkeypatch.setattr(DualFrankaEnv, "_go_to_rest", forbidden)


@pytest.fixture
def dummy_env():
    import gymnasium as gym
    from omegaconf import OmegaConf
    from rlinf.robotics.discovery import RobotInfo
    from rlinf.robotics.robots.dual_franka import DualFrankaConfig

    from robots.dual_franka.rpent_env import register_rpent_dual_franka_env
    from robots.dual_franka.runtime_config import load_runtime_config

    runtime = load_runtime_config(None, task_description="offline contract")
    assert runtime.rlinf.env.eval.auto_reset is False
    hardware = DualFrankaConfig(
        **OmegaConf.to_container(
            runtime.rlinf.cluster.node_groups[0].hardware.configs[0]
        )
    )
    override = OmegaConf.to_container(runtime.rlinf.env.eval.override_cfg)
    override["is_dummy"] = True
    register_rpent_dual_franka_env()
    register_rpent_dual_franka_env()
    env = gym.make(
        runtime.rlinf.env.eval.init_params.id,
        override_cfg=override,
        worker_info=None,
        env_idx=0,
        robot_info=RobotInfo(type="DualFranka", model="DualFranka", config=hardware),
        env_cfg=OmegaConf.to_container(runtime.rlinf.env.eval),
    )
    obs, _ = env.reset(seed=7)
    assert env.action_space.shape == (20,)
    assert env.observation_space.contains(obs)
    env.step(np.zeros(20, dtype=np.float32))
    try:
        yield env
    finally:
        env.close()


class CameraReading:
    def __init__(self, info, value):
        self.camera_info = info
        self.depth_scale = 0.001
        self.frame = np.full((4, 6, 3), [value, value + 1, value + 2], dtype=np.uint8)

    def get_observation(self, **kwargs):
        return {
            "frame": self.frame.copy(),
            "depth": np.full((4, 6), 0.42, dtype=np.float32),
        }


@pytest.fixture
def reading_env(dummy_env):
    from rlinf.robotics.parts.arms.franka import FrankaRobotState

    env = dummy_env.unwrapped
    env.config.is_dummy = False
    env._cameras = {
        info.name: CameraReading(dataclasses.replace(info, enable_depth=True), 10 + i)
        for i, info in enumerate(env._camera_infos())
    }
    env._last_camera_frame = {}
    env.camera_player = SimpleNamespace(
        put_frame=lambda frames: None, stop=lambda: None
    )
    for side in ("left", "right"):
        state = FrankaRobotState(tcp_pose=np.array([0.4, 0.2, 0.3, 0, 0, 0, 1]))
        setattr(env, f"_{side}_arm", SimpleNamespace(get_state=lambda: state))
        setattr(env, f"_{side}_hand", SimpleNamespace(position=0.5, is_open=True))
    env._arm_executors = tuple(ThreadPoolExecutor(max_workers=1) for _ in range(2))
    env.get_tcp_pose()
    return env


def test_raw_policy_and_worker_observations_use_latest_acquisition(reading_env):
    import gymnasium as gym
    from rlinf.envs.real.env import RealWorldEnv

    from robots.dual_franka.env_server import _create_worker_class

    env = reading_env
    env._get_camera_observation()
    snapshot = env.get_raw_camera_snapshot()
    for name, camera in env._cameras.items():
        np.testing.assert_array_equal(
            snapshot["raw_frames"][name], camera.frame[..., ::-1]
        )
        np.testing.assert_allclose(snapshot["raw_depths"][name], 0.42)
    vector = gym.vector.SyncVectorEnv([lambda: env])
    wrapped = object.__new__(RealWorldEnv)
    wrapped.env = vector
    wrapped.main_image_key = "left_wrist_0_rgb"
    wrapped.task_descriptions = [env.task_description]
    worker = object.__new__(_create_worker_class())
    worker.env = wrapped
    worker._perception_cameras = {}
    worker._perception_camera_meta = {}
    try:
        obs = worker.get_observation()
        assert obs["states"].shape == (20,)
        assert obs["main_images"].shape == (224, 224, 3)
        assert obs["extra_view_images"].shape == (2, 224, 224, 3)
        assert obs["main_depths"].shape == (224, 224)
        for camera in env._cameras.values():
            camera.frame[:] = 99
        fresh = worker.get_observation()
        assert np.all(fresh["main_images"] == 99)
        assert all(np.all(frame == 99) for frame in fresh["raw_camera_frames"].values())
        assert not np.all(obs["main_images"] == 99)
    finally:
        vector.close()
