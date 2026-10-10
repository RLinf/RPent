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
import queue
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
    try:
        yield env
    finally:
        env.close()


class CameraReading:
    def __init__(self, info, value):
        self.camera_info = info
        self.depth_scale = 0.001
        self.calls = 0
        self.fail = False
        self.frame = np.full((4, 6, 3), [value, value + 1, value + 2], dtype=np.uint8)

    def get_observation(self, **kwargs):
        self.calls += 1
        if self.fail:
            raise queue.Empty
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


def test_registered_environment_uses_official_wrappers_and_closes_on_failure(
    dummy_env,
    monkeypatch,
):
    import gymnasium as gym

    from robots.dual_franka import rpent_env as adapter

    obs, _ = dummy_env.reset(seed=7)
    assert dummy_env.action_space.shape == (20,)
    assert dummy_env.observation_space.contains(obs)
    assert obs["state"]["tcp_pose_rot6d"].shape == (18,)
    dummy_env.step(np.zeros(20, dtype=np.float32))
    closed = []
    original_close = adapter.RPentDualFrankaEnv.close

    def close(env):
        closed.append(True)
        original_close(env)

    def fail(*args):
        raise ValueError("invalid wrapper config")

    monkeypatch.setattr(adapter.RPentDualFrankaEnv, "close", close)
    monkeypatch.setattr(adapter, "build_stack", fail)
    with pytest.raises(ValueError, match="invalid wrapper config"):
        gym.make(dummy_env.spec.id, **dummy_env.spec.kwargs)
    assert closed == [True]


def test_raw_policy_and_worker_observations_use_latest_acquisition(reading_env):
    import gymnasium as gym
    from rlinf.envs.real.env import RealWorldEnv
    from rlinf.envs.real.franka.dual_franka_tcp import DualFrankaTCPEnv

    from robots.dual_franka.env_server import _create_worker_class

    env = reading_env
    frames, depths = env._get_camera_observation()
    snapshot = env.get_raw_camera_snapshot()
    assert all(camera.calls == 1 for camera in env._cameras.values())
    for name, camera in env._cameras.items():
        np.testing.assert_array_equal(
            snapshot["raw_frames"][name], camera.frame[..., ::-1]
        )
        np.testing.assert_allclose(snapshot["raw_depths"][name], 0.42)
    official_frames, official_depths = DualFrankaTCPEnv._get_camera_observation(env)
    for name in frames:
        np.testing.assert_array_equal(frames[name], official_frames[name])
        np.testing.assert_array_equal(depths[name], official_depths[name])

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


@pytest.mark.parametrize("with_depth", [False, True])
def test_extra_camera_preserves_official_color_and_depth_units(monkeypatch, with_depth):
    from rlinf.robotics.parts.cameras import CameraInfo, RealSenseCamera

    from robots.dual_franka.env_server import _create_worker_class
    from robots.franka import rpent_env

    camera = RealSenseCamera(
        CameraInfo("extra_rgb", "offline", enable_depth=with_depth)
    )
    camera._depth_scale = 0.001
    frame = np.full((4, 6, 3), [10, 20, 30], dtype=np.uint8)
    if with_depth:
        frame = np.concatenate(
            [frame, np.full((4, 6, 1), 420, dtype=np.float32)], axis=-1
        )
    monkeypatch.setattr(camera, "get_frame", lambda **kwargs: frame.copy())
    monkeypatch.setattr(rpent_env, "realsense_color_intrinsics", lambda _: {"fx": 10.0})
    worker = object.__new__(_create_worker_class())
    worker._perception_cameras = {"extra": camera}
    worker._perception_camera_meta = {}
    result = worker._capture_perception_camera_snapshot()
    assert result["raw_frames"]["extra_rgb"][0, 0].tolist() == [30, 20, 10]
    if with_depth:
        np.testing.assert_allclose(result["raw_depths"]["extra_rgb"], 0.42)
    else:
        assert result["raw_depths"] == {}


@pytest.mark.parametrize("is_open", [False, True])
def test_zero_gripper_command_keeps_either_hand_state(reading_env, is_open):
    def forbidden():
        pytest.fail("zero command issued an open/close action")

    hand = SimpleNamespace(is_open=is_open, open=forbidden, close=forbidden)
    assert reading_env._gripper_action(0, hand, 0.0) is False
    assert hand.is_open is is_open


def test_remote_camera_metadata_uses_official_proxy(dummy_env):
    from rlinf.robotics.parts.cameras import CameraInfo
    from rlinf.robotics.placement.handles import remote_view_of

    from robots.dual_franka.rpent_env import RPentRealSenseCamera

    info = CameraInfo(
        "right_wrist_0_rgb",
        "remote-camera",
        camera_type="rpent_realsense",
        enable_depth=True,
    )
    intrinsics = {"fx": 600.0, "fy": 601.0}

    def reply(value):
        return SimpleNamespace(wait=lambda: [value])

    camera = RPentRealSenseCamera(info, node_rank=1)
    camera._group = SimpleNamespace(
        attribute=lambda name: reply({"camera_info": info, "depth_scale": 0.001}[name]),
        get_color_intrinsics=lambda: reply(intrinsics),
    )
    camera.__class__ = remote_view_of(RPentRealSenseCamera)
    env = dummy_env.unwrapped
    env._cameras = {info.name: camera}
    env._rpent_camera_snapshot = {
        "raw_frames": {info.name: np.zeros((2, 2, 3), dtype=np.uint8)},
        "raw_depths": {info.name: np.full((2, 2), 0.4, dtype=np.float32)},
    }
    metadata = env.get_raw_camera_metadata()[info.name]
    assert metadata["serial_number"] == "remote-camera"
    assert metadata["color_intrinsics"] == intrinsics
    assert metadata["projection_supported"] is True
