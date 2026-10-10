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

"""Offline contracts using installed official RLinf, without Ray or hardware.

Install the pinned official RLinf dependency, then run from the RPent root::

    RPENT_RUN_RLINF_CONTRACT=1 python -m pytest \
        tests/integration_tests/robots/dual_franka/test_rlinf_runtime.py -v

The real environment, observation conversion and Gym registration are exercised.
Only camera/arm/hand readings are faked; hardware setup, connections and Ray
startup fail immediately if accidentally reached. This is not a live-robot test.
"""

from __future__ import annotations

import dataclasses
import os
import queue
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("RPENT_RUN_RLINF_CONTRACT") != "1",
    reason="set RPENT_RUN_RLINF_CONTRACT=1 with the official RLinf installed",
)


@pytest.fixture(autouse=True)
def no_hardware_or_ray(monkeypatch):
    import ray
    from rlinf.envs.real.env import RealWorldEnv
    from rlinf.envs.real.franka.dual_base import DualFrankaEnv
    from rlinf.robotics.parts.base import Connection

    def forbidden(*_args, **_kwargs):
        pytest.fail("offline RLinf contract attempted hardware or Ray startup")

    monkeypatch.setattr(ray, "init", forbidden)
    monkeypatch.setattr(RealWorldEnv, "__init__", forbidden)
    monkeypatch.setattr(RealWorldEnv, "realworld_setup", forbidden)
    monkeypatch.setattr(Connection, "connect", forbidden)
    monkeypatch.setattr(DualFrankaEnv, "_setup_hardware", forbidden)
    monkeypatch.setattr(DualFrankaEnv, "_go_to_rest", forbidden)


@pytest.fixture
def runtime():
    from robots.dual_franka.runtime_config import load_runtime_config

    return load_runtime_config(None, task_description="offline adapter contract")


@pytest.fixture
def dummy_env(runtime):
    import gymnasium as gym
    from omegaconf import OmegaConf
    from rlinf.robotics.discovery import RobotInfo
    from rlinf.robotics.robots.dual_franka import DualFrankaConfig

    from robots.dual_franka.rpent_env import register_rpent_dual_franka_env

    hardware = DualFrankaConfig(
        **OmegaConf.to_container(
            runtime.rlinf.cluster.node_groups[0].hardware.configs[0]
        )
    )
    override = OmegaConf.to_container(runtime.rlinf.env.eval.override_cfg)
    override["is_dummy"] = True
    register_rpent_dual_franka_env()
    env = gym.make(
        runtime.rlinf.env.eval.init_params.id,
        override_cfg=override,
        worker_info=None,
        robot_info=RobotInfo(type="DualFranka", model="DualFranka", config=hardware),
        env_idx=0,
        env_cfg=OmegaConf.to_container(runtime.rlinf.env.eval),
    )
    try:
        yield env
    finally:
        env.close()


class _CameraReading:
    def __init__(self, info, value):
        self.camera_info = info
        self.depth_scale = 0.001
        self.calls = 0
        self.fail = False
        self.frame = np.full((4, 6, 3), [value, value + 1, value + 2], dtype=np.uint8)
        self.depth = np.full((4, 6), 0.42, dtype=np.float32)

    def get_observation(self, **_kwargs):
        self.calls += 1
        if self.fail:
            raise queue.Empty
        reading = {"frame": self.frame.copy()}
        if self.camera_info.enable_depth:
            reading["depth"] = self.depth.copy()
        return reading


@pytest.fixture
def reading_env(dummy_env):
    from rlinf.robotics.parts.arms.franka import FrankaRobotState

    env = dummy_env.unwrapped
    env.config.is_dummy = False
    env._cameras = {
        info.name: _CameraReading(
            dataclasses.replace(info, enable_depth=True), value=10 + index * 10
        )
        for index, info in enumerate(env._camera_infos())
    }
    env._last_camera_frame = {}
    env.camera_player = SimpleNamespace(
        put_frame=lambda _frames: None, stop=lambda: None
    )
    left = FrankaRobotState(tcp_pose=np.array([0.4, -0.2, 0.3, 0, 0, 0, 1]))
    right = FrankaRobotState(tcp_pose=np.array([0.5, 0.2, 0.4, 0, 0, 0, 1]))
    env._left_arm = SimpleNamespace(get_state=lambda: left)
    env._right_arm = SimpleNamespace(get_state=lambda: right)
    env._left_hand = SimpleNamespace(position=0.25, is_open=False)
    env._right_hand = SimpleNamespace(position=0.75, is_open=True)
    env._arm_executors = (
        ThreadPoolExecutor(max_workers=1),
        ThreadPoolExecutor(max_workers=1),
    )
    env.get_tcp_pose()
    return env


def test_official_config_and_registered_dummy_environment(runtime, dummy_env):
    from rlinf.envs.real.franka.dual_franka_tcp import DualFrankaTCPEnv
    from rlinf.robotics.robots.dual_franka import DualFrankaConfig

    from robots.dual_franka.rpent_env import RPentDualFrankaEnv

    assert not hasattr(DualFrankaTCPEnv, "get_raw_camera_snapshot")
    assert "left_camera_node_rank" not in {
        field.name for field in dataclasses.fields(DualFrankaConfig)
    }
    assert isinstance(dummy_env.unwrapped, RPentDualFrankaEnv)
    assert isinstance(dummy_env.unwrapped, DualFrankaTCPEnv)
    assert dummy_env.action_space.shape == (20,)
    obs, _ = dummy_env.reset(seed=7)
    assert dummy_env.observation_space.contains(obs)
    assert obs["state"]["tcp_pose_rot6d"].shape == (18,)
    assert set(obs["frames"]) == {"base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"}
    assert runtime.rlinf.env.eval.auto_reset is False
    _, _, _, _, _ = dummy_env.step(dummy_env.action_space.sample())


def test_raw_snapshot_matches_official_policy_processing(reading_env):
    from rlinf.envs.real.franka.dual_franka_tcp import DualFrankaTCPEnv

    env = reading_env
    frames, depths = env._get_camera_observation()
    snapshot = env.get_raw_camera_snapshot()
    assert all(camera.calls == 1 for camera in env._cameras.values())
    for name, camera in env._cameras.items():
        np.testing.assert_array_equal(
            snapshot["raw_frames"][name], camera.frame[..., ::-1]
        )
        np.testing.assert_array_equal(snapshot["raw_depths"][name], camera.depth)
        assert snapshot["camera_meta"][name]["depth_unit"] == "m"

    original_frames, original_depths = DualFrankaTCPEnv._get_camera_observation(env)
    for name in frames:
        np.testing.assert_array_equal(frames[name], original_frames[name])
        np.testing.assert_array_equal(depths[name], original_depths[name])
    name = next(iter(frames))
    snapshot["raw_frames"][name][:] = 0
    assert np.any(env.get_raw_camera_snapshot()["raw_frames"][name])
    assert all(camera.calls == 2 for camera in env._cameras.values())


def test_camera_recovery_never_reuses_stale_projection_snapshot(reading_env):
    env = reading_env
    with pytest.raises(RuntimeError, match="No fresh complete camera snapshot"):
        env.get_raw_camera_snapshot()
    env._get_camera_observation()
    camera = next(iter(env._cameras.values()))
    camera.fail = True
    policy_frames, _ = env._get_camera_observation()
    assert policy_frames
    with pytest.raises(RuntimeError, match="No fresh complete camera snapshot"):
        env.get_raw_camera_snapshot()
    camera.fail = False
    camera.frame[:] = 77
    env._get_camera_observation()
    name = camera.camera_info.name
    assert np.all(env.get_raw_camera_snapshot()["raw_frames"][name] == 77)


@pytest.mark.parametrize("with_depth", [False, True])
def test_extra_camera_uses_official_depth_conversion_once(monkeypatch, with_depth):
    from rlinf.robotics.parts.cameras import CameraInfo, RealSenseCamera

    from robots.dual_franka.env_server import _create_worker_class
    from robots.franka import rpent_env

    camera = RealSenseCamera(
        CameraInfo("extra_rgb", "offline-camera", enable_depth=with_depth)
    )
    camera._depth_scale = 0.001
    frame = np.full((4, 6, 3), [10, 20, 30], dtype=np.uint8)
    if with_depth:
        frame = np.concatenate(
            [frame, np.full((4, 6, 1), 420, dtype=np.float32)], axis=-1
        )
    calls = []

    def get_frame(**kwargs):
        calls.append(kwargs["timeout"])
        return frame.copy()

    monkeypatch.setattr(camera, "get_frame", get_frame)
    intrinsics = {"width": 6, "height": 4, "fx": 10.0, "fy": 11.0}
    monkeypatch.setattr(rpent_env, "realsense_color_intrinsics", lambda _: intrinsics)
    worker = object.__new__(_create_worker_class())
    worker._perception_cameras = {"extra": camera}
    worker._perception_camera_meta = {}

    result = worker._capture_perception_camera_snapshot()

    assert calls == [2]
    rgb = result["raw_frames"]["extra_rgb"]
    assert rgb.shape == (4, 6, 3)
    assert rgb.dtype == np.uint8
    assert rgb[0, 0].tolist() == [30, 20, 10]
    assert result["camera_meta"]["extra_rgb"]["color_intrinsics"] == intrinsics
    if with_depth:
        depth = result["raw_depths"]["extra_rgb"]
        assert depth.shape == (4, 6)
        assert depth.dtype == np.float32
        np.testing.assert_allclose(depth, 0.42)
    else:
        assert result["raw_depths"] == {}
        assert result["camera_meta"]["extra_rgb"]["depth_available"] is False


def test_worker_observation_uses_official_wrapper_and_latest_acquisition(reading_env):
    import gymnasium as gym
    from rlinf.envs.real.env import RealWorldEnv

    from robots.dual_franka.env_server import _create_worker_class

    env = reading_env
    vector_env = gym.vector.SyncVectorEnv([lambda: env])
    # Bypass only the process/hardware-owning constructor, not conversion logic.
    wrapped = object.__new__(RealWorldEnv)
    wrapped.env = vector_env
    wrapped.main_image_key = "left_wrist_0_rgb"
    wrapped.task_descriptions = [env.task_description]
    worker = object.__new__(_create_worker_class())
    worker.env = wrapped
    worker._perception_cameras = {}
    worker._perception_camera_meta = {}
    try:
        obs = worker.get_observation()
        assert obs["states"].shape == (20,)
        np.testing.assert_allclose(obs["states"][:2], [0.25, 0.75])
        assert obs["main_images"].shape == (224, 224, 3)
        assert obs["extra_view_images"].shape == (2, 224, 224, 3)
        assert obs["main_depths"].shape == (224, 224)
        assert all(camera.calls == 1 for camera in env._cameras.values())
        for camera in env._cameras.values():
            camera.frame[:] = 99
        refreshed = worker.get_observation()
        assert np.all(refreshed["main_images"] == 99)
        assert all(
            np.all(frame == 99) for frame in refreshed["raw_camera_frames"].values()
        )
        assert all(camera.calls == 2 for camera in env._cameras.values())
        assert not np.all(obs["main_images"] == 99)
    finally:
        vector_env.close()


@pytest.mark.parametrize("is_open", [False, True])
def test_official_zero_gripper_command_leaves_either_hand_state_unchanged(
    reading_env, is_open
):
    def forbidden():
        pytest.fail("zero gripper command issued an open/close action")

    hand = SimpleNamespace(is_open=is_open, open=forbidden, close=forbidden)
    assert reading_env._gripper_action(0, hand, 0.0) is False
    assert hand.is_open is is_open


def test_remote_camera_metadata_uses_official_proxy_without_local_sdk_profile(
    dummy_env,
):
    from rlinf.robotics.parts.cameras import Camera, CameraInfo, RealSenseCamera
    from rlinf.robotics.placement.handles import PartWorkerHost, remote_view_of

    from robots.dual_franka.rpent_env import RPentRealSenseCamera

    info = CameraInfo(
        name="right_wrist_0_rgb",
        serial_number="remote-camera",
        camera_type="rpent_realsense",
        enable_depth=True,
    )
    intrinsics = {
        "width": 640,
        "height": 480,
        "fx": 600.0,
        "fy": 601.0,
        "ppx": 320.0,
        "ppy": 240.0,
    }
    calls = []

    class Reply:
        def __init__(self, value):
            self.value = value

        def wait(self):
            return [self.value]

    class MetadataTransport:
        def attribute(self, name):
            calls.append(("attribute", name))
            return Reply({"camera_info": info, "depth_scale": 0.001}[name])

        def get_color_intrinsics(self):
            calls.append(("method", "get_color_intrinsics"))
            return Reply(intrinsics)

    camera = RPentRealSenseCamera(info, node_rank=1)
    camera._group = MetadataTransport()
    # Connection.connect changes the declared part to this same official view;
    # only the worker-group transport is replaced, without opening hardware.
    camera.__class__ = remote_view_of(RPentRealSenseCamera)
    assert isinstance(camera, RPentRealSenseCamera)
    assert not hasattr(camera, "profile")
    assert Camera.backend("realsense") is RealSenseCamera
    worker_cls = PartWorkerHost.worker_class(RPentRealSenseCamera)
    assert "get_color_intrinsics" in worker_cls.__dict__

    env = dummy_env.unwrapped
    env._cameras = {info.name: camera}
    env._rpent_camera_snapshot = {
        "raw_frames": {info.name: np.zeros((480, 640, 3), dtype=np.uint8)},
        "raw_depths": {info.name: np.full((480, 640), 0.4, dtype=np.float32)},
    }
    metadata = env.get_raw_camera_metadata()[info.name]
    assert metadata["serial_number"] == "remote-camera"
    assert metadata["camera_type"] == "realsense"
    assert metadata["depth_scale"] == 0.001
    assert metadata["color_intrinsics"] == intrinsics
    assert metadata["projection_supported"] is True
    assert calls == [
        ("attribute", "camera_info"),
        ("method", "get_color_intrinsics"),
        ("attribute", "depth_scale"),
    ]
