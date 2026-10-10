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

"""Offline observation tests without starting robot, camera or Ray resources."""

from __future__ import annotations

import importlib.util
import logging
import queue
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace

import cv2
import gymnasium as gym
import numpy as np
import pytest


@pytest.fixture
def adapter(fake_rlinf_realworld_modules, monkeypatch):
    cameras = sys.modules["rlinf.robotics.parts.cameras"]
    official_realsense = cameras.RealSenseCamera

    @dataclass
    class CameraInfo:
        name: str
        serial_number: str
        camera_type: str = "realsense"
        resolution: tuple[int, int] = (640, 480)
        fps: int = 15
        enable_depth: bool = False
        crop_region: tuple[float, float, float, float] | None = None

    class Camera:
        backends = {"realsense": official_realsense}

        @classmethod
        def register(cls, name):
            def add(driver):
                assert name not in cls.backends
                cls.backends[name] = driver
                return driver

            return add

    class DualFrankaTCPEnv:
        def __init__(self, **kwargs):
            self.constructor_args = kwargs
            self.closed = False

        def _camera_infos(self):
            return self.declarations

        def _crop_frame(self, frame, size, interpolation=cv2.INTER_LINEAR):
            height, width = frame.shape[:2]
            side = min(height, width)
            x, y = (width - side) // 2, (height - side) // 2
            cropped = frame[y : y + side, x : x + side]
            return cropped, cv2.resize(cropped, size, interpolation=interpolation)

        def close(self):
            self.closed = True

    monkeypatch.setattr(cameras, "CameraInfo", CameraInfo)
    monkeypatch.setattr(cameras, "Camera", Camera, raising=False)
    monkeypatch.setattr(
        sys.modules["rlinf.envs.real.franka.dual_franka_tcp"],
        "DualFrankaTCPEnv",
        DualFrankaTCPEnv,
        raising=False,
    )
    path = Path(__file__).parents[4] / "robots/dual_franka/rpent_env.py"
    spec = importlib.util.spec_from_file_location("_test_dual_franka_adapter", path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def make_camera(adapter, name, readings, *, depth=False, realsense=False):
    camera = adapter.RPentRealSenseCamera() if realsense else SimpleNamespace()
    camera.camera_info = adapter.CameraInfo(
        name,
        f"serial-{name}",
        camera_type="rpent_realsense" if realsense else "lumos",
        resolution=(6, 4),
        enable_depth=depth,
    )
    camera.depth_scale = 0.001 if depth else 1.0
    camera.calls = []
    observations = iter(readings)

    def read(**kwargs):
        camera.calls.append(kwargs)
        result = next(observations)
        if isinstance(result, Exception):
            raise result
        return result

    camera.get_observation = read
    if realsense:
        camera.get_color_intrinsics = lambda: {
            "width": 6,
            "height": 4,
            "fx": 5.0,
            "fy": 5.0,
            "ppx": 3.0,
            "ppy": 2.0,
        }
    return camera


def make_env(adapter, cameras):
    env = adapter.RPentDualFrankaEnv()
    env._cameras = cameras
    env._last_camera_frame = {}
    env._logger = logging.getLogger(__name__)
    env.observation_space = {
        "frames": {name: SimpleNamespace(shape=(2, 2, 3)) for name in cameras}
    }
    env.camera_player = SimpleNamespace(put_frame=lambda frames: None)
    return env


def test_camera_declarations_preserve_names_order_depth_and_official_backend(adapter):
    env = adapter.RPentDualFrankaEnv()
    original = [
        adapter.CameraInfo("base_0_rgb", "base"),
        adapter.CameraInfo("left_wrist_0_rgb", "left", camera_type="lumos"),
        adapter.CameraInfo("right_wrist_0_rgb", "right", camera_type="lumos"),
    ]
    env.declarations = original
    declarations = env._camera_infos()
    assert [item.name for item in declarations] == [item.name for item in original]
    assert [item.serial_number for item in declarations] == ["base", "left", "right"]
    assert not any(item.enable_depth for item in declarations)
    assert declarations[0].camera_type == "rpent_realsense"
    assert original[0].camera_type == "realsense"
    assert declarations[1:] == original[1:]
    assert adapter.Camera.backends["realsense"] is adapter.RealSenseCamera


def test_policy_and_raw_rgbd_share_one_acquisition_and_snapshot_owns_arrays(adapter):
    bgr = np.arange(72, dtype=np.uint8).reshape(4, 6, 3)
    depth = np.arange(24, dtype=np.float32).reshape(4, 6) / 100.0
    camera = make_camera(
        adapter,
        "base_0_rgb",
        [{"frame": bgr, "depth": depth}],
        depth=True,
        realsense=True,
    )
    env = make_env(adapter, {"base_0_rgb": camera})
    frames, depths = env._get_camera_observation()
    snapshot = env.get_raw_camera_snapshot()
    np.testing.assert_array_equal(
        frames["base_0_rgb"], cv2.resize(bgr[:, 1:5], (2, 2))[..., ::-1]
    )
    np.testing.assert_array_equal(
        depths["base_0_rgb"],
        cv2.resize(depth[:, 1:5], (2, 2), interpolation=cv2.INTER_NEAREST),
    )
    np.testing.assert_array_equal(snapshot["raw_frames"]["base_0_rgb"], bgr[..., ::-1])
    np.testing.assert_array_equal(snapshot["raw_depths"]["base_0_rgb"], depth)
    meta = snapshot["camera_meta"]["base_0_rgb"]
    assert meta["depth_unit"] == "m"
    assert meta["rgb_shape"] == [4, 6, 3]
    assert meta["color_intrinsics"]["width"] == 6
    assert meta["projection_supported"] is True
    snapshot["raw_frames"]["base_0_rgb"][:] = 0
    snapshot["raw_depths"]["base_0_rgb"][:] = 0
    again = env.get_raw_camera_snapshot()
    np.testing.assert_array_equal(again["raw_frames"]["base_0_rgb"], bgr[..., ::-1])
    np.testing.assert_array_equal(again["raw_depths"]["base_0_rgb"], depth)
    assert camera.calls == [{"timeout": 0.5, "max_age": 0.5}]


def test_stalled_camera_keeps_policy_fallback_but_invalidates_raw_snapshot(adapter):
    first = {"frame": np.full((4, 6, 3), 15, dtype=np.uint8)}
    recovered = {"frame": np.full((4, 6, 3), 80, dtype=np.uint8)}
    camera = make_camera(
        adapter, "right_wrist_0_rgb", [first, queue.Empty(), recovered]
    )
    env = make_env(adapter, {"right_wrist_0_rgb": camera})
    env._get_camera_observation()
    assert env.get_raw_camera_snapshot()["raw_frames"]["right_wrist_0_rgb"].max() == 15
    frames, _ = env._get_camera_observation()
    assert frames["right_wrist_0_rgb"].max() == 15
    with pytest.raises(RuntimeError, match="No fresh complete camera snapshot"):
        env.get_raw_camera_snapshot()
    with pytest.raises(RuntimeError, match="No fresh complete camera snapshot"):
        env.get_raw_camera_metadata()
    env._get_camera_observation()
    assert env.get_raw_camera_snapshot()["raw_frames"]["right_wrist_0_rgb"].max() == 80


def test_failed_first_acquisition_has_no_fallback_or_partial_snapshot(adapter):
    valid = {"frame": np.zeros((4, 6, 3), dtype=np.uint8)}
    env = make_env(
        adapter,
        {
            "left_wrist_0_rgb": make_camera(adapter, "left_wrist_0_rgb", [valid]),
            "right_wrist_0_rgb": make_camera(
                adapter, "right_wrist_0_rgb", [queue.Empty()]
            ),
        },
    )
    with pytest.raises(RuntimeError, match="right_wrist_0_rgb stalled"):
        env._get_camera_observation()
    with pytest.raises(RuntimeError, match="No fresh complete camera snapshot"):
        env.get_raw_camera_snapshot()


def test_rgb_only_camera_does_not_claim_depth_or_projection_support(adapter):
    env = make_env(
        adapter,
        {
            "left_wrist_0_rgb": make_camera(
                adapter,
                "left_wrist_0_rgb",
                [{"frame": np.zeros((4, 6, 3), dtype=np.uint8)}],
            )
        },
    )
    _, depths = env._get_camera_observation()
    meta = env.get_raw_camera_metadata()["left_wrist_0_rgb"]
    assert depths == {}
    assert meta["camera_type"] == "lumos"
    assert meta["color_intrinsics"] is None
    assert meta["depth_available"] is False
    assert meta["depth_unit"] is None
    assert meta["projection_supported"] is False


def test_realsense_metadata_reads_only_the_owned_profile(adapter, monkeypatch):
    intrinsics = SimpleNamespace(
        width=640,
        height=480,
        fx=600.0,
        fy=601.0,
        ppx=320.0,
        ppy=240.0,
        model="brown_conrady",
        coeffs=[0.0] * 5,
    )
    profile = SimpleNamespace(get_intrinsics=lambda: intrinsics)
    profile.as_video_stream_profile = lambda: profile
    realsense = ModuleType("pyrealsense2")
    realsense.stream = SimpleNamespace(color=object())
    monkeypatch.setitem(sys.modules, "pyrealsense2", realsense)
    camera = adapter.RPentRealSenseCamera()
    camera.profile = SimpleNamespace(get_stream=lambda stream: profile)
    metadata = camera.get_color_intrinsics()
    assert metadata["fx"] == 600.0
    assert metadata["fy"] == 601.0
    assert metadata["width"] == 640
    assert metadata["coeffs"] == [0.0] * 5


def test_factory_uses_standard_wrapper_stack_and_closes_on_failure(
    adapter, monkeypatch
):
    seen = []
    wrapped = object()

    def build(env, config):
        seen.append((env, config))
        return wrapped

    monkeypatch.setattr(adapter, "build_stack", build)
    kwargs = {
        "override_cfg": {},
        "worker_info": None,
        "robot_info": None,
        "env_idx": 0,
        "env_cfg": {},
    }
    assert adapter.create_rpent_dual_franka_env(**kwargs) is wrapped
    assert seen[0][0].constructor_args == {
        key: value for key, value in kwargs.items() if key != "env_cfg"
    }

    def fail(env, config):
        seen.append((env, config))
        raise ValueError("invalid wrapper config")

    monkeypatch.setattr(adapter, "build_stack", fail)
    with pytest.raises(ValueError, match="invalid wrapper config"):
        adapter.create_rpent_dual_franka_env(**kwargs)
    assert seen[-1][0].closed


def test_registration_is_idempotent_and_keeps_official_id(adapter, monkeypatch):
    env_id = "RPentDualFrankaTCPEnv-v1"
    monkeypatch.delitem(gym.registry, env_id, raising=False)
    before = dict(gym.registry)
    adapter.register_rpent_dual_franka_env()
    adapter.register_rpent_dual_franka_env()
    assert gym.spec(env_id).entry_point == (
        "robots.dual_franka.rpent_env:create_rpent_dual_franka_env"
    )
    assert {
        key: value for key, value in gym.registry.items() if key != env_id
    } == before
    monkeypatch.delitem(gym.registry, env_id)
