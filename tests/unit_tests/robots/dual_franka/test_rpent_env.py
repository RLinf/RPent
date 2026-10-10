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

"""Core RGB-D regressions without optional RLinf hardware dependencies."""

import importlib.util
import logging
import queue
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest


@pytest.fixture
def adapter(fake_rlinf_realworld_modules, monkeypatch):
    cameras = sys.modules["rlinf.robotics.parts.cameras"]
    monkeypatch.setattr(
        cameras,
        "Camera",
        SimpleNamespace(register=lambda name: lambda driver: driver),
        raising=False,
    )
    monkeypatch.setattr(
        sys.modules["rlinf.envs.real.franka.dual_franka_tcp"],
        "DualFrankaTCPEnv",
        object,
        raising=False,
    )
    path = Path(__file__).parents[4] / "robots/dual_franka/rpent_env.py"
    spec = importlib.util.spec_from_file_location("_test_dual_franka_adapter", path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def camera_env(adapter):
    camera = adapter.RPentRealSenseCamera()
    camera.camera_info = SimpleNamespace(
        serial_number="offline",
        camera_type="rpent_realsense",
        enable_depth=True,
    )
    camera.depth_scale = 0.001
    camera.get_color_intrinsics = lambda: {"fx": 5.0, "fy": 5.0}
    env = adapter.RPentDualFrankaEnv()
    env._cameras = {"base_0_rgb": camera}
    env._last_camera_frame = {}
    env._logger = logging.getLogger(__name__)
    env.observation_space = {"frames": {"base_0_rgb": SimpleNamespace(shape=(2, 2, 3))}}
    # Official crop/wrapper behavior is exercised by the real-RLinf contract.
    env._crop_frame = lambda frame, size, **kwargs: (frame, cv2.resize(frame, size))
    env.camera_player = SimpleNamespace(put_frame=lambda frames: None)
    return camera, env


@pytest.mark.parametrize("with_depth", [False, True])
def test_policy_and_projection_share_one_read_and_own_snapshot_arrays(
    camera_env,
    with_depth,
):
    camera, env = camera_env
    bgr = np.full((4, 6, 3), [10, 20, 30], dtype=np.uint8)
    depth = np.full((4, 6), 0.42, dtype=np.float32)
    calls = []

    def read(**kwargs):
        calls.append(kwargs)
        return {"frame": bgr, **({"depth": depth} if with_depth else {})}

    camera.get_observation = read
    env._get_camera_observation()
    snapshot = env.get_raw_camera_snapshot()
    np.testing.assert_array_equal(snapshot["raw_frames"]["base_0_rgb"], bgr[..., ::-1])
    metadata = snapshot["camera_meta"]["base_0_rgb"]
    assert metadata["depth_available"] is with_depth
    assert metadata["projection_supported"] is with_depth
    if with_depth:
        np.testing.assert_allclose(snapshot["raw_depths"]["base_0_rgb"], 0.42)
        assert metadata["depth_unit"] == "m"
        snapshot["raw_depths"]["base_0_rgb"][:] = 0
        np.testing.assert_allclose(
            env.get_raw_camera_snapshot()["raw_depths"]["base_0_rgb"],
            0.42,
        )
    else:
        assert snapshot["raw_depths"] == {}
    snapshot["raw_frames"]["base_0_rgb"][:] = 0
    bgr[:] = 99
    assert env.get_raw_camera_snapshot()["raw_frames"]["base_0_rgb"][0, 0].tolist() == [
        30,
        20,
        10,
    ]
    assert len(calls) == 1


@pytest.mark.parametrize("cached", [False, True])
def test_camera_failure_invalidates_projection_until_recovery(camera_env, cached):
    camera, env = camera_env
    frame = np.full((4, 6, 3), 15, dtype=np.uint8)
    camera.get_observation = lambda **kwargs: {"frame": frame}
    if cached:
        env._get_camera_observation()

    def stalled(**kwargs):
        raise queue.Empty

    camera.get_observation = stalled
    if cached:
        frames, _ = env._get_camera_observation()
        assert frames["base_0_rgb"].max() == 15
    else:
        with pytest.raises(RuntimeError, match="stalled"):
            env._get_camera_observation()
    with pytest.raises(RuntimeError, match="No fresh complete camera snapshot"):
        env.get_raw_camera_snapshot()
    frame[:] = 80
    camera.get_observation = lambda **kwargs: {"frame": frame}
    env._get_camera_observation()
    assert env.get_raw_camera_snapshot()["raw_frames"]["base_0_rgb"].max() == 80
