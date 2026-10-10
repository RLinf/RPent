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

"""Operator evidence must refresh without invoking reset or step."""

import queue
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from robots.dual_franka.env_server import _create_worker_class


@pytest.fixture
def worker(monkeypatch):
    for name, attrs in {
        "rlinf.envs.real.env": {"RealWorldEnv": object},
        "rlinf.robotics.parts.cameras": {
            "Camera": object,
            "CameraInfo": object,
        },
        "rlinf.scheduler": {"Worker": object},
        "robots.franka.rpent_env": {"realsense_color_intrinsics": lambda camera: {}},
    }.items():
        module = ModuleType(name)
        module.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, module)
    cls = _create_worker_class()
    return cls.__new__(cls)


def test_worker_reads_before_reset_and_refreshes_cached_frames(worker):
    reads = []

    def raw_observation():
        reads.append(True)
        return {"main_images": np.full((4, 4, 3), len(reads), dtype=np.uint8)}

    raw = SimpleNamespace(
        config=SimpleNamespace(is_dummy=True), _get_observation=raw_observation
    )
    raw.unwrapped = raw
    worker.env = SimpleNamespace(
        env=SimpleNamespace(envs=[raw], call=lambda *args: [lambda: {}]),
        _wrap_obs=lambda obs: obs,
        reset=lambda: pytest.fail("reading evidence must not reset"),
        step=lambda *args, **kwargs: pytest.fail("reading evidence must not move"),
    )
    worker.last_obs = None
    worker._capture_perception_camera_snapshot = lambda: {
        "raw_frames": {},
        "raw_depths": {},
    }
    first = worker.get_observation()
    second = worker.get_observation()
    assert len(reads) == 2
    assert np.all(first["main_images"] == 1)
    assert np.all(second["main_images"] == 2)
    assert first["main_images"].shape == (4, 4, 3)

    def unavailable(timeout):
        raise queue.Empty

    del worker._capture_perception_camera_snapshot
    worker._perception_cameras = {"d455": SimpleNamespace(get_observation=unavailable)}
    with pytest.raises(RuntimeError, match="refusing cached RGBD"):
        worker.get_observation()


@pytest.mark.parametrize("with_depth", [False, True])
def test_perception_uses_official_rgbd_reading_without_rescaling_depth(
    worker, with_depth
):
    frame = np.full((4, 6, 3), [10, 20, 30], dtype=np.uint8)
    depth = np.full((4, 6), 0.42, dtype=np.float32)
    calls = []

    def observation(*, timeout):
        calls.append(timeout)
        return {"frame": frame, **({"depth": depth} if with_depth else {})}

    info = SimpleNamespace(
        camera_type="realsense", serial_number="mock-extra", enable_depth=with_depth
    )
    worker._perception_cameras = {
        "extra": SimpleNamespace(
            get_observation=observation,
            get_frame=lambda **kwargs: pytest.fail("use the official RGB-D interface"),
            camera_info=info,
            depth_scale=0.001,
        )
    }
    worker._perception_camera_meta = {}
    result = worker._capture_perception_camera_snapshot()

    assert calls == [2]
    np.testing.assert_array_equal(result["raw_frames"]["extra_rgb"], frame[..., ::-1])
    assert result["raw_frames"]["extra_rgb"].dtype == np.uint8
    meta = result["camera_meta"]["extra_rgb"]
    assert meta["rgb_shape"] == [4, 6, 3]
    assert meta["serial_number"] == "mock-extra"
    assert meta["depth_available"] is with_depth
    assert meta["depth_enabled"] is with_depth
    assert worker._perception_camera_meta["extra_rgb"] == meta
    if with_depth:
        assert meta["depth_shape"] == [4, 6]
        np.testing.assert_array_equal(result["raw_depths"]["extra_rgb"], depth)
        assert result["raw_depths"]["extra_rgb"].dtype == np.float32
        depth[:] = 9
        np.testing.assert_allclose(result["raw_depths"]["extra_rgb"], 0.42)
    else:
        assert meta["depth_shape"] is None
        assert result["raw_depths"] == {}
    frame[:] = 0
    assert result["raw_frames"]["extra_rgb"][0, 0].tolist() == [30, 20, 10]
