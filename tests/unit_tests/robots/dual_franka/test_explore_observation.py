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

import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from robots.dual_franka.env_server import _create_worker_class


def test_worker_reads_before_reset_and_refreshes_cached_frames(monkeypatch):
    for name, attrs in {
        "rlinf.envs.real.env": {"RealWorldEnv": object},
        "rlinf.robotics.parts.cameras": {
            "Camera": object,
            "CameraInfo": object,
        },
        "rlinf.scheduler": {"Worker": object},
    }.items():
        module = ModuleType(name)
        module.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, module)
    cls = _create_worker_class()
    worker = cls.__new__(cls)
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
