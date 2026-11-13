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

"""Public RLinf part fakes; no Gym wrappers or physical devices."""

import sys
import time
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from robots.lekiwi.runtime_config import load_runtime_config


class FakeBase:
    def __init__(self):
        self.velocity = np.zeros(3)
        self.commands = []
        self.stops = 0

    def send_action(self, action):
        velocity = np.array(action["velocity"])
        self.commands.append((time.monotonic(), velocity))
        self.velocity = velocity * 0.7
        return {"velocity": velocity}

    def get_observation(self):
        return {"velocity": self.velocity.copy()}

    def reset(self):
        self.stops += 1
        self.velocity = np.zeros(3)


class FakeCamera:
    def __init__(self, info):
        self.camera_info = info
        self.connected = False
        self.releases = 0

    @classmethod
    def of(cls, info):
        return cls(info)

    def connect(self):
        self.connected = True

    def disconnect(self):
        self.connected = False
        self.releases += 1

    def is_ready(self, timeout):
        return True

    def get_frame(self, **kwargs):
        assert self.connected
        return np.full((48, 64, 3), [10, 30, 80], dtype=np.uint8)


@pytest.fixture
def robot_parts(monkeypatch):
    from pathlib import Path

    from robots.lekiwi.env_server import LeKiwiEnvBackend

    instances = []

    class Robot:
        @classmethod
        def build(cls, **kwargs):
            assert kwargs["node_rank"] is None
            result = cls()
            result.base = FakeBase()
            result.cameras = {
                name: FakeCamera(info) for name, info in kwargs["cameras"].items()
            }
            result.connected = False
            result.releases = 0
            result.build_kwargs = kwargs
            instances.append(result)
            return result

        def child(self, name):
            return self.base if name == "base" else self.cameras[name]

        def connect(self):
            self.connected = True
            for camera in self.cameras.values():
                camera.connect()

        def disconnect(self):
            self.connected = False
            self.releases += 1
            for camera in self.cameras.values():
                camera.disconnect()

    lekiwi = ModuleType("rlinf.robotics.robots.lekiwi")
    lekiwi.LeKiwiRobot = Robot
    monkeypatch.setitem(sys.modules, lekiwi.__name__, lekiwi)
    cameras = ModuleType("rlinf.robotics.parts.cameras")
    cameras.Camera = FakeCamera
    cameras.CameraInfo = lambda **kw: SimpleNamespace(resolution=(640, 480), **kw)
    monkeypatch.setitem(sys.modules, cameras.__name__, cameras)
    cfg = load_runtime_config(
        Path(__file__).parents[4] / "robots/lekiwi/config/example.yaml"
    )
    cfg.drive.step_frequency = 50
    backends = []

    def build(**kwargs):
        backend = LeKiwiEnvBackend(cfg, **kwargs)
        backends.append(backend)
        return backend, instances[-1]

    yield SimpleNamespace(
        build=build, instances=instances, cfg=cfg, Robot=Robot, Camera=FakeCamera
    )
    for backend in backends:
        backend.close_env()
