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

"""RoboTwin localization capture and observation artifacts."""

from __future__ import annotations

import importlib.util
import sys
import threading
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from robots.robotwin import toolkit, tools
from robots.robotwin.robot_spec import ROBOTWIN_CAMERA_NAMES
from rpent.session import EnvState
from rpent.tools.localization.localization import GL_FROM_CV, LocalizationFrame


class _Camera:
    """SAPIEN-like camera: 2x3 picture of a wall 2 m in front of the camera."""

    def __init__(self, x):
        self.model = np.eye(4)
        self.model[0, 3] = x

    def get_picture(self, name):
        if name == "Color":
            return np.full((2, 3, 4), 0.5, np.float32)
        rows, cols = np.indices((2, 3), dtype=np.float64)
        position = np.stack(
            [
                (cols + 0.5 - 1.5) * 2 / 4,
                -(rows + 0.5 - 1) * 2 / 4,
                -2 + 0 * rows,
                0.5 + 0 * rows,
            ],
            axis=-1,
        )
        position[0, 0, 3] = 1.0  # background
        return position.astype(np.float32)

    def get_intrinsic_matrix(self):
        return np.array([[4.0, 0, 1.5], [0, 4.0, 1.0], [0, 0, 1]])

    def get_model_matrix(self):
        return self.model

    def get_extrinsic_matrix(self):
        return np.linalg.inv(self.model @ GL_FROM_CV)[:3]

    def get_height(self):
        return 2

    def get_width(self):
        return 3


def _load_agent_env(monkeypatch):
    """Import the RPent adapter without RLinf, which unit tests do not install."""
    for name in ("rlinf", "rlinf.envs", "rlinf.envs.sim", "rlinf.envs.sim.robotwin"):
        module = ModuleType(name)
        module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)
    base = ModuleType("rlinf.envs.sim.robotwin.robotwin_env")
    base.RoboTwinEnv = object
    monkeypatch.setitem(sys.modules, base.__name__, base)
    if importlib.util.find_spec("torch") is None:
        monkeypatch.setitem(sys.modules, "torch", ModuleType("torch"))
    spec = importlib.util.spec_from_file_location(
        "offline_robotwin_rlinf_env", Path(toolkit.__file__).with_name("rlinf_env.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.RoboTwinAgentEnv


def test_capture_takes_one_picture_set_under_the_environment_lock(monkeypatch):
    agent_env = _load_agent_env(monkeypatch)
    head, left, right = _Camera(0.0), _Camera(1.0), _Camera(2.0)
    observations = []
    lock = threading.Lock()

    def get_obs():
        assert lock.locked()
        observations.append(True)

    task = SimpleNamespace(
        get_obs=get_obs,
        take_action_cnt=5,
        cameras=SimpleNamespace(
            static_camera_list=[head],
            static_camera_name=["head_camera"],
            collect_wrist_camera=True,
            left_camera=left,
            right_camera=right,
        ),
    )
    env = agent_env.__new__(agent_env)
    env.venv = SimpleNamespace(envs=[SimpleNamespace(lock=lock, task=task)])
    records = env.capture_localization(["head", "left_wrist", "right_wrist"])
    assert len(observations) == 1
    frames = [LocalizationFrame.from_canonical(record) for record in records]
    assert [frame.camera for frame in frames] == ["head", "left_wrist", "right_wrist"]
    for frame, x in zip(frames, (0.0, 1.0, 2.0)):
        assert not frame.valid_mask[0, 0] and frame.valid_mask.sum() == 5
        # Wall points of the OpenGL camera lie at world z = -2.
        np.testing.assert_allclose(frame.world_xyz[frame.valid_mask][:, 2], -2.0)
        np.testing.assert_allclose(frame.world_xyz[1, 1], [x, -0.25, -2])
        assert frame.metadata["env_step"] == 5
    with pytest.raises(ValueError, match="unknown RoboTwin camera"):
        env.capture_localization(["front"])


def test_full_observation_saves_world_mask_and_calibration(tmp_path):
    def frame(camera):
        return LocalizationFrame.from_canonical(
            {
                "camera": camera,
                "rgb": np.full((2, 3, 3), 7, np.uint8),
                "depth_m": [[np.nan, 1, 1], [1, 1, 1]],
                "intrinsic_K": np.eye(3),
                "T_world_from_cv": np.eye(4),
            }
        )

    requests = []

    def capture_frames(cameras):
        requests.append(cameras)
        return [frame(camera) for camera in cameras]

    robot_toolkit = toolkit.RoboTwinToolkit.__new__(toolkit.RoboTwinToolkit)
    robot_toolkit._primitives = SimpleNamespace(
        env=SimpleNamespace(
            capture_frames=capture_frames,
            get_task_language=lambda: "offline",
            last_info={"robot_state": {}},
            server_meta={"task_name": "offline"},
        )
    )
    observation = robot_toolkit._capture_full_observation()
    assert requests == [list(ROBOTWIN_CAMERA_NAMES)]
    state = EnvState(tmp_path)
    record = tools.dump_observation(observation, env_state=state, status={}, log=None)
    for camera in ROBOTWIN_CAMERA_NAMES:
        valid = state.load(f"{camera}_valid_mask.npy", step=0)
        world = state.load(f"{camera}_world_xyz.npy", step=0)
        meta = state.load(f"{camera}_camera_meta.json", step=0)
        assert valid.dtype == np.bool_ and not valid[0, 0] and valid.sum() == 5
        assert np.isnan(world[0, 0]).all()
        assert meta["camera"] == camera and meta["valid_pixels"] == 5
        assert "valid_mask" in record.state["artifacts"][camera]
