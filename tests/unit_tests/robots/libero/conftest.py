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

from __future__ import annotations

import numpy as np
import pytest

from robots.libero.toolkit import LiberoToolkit
from rpent.dashboard.events import NullDashboardEventSink
from rpent.memory import MemoryManager
from rpent.robots.components.sam3_client import Sam3Result


class FakeEnv:
    return_all_frames = False

    def __init__(self):
        self.actions = []
        self.reset_calls = 0
        self.terminated = self.truncated = False
        self.after_step = lambda: None
        self.image = np.zeros((8, 8, 3), dtype=np.uint8)
        self.pos = np.array([0.0, 0.0, 0.3])

    def obs(self):
        return {
            "main_images": self.image,
            "states": [*self.pos, 0, 0, 0, 0.04, -0.04],
            "task_descriptions": "original task",
        }

    def reset(self):
        self.reset_calls += 1
        self.terminated = self.truncated = False
        return self.obs(), {}

    def step(self, action):
        self.actions.append(action.copy())
        self.after_step()
        return self.obs(), 0, self.terminated, self.truncated, {}

    def chunk_step(self, actions, *, return_all_frames=False):
        observations = [self.step(action)[0] for action in actions]
        return (
            observations if return_all_frames else observations[-1],
            0,
            np.zeros(len(actions), dtype=bool),
            np.zeros(len(actions), dtype=bool),
            {},
        )

    def raw_obs(self):
        return {
            "robot0_eef_pos": self.pos,
            "robot0_eef_quat": [1, 0, 0, 0],
            "robot0_gripper_qpos": [0.04, -0.04],
            "agentview_image": self.image,
            "agentview_depth": np.ones((8, 8)),
            "robot0_eye_in_hand_image": self.image,
            "robot0_eye_in_hand_depth": np.ones((8, 8)),
        }

    def get_task_language(self):
        return "original task"

    def get_camera_meta(self, camera_name, height, width):
        return {
            "intrinsic_K": [[4, 0, 4], [0, 4, 4], [0, 0, 1]],
            "extrinsic_cam2world": np.eye(4).tolist(),
        }

    def render_camera(self, **kwargs):
        return self.image, np.ones((8, 8))


class FakeModel:
    def __init__(self):
        self.instructions = []

    def predict(self, obs, *, options):
        self.instructions.append(obs["task_descriptions"])
        assert options == {"mode": "eval"}
        return np.zeros((3, 7), dtype=np.float32)


class FakeSam:
    def segment(self, image, **kwargs):
        return Sam3Result(
            found=True,
            mask=np.ones((8, 8), dtype=bool),
            score=0.8,
            box=[0, 0, 8, 8],
            mask_shape=(8, 8),
        )


@pytest.fixture
def make_toolkit(tmp_path):
    instances = []

    def make(*, mode="evaluation", attempts=0, molmo_client=None):
        output = tmp_path / str(len(instances))
        env, model = FakeEnv(), FakeModel()
        toolkit = LiberoToolkit(
            runtime_kwargs={
                "env": env,
                "model": model,
                "sam3_client": FakeSam(),
                "molmo_client": molmo_client,
            },
            output_dir=output,
            state_output_dir=output / "sessions" / "session_001",
            memory=MemoryManager(output / "memory"),
            dashboard_events=NullDashboardEventSink(),
            mode=mode,
            attempts_per_session=attempts,
        )
        instances.append(toolkit)
        return toolkit, env, model

    yield make
    for toolkit in instances:
        # Avoid video encoding in CPU contract tests; recording contents are asserted explicitly.
        toolkit._frames.clear()
        if toolkit._scheduler._state != "closed":
            toolkit.close()
