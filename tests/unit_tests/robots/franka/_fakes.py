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


"""CPU clients shared by behavior and pre-native compatibility tests."""

from __future__ import annotations

import numpy as np


class FakeEnv:
    def __init__(self) -> None:
        self.moves: list[np.ndarray] = []
        self.rotations: list[np.ndarray] = []
        self.gripper_open = True
        self.chunks: list[np.ndarray] = []
        self.observation_calls = 0

    def reset(self):
        return {"ok": True}

    def move_delta(self, value):
        self.moves.append(np.asarray(value))
        return {"ok": True}

    def rotate_delta(self, value):
        self.rotations.append(np.asarray(value))
        return {"ok": True}

    def set_gripper(self, *, open: bool):
        self.gripper_open = open
        return {"ok": True, "open": open}

    def _obs(self):
        return {
            "main_images": np.zeros((8, 8, 3), dtype=np.uint8),
            "extra_view_images": np.ones((1, 8, 8, 3), dtype=np.uint8),
            "main_depths": np.ones((8, 8), dtype=np.float32),
            "extra_view_depths": np.ones((1, 8, 8), dtype=np.float32) * 2,
            "states": np.zeros(8, dtype=np.float32),
        }

    def get_observation(self):
        self.observation_calls += 1
        return self._obs()

    def get_robot_state(self):
        return {"tcp_pose": [0.5, 0.0, 0.2, 0.0, 0.0, 0.0, 1.0]}

    def get_camera_meta(self):
        return {"depth_unit": "m", "cameras": {"wrist_1": {"fx": 100.0}}}

    def chunk_step(self, actions):
        self.chunks.append(np.asarray(actions))
        return {"terminated": False, "truncated": False, "observation": self._obs()}


class FakeModel:
    def predict(self, observation, options=None):
        assert observation["task_descriptions"] == "pick up the cube"
        assert options == {"mode": "eval"}
        return np.zeros((2, 7), dtype=np.float32)
