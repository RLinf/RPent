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
        self.moves: list[tuple[str, np.ndarray]] = []
        self.rotations: list[tuple[str, np.ndarray]] = []
        self.grippers: list[tuple[str, bool]] = []
        self.chunks: list[np.ndarray] = []
        self.observation_calls = 0

    def reset(self):
        return {"ok": True}

    def move_delta(self, arm, value):
        self.moves.append((arm, np.asarray(value)))
        return {"ok": True, "arm": arm}

    def rotate_delta(self, arm, value):
        self.rotations.append((arm, np.asarray(value)))
        return {"ok": True, "arm": arm}

    def set_gripper(self, arm, *, open: bool):
        self.grippers.append((arm, open))
        return {"ok": True, "arm": arm, "open": open}

    def _obs(self):
        return {
            "main_images": np.zeros((8, 8, 3), dtype=np.uint8),
            "extra_view_images": np.ones((2, 8, 8, 3), dtype=np.uint8),
            "main_depths": np.ones((8, 8), dtype=np.float32),
            "extra_view_depths": np.ones((2, 8, 8), dtype=np.float32) * 2,
            "d455_images": np.ones((8, 8, 3), dtype=np.uint8) * 3,
            "d455_depths": np.ones((8, 8), dtype=np.float32) * 4,
            "raw_camera_frames": {
                "left_wrist_0_rgb": np.full((10, 12, 3), 5, dtype=np.uint8),
                "base_0_rgb": np.full((10, 12, 3), 7, dtype=np.uint8),
                "right_wrist_0_rgb": np.full((10, 12, 3), 6, dtype=np.uint8),
            },
            "raw_camera_depths": {
                "left_wrist_0_rgb": np.full((10, 12), 8, dtype=np.float32),
                "base_0_rgb": np.full((10, 12), 9, dtype=np.float32),
                "right_wrist_0_rgb": np.full((10, 12), 11, dtype=np.float32),
            },
            "states": np.zeros(20, dtype=np.float32),
        }

    def get_observation(self):
        self.observation_calls += 1
        return self._obs()

    def get_robot_state(self):
        return {
            "left_arm": {"tcp_pose": [0.5, -0.2, 0.5, 0.0, 0.0, 0.0, 1.0]},
            "right_arm": {"tcp_pose": [0.5, 0.2, 0.5, 0.0, 0.0, 0.0, 1.0]},
        }

    def get_camera_meta(self):
        return {
            "cameras": {"left_wrist_0_rgb": {"serial": "left", "type": "zed"}},
            "observation_camera_map": {"main": "left_wrist_0_rgb"},
        }

    def chunk_step(self, actions):
        self.chunks.append(np.asarray(actions))
        return {"terminated": False, "truncated": False, "observation": self._obs()}
