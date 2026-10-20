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

from pathlib import Path
from typing import Any

import numpy as np

from robots.behavior.rlinf_env import OfficialBehaviorBackend
from robots.behavior.schemas import ACTION_DIM, RAW_PROPRIO_SEGMENTS


class _FakeOfficialBehaviorEnv:
    def __init__(
        self,
        *_args: Any,
        terminated_on_step: int | None = None,
        truncated_on_step: int | None = None,
        success_on_step: int | None = None,
        **_kwargs: Any,
    ) -> None:
        self.actions: list[np.ndarray] = []
        self.closed = False
        self.terminated_on_step = terminated_on_step
        self.truncated_on_step = truncated_on_step
        self.success_on_step = success_on_step
        self.raw = np.zeros(256, dtype=np.float32)
        self.raw[RAW_PROPRIO_SEGMENTS["trunk"]] = np.asarray(
            [0.11, 0.12, 0.13, 0.14],
            dtype=np.float32,
        )
        self.raw[RAW_PROPRIO_SEGMENTS["left_arm"]] = np.linspace(
            0.21,
            0.27,
            7,
            dtype=np.float32,
        )
        self.raw[RAW_PROPRIO_SEGMENTS["right_arm"]] = np.linspace(
            -0.31,
            -0.37,
            7,
            dtype=np.float32,
        )
        self.raw[RAW_PROPRIO_SEGMENTS["left_gripper"]] = 0.05
        self.raw[RAW_PROPRIO_SEGMENTS["right_gripper"]] = 0.06

    def _obs(self) -> dict[str, Any]:
        return {
            "main_images": np.zeros((8, 8, 3), dtype=np.uint8),
            "wrist_images": np.zeros((2, 8, 8, 3), dtype=np.uint8),
            "states": self.raw.copy(),
            "task_descriptions": "turn on the radio",
        }

    def env_reset(self) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        return [self._obs()], [{"done": {"success": False}}]

    def env_chunk_step(
        self,
        chunk_actions: Any,
    ) -> tuple[
        list[list[dict[str, Any]]],
        np.ndarray,
        np.ndarray,
        np.ndarray,
        list[list[dict[str, Any]]],
    ]:
        if hasattr(chunk_actions, "detach"):
            action_array = chunk_actions.detach().cpu().numpy().astype(np.float32)
        else:
            action_array = np.asarray(chunk_actions, dtype=np.float32)
        assert action_array.shape == (1, 1, ACTION_DIM)
        self.actions.append(action_array[0, 0].copy())
        step_index = len(self.actions)
        terminated = self.terminated_on_step == step_index
        truncated = self.truncated_on_step == step_index
        success = self.success_on_step == step_index
        return (
            [[self._obs()]],
            np.asarray([[0.0]], dtype=np.float32),
            np.asarray([[terminated]], dtype=bool),
            np.asarray([[truncated]], dtype=bool),
            [[{"done": {"success": success}}]],
        )

    def close(self) -> None:
        self.closed = True


def _both_hand_request() -> dict[str, Any]:
    return {
        "hand": "both",
        "targets": {
            "left": {"delta_xyz": [0.01, 0.0, 0.0], "frame": "world"},
            "right": {"delta_xyz": [-0.01, 0.0, 0.0], "frame": "eef"},
        },
        "visual_hand_checks": {
            "left": {
                "camera": "left_wrist",
                "frame_id": "left-frame",
                "selected_hand": "left",
                "assessment": "selected_hand_visually_confirmed",
            },
            "right": {
                "camera": "right_wrist",
                "frame_id": "right-frame",
                "selected_hand": "right",
                "assessment": "selected_hand_visually_confirmed",
            },
        },
    }


def _visual_check(hand: str) -> dict[str, str]:
    return {
        "camera": f"{hand}_wrist",
        "frame_id": f"{hand}-frame",
        "selected_hand": hand,
        "assessment": "selected_hand_visually_confirmed",
    }


def _official_backend(
    tmp_path: Path,
    fake_env: _FakeOfficialBehaviorEnv | None = None,
) -> tuple[OfficialBehaviorBackend, _FakeOfficialBehaviorEnv]:
    env = fake_env or _FakeOfficialBehaviorEnv()
    backend = OfficialBehaviorBackend(
        meta={
            "task_name": "turning_on_radio",
            "task_language": "turn on the radio",
            "activity_definition_id": 0,
            "activity_instance_id": 242,
            "public_seed": 0,
            "scene_model": "house_double_floor_lower",
            "max_episode_steps": 64,
        },
        output_dir=tmp_path,
        behavior_env_cls=lambda *_args, **_kwargs: env,
        cfg=object(),
    )
    backend.reset()
    return backend, env
