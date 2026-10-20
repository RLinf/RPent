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

import json
from pathlib import Path
from typing import Any

import numpy as np

from robots.behavior.toolkit import BehaviorToolkit
from rpent.dashboard.events import NullDashboardEventSink
from rpent.memory import MemoryManager
from rpent.robots import RunConfig


class _FakeModel:
    def predict(self, observation: dict[str, Any], *, options: dict[str, Any]) -> Any:
        assert observation["task_descriptions"] == "turn on the radio"
        assert options == {"mode": "eval"}
        return np.zeros((32, 23), dtype=np.float32)


class _FakeChunkEnv:
    total_env_steps = 0
    official_success_latched = False
    official_success_receipt = None

    def __init__(self) -> None:
        self.return_all_frames: list[bool] = []

    def chunk_step(
        self,
        actions: Any,
        *,
        return_all_frames: bool = False,
    ) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        array = np.asarray(actions)
        self.return_all_frames.append(bool(return_all_frames))
        self.total_env_steps += int(array.shape[0])
        frames = [
            {
                "main_images": np.full((16, 16, 3), idx, dtype=np.uint8),
                "task_descriptions": "turn on the radio",
            }
            for idx in range(int(array.shape[0]))
        ]
        obs: Any = frames if return_all_frames else frames[-1]
        return (
            obs,
            0.0,
            False,
            False,
            {
                "executed_steps": int(array.shape[0]),
                "_rpent": {"total_env_steps": self.total_env_steps},
            },
        )


def test_behavior_pi0_chunk_records_streaming_episode_video(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    session_dir = run_dir / "sessions" / "session_001"
    toolkit = BehaviorToolkit(
        runtime_kwargs={
            "env": _FakeChunkEnv(),
            "model": _FakeModel(),
            "task_name": "turning_on_radio",
            "public_seed": 0,
            "max_episode_steps": 64,
            "initial_observation": {
                "main_images": np.zeros((16, 16, 3), dtype=np.uint8),
                "task_descriptions": "turn on the radio",
            },
        },
        dashboard_events=NullDashboardEventSink(),
        memory=MemoryManager(tmp_path / "memory"),
        config=RunConfig(
            recipe_tag="turning_on_radio_s0",
            output_dir=run_dir,
            prompt_vars={
                "mode": "explore",
                "task_name": "turning_on_radio",
                "public_seed": 0,
                "memory_dir": str(tmp_path / "memory"),
            },
            task_desc={},
        ),
        state_output_dir=session_dir,
    )

    result = toolkit.primitives.pi0_nav_pick(
        instruction="turn on the radio",
        chunks=1,
    )
    toolkit.execute_tool("finish", {"status": "incomplete", "summary": "done"})

    assert result["chunks_used"] == 1
    assert result["env_steps_used"] == 32
    assert toolkit.primitives.env.return_all_frames == [True]
    video_path = session_dir / "episode.mp4"
    assert video_path.is_file()
    assert video_path.stat().st_size > 0
    assert (
        "episode.mp4"
        in json.loads((session_dir / "states.json").read_text())["run_artifacts"]
    )


def test_pi0_chunk_keeps_gym_termination_separate_from_success(tmp_path: Path) -> None:
    class SuccessfulTerminalEnv(_FakeChunkEnv):
        def chunk_step(self, actions: Any, *, return_all_frames: bool = False):
            obs, reward, _, _, info = super().chunk_step(
                actions, return_all_frames=return_all_frames
            )
            return obs, reward, True, True, {**info, "done": {"success": True}}

    toolkit = BehaviorToolkit(
        runtime_kwargs={
            "env": SuccessfulTerminalEnv(),
            "model": _FakeModel(),
            "task_name": "turning_on_radio",
            "output_dir": tmp_path / "run",
            "initial_observation": {
                "main_images": np.zeros((16, 16, 3), dtype=np.uint8),
            },
        },
        memory=MemoryManager(tmp_path / "memory"),
    )
    try:
        result = toolkit.primitives.pi0_nav_pick(
            instruction="turn on the radio", chunks=1
        )
        assert result["stop_reason"] == "official_task_success"
        assert result["task_success"] is True
        assert result["terminated"] is True
        assert result["truncated"] is True

        toolkit.get_env_state(
            command={"action": "pi0_nav_pick"}, result=result, elapsed_s=0.1
        )
        record = toolkit.state.latest_record()
        assert record is not None
        assert record.terminated is True
        assert record.truncated is True
    finally:
        toolkit.close()
