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

"""Offline Franka primitive and state-capture tests."""

from __future__ import annotations

from pathlib import Path
from threading import Event
from types import SimpleNamespace

import numpy as np

from robots.franka import toolkit as franka_toolkit
from robots.franka import tools
from robots.franka.perception import back_project
from robots.franka.runtime_config import set_calibration_path
from robots.franka.toolkit import FrankaRuntime, FrankaToolkit
from robots.franka.tools import (
    dump_state,
    view_camera_meta,
    view_env_state,
)
from rpent.dashboard.events import StepRecordEvent
from rpent.memory import MemoryManager
from rpent.session import EnvState
from rpent.tools import ToolContext


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


def _context(env: FakeEnv, *, model=None, state=None):
    return ToolContext(
        robot=FrankaRuntime(env=env, model=model, task_description="default task"),
        state=state,
        memory=None,
        output_dir=Path("."),
        record_frame=lambda frame: None,
        _cancel_event=Event(),
    )


def test_vec3_validation_and_motion_forwarding():
    env = FakeEnv()
    ctx = _context(env)

    tools.move_delta.handler([0.01, 0.0, -0.02], ctx=ctx)
    tools.rotate_delta.handler([0.0, 0.0, 0.1], ctx=ctx)

    np.testing.assert_allclose(env.moves[0], [0.01, 0.0, -0.02])
    np.testing.assert_allclose(env.rotations[0], [0.0, 0.0, 0.1])


def test_dump_state_saves_canonical_rgbd_artifacts(tmp_path: Path):
    env = FakeEnv()
    ctx = _context(env)
    state = EnvState(tmp_path)

    record = dump_state(
        ctx.robot,
        state,
        command={"action": "move_delta"},
        result={"ok": True},
        elapsed_s=0.2,
    )

    assert record.artifacts == {
        "camera.png",
        "camera_depth.npy",
        "camera_meta.json",
        "wrist.png",
        "wrist_depth.npy",
    }
    output = view_env_state.handler(ctx=_context(env, state=state))
    assert output.images == [
        state.load_bytes("wrist.png"),
        state.load_bytes("camera.png"),
    ]
    assert output.data["images"] == ["wrist", "camera"]
    assert (
        view_camera_meta.handler(ctx=_context(env, state=state)).data["camera_meta"][
            "depth_unit"
        ]
        == "m"
    )


def test_vla_grasp_runs_bounded_chunks():
    env = FakeEnv()
    ctx = _context(env, model=FakeModel())

    result = tools.vla_grasp.handler("pick up the cube", max_chunks=3, ctx=ctx)

    assert result.data["chunks_executed"] == 3
    assert len(env.chunks) == 3
    # Obs is fetched once, then threaded from each chunk_step result.
    assert env.observation_calls == 1


def test_back_project_reads_rpent_state_artifacts(tmp_path: Path):
    state = EnvState(tmp_path)
    with state.record_step(
        state={"raw_base_state": {"tcp_pose": [0, 0, 0, 0, 0, 0, 1]}}
    ) as step:
        state.save("wrist_depth.npy", np.full((4, 4), 0.5), step=step)
        state.save(
            "camera_meta.json",
            {
                "observation_camera_map": {"main": "wrist_cam"},
                "cameras": {
                    "wrist_cam": {"intrinsic_K": [[100, 0, 2], [0, 100, 2], [0, 0, 1]]}
                },
            },
            step=step,
        )

    set_calibration_path(
        Path(__file__).parent / "fixtures" / "hand_eye_calibration.json"
    )
    result = back_project(row=2, col=2, camera="wrist", state=state)

    assert result["coordinate_frame"] == "franka_base"
    assert result["depth_m"] == 0.5
    assert len(result["point_base"]) == 3


def test_toolkit_factory_validation_capture_and_cancellation(tmp_path, monkeypatch):
    monkeypatch.setattr(franka_toolkit, "get_output_dir", lambda: tmp_path)
    env = FakeEnv()
    # RPC state contains NumPy arrays; native result text must still be JSON.
    get_robot_state = env.get_robot_state
    env.get_robot_state = lambda: {**get_robot_state(), "states": np.zeros(8)}
    events = []
    kwargs = {"env": env, "model": None, "task_description": "test task"}
    toolkit = FrankaToolkit(
        runtime_kwargs=kwargs,
        dashboard_events=SimpleNamespace(enabled=True, emit=events.append),
        memory=MemoryManager(root=tmp_path / "memory"),
    )
    assert kwargs == {"env": env, "model": None, "task_description": "test task"}
    assert env.observation_calls == 1
    assert len(events) == 1 and isinstance(events[0], StepRecordEvent)
    assert all(
        "ctx" not in tool.input_schema["properties"] for tool in toolkit.list_tools()
    )
    for delta in ([1, 2], [0, 0, float("inf")]):
        result = toolkit.execute_tool("move_delta", {"delta_xyz": delta})
        assert result.is_error
    assert env.moves == []
    result = toolkit.execute_tool("move_delta", {"delta_xyz": [0.01, 0, 0]})
    assert not result.is_error
    assert result.data["images"] == ["wrist", "camera"]
    assert len(result.images) == 2
    assert "states" in result.to_text()
    assert len(events) == 2
    assert events[-1].record.command["action"] == "move_delta"
    read = toolkit.execute_tool("view_env_state", {})
    assert read.data == result.data and read.images == result.images
    assert len(events) == 2
    toolkit.cancel_active_and_wait()
    assert toolkit.execute_tool("move_delta", {"delta_xyz": [0, 0, 0]}).is_error
    assert len(env.moves) == 1
    toolkit.resume_calls()
    assert not toolkit.execute_tool("view_env_state", {}).is_error
    assert not toolkit.execute_tool(
        "finish", {"status": "success", "summary": "done"}
    ).is_error
    assert toolkit.finish_result == {"status": "success", "summary": "done"}
    assert len(events) == 2
    toolkit.close()
