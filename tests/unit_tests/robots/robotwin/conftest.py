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

"""Offline RoboTwin clients for testing the real native tools."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from robots.robotwin.robot_spec import ROBOTWIN_CAMERA_NAMES
from robots.robotwin.toolkit import RoboTwinToolkit
from rpent.dashboard.events import NullDashboardEventSink
from rpent.memory import MemoryManager


class FakeRoboTwinEnv:
    def __init__(self):
        self.terminated = False
        self.truncated = False
        self.server_meta = {"task_name": "stack_blocks"}
        self.execution_capabilities = {"chunk_step_all_frames": True}
        self.last_info = {
            "episode_status": {
                "eval_success": False,
                "take_action_cnt": 0,
                "step_lim": 200,
                "actual_seed": 7,
            },
            "robot_state": {
                "qpos_target14": [0.0] * 14,
                "left_eef_pose": [0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0],
                "right_eef_pose": [0.4, 0.5, 0.6, 1.0, 0.0, 0.0, 0.0],
                "left_gripper": 0.0,
                "right_gripper": 0.0,
            },
        }
        self.last_reset_info = {
            "actual_seed": 7,
            "instruction": self.get_task_language(),
        }
        self.renders = []
        self.plans = []
        self.steps = []
        self.chunks = []
        self.path = np.arange(36, dtype=float).reshape(6, 6) / 100
        self.plan_status = "Success"
        self.on_step = lambda: None
        self.on_chunk = lambda: None
        self.success_at = None

    def get_task_language(self):
        return "Stack the red block on the blue block."

    def render_camera(self, camera, *, depth=False):
        self.renders.append((camera, depth))
        rgb = np.full(
            (4, 5, 3), ROBOTWIN_CAMERA_NAMES.index(camera) * 50, dtype=np.uint8
        )
        return (rgb, np.ones((4, 5))) if depth else rgb

    def get_camera_meta(self, camera):
        return {
            "intrinsic_K": [[2.0, 0, 0], [0, 2.0, 0], [0, 0, 1]],
            "cam2world_gl": np.eye(4).tolist(),
            "height": 4,
            "width": 5,
        }

    def plan_arm_path(self, arm, target_pose):
        self.plans.append((arm, target_pose.copy()))
        return {"status": self.plan_status, "position": self.path.copy()}

    def _advance(self, requested):
        status = self.last_info["episode_status"]
        count = min(requested, status["step_lim"] - status["take_action_cnt"])
        if self.success_at is not None:
            count = min(count, self.success_at - status["take_action_cnt"])
        status["take_action_cnt"] += count
        status["eval_success"] = status["take_action_cnt"] == self.success_at
        self.terminated = status["eval_success"]
        self.truncated = status["take_action_cnt"] >= status["step_lim"]
        self.last_info["executed_actions"] = count
        return count

    def step(self, action, *, action_type):
        assert not self.terminated and not self.truncated
        assert action_type == "qpos"
        self.steps.append(action.copy())
        self._advance(1)
        state = self.last_info["robot_state"]
        state["qpos_target14"] = action.tolist()
        state["left_gripper"] = action[6]
        state["right_gripper"] = action[13]
        self.on_step()
        return (
            {"main_images": np.zeros((4, 5, 3), dtype=np.uint8)},
            0,
            self.terminated,
            self.truncated,
            self.last_info,
        )

    def chunk_step(self, actions, *, action_type, return_all_frames):
        assert not self.terminated and not self.truncated
        assert action_type == "ee"
        self.chunks.append((actions.copy(), return_all_frames))
        executed = self._advance(len(actions))
        rgb = np.zeros((4, 5, 3), dtype=np.uint8)
        payload = {"main_images": rgb}
        if return_all_frames:
            payload = {
                "frames": [rgb.copy() for _ in range(executed)],
                "final": payload,
            }
        self.on_chunk()
        return payload, 0, self.terminated, self.truncated, self.last_info


class FakeLingBot:
    def __init__(self):
        self.observations = []
        self.on_infer = lambda: None

    def infer(self, observation):
        self.observations.append(observation)
        self.on_infer()
        return np.arange(60 * 16, dtype=float).reshape(60, 16)


@pytest.fixture
def robotwin(tmp_path):
    env = FakeRoboTwinEnv()
    model = FakeLingBot()
    toolkit = RoboTwinToolkit(
        runtime_kwargs={"env": env, "model": model, "seed": 7},
        output_dir=tmp_path / "run",
        dashboard_events=NullDashboardEventSink(),
        memory=MemoryManager(tmp_path / "memory"),
    )
    yield SimpleNamespace(
        env=env, model=model, toolkit=toolkit, output_dir=tmp_path / "run"
    )
    toolkit._frames.clear()
    toolkit.close()
