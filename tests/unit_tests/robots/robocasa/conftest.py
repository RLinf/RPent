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

"""Small CPU fakes for the RoboCasa client boundaries."""

from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from robots.robocasa.robot_spec import get_toolkit
from rpent.dashboard.events import NullDashboardEventSink
from rpent.robots import RunConfig


class FakeEnv:
    def __init__(self):
        self.actions = []
        self.reset_calls = 0
        self.success = False
        self.language = "Open the left drawer."
        self.on_step = lambda: None
        self.eef_pos = np.array([0.0, 0.0, 1.0])
        self.eef_quat = np.array([0.0, 0.0, 0.0, 1.0])
        self.gripper_qpos = np.array([0.02, -0.02])
        self.base_pos = np.zeros(3)
        self.base_yaw = 0.0

    @property
    def current_raw_obs(self):
        return {
            "language": self.language,
            "robot0_base_pos": self.base_pos.copy(),
            "robot0_base_quat": np.array(
                [
                    0,
                    0,
                    np.sin(self.base_yaw / 2),
                    np.cos(self.base_yaw / 2),
                ]
            ),
            "robot0_gripper_qpos": self.gripper_qpos.copy(),
            "robot0_base_to_eef_pos": self.eef_pos - self.base_pos,
            "robot0_base_to_eef_quat": self.eef_quat.copy(),
        }

    @property
    def terminated(self):
        return self.success

    def reset(self):
        self.reset_calls += 1
        self.eef_pos = np.array([0.0, 0.0, 1.0])
        self.base_pos = np.zeros(3)
        self.base_yaw = 0.0
        self.success = False

    def step(self, action):
        self.actions.append(action.copy())
        self.eef_pos += action[:3] * 0.05
        self.base_pos[:2] += (
            np.array(
                [
                    np.cos(self.base_yaw) * action[7]
                    - np.sin(self.base_yaw) * action[8],
                    np.sin(self.base_yaw) * action[7]
                    + np.cos(self.base_yaw) * action[8],
                ]
            )
            * 0.01
        )
        self.base_yaw += action[9] * 0.1
        self.on_step()

    def render_camera(self, camera_name, height=4, width=4, depth=False):
        colors = {"agentview": 1, "navview": 2, "wrist": 3}
        rgb = np.full((4, 4, 3), colors.get(camera_name, 4), dtype=np.uint8)
        rgb[..., 0] = len(self.actions) % 256
        return (rgb, np.ones((4, 4))) if depth else rgb

    def world_map(self, *args):
        world = np.ones((4, 4, 3))
        world[..., 2] = 0.9
        return world

    def get_camera_meta(self, camera):
        return {"camera": camera}

    def get_task_language(self):
        return self.language

    def get_task_progress(self):
        return {"steps": len(self.actions)}

    def get_success_criteria_text(self):
        return "offline success criteria"

    def check_success(self):
        return self.success

    def grasp_contact(self):
        return False, None

    def reassemble_env_action(self, action):
        return np.concatenate(
            [
                action[name]
                for name in (
                    "action.end_effector_position",
                    "action.end_effector_rotation",
                    "action.gripper_close",
                    "action.base_motion",
                    "action.control_mode",
                )
            ]
        )


class FakeModel:
    def __init__(self):
        self.calls = []
        self.resets = 0
        self.on_predict = lambda: None

    def get_modality_config(self):
        return {"video_delta_indices": [-2, 0], "hist_maxlen": 3}

    def predict(self, obs, options):
        self.calls.append((deepcopy(obs), deepcopy(options)))
        self.on_predict()
        return {
            "action.end_effector_position": np.zeros((1, 2, 3)),
            "action.end_effector_rotation": np.zeros((1, 2, 3)),
            "action.gripper_close": np.ones((1, 2, 1)),
            "action.base_motion": np.ones((1, 2, 4)),
            "action.control_mode": np.ones((1, 2, 1)),
        }

    def reset_session(self):
        self.resets += 1


@pytest.fixture
def make_toolkit(monkeypatch, tmp_path):
    for name in (
        "RLDX_MAX_CHUNKS",
        "RLDX_ACTION_STEPS_PER_CHUNK",
        "RLDX_SETTLE_PATIENCE",
        "RLDX_ALLOW_RESET",
        "RLDX_KEEP_HEAVY_NPY",
        "RLDX_VIDEO_DIR",
    ):
        monkeypatch.delenv(name, raising=False)
    runs = []

    def make(**settings):
        for name, value in settings.items():
            monkeypatch.setenv(name, str(value))
        env = FakeEnv()
        model = FakeModel()
        output_dir = tmp_path / f"run-{len(runs)}"
        config = RunConfig(
            recipe_tag="OpenDrawer_s1",
            output_dir=output_dir,
            prompt_vars={"memory_dir": str(tmp_path / "memory")},
            task_desc={},
        )
        robot_toolkit = get_toolkit(
            runtime_kwargs={"env": env, "model": model, "hi_res": 8},
            dashboard_events=NullDashboardEventSink(),
            config=config,
        )
        # The simulator package owns action conversion; these tests exercise the
        # rollout, frame history, and RPC-independent toolkit execution.
        robot_toolkit._robot._rldx._unmap = lambda action: action
        run = SimpleNamespace(
            toolkit=robot_toolkit, env=env, model=model, config=config
        )
        runs.append(run)
        return run

    yield make
    for run in runs:
        run.toolkit._frames.clear()
        if run.toolkit._scheduler._state != "closed":
            run.toolkit.close()
