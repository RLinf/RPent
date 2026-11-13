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

"""Offline RoboLab registration, observation, and control contracts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pytest

import robots.robolab.robot_spec as robolab_spec
from robots.robolab.toolkit import RoboLabToolkit
from rpent.dashboard.events import NullDashboardEventSink
from rpent.evaluation import RunFinalizationContext
from rpent.memory import MemoryManager
from rpent.robots.base import get_robot_spec
from rpent.robots.components import am_runtime
from rpent.robots.components.action_spec import box_action_spec, validate_action
from rpent.robots.components.pi05_vla_client import Pi05VLAClient


def test_robolab_registers_and_parses_without_simulator(tmp_path):
    spec = get_robot_spec("robolab")
    parser = argparse.ArgumentParser()
    spec.add_cli_args(parser, False)
    args = parser.parse_args(["--task", "BananaInBowlTask", "--policy-backend", "none"])
    args.output_dir = tmp_path / "run"
    args.memory_dir = tmp_path / "memory"
    config = spec.parse_config(args)
    assert config.recipe_tag == "BananaInBowlTask_s0"
    assert config.task_desc == {"task": "BananaInBowlTask", "seed": 0}
    assert config.enable_direct_action
    assert "BananaInBowlTask" in spec.prompts.render(
        "user", variables=config.prompt_vars
    )
    spec.prepare_memory(args, config)
    assert (tmp_path / "memory").is_dir()


def test_robolab_rejects_non_task_path():
    spec = get_robot_spec("robolab")
    parser = argparse.ArgumentParser()
    spec.add_cli_args(parser, False)
    args = parser.parse_args(
        ["--task", "../BananaInBowlTask", "--policy-backend", "none"]
    )
    args.output_dir = None
    args.memory_dir = None
    with pytest.raises(ValueError, match="task class name"):
        spec.parse_config(args)


def test_runtime_keeps_virtualenv_python_symlinks(tmp_path, monkeypatch):
    executable = tmp_path / "python"
    executable.symlink_to(sys.executable)

    class _Daemon:
        def __init__(self, *, name, cmd, **kwargs):
            self.cmd = cmd

        def start(self):
            pass

    monkeypatch.setattr(robolab_spec, "ProcessDaemon", _Daemon)
    monkeypatch.setattr(am_runtime, "ProcessDaemon", _Daemon)
    parser = argparse.ArgumentParser()
    robolab_spec.get_robot_spec().add_cli_args(parser, False)
    args = parser.parse_args(
        [
            "--task",
            "BananaInBowlTask",
            "--sim-python",
            str(executable),
            "--am-python",
            str(executable),
            "--am-checkpoint",
            "/external/pi05",
        ]
    )
    env_daemon, _ = robolab_spec._spawn_env(args, tmp_path)
    vla_daemon, _ = robolab_spec._spawn_vla(args, tmp_path)
    assert Path(env_daemon.cmd[0]) == executable
    assert Path(vla_daemon.cmd[0]) == executable
    model_path = vla_daemon.cmd[vla_daemon.cmd.index("--model-path") + 1]
    assert model_path == "/external/pi05"


@pytest.mark.parametrize("native_success", [None, False, True])
def test_result_preserves_native_unfinished_status(tmp_path, native_success):
    (tmp_path / "states.json").write_text(
        json.dumps({"steps": [{"state": {"success": native_success}}]}),
        encoding="utf-8",
    )
    context = RunFinalizationContext(
        output_dir=tmp_path,
        robot_name="robolab",
        task_desc={"task": "BananaInBowlTask", "seed": 0},
        environment_success=bool(native_success),
        agent_error=None,
        elapsed_s=1.0,
        planner="api",
        model="offline",
        reasoning_effort="none",
        max_turns=2,
        planner_timeout_s=None,
        finish_result={"status": "stuck"},
        stats={},
    )
    path = robolab_spec.finalize_run(context)
    result = json.loads(path.read_text(encoding="utf-8"))
    assert result["success"] is native_success
    assert result["environment_result_available"] is (native_success is not None)


def test_robolab_pi05_encoder_preserves_droid_contract():
    obs = {
        "main_images": np.zeros((224, 224, 3), dtype=np.uint8),
        "wrist_images": np.ones((224, 224, 3), dtype=np.uint8),
        "states": np.arange(8, dtype=np.float32),
        "task_descriptions": "Put the banana in the bowl.",
    }
    client = Pi05VLAClient(object(), embodiment="robolab")
    encoded = client.encode_obs(obs)
    assert encoded["main_images"].shape == (1, 224, 224, 3)
    assert encoded["wrist_images"].shape == (1, 224, 224, 3)
    assert encoded["states"].shape == (1, 8)
    assert encoded["task_descriptions"] == [obs["task_descriptions"]]


class _FakeEnv:
    action_specs = {
        "default": box_action_spec(
            np.full(8, -10), np.full(8, 10), "DROID joint positions"
        )
    }

    def __init__(self):
        self.last_obs = {
            "main_images": np.zeros((224, 224, 3), dtype=np.uint8),
            "wrist_images": np.ones((224, 224, 3), dtype=np.uint8),
            "states": np.zeros(8, dtype=np.float32),
            "task_descriptions": "Put the banana in the bowl.",
        }
        self.terminated = False
        self.truncated = False
        self.last_info = {}
        self.actions = []

    def get_status(self):
        return {
            "steps": len(self.actions),
            "step_limit": 10,
            "success": True if len(self.actions) >= 2 else None,
            "terminated": self.terminated,
            "truncated": self.truncated,
        }

    def render_camera(self, name):
        return self.last_obs[
            "main_images" if name == "over_shoulder_left_camera" else "wrist_images"
        ]

    def validate_action(self, action):
        return validate_action(action, self.action_specs["default"])

    def step(self, action):
        self.actions.append(np.asarray(action))
        self.terminated = len(self.actions) >= 2
        self.last_info = self.get_status()
        return (
            self.last_obs,
            float(self.terminated),
            self.terminated,
            False,
            self.last_info,
        )


class _FakeModel:
    def predict(self, obs):
        assert obs["task_descriptions"] == "Put the banana in the bowl."
        return np.ones((15, 8), dtype=np.float32)


def test_policy_tool_executes_until_native_success_and_records_state(tmp_path):
    env = _FakeEnv()
    toolkit = RoboLabToolkit(
        env=env,
        model=_FakeModel(),
        dashboard_events=NullDashboardEventSink(),
        memory=MemoryManager(tmp_path / "memory"),
        output_dir=tmp_path / "run",
    )
    assert toolkit.state.get(0).state["success"] is None
    result = toolkit.execute_tool("pi05_act", {"steps": 5})
    assert result.data["state"]["success"] is True
    assert len(env.actions) == 2
    assert toolkit.solved()
    assert toolkit.state.get(1).terminated
    assert {"main.png", "wrist.png"} <= toolkit.state.get(1).artifacts
