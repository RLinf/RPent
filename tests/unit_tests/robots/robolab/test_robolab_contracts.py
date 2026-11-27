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
from types import SimpleNamespace

import numpy as np
import pytest

import robots.robolab.robot_spec as robolab_spec
from components.models.openpi.client import Pi05VLAClient
from robots.robolab.env_server import RoboLabEnvFacade
from robots.robolab.toolkit import RoboLabToolkit
from rpent.components.env.action_spec import box_action_spec, validate_action
from rpent.components.models import backends as am_runtime
from rpent.dashboard.events import NullDashboardEventSink
from rpent.evaluation import RunFinalizationContext
from rpent.memory import MemoryManager
from rpent.robots.base import get_robot_spec


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


@pytest.mark.parametrize("env_gpu", [None, 1])
def test_runtime_keeps_virtualenv_python_symlinks(tmp_path, monkeypatch, env_gpu):
    executable = tmp_path / "python"
    executable.symlink_to(sys.executable)

    class _Daemon:
        def __init__(self, *, name, cmd, **kwargs):
            self.cmd = cmd
            self.env_overrides = kwargs["env_overrides"]

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
            "--cuda-device",
            "2",
            "--device",
            "cuda:2",
            *(["--env-cuda-device", str(env_gpu)] if env_gpu is not None else []),
        ]
    )
    env_daemon, _ = robolab_spec._spawn_env(args, tmp_path)
    vla_daemon, _ = robolab_spec._spawn_vla(args, tmp_path)
    assert Path(env_daemon.cmd[0]) == executable
    assert Path(vla_daemon.cmd[0]) == executable
    model_path = vla_daemon.cmd[vla_daemon.cmd.index("--model-path") + 1]
    assert model_path == "/external/pi05"
    assert env_daemon.cmd[env_daemon.cmd.index("--device") + 1] == "cuda:0"
    assert env_daemon.env_overrides["CUDA_VISIBLE_DEVICES"] == str(
        env_gpu if env_gpu is not None else 2
    )
    assert vla_daemon.env_overrides["CUDA_VISIBLE_DEVICES"] == "2"


def test_external_env_connection_ignores_local_gpu(monkeypatch, tmp_path):
    parser = argparse.ArgumentParser()
    robolab_spec.get_robot_spec().add_cli_args(parser, False)
    args = parser.parse_args(
        [
            "--task",
            "BananaInBowlTask",
            "--policy-backend",
            "none",
            "--env-endpoint",
            "http://remote:1234",
            "--device",
            "cuda:7",
        ]
    )

    class RPC:
        def call(self, method, **kwargs):
            if method == "env.get_env_meta":
                return {"task": "BananaInBowlTask", "seed": 0}
            assert method == "env.reset"
            return {"states": np.zeros(8)}

    rpc = RPC()
    monkeypatch.setattr(robolab_spec, "make_rpc_client", lambda endpoint: rpc)
    monkeypatch.setattr(
        robolab_spec,
        "try_wait_server",
        lambda *args, post_fn, **kwargs: post_fn(),
    )
    daemons, runtime = robolab_spec._init_runtime(
        args, tmp_path, NullDashboardEventSink(), None
    )
    assert daemons == []
    assert runtime["env"].last_obs["states"].shape == (8,)
    assert runtime["model"] is None


def test_explicit_memory_repository_sets_prompt_and_runtime_root(monkeypatch, tmp_path):
    root = tmp_path / "downloaded" / "robolab"
    root.mkdir(parents=True)
    args = argparse.Namespace(memory_repo="team/notes@revision", memory_dir=None)
    config = SimpleNamespace(prompt_vars={"memory_dir": str(tmp_path / "unused")})

    def prepare(source, robot):
        assert (source, robot) == ("team/notes@revision", "robolab")
        return root

    monkeypatch.setattr(robolab_spec, "prepare_explicit_source", prepare)
    robolab_spec._prepare_memory(args, config)
    assert config.prompt_vars["memory_dir"] == args.memory_dir == str(root)
    assert not (tmp_path / "unused").exists()


@pytest.mark.parametrize("success", [False, True])
@pytest.mark.parametrize("chunk", [False, True])
def test_native_episode_survives_artifact_reset_and_freezes(success, chunk):
    class RLinfEnv:
        task_descriptions = ["Task"]
        action_bounds = {"low": np.full(8, -10), "high": np.full(8, 10)}

        def __init__(self):
            self.calls = 0
            self.status = {
                "steps": 0,
                "step_limit": 3,
                "success": None,
                "terminated": False,
                "truncated": False,
            }

        def get_status(self):
            return [self.status.copy()]

        def step(self, action):
            assert tuple(action.shape) == (1, 8)
            self.calls += 1
            self.status["steps"] = [1, 0, 1, 2, 3][self.calls - 1]
            if self.calls == 5:
                self.status.update(
                    success=success, terminated=success, truncated=not success
                )
            return (
                {"states": np.zeros((1, 8))},
                np.array([float(self.status["terminated"])]),
                np.array([self.status["terminated"]]),
                np.array([self.status["truncated"]]),
                {},
            )

    env = RLinfEnv()
    facade = RoboLabEnvFacade(env, {"task": "Task", "seed": 0})
    assert facade.get_status()["success"] is None
    assert facade.get_task_language() == "Task"
    if chunk:
        result = facade.chunk_step(np.zeros((10, 8)), return_all_frames=True)
        assert len(result[0]) == 5
        assert result[0][0]["states"].shape == (8,)
    else:
        for steps in [1, 0, 1, 2]:
            result = facade.step(np.zeros(8))
            assert result[0]["states"].shape == (8,)
            assert result[2:4] == (False, False)
            assert result[4]["steps"] == steps
            assert result[4]["success"] is None
            assert facade.get_status() == result[4]
        result = facade.step(np.zeros(8))
    assert result[1:4] == (float(success), success, not success)
    assert result[4] == {
        "steps": 3,
        "step_limit": 3,
        "success": success,
        "terminated": success,
        "truncated": not success,
    }
    assert facade.get_status() == result[4]
    with pytest.raises(RuntimeError, match="episode has ended"):
        facade.step(np.zeros(8))
    assert env.calls == 5


@pytest.mark.parametrize(
    "metrics", [{}, {"success": None}, {"success": False}, {"success": True}]
)
def test_result_uses_current_run_metrics_in_reused_directory(tmp_path, metrics):
    native_success = metrics.get("success")
    (tmp_path / "states.json").write_text(
        json.dumps({"steps": [{"state": {"success": not native_success}}]}),
        encoding="utf-8",
    )
    context = RunFinalizationContext(
        output_dir=tmp_path,
        robot_name="robolab",
        task_desc={"task": "BananaInBowlTask", "seed": 0},
        environment_success=bool(native_success) if metrics else None,
        environment_metrics=metrics,
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
    assert toolkit.evaluation_metrics(success=False) == {"success": None}
    assert toolkit.state.get(0).state["success"] is None
    result = toolkit.execute_tool("pi05_act", {"steps": 5})
    assert result.data["state"]["success"] is True
    assert len(env.actions) == 2
    assert toolkit.evaluation_metrics(success=True) == {"success": True}
    assert toolkit.solved()
    assert toolkit.state.get(1).terminated
    assert {"main.png", "wrist.png"} <= toolkit.state.get(1).artifacts
