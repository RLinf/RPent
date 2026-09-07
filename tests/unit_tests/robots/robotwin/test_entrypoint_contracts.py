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

"""RoboTwin CLI and Dashboard wiring with offline client boundaries."""

from __future__ import annotations

import argparse
import json
import sys
from types import SimpleNamespace

import pytest

from robots.robotwin import robot_spec
from robots.robotwin.env_client import RoboTwinEnvClient
from robots.robotwin.vla_client import LingBotVLAClient
from rpent.cli import dashboard as dashboard_cli
from rpent.cli import main as cli
from rpent.dashboard.events import NullDashboardEventSink, StepRecordEvent
from rpent.dashboard.state import ClaimedTask
from rpent.planner.base import PlannerResult
from rpent.session import EnvState


@pytest.mark.parametrize("components", [{"env"}, {"vla"}, None])
def test_runtime_builds_native_clients_and_resets_exactly_once(
    monkeypatch, tmp_path, components, robotwin
):
    env = robotwin.env
    metadata = robot_spec.env_runtime_contract(
        task_name="stack_blocks",
        task_config="demo_randomized",
        seed=7,
        max_episode_steps=200,
    )
    calls = []

    def call(name, **kwargs):
        calls.append(name)
        if name == "env.get_env_meta":
            return metadata
        assert name == "env.reset"
        return {}, {**env.last_info, "instruction": env.get_task_language()}

    rpc = SimpleNamespace(call=call)
    daemons = {name: object() for name in ("env", "vla")}

    def spawn(owned, events, name, starter):
        owned[name] = daemons[name]
        return daemons[name], rpc

    monkeypatch.setattr(robot_spec, "try_spawn_server", spawn)
    monkeypatch.setattr(robot_spec, "try_wait_server", lambda *a, post_fn: post_fn())
    monkeypatch.setattr(
        robot_spec,
        "_spawn_vla_server",
        lambda *a: (daemons["vla"], ("localhost", 9000)),
    )
    monkeypatch.setattr(robot_spec, "_wait_for_tcp", lambda *a, **kw: None)
    contracts = []
    monkeypatch.setattr(
        LingBotVLAClient,
        "validate_contract",
        lambda self, contract: contracts.append(contract),
    )
    args = argparse.Namespace(
        task_name="stack_blocks",
        task_config="demo_randomized",
        seed=7,
        max_episode_steps=200,
    )
    owned, resources = robot_spec._init_runtime(
        args, tmp_path, NullDashboardEventSink(), components
    )
    selected = {"env", "vla"} if components is None else components
    assert set(owned) == {daemons[name] for name in selected}
    assert set(resources) == (
        {"env", "seed", "seed_mode"} if "env" in selected else set()
    ) | ({"model"} if "vla" in selected else set())
    if "env" in selected:
        assert isinstance(resources["env"], RoboTwinEnvClient)
        assert resources["seed"] == 7 and resources["seed_mode"] == "exact"
        assert calls == ["env.get_env_meta", "env.reset"]
    if "vla" in selected:
        assert isinstance(resources["model"], LingBotVLAClient)
        assert contracts == [robot_spec.vla_runtime_contract()]


@pytest.fixture
def offline_planner(monkeypatch):
    toolkits = []
    videos = []
    save = EnvState.save

    def save_artifact(state, name, value, **kwargs):
        if name.endswith(".mp4"):
            videos.append((name, len(value)))
            return None
        return save(state, name, value, **kwargs)

    class Planner:
        def solve(self, *, toolkit, **kwargs):
            toolkits.append(toolkit)
            assert not toolkit.execute_tool(
                "release", {"arm": "left", "steps": 1}
            ).is_error
            assert not toolkit.execute_tool(
                "finish", {"status": "success", "summary": "planner claim"}
            ).is_error
            return PlannerResult(
                finish_result=toolkit.finish_result,
                messages=[],
                stats={"tool_calls": 2},
            )

    monkeypatch.setattr(EnvState, "save", save_artifact)
    return SimpleNamespace(planner=Planner(), toolkits=toolkits, videos=videos)


@pytest.mark.parametrize("success", [True, False])
def test_cli_constructs_native_toolkit_and_verifies_finish(
    robotwin, offline_planner, monkeypatch, tmp_path, success
):
    run = robotwin
    run.env.success_at = 1 if success else None
    stopped = []
    daemon = SimpleNamespace(stop=lambda: stopped.append(True))
    monkeypatch.setattr(
        robot_spec,
        "_init_runtime",
        lambda *args: ([daemon], {"env": run.env, "model": run.model, "seed": 7}),
    )
    monkeypatch.setattr(
        cli, "build_planner", lambda *args, **kwargs: offline_planner.planner
    )
    output_dir = tmp_path / "cli"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "rpent",
            "--robot",
            "robotwin",
            "--task-name",
            "stack_blocks",
            "--seed",
            "7",
            "--planner",
            "codex",
            "--memory-profile",
            "local",
            "--memory-dir",
            str(tmp_path / "memory"),
            "--output-dir",
            str(output_dir),
        ],
    )
    assert cli.main() == 0
    assert stopped == [True]
    assert offline_planner.videos == [("episode.mp4", 1)]
    result = json.loads(
        (output_dir / "transcript_robotwin_stack_blocks_s7.json").read_text()
    )
    assert result["finish"]["status"] == ("success" if success else "failure")
    assert result["task_name"] == "stack_blocks" and result["requested_seed"] == 7
    assert offline_planner.toolkits[0].solved() is success
    assert (
        offline_planner.toolkits[0]
        .execute_tool("release", {"arm": "left", "steps": 1})
        .is_error
    )


def test_dashboard_combines_shared_model_with_task_environment(
    robotwin, offline_planner, monkeypatch, tmp_path
):
    run = robotwin
    stopped = []
    daemon = SimpleNamespace(stop=lambda: stopped.append(True))
    components = []

    def init_runtime(args, output_dir, events, selected):
        components.append(selected)
        return [daemon], {"env": run.env, "seed": 7, "seed_mode": "exact"}

    monkeypatch.setattr(robot_spec, "_init_runtime", init_runtime)
    monkeypatch.setattr(
        dashboard_cli, "build_planner", lambda *args, **kwargs: offline_planner.planner
    )
    events = []
    state = SimpleNamespace(
        enabled=True, task_replacement_requested=False, emit=events.append
    )
    output_dir = tmp_path / "dashboard-task"
    args = argparse.Namespace(
        robot_name="robotwin",
        task_name="stack_blocks",
        task_config="demo_randomized",
        seed=7,
        memory_dir=tmp_path / "memory",
        verbose=False,
        explore=False,
        planner="codex",
        base_url=None,
        model="offline",
        max_tokens=128,
        planner_timeout_s=10,
        reasoning_effort="high",
        claude_code_max_budget_usd=None,
        no_images=False,
        max_turns=2,
    )
    error = dashboard_cli._run_dashboard_task(
        args=args,
        robot_spec=robot_spec.get_robot_spec(),
        state=state,
        claimed=ClaimedTask(number=1, request={}, output_dir=output_dir),
        shared_runtime_kwargs={"model": run.model},
        unique_components={"env"},
        session_root=tmp_path / "session",
    )
    assert error is None
    assert components == [{"env"}]
    assert stopped == [True]
    assert offline_planner.videos == [("action_release.mp4", 1), ("episode.mp4", 1)]
    assert [
        event.record.step_idx for event in events if isinstance(event, StepRecordEvent)
    ] == [0, 1]
    tk = offline_planner.toolkits[0]
    assert tk._robot.env is run.env
    assert tk._robot.model is run.model
    assert tk.execute_tool("release", {"arm": "left", "steps": 1}).is_error
    assert (output_dir / "transcript_robotwin_stack_blocks_s7.json").exists()
