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

"""RoboCasa CLI and Dashboard wiring with offline client boundaries."""

from __future__ import annotations

import argparse
import json
import sys
from types import SimpleNamespace

import pytest

from robots.robocasa import robot_spec
from robots.robocasa.env_client import RoboCasaEnvClient
from robots.robocasa.vla_client import RoboCasaVLAClient
from rpent.cli import dashboard as dashboard_cli
from rpent.cli import main as cli
from rpent.dashboard.events import NullDashboardEventSink, StepRecordEvent
from rpent.dashboard.state import ClaimedTask
from rpent.planner.base import PlannerResult
from rpent.session import EnvState


@pytest.mark.parametrize("components", [{"env"}, {"vla"}, None])
def test_runtime_builds_native_resources_for_shared_and_unique_components(
    monkeypatch, tmp_path, components
):
    metadata = {
        "task_name": "OpenDrawer",
        "split": "target",
        "seed": 1,
        "camera_h": 256,
        "camera_w": 256,
    }
    rpc = SimpleNamespace(
        call=lambda name, **kwargs: metadata if name == "env.get_env_meta" else {}
    )
    daemons = {name: object() for name in ("env", "vla")}

    def spawn(owned, events, name, starter):
        owned[name] = daemons[name]
        return daemons[name], rpc

    def wait(owned, events, name, client, daemon, timeout, *, post_fn):
        return post_fn()

    monkeypatch.setattr(robot_spec, "try_spawn_server", spawn)
    monkeypatch.setattr(robot_spec, "try_wait_server", wait)
    args = argparse.Namespace(
        task_name="OpenDrawer", split="target", seed=1, hi_res=1024
    )
    owned, resources = robot_spec._init_runtime(
        args, tmp_path, NullDashboardEventSink(), components
    )
    selected = {"env", "vla"} if components is None else components
    assert set(owned) == {daemons[name] for name in selected}
    assert set(resources) == ({"env", "hi_res"} if "env" in selected else set()) | (
        {"model"} if "vla" in selected else set()
    )
    if "env" in selected:
        assert isinstance(resources["env"], RoboCasaEnvClient)
        assert resources["hi_res"] == 1024
    if "vla" in selected:
        assert isinstance(resources["model"], RoboCasaVLAClient)


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
            assert not toolkit.execute_tool("release", {"steps": 1}).is_error
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
def test_cli_constructs_native_toolkit_and_finalizes_environment_result(
    make_toolkit, offline_planner, monkeypatch, tmp_path, success
):
    run = make_toolkit()
    run.env.on_step = lambda: setattr(run.env, "success", success)
    stopped = []
    daemon = SimpleNamespace(stop=lambda: stopped.append(True))
    monkeypatch.setattr(
        robot_spec,
        "_init_runtime",
        lambda *args: ([daemon], {"env": run.env, "model": run.model}),
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
            "robocasa",
            "--task-name",
            "OpenDrawer",
            "--seed",
            "1",
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
    result = json.loads((output_dir / "result.json").read_text())
    assert result["valid"] is True
    assert result["success"] is success
    assert result["success_source"] == "state.success"
    assert result["task_name"] == "OpenDrawer" and result["seed"] == 1
    assert offline_planner.toolkits[0].execute_tool("release", {"steps": 1}).is_error


def test_dashboard_combines_shared_model_with_task_environment(
    make_toolkit, offline_planner, monkeypatch, tmp_path
):
    run = make_toolkit()
    stopped = []
    daemon = SimpleNamespace(stop=lambda: stopped.append(True))
    components = []

    def init_runtime(args, output_dir, events, selected):
        components.append(selected)
        return [daemon], {"env": run.env, "hi_res": None}

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
        robot_name="robocasa",
        task_name="OpenDrawer",
        split="target",
        seed=1,
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
    assert tk._robot._rldx._vla_client is run.model
    assert tk.execute_tool("release", {"steps": 1}).is_error
    assert (output_dir / "transcript_OpenDrawer_target_s1.json").exists()
