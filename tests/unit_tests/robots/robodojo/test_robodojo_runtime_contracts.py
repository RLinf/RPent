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

"""Offline RoboDojo configuration and read-only dispatch contracts."""

import argparse
import json
import os
import sys
from functools import partial
from types import SimpleNamespace

import numpy as np
import pytest

from robots.robodojo import robot_spec, tools
from rpent.tools.toolkit import _is_readonly


def _args(*flags):
    parser = argparse.ArgumentParser()
    robot_spec._add_cli_args(parser, False)
    return parser.parse_args(["--task", "put_bottles_into_dustbin", *flags])


@pytest.mark.parametrize("success", [True, False, None])
@pytest.mark.parametrize("mode", ["dev", "eval-fair"])
def test_finalize_run_writes_environment_result(tmp_path, success, mode):
    from rpent.evaluation import RunFinalizationContext

    task_desc = {"task": "put_bottles_into_dustbin", "layout": 2}
    if mode == "eval-fair":
        task_desc["mode"] = mode
    context = RunFinalizationContext(
        output_dir=tmp_path,
        robot_name="robodojo",
        task_desc=task_desc,
        environment_success=success,
        agent_error="private error details",
        elapsed_s=12.34,
        planner="flash" if mode == "eval-fair" else "api",
        model=None,
        reasoning_effort="low",
        max_turns=20,
        planner_timeout_s=60,
        finish_result={"done": True},
        stats={},
    )
    path = robot_spec.get_robot_spec().finalize_run(context)
    assert path == tmp_path / "result.json"
    record = json.loads(path.read_text())
    assert record == {
        "task": task_desc["task"],
        "layout": 2,
        "mode": mode,
        "success": success,
        "environment_result_available": success is not None,
        "agent_error_present": True,
        "elapsed_s": 12.3,
        "planner": {
            "backend": context.planner,
            "model": None,
            "reasoning_effort": "low",
            "max_turns": 20,
        },
        "planner_timeout_s": 60,
    }
    assert not (tmp_path / ".result.json.tmp").exists()


def test_explicit_paths_do_not_read_workspace_or_mutate_parent(monkeypatch, tmp_path):
    monkeypatch.setenv("ROBODOJO_WORKSPACE", str(tmp_path))
    monkeypatch.setenv("PYTHONPATH", "inherited")
    args = _args(
        "--source-root",
        str(tmp_path / "source"),
        "--xpolicylab-root",
        str(tmp_path / "policy"),
    )
    before = dict(os.environ)
    env = robot_spec._runtime_overrides(args)
    assert str(tmp_path / "source") in env["PYTHONPATH"].split(os.pathsep)
    assert str(tmp_path / "policy") in env["PYTHONPATH"].split(os.pathsep)
    assert env["ROBODOJO_PI05_POLICY_ROOT"] == str(tmp_path / "policy/policy/Pi_05")
    assert dict(os.environ) == before
    assert args.sim_python == args.pi05_python == sys.executable


def test_local_spawn_requires_source_root():
    with pytest.raises(ValueError, match="--source-root"):
        robot_spec._runtime_overrides(_args())


def test_task_inventory_uses_explicit_checkout(tmp_path, monkeypatch):
    from robots.robodojo import tasks

    configs = tmp_path / "task/RoboDojo/config"
    configs.mkdir(parents=True)
    (configs / "example.yml").write_text("{}")
    (configs / "_task.yml").write_text("{}")
    monkeypatch.setenv("ROBODOJO_SOURCE_ROOT", "/ignored")
    assert tasks.list_tasks(tmp_path) == ["example"]
    assert tasks.validate_task("example", tmp_path) is None
    assert "unknown RoboDojo task" in tasks.validate_task("other", tmp_path)
    assert tasks.list_tasks() == []
    assert tasks.task_summary("remote") == {"task": "remote"}


@pytest.mark.parametrize("component", ["env", "vla"])
def test_borrowed_endpoint_needs_no_local_paths(monkeypatch, tmp_path, component):
    client = object()
    monkeypatch.setattr(robot_spec, "make_rpc_client", lambda endpoint: client)
    args = _args(f"--{component}-endpoint", "http://example:1234")
    assert getattr(robot_spec, f"_spawn_{component}_server")(args, tmp_path) == (
        None,
        client,
    )


@pytest.mark.parametrize(
    "component,flag", [("env", "--sim-python"), ("vla", "--pi05-python")]
)
def test_spawn_uses_explicit_interpreter_and_child_paths(
    monkeypatch, tmp_path, component, flag
):
    recorded = {}

    class Daemon:
        def __init__(self, **kwargs):
            recorded.update(kwargs)

        def start(self):
            recorded["started"] = True

    monkeypatch.setattr(robot_spec, "ProcessDaemon", Daemon)
    args = _args("--source-root", str(tmp_path), flag, sys.executable)
    getattr(robot_spec, f"_spawn_{component}_server")(args, tmp_path)
    assert recorded["cmd"][0] == sys.executable
    assert recorded["started"]
    assert "--parent-watch" in recorded["cmd"]
    directory_flag = "--output-dir" if component == "vla" else "--save-dir"
    assert recorded["cmd"][recorded["cmd"].index(directory_flag) + 1] == str(tmp_path)
    if component == "vla":
        cmd = recorded["cmd"]
        assert cmd[2:4] == ["-m", "rpent.robots.components.xpolicylab_vla_server"]
        assert cmd[cmd.index("--policy-root") + 1] == str(
            tmp_path / "XPolicyLab/policy/Pi_05"
        )
    assert recorded["env_overrides"]["ROBODOJO_PI05_POLICY_ROOT"] == str(
        tmp_path / "XPolicyLab/policy/Pi_05"
    )


@pytest.mark.parametrize("lifting_arm", ["left", "right"])
def test_pi0_pick_monitors_both_arms_and_preserves_policy_input(lifting_arm):
    def observation(z):
        state = {}
        for arm in ("left", "right"):
            state[f"{arm}_ee_pose"] = [0, 0, z if arm == lifting_arm else 1]
            state[f"{arm}_ee_joint_state"] = [0.2]
        return {
            "state": state,
            "vision": {
                name: {"color": np.zeros((2, 2, 3))}
                for name in ("cam_head", "cam_left_wrist", "cam_right_wrist")
            },
        }

    obs = observation(1)
    frames = iter([observation(0.9), observation(0.96)])
    actions = np.zeros((2, 14))
    executed = []

    def predict(value):
        assert value is obs
        assert value["instruction"] == "pick bottle"
        assert len(value["vision"]) == 3
        return actions

    def step(action):
        executed.append(action)
        return next(frames), 0, False, {"status": {"step": 1, "step_limit": 10}}

    primitives = SimpleNamespace(
        env=SimpleNamespace(get_obs=lambda: obs, step=step),
        vla_client=SimpleNamespace(predict=predict),
        _check_cancelled=lambda: None,
    )
    result = tools.pi0_pick(primitives, None, "pick bottle", arm="right")
    assert result["success"] is True
    assert result["chunks_used"] == 1
    assert set(result["arms"]) == {"left", "right"}
    assert result["arms"][lifting_arm]["peak_lift_m"] == 0.06
    np.testing.assert_array_equal(executed, actions)


def test_readers_are_readonly_and_do_not_act():
    obs = {
        "vision": {
            "cam_head": {
                "depth": np.ones((2, 2)),
                "color": np.zeros((2, 2, 3)),
                "intrinsic_matrix": np.eye(3),
                "extrinsic_matrix": np.eye(4),
            }
        }
    }
    primitives = SimpleNamespace(
        _last_obs=obs,
        sam3_client=SimpleNamespace(
            segment=lambda *a, **kw: SimpleNamespace(found=False)
        ),
    )
    for name in [
        "back_project",
        "segment",
        "view_env_state",
        "get_reward_details",
        "get_safety_status",
    ]:
        assert _is_readonly(partial(getattr(tools, name)))
    assert not _is_readonly(tools.move_to)
    assert tools.back_project(primitives, None, 0, 0)["world_xyz"] == [0, 0, -1]
    assert tools.segment(primitives, None, "object")["found"] is False


@pytest.mark.parametrize("task", ["put_bottles_into_dustbin", "stack_bowls_random"])
@pytest.mark.parametrize("groups", [None, frozenset({"general"}), frozenset()])
def test_tool_group_hook_filters_schemas_and_dispatch(
    monkeypatch, tmp_path, task, groups
):
    from robots.robodojo import toolkit as module
    from rpent.dashboard.events import NullDashboardEventSink
    from rpent.memory import MemoryManager
    from rpent.robots.base import get_toolkit
    from rpent.robots.robot_spec import RunConfig
    from rpent.utils import logging

    monkeypatch.setattr(logging, "_output_dir", tmp_path / "output")
    monkeypatch.setattr(module, "get_output_dir", lambda: tmp_path / "output")
    env = SimpleNamespace(
        get_obs=lambda: {"vision": {}, "state": {}},
        get_status=lambda: {"step": 0, "step_limit": 10},
        get_task_language=lambda: "instruction",
        get_reward_details=lambda: {"score": 100},
    )
    toolkit = get_toolkit(
        "robodojo",
        runtime_kwargs={"env": env, "task": task},
        dashboard_events=NullDashboardEventSink(),
        config=RunConfig(
            recipe_tag=task,
            output_dir=tmp_path / "output",
            prompt_vars={"memory_dir": str(tmp_path / "memory")},
            task_desc={"task": task},
        ),
        allowed_tool_groups=groups,
    )
    assert isinstance(toolkit, module.RoboDojoToolkit)
    assert isinstance(toolkit.memory, MemoryManager)
    assert toolkit.memory.root == tmp_path / "memory"
    names = {spec["name"] for spec in toolkit.get_tools_spec()}
    robot_names = {spec["name"] for spec in tools.TOOLS_SPEC}
    assert sum(map(len, tools.TOOL_GROUPS.values())) == len(robot_names)
    assert set().union(*tools.TOOL_GROUPS.values()) == robot_names
    expected = (
        robot_names
        if groups is None
        else set().union(*(tools.TOOL_GROUPS[group] for group in groups))
    )
    if task != "put_bottles_into_dustbin":
        expected = expected - {"place_in_bin"}
    assert names & robot_names == expected
    assert "finish" in names
    result = toolkit.execute_tool("get_reward_details", {}).result
    assert result == (
        {"score": 100}
        if groups is None
        else {"error": "unknown tool: get_reward_details"}
    )


@pytest.mark.parametrize("missing", ["runtime_kwargs", "dashboard_events", "config"])
def test_registry_toolkit_factory_requires_cli_arguments(missing):
    from rpent.robots.base import get_toolkit

    kwargs = {"runtime_kwargs": {}, "dashboard_events": None, "config": None}
    del kwargs[missing]
    with pytest.raises(TypeError, match=f"required keyword-only argument: '{missing}'"):
        get_toolkit("robodojo", **kwargs)


def test_registry_toolkit_factory_rejects_unknown_arguments():
    from rpent.robots.base import get_toolkit

    with pytest.raises(
        TypeError, match="unexpected keyword argument 'primitives_kwargs'"
    ):
        get_toolkit(
            "robodojo",
            runtime_kwargs={},
            dashboard_events=None,
            config=None,
            primitives_kwargs={},
        )


def test_tool_group_hook_rejects_unknown_group():
    from robots.robodojo.toolkit import RoboDojoToolkit

    with pytest.raises(ValueError, match="Unknown RoboDojo tool groups"):
        RoboDojoToolkit(
            primitives_kwargs={},
            dashboard_events=None,
            memory=None,
            allowed_tool_groups=frozenset({"typo"}),
        )


def test_task_guidance_is_scoped_and_tools_are_injected():
    bundle = robot_spec.get_robot_spec().prompts
    for task in ("put_bottles_into_dustbin", "stack_bowls_random"):
        variables = {
            "task": task,
            "output_dir": "/output",
            "layout": 0,
            "env_cfg_type": "arx_x5",
            "action_type": "joint",
            "task_summary": "scene",
        }
        system = bundle.render("system", variables=variables)
        user = bundle.render("user", variables=variables)
        assert "dustbin" not in system and "place_in_bin" not in system
        assert ("place_in_bin" in user) == (task == "put_bottles_into_dustbin")
        assert ("bottles_on_bin_bottom" in user) == (task == "put_bottles_into_dustbin")
        assert "planner supplies structured tools" in system
        assert "JSON-RPC" not in system + user
        assert "MCP URL" not in system + user


def test_place_in_bin_keeps_backend_phase_sequence(monkeypatch):
    calls = []

    def move(primitives, state, xyz, **kwargs):
        calls.append((xyz, kwargs["gripper"]))
        return {"reached": True}

    monkeypatch.setattr(tools, "move_to", move)
    monkeypatch.setattr(
        tools, "set_gripper", lambda *args: {"status": {"success": True}}
    )
    result = tools.place_in_bin(None, None, "left", [0.2, 0.3, 0.4])
    assert calls == [
        ([0.2, 0.3, 0.95], 1),
        ([0.2, 0.3, 0.78], 1),
        ([0.2, 0.3, 0.95], -1),
    ]
    assert result["released"] is True
    assert result["phases"]["release"]["status"]["success"] is True
