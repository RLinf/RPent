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
from types import SimpleNamespace

import numpy as np
import pytest

from robots.robodojo import robot_spec, tools
from rpent.tools import Tool, iter_tools


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
        "seed": 0,
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
    assert dict(os.environ) == before
    assert args.sim_python == sys.executable


def test_local_spawn_uses_packaged_source_root(monkeypatch, tmp_path):
    calls = []

    def resolve(command, **kwargs):
        calls.append(command)
        return str(tmp_path / "packaged") + "\n"

    monkeypatch.setattr(robot_spec.subprocess, "check_output", resolve)
    monkeypatch.setenv("ROBODOJO_ASSETS_ROOT", str(tmp_path / "data"))
    overrides = robot_spec._runtime_overrides(_args())
    assert calls[0][0] == sys.executable
    assert overrides["ROBODOJO_SOURCE_ROOT"] == str(tmp_path / "packaged")
    assert overrides["ROBODOJO_ROOT"] == str(tmp_path / "packaged")
    assert overrides["ROBODOJO_ASSETS_ROOT"] == str(tmp_path / "data")


@pytest.mark.parametrize("inherited", [None, "", "2", "2,3", "GPU-example"])
@pytest.mark.parametrize("explicit", [None, 0, 3])
@pytest.mark.parametrize("component", ["env", "sam3", "vla"])
@pytest.mark.parametrize("backend", ["vpp2", "awomo05"])
def test_cuda_selection_is_only_overridden_explicitly(
    monkeypatch, tmp_path, inherited, explicit, component, backend
):
    if inherited is None:
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    else:
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", inherited)
    recorded = {}

    class Daemon:
        def __init__(self, **kwargs):
            recorded.update(kwargs)

        def start(self):
            pass

    monkeypatch.setattr(robot_spec, "ProcessDaemon", Daemon)
    flags = [] if explicit is None else ["--cuda-device", str(explicit)]
    args = _args("--source-root", str(tmp_path), "--vla-backend", backend, *flags)
    assert args.cuda_device == explicit
    if component == "vla":
        robot_spec._spawn_model_server(
            args, tmp_path, model_path=str(tmp_path), rpc_options={}
        )
    else:
        getattr(robot_spec, f"_spawn_{component}_server")(args, tmp_path)
    overrides = recorded.get("env_overrides", {})
    if component != "sam3":
        assert ("CUDA_VISIBLE_DEVICES" in overrides) == (explicit is not None)
        child_env = dict(os.environ, **overrides)
        assert child_env.get("CUDA_VISIBLE_DEVICES") == (
            inherited if explicit is None else str(explicit)
        )
    flag = "--cuda-device"
    cmd = recorded["cmd"]
    if component != "vla":
        assert (flag in cmd) == (explicit is not None)
        if explicit is not None:
            assert cmd[cmd.index(flag) + 1] == str(explicit)
    assert "None" not in cmd
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == inherited


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
    result = (
        robot_spec._spawn_model_server(args, tmp_path, model_path=None, rpc_options={})
        if component == "vla"
        else robot_spec._spawn_env_server(args, tmp_path)
    )
    assert result == (None, client)


def test_env_spawn_uses_explicit_interpreter_and_child_paths(monkeypatch, tmp_path):
    recorded = {}

    class Daemon:
        def __init__(self, **kwargs):
            recorded.update(kwargs)

        def start(self):
            recorded["started"] = True

    monkeypatch.setattr(robot_spec, "ProcessDaemon", Daemon)
    args = _args("--source-root", str(tmp_path), "--sim-python", sys.executable)
    robot_spec._spawn_env_server(args, tmp_path)
    assert recorded["cmd"][0] == sys.executable
    assert recorded["started"]
    assert "--parent-watch" in recorded["cmd"]
    assert recorded["cmd"][recorded["cmd"].index("--save-dir") + 1] == str(tmp_path)
    assert recorded["env_overrides"]["ROBODOJO_SOURCE_ROOT"] == str(tmp_path)


@pytest.mark.parametrize("backend", ["vpp2", "awomo05"])
def test_runtime_connects_selected_backend(monkeypatch, tmp_path, backend):
    from robots.robodojo.awomo05_client import Awomo05Client
    from robots.robodojo.vpp2_client import VPP2Client
    from rpent.dashboard.events import NullDashboardEventSink

    rpc = object()
    monkeypatch.setattr(robot_spec, "_spawn_model_server", lambda *a, **kw: (None, rpc))
    monkeypatch.setattr(robot_spec, "try_wait_server", lambda *a, post_fn: post_fn())
    daemons, kwargs = robot_spec._init_runtime(
        _args("--vla-backend", backend, "--vla-endpoint", "socket://localhost:1234"),
        tmp_path,
        NullDashboardEventSink(),
        {"vla"},
    )
    assert daemons == []
    client = kwargs["vla_client"]
    assert isinstance(client, VPP2Client if backend == "vpp2" else Awomo05Client)
    assert client._client is rpc


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
    result = tools.pi0_pick(primitives, None, "pick bottle", arm="right").data
    assert result["success"] is True
    assert result["chunks_used"] == 1
    assert set(result["arms"]) == {"left", "right"}
    assert result["arms"][lifting_arm]["peak_lift_m"] == 0.06
    np.testing.assert_array_equal(executed, actions)


def test_pi0_pick_stops_the_chunk_when_the_grasp_is_held():
    """The actions after a held grasp would reopen the gripper."""

    def observation(z):
        state = {}
        for arm in ("left", "right"):
            state[f"{arm}_ee_pose"] = [0, 0, z]
            state[f"{arm}_ee_joint_state"] = [0.2]
        return {
            "state": state,
            "vision": {
                name: {"color": np.zeros((2, 2, 3))}
                for name in ("cam_head", "cam_left_wrist", "cam_right_wrist")
            },
        }

    obs = observation(1.0)
    frames = iter([observation(0.94), observation(0.99), observation(0.99)])
    executed = []

    def step(action):
        executed.append(action)
        return next(frames), 0, False, {"status": {"step": 1, "step_limit": 99}}

    primitives = SimpleNamespace(
        env=SimpleNamespace(get_obs=lambda: obs, step=step),
        vla_client=SimpleNamespace(predict=lambda _: np.zeros((3, 14))),
        _check_cancelled=lambda: None,
    )
    result = tools.pi0_pick(primitives, None, "pick bottle", arm="right").data
    assert result["success"] is True
    # Three actions were predicted and the grasp is held after the second.
    assert len(executed) == 2


@pytest.mark.parametrize(
    "name", ["move_to", "set_gripper", "stabilize", "place_in_bin"]
)
def test_removed_motion_tools_have_no_implementation(name):
    assert not hasattr(tools, name)
    assert name not in {definition.name for definition in iter_tools(tools)}


@pytest.fixture
def saved_camera(tmp_path, fake_env):
    from rpent.session import EnvState

    def save(camera):
        depth = camera.get("depth", camera.get("distance_to_image_plane"))
        camera = {"color": np.zeros((*depth.shape, 3), dtype=np.uint8), **camera}
        env = fake_env(limit=10)
        env.get_obs = lambda: {"vision": {"cam_head": camera}, "instruction": "Pick."}
        state = EnvState(tmp_path)
        tools.dump_state(SimpleNamespace(env=env), state)
        return state

    return save


def test_readers_are_readonly_and_do_not_act(saved_camera):
    from rpent.robots.components.sam3_client import Sam3Result

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
        sam3_client=SimpleNamespace(segment=lambda *a, **kw: Sam3Result(found=False)),
    )
    for name in [
        "back_project",
        "segment",
        "view_env_state",
    ]:
        assert getattr(tools, name).readonly
    assert not isinstance(tools.get_reward_details, Tool)
    assert not isinstance(tools.get_safety_status, Tool)
    state = saved_camera(obs["vision"]["cam_head"])
    assert tools.back_project(state, 0, 0).data["world_xyz"] == [0, 0, -1]
    assert tools.segment(primitives, None, "object").data["found"] is False


def test_back_project_row_axis_follows_the_usd_camera_frame(saved_camera):
    """Image rows grow downward, the camera frame's +Y grows upward."""
    obs = {
        "vision": {
            "cam_head": {
                "depth": np.full((3, 3), 2.0),
                "intrinsic_matrix": np.array(
                    [[1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [0.0, 0.0, 1.0]]
                ),
                "extrinsic_matrix": np.eye(4),
            }
        }
    }
    state = saved_camera(obs["vision"]["cam_head"])
    above = tools.back_project(state, 0, 1).data["world_xyz"]
    below = tools.back_project(state, 2, 1).data["world_xyz"]
    assert above == [0.0, 2.0, -2.0]
    assert below == [0.0, -2.0, -2.0]


def test_back_project_applies_camera_to_world_transform(saved_camera):
    camera = {
        "distance_to_image_plane": np.full((3, 3), 2.0),
        "intrinsic_matrix": np.array(
            [[2.0, 0.0, 1.0], [0.0, 2.0, 1.0], [0.0, 0.0, 1.0]]
        ),
        "extrinsic_matrix": np.array(
            [[0, -1, 0, 3], [1, 0, 0, 4], [0, 0, 1, 5], [0, 0, 0, 1]]
        ),
    }
    state = saved_camera(camera)
    # Pixel (2, 2) is (1, -1, -2) in USD camera coordinates, then rotated
    # 90 degrees about Z and translated by (3, 4, 5).
    result = tools.back_project(state, 2, 2).data
    assert result["world_xyz"] == [4.0, 5.0, 3.0]


@pytest.mark.parametrize("depth", [0.0, -1.0, np.nan, np.inf])
def test_back_project_rejects_invalid_depth(depth, saved_camera):
    camera = {
        "depth": np.array([[depth]]),
        "intrinsic_matrix": np.eye(3),
        "extrinsic_matrix": np.eye(4),
    }
    state = saved_camera(camera)
    result = tools.back_project(state, 0, 0).data
    assert "invalid depth" in result["error"]
    assert "world_xyz" not in result


@pytest.mark.parametrize("task", ["put_bottles_into_dustbin", "stack_bowls_random"])
def test_toolkit_exposes_new_primitives_without_legacy_actions(
    monkeypatch, tmp_path, task, fake_env
):
    from robots.robodojo import toolkit as module
    from rpent.dashboard.events import NullDashboardEventSink
    from rpent.memory import MemoryManager
    from rpent.robots.base import get_toolkit
    from rpent.robots.robot_spec import RunConfig
    from rpent.utils import logging

    monkeypatch.setattr(logging, "_output_dir", tmp_path / "output")
    monkeypatch.setattr(module, "get_output_dir", lambda: tmp_path / "output")
    env = fake_env(limit=10)
    env.get_obs = lambda: {"vision": {}, "state": {}}
    env.get_task_language = lambda: "instruction"
    env.get_reward_details = lambda: {"score": 100}
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
    )
    assert isinstance(toolkit, module.RoboDojoToolkit)
    assert isinstance(toolkit.memory, MemoryManager)
    assert toolkit.memory.root == tmp_path / "memory"
    names = {spec.name for spec in toolkit.list_tools()}
    robot_names = {spec.name for spec in iter_tools(tools)}
    assert names & robot_names == set(module.OBSERVATION_TOOLS)
    assert (module.MOTION_TOOLS - {"vpp2_act"}) | {"render"} <= names
    assert "vpp2_act" not in names  # This component-only toolkit has no model.
    assert "reset" not in names
    assert "finish" in names
    for name in (
        "pi0_pick",
        "stabilize",
        "place_in_bin",
        "get_reward_details",
        "get_safety_status",
    ):
        assert name not in names
        assert toolkit.execute_tool(name, {}).data == {"error": f"unknown tool: {name}"}
    result = toolkit.execute_tool(
        "finish", {"status": "stuck", "summary": "Observation only"}
    ).data
    assert result["_finish"] is True
    assert result["status"] == "stuck"
    assert result["success"] is None
    assert set(robot_spec.ROBODOJO_DASHBOARD_SPEC["primitives"]) == {
        "view_env_state",
        "segment",
        "back_project",
        "render",
        "sample_world_xyz",
        "query_world_map",
        "move_to",
        "rotate_wrist",
        "set_gripper",
        "release",
        "vpp2_act",
        "awomo05_act",
    }


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
            "memory_dir": "/memory",
            "recipe_tag": task,
            "seed": 0,
        }
        system = bundle.render("system", variables=variables)
        user = bundle.render("user", variables=variables)
        assert "move_to" in system and "set_gripper" in system
        for name in ("pi0_pick", "stabilize", "place_in_bin"):
            assert name not in system + user
        assert "bottles_on_bin_bottom" not in user
        assert "registered tools" in system
        assert "JSON-RPC" not in system + user
        assert "MCP URL" not in system + user


@pytest.mark.parametrize("backend", ["vpp2", "awomo05"])
def test_default_runtime_starts_env_sam3_and_model(monkeypatch, tmp_path, backend):
    from rpent.dashboard.events import NullDashboardEventSink

    started = []
    waited = []
    for name in ("env", "sam3", "vla"):

        def spawn(*args, component=name, **kwargs):
            started.append(component)
            return None, object()

        monkeypatch.setattr(
            robot_spec, f"_spawn_{'model' if name == 'vla' else name}_server", spawn
        )

    def wait(daemons, events, component, *args, **kwargs):
        assert started == ["env", "vla", "sam3"]
        waited.append(component)
        return {}

    monkeypatch.setattr(robot_spec, "try_wait_server", wait)
    daemons, kwargs = robot_spec._init_runtime(
        _args("--vla-backend", backend, "--vla-endpoint", "socket://localhost:1234"),
        tmp_path,
        NullDashboardEventSink(),
        None,
    )
    assert started == ["env", "vla", "sam3"]
    assert waited == ["env", "sam3", "vla"]
    assert daemons == []
    assert "vla_client" not in kwargs
    assert "vla" in {
        item["name"]
        for item in robot_spec.ROBODOJO_DASHBOARD_SPEC["runtime_components"]
    }


def test_flash_rejected_before_services_start(monkeypatch, tmp_path):
    from rpent.dashboard.events import NullDashboardEventSink

    args = _args()
    args.planner = "flash"
    monkeypatch.setattr(
        robot_spec, "_spawn_env_server", lambda *a: pytest.fail("simulator started")
    )
    assert robot_spec.get_robot_spec().run_flash is None
    with pytest.raises(ValueError, match="Flash replay is unavailable"):
        robot_spec.get_robot_spec().validate_args(args)
    with pytest.raises(ValueError, match="Flash replay is unavailable"):
        robot_spec._init_runtime(args, tmp_path, NullDashboardEventSink(), None)


@pytest.fixture
def runtime_services(monkeypatch):
    from robots.robodojo import env_client, vpp2_server

    events = []
    failure = {}

    def fail(phase):
        if failure.get("phase") == phase:
            raise failure["error"]

    def spawn(name, config):
        events.append(("spawn", name))
        if name == "vla":
            fail("spawn")

        def call(method, **kwargs):
            events.append((method, name))
            assert method == "healthz"
            return {"status": "ok"}

        rpc = SimpleNamespace(call=call, close=lambda: events.append(("close", name)))
        # The environment is owned, while the model service is borrowed.
        daemon = (
            SimpleNamespace(
                name=name, poll=lambda: None, stop=lambda: events.append(("stop", name))
            )
            if name == "env"
            else None
        )
        return daemon, rpc

    for component, suffix in (
        ("env", "env"),
        ("sam3", "sam3"),
        ("vla", "model"),
    ):
        monkeypatch.setattr(
            robot_spec,
            f"_spawn_{suffix}_server",
            lambda config, output, name=component, **kwargs: spawn(name, config),
        )

    def environment(rpc, **kwargs):
        events.append(("connect", "env"))
        events.append(("vpp2_history", kwargs["record_vpp2_history"]))
        return rpc

    monkeypatch.setattr(env_client, "RoboDojoEnvClient", environment)
    from robots.robodojo import awomo05_client

    original = awomo05_client.Awomo05Client

    def connect(rpc):
        fail("connect")
        return original(rpc)

    monkeypatch.setattr(awomo05_client, "Awomo05Client", connect)
    from robots.robodojo import vpp2_client

    original_vpp2 = vpp2_client.VPP2Client

    def connect_vpp2(rpc):
        fail("connect")
        return original_vpp2(rpc)

    monkeypatch.setattr(vpp2_client, "VPP2Client", connect_vpp2)
    monkeypatch.setattr(
        vpp2_server,
        "model_paths",
        lambda *args: pytest.fail("unexpected VPP2 weight lookup"),
    )
    return events, failure


@pytest.mark.parametrize("backend", ["vpp2", "awomo05"])
@pytest.mark.parametrize("phase", ["spawn", "connect"])
@pytest.mark.parametrize("error_type", [ValueError, KeyboardInterrupt, SystemExit])
def test_runtime_failure_cleans_owned_processes_and_pending_connections(
    runtime_services, tmp_path, backend, phase, error_type
):
    from rpent.dashboard.events import NullDashboardEventSink

    events, failure = runtime_services
    error = error_type("interrupted startup")
    failure.update(phase=phase, error=error)
    config = _args("--vla-backend", backend, "--vla-endpoint", "http://model")
    with pytest.raises(
        RuntimeError if error_type is ValueError else error_type
    ) as caught:
        robot_spec._init_runtime(
            config, tmp_path, NullDashboardEventSink(), {"env", "vla"}
        )
    if error_type is not ValueError:
        assert caught.value is error
    assert [(op, name) for op, name in events if op == "stop"] == [("stop", "env")]
    assert {name for op, name in events if op == "close"} == (
        {"env"} if phase == "spawn" else {"env", "vla"}
    )
    assert (("connect", "env") in events) == (phase == "connect")


@pytest.mark.parametrize("backend", ["vpp2", "awomo05"])
def test_borrowed_model_rpc_uses_backend_session_options(
    monkeypatch, tmp_path, backend
):
    from rpent.dashboard.events import NullDashboardEventSink

    called = []
    monkeypatch.setattr(
        robot_spec,
        "make_rpc_client",
        lambda endpoint, **kw: called.append((endpoint, kw)),
    )
    monkeypatch.setattr(robot_spec, "try_wait_server", lambda *a, **kw: {})
    config = _args("--vla-backend", backend, "--vla-endpoint", "socket://model:1234")
    daemons, _ = robot_spec._init_runtime(
        config, tmp_path, NullDashboardEventSink(), {"vla"}
    )
    assert daemons == []
    assert called == [
        (
            "socket://model:1234",
            {"enable_sessions": True} if backend == "awomo05" else {},
        )
    ]


@pytest.mark.parametrize("backend", ["vpp2", "awomo05"])
def test_backends_accept_shared_sam3_endpoint(runtime_services, tmp_path, backend):
    from rpent.dashboard.events import NullDashboardEventSink

    events, _ = runtime_services
    config = _args("--vla-backend", backend, "--sam3-endpoint", "http://sam3")
    robot_spec._validate_args(config)
    _, runtime = robot_spec._init_runtime(
        config, tmp_path, NullDashboardEventSink(), {"sam3"}
    )
    assert "sam3_client" in runtime
    assert ("spawn", "sam3") in events
    assert not any(name == "vla" for _, name in events)


@pytest.mark.parametrize(
    "backend,variable", [("vpp2", "VPP2_MODEL_PATH"), ("awomo05", "AWOMO05_MODEL_PATH")]
)
def test_resource_root_uses_selected_backend_environment(
    monkeypatch, backend, variable
):
    from robots.robodojo.robot_spec import _vla_model_path

    monkeypatch.setenv("VPP2_MODEL_PATH", "/models/vpp2")
    monkeypatch.setenv("AWOMO05_MODEL_PATH", "/models/awomo")
    config = _args("--vla-backend", backend)
    assert (
        _vla_model_path(config)
        == {"VPP2_MODEL_PATH": "/models/vpp2", "AWOMO05_MODEL_PATH": "/models/awomo"}[
            variable
        ]
    )
    config.vla_model_path = "/explicit"
    assert _vla_model_path(config) == "/explicit"


@pytest.mark.parametrize("backend", ["vpp2", "awomo05"])
def test_missing_model_resources_fail_before_any_service_starts(
    monkeypatch, tmp_path, backend
):
    from robots.robodojo import robot_spec
    from rpent.dashboard.events import NullDashboardEventSink

    def unexpected_start(*args, **kwargs):
        pytest.fail("service started before resource validation")

    for name in ("env", "sam3", "model"):
        monkeypatch.setattr(robot_spec, f"_spawn_{name}_server", unexpected_start)
    config = _args(
        "--vla-backend", backend, "--vla-model-path", str(tmp_path / "missing")
    )
    with pytest.raises(FileNotFoundError, match="resource missing"):
        robot_spec._init_runtime(config, tmp_path, NullDashboardEventSink(), None)


@pytest.mark.parametrize("backend", ["vpp2", "awomo05"])
@pytest.mark.parametrize("phase", ["start", "connect"])
def test_model_startup_failure_stops_owned_process(
    tmp_path, monkeypatch, backend, phase
):
    from robots.robodojo import robot_spec

    events = []

    class Daemon:
        def __init__(self, **kwargs):
            pass

        def start(self):
            events.append("start")
            if phase == "start":
                raise RuntimeError("model startup failed")

        def stop(self):
            events.append("stop")

    def fail(*args, **kwargs):
        raise RuntimeError("model startup failed")

    monkeypatch.setattr(robot_spec, "ProcessDaemon", Daemon)
    monkeypatch.setattr(robot_spec, "pick_free_port", lambda: 9205)
    monkeypatch.setattr(robot_spec, "make_rpc_client", fail)
    config = _args(
        "--vla-backend",
        backend,
        "--vla-model-path",
        str(tmp_path),
    )
    with pytest.raises(RuntimeError, match="model startup failed"):
        robot_spec._spawn_model_server(
            config,
            tmp_path,
            model_path=str(tmp_path),
            rpc_options={"enable_sessions": True} if backend == "awomo05" else {},
        )
    assert events == ["start", "stop"]


@pytest.mark.parametrize(
    "flag,value", [("--action-type", "ee"), ("--env-cfg-type", "other")]
)
def test_awomo_requires_native_joint_robot(flag, value):
    with pytest.raises(ValueError):
        robot_spec._validate_args(
            _args(
                "--vla-backend",
                "awomo05",
                "--vla-endpoint",
                "socket://model:9005",
                flag,
                value,
            )
        )


@pytest.mark.parametrize("backend", ["vpp2", "awomo05"])
def test_model_spawn_owns_process_and_borrows_endpoint(monkeypatch, tmp_path, backend):
    recorded = {}

    class Daemon:
        def __init__(self, **kwargs):
            recorded.update(kwargs)

        def start(self):
            recorded["started"] = True

    monkeypatch.setattr(robot_spec, "ProcessDaemon", Daemon)
    monkeypatch.setattr(robot_spec, "pick_free_port", lambda: 9205)
    monkeypatch.setattr(
        robot_spec, "make_rpc_client", lambda endpoint, **kw: (endpoint, kw)
    )
    config = _args("--vla-backend", backend)
    options = {"enable_sessions": True} if backend == "awomo05" else {}
    daemon, rpc = robot_spec._spawn_model_server(
        config, tmp_path, model_path=str(tmp_path), rpc_options=options
    )
    assert daemon and recorded["started"]
    assert recorded["cmd"][:4] == [
        sys.executable,
        "-u",
        "-m",
        f"robots.robodojo.{backend}_server",
    ]
    assert "--parent-watch" in recorded["cmd"]
    assert rpc == ("socket://127.0.0.1:9205", options)
    recorded.clear()
    config.vla_endpoint = "socket://borrowed:9205"
    assert robot_spec._spawn_model_server(
        config, tmp_path, model_path="/missing", rpc_options=options
    ) == (None, (config.vla_endpoint, options))
    assert not recorded


@pytest.mark.parametrize(
    "backend_args,backend",
    [
        ([], "vpp2"),
        (["--vla-backend", "vpp2"], "vpp2"),
        (["--vla-backend", "awomo05"], "awomo05"),
    ],
)
def test_missing_model_path_fails_before_services_start(
    monkeypatch, tmp_path, backend_args, backend
):
    from rpent.dashboard.events import NullDashboardEventSink

    monkeypatch.delenv("VPP2_MODEL_PATH", raising=False)
    monkeypatch.delenv("AWOMO05_MODEL_PATH", raising=False)
    config = _args(*backend_args)
    assert config.vla_backend == backend
    with pytest.raises(SystemExit):
        _args("--vla-backend", "unknown")
    robot_spec._validate_args(config)
    for name in ("env", "sam3", "model"):
        monkeypatch.setattr(
            robot_spec,
            f"_spawn_{name}_server",
            lambda *a, **kw: pytest.fail("service started"),
        )
    with pytest.raises(ValueError, match="MODEL_PATH"):
        robot_spec._init_runtime(config, tmp_path, NullDashboardEventSink(), None)


@pytest.mark.parametrize(
    "backend",
    ["vpp2", "awomo05"],
)
def test_robodojo_backends_start_the_same_shared_services(
    monkeypatch, tmp_path, backend
):
    from dataclasses import replace

    from robots.robodojo import robot_spec
    from rpent.cli import dashboard
    from rpent.cli.main import _build_argparser
    from rpent.dashboard.server import DashboardServer

    parser = _build_argparser()
    robot_spec._add_cli_args(parser, True)
    args = parser.parse_args(
        [
            "--robot",
            "robodojo",
            "--dashboard",
            "--planner",
            "api",
            "--model",
            "test",
            "--vla-backend",
            backend,
            "--vla-endpoint",
            "http://policy",
            "--memory-profile",
            "local",
            "--output-dir",
            str(tmp_path),
        ]
    )
    started = []

    def init_runtime(args, output_dir, state, components):
        started.append(components)
        return [], {}

    def start(server):
        assert {
            component["name"]
            for component in server._state.dashboard_spec["runtime_components"]
        } == {"env", "vla", "sam3"}
        server._state.request_shutdown()
        return "http://127.0.0.1:12345"

    monkeypatch.setattr(DashboardServer, "start", start)
    spec = replace(robot_spec.get_robot_spec(), init_runtime=init_runtime)
    assert dashboard.run_dashboard_session(args, spec, parser=parser) == 0
    assert started == [{"vla", "sam3"}]
