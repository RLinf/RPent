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

import importlib
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from rpent.cli.main import _build_argparser
from rpent.dashboard.events import NullDashboardEventSink
from rpent.robots.components.action_spec import box_action_spec, direct_action_tool_spec
from rpent.utils import templates


@pytest.fixture(params=["libero", "robocasa", "robotwin"])
def robot(request):
    return request.param


def make_primitive(robot, action_size=None):
    """Keep the real primitive and env client, replacing only RPC and rendering."""
    client_module = importlib.import_module(f"robots.{robot}.env_client")
    class_name = {"libero": "Libero", "robocasa": "RoboCasa", "robotwin": "RoboTwin"}[
        robot
    ]
    client_cls = getattr(client_module, class_name + "EnvClient")
    env = client_cls.__new__(client_cls)
    obs = {"states": np.zeros(8), "main_images": np.zeros((2, 2, 3), dtype=np.uint8)}
    info = {
        "executed_actions": 1,
        "episode_status": {
            "eval_success": False,
            "take_action_cnt": 1,
            "step_lim": 100,
            "actual_seed": 0,
        },
    }
    step_result = (
        (obs, 0.0, False, info)
        if robot == "robocasa"
        else (obs, 0.0, False, False, info)
    )
    size = action_size or {"libero": 7, "robocasa": 12, "robotwin": 14}[robot]
    specs = {
        "default": box_action_spec([-1] * size, [1] * size, "Test environment controls")
    }
    if robot == "robotwin":
        specs = {
            "qpos": box_action_spec(
                [-np.inf] * size, [np.inf] * size, "Test joint controls"
            ),
            "ee": box_action_spec([-np.inf] * 16, [np.inf] * 16, "Test pose controls"),
        }

    def call(method, **kwargs):
        return specs if method == "env.get_action_spec" else step_result

    env._client = SimpleNamespace(call=Mock(side_effect=call))
    env.last_obs = obs
    env.last_reset_info = {}
    if robot != "robocasa":
        env.terminated = env.truncated = False
    if robot == "libero":
        from robots.libero.tools import LiberoPrimitives

        primitive = LiberoPrimitives(env, None, None, lambda: None)
    elif robot == "robocasa":
        from robots.robocasa.primitives import RoboCasaPrimitives

        primitive = RoboCasaPrimitives.__new__(RoboCasaPrimitives)
        primitive.env = env
        primitive._check_cancelled = lambda: None
        primitive._vla_desync = False
        env.render_camera = lambda **kwargs: obs["main_images"]
    else:
        from robots.robotwin.primitives import RoboTwinPrimitives

        primitive = RoboTwinPrimitives(
            env=env, model=None, seed=0, check_cancelled=lambda: None
        )
    primitive.start_recording()
    return primitive


@pytest.mark.parametrize("enabled", [False, True])
def test_cli_factory_gates_tool_and_preserves_action_records(
    robot, enabled, monkeypatch, tmp_path
):
    module = importlib.import_module(f"robots.{robot}.toolkit")
    spec = importlib.import_module(f"robots.{robot}.robot_spec")
    class_name = {"libero": "Libero", "robocasa": "RoboCasa", "robotwin": "RoboTwin"}[
        robot
    ]
    toolkit_cls = getattr(module, class_name + "Toolkit")
    primitive = make_primitive(robot)
    (tmp_path / "memory").mkdir()
    (tmp_path / "memory" / "MEMORY.md").write_text("Offline corpus")
    monkeypatch.setattr(templates, "get_output_dir", lambda: tmp_path)
    monkeypatch.setattr(module, "get_output_dir", lambda: tmp_path)
    if robot == "robotwin":
        monkeypatch.setattr(module, "RoboTwinPrimitives", lambda **kwargs: primitive)
    else:
        monkeypatch.setattr(
            toolkit_cls,
            "init_primitives",
            lambda self, **kwargs: setattr(self, "_primitives", primitive),
        )

    def capture(self, *, command, result, elapsed_s):
        with self.state.record_step(
            state={}, command=command, result=result, elapsed_s=elapsed_s
        ):
            pass
        return result

    monkeypatch.setattr(toolkit_cls, "get_env_state", capture)
    parser = _build_argparser()
    spec._add_cli_args(parser, use_dashboard=False)
    args = [
        "--robot",
        robot,
        "--seed",
        "0",
        "--memory-profile",
        "local",
        "--output-dir",
        str(tmp_path),
        "--memory-dir",
        str(tmp_path / "memory"),
    ]
    args += (
        ["--suite", "libero_spatial", "--task", "0"]
        if robot == "libero"
        else ["--task-name", "test_task"]
    )
    if enabled:
        args.append("--enable-direct-action")
    config = spec._parse_config(parser.parse_args(args))
    toolkit = spec.get_toolkit(
        runtime_kwargs={}, dashboard_events=NullDashboardEventSink(), config=config
    )
    names = {s["name"] for s in toolkit.get_tools_spec()}
    assert ("execute_action" in names) == enabled
    assert "move_to" in names
    assert {"libero": "pi0_pick", "robocasa": "rldx_skill", "robotwin": "lingbot_act"}[
        robot
    ] in names
    values = [0.1] * {"libero": 7, "robocasa": 12, "robotwin": 14}[robot]
    result = toolkit.execute_tool("execute_action", {"values": values}).result
    if not enabled:
        assert "unknown tool" in result["error"]
        assert all(
            call.args[0] != "env.step"
            for call in primitive.env._client.call.call_args_list
        )
        return
    assert "error" not in result
    call = primitive.env._client.call.call_args
    assert call.args[0] == "env.step"
    np.testing.assert_array_equal(call.kwargs["args"][0], values)
    assert primitive.env.last_obs["main_images"].shape == (2, 2, 3)
    assert primitive.recorded_frame_count() == 1
    assert toolkit.state.latest_record().command == {
        "action": "execute_action",
        "values": values,
    }
    if robot == "libero":
        assert primitive._last_obs is primitive.env.last_obs
    elif robot == "robocasa":
        assert primitive._vla_desync is True
    else:
        assert primitive.native_actions == 1
        assert primitive.policy_actions == 0


@pytest.mark.parametrize("invalid", [[], [float("nan")], [float("inf")], [[0.0]]])
def test_invalid_action_never_reaches_environment(robot, invalid):
    primitive = make_primitive(robot)
    dim = {"libero": 7, "robocasa": 12, "robotwin": 14}[robot]
    values = invalid * dim
    with pytest.raises(ValueError):
        primitive.execute_action(values)
    assert all(
        call.args[0] != "env.step" for call in primitive.env._client.call.call_args_list
    )


@pytest.mark.parametrize("robot", ["libero", "robocasa"])
def test_out_of_range_controls_are_rejected(robot):
    primitive = make_primitive(robot)
    with pytest.raises(ValueError, match="environment.*bounds"):
        primitive.execute_action([1.1] * (7 if robot == "libero" else 12))
    assert all(
        call.args[0] != "env.step" for call in primitive.env._client.call.call_args_list
    )


def test_robotwin_ee_action_is_forwarded_with_its_native_type():
    primitive = make_primitive("robotwin")
    values = [0.1, 0.2, 0.3, 1, 0, 0, 0, 1] * 2
    primitive.execute_action(values, action_type="ee")
    call = primitive.env._client.call.call_args
    np.testing.assert_array_equal(call.kwargs["args"][0], values)
    assert call.kwargs["kwargs"] == {"action_type": "ee"}


def test_robotwin_rejects_wrong_layout_and_terminal_episode():
    primitive = make_primitive("robotwin")
    with pytest.raises(ValueError):
        primitive.execute_action([0] * 14, action_type="ee")
    primitive.env.terminated = True
    with pytest.raises(RuntimeError, match="terminal"):
        primitive.execute_action([0] * 14)
    assert all(
        call.args[0] != "env.step" for call in primitive.env._client.call.call_args_list
    )


def test_robocasa_reseeds_vla_after_direct_action():
    primitive = make_primitive("robocasa")
    primitive._rldx = SimpleNamespace(run=Mock(return_value={}))
    primitive.execute_action([0.0] * 12)
    primitive.env.last_obs["language"] = "open drawer"
    kwargs = {
        "base_clip": None,
        "max_chunks": 1,
        "use_prompt": False,
        "prompt": None,
        "force_reset": False,
        "n_action_steps": 1,
        "settle_patience": 1,
        "settle_eps": 0.0,
    }
    primitive.run_rldx_skill(**kwargs)
    assert primitive._rldx.run.call_args.kwargs["force_reset"] is True
    primitive.run_rldx_skill(**kwargs)
    assert primitive._rldx.run.call_args.kwargs["force_reset"] is False


def test_cancelled_direct_action_does_not_step(robot):
    primitive = make_primitive(robot)
    primitive._check_cancelled = Mock(side_effect=RuntimeError("cancelled"))
    with pytest.raises(RuntimeError, match="cancelled"):
        primitive.execute_action(
            [0.0] * {"libero": 7, "robocasa": 12, "robotwin": 14}[robot]
        )
    assert all(
        call.args[0] != "env.step" for call in primitive.env._client.call.call_args_list
    )


@pytest.mark.parametrize("robot", ["libero", "robocasa"])
@pytest.mark.parametrize("size", [3, 9, 20])
def test_direct_action_uses_environment_dimensions(robot, size):
    primitive = make_primitive(robot, action_size=size)
    tool = primitive.env.get_direct_action_tool_spec()
    values_schema = tool["input_schema"]["properties"]["values"]
    assert values_schema["minItems"] == values_schema["maxItems"] == size
    primitive.execute_action([0.25] * size)
    call = primitive.env._client.call.call_args
    assert call.args[0] == "env.step"
    np.testing.assert_array_equal(call.kwargs["args"][0], [0.25] * size)
    with pytest.raises(ValueError, match=f"{size} finite"):
        primitive.execute_action([0.0] * (size + 1))
    assert (
        sum(
            c.args[0] == "env.get_action_spec"
            for c in primitive.env._client.call.call_args_list
        )
        == 1
    )


def test_direct_action_uses_per_coordinate_environment_bounds():
    primitive = make_primitive("libero", action_size=3)
    primitive.env.action_specs = {
        "default": box_action_spec(
            [-2, 0, -np.inf], [2, 5, np.inf], "Custom controller"
        )
    }
    primitive.execute_action([1.5, 4.0, -10.0])
    with pytest.raises(ValueError, match="environment.*bounds"):
        primitive.execute_action([1.5, -0.1, 0.0])
    assert (
        sum(c.args[0] == "env.step" for c in primitive.env._client.call.call_args_list)
        == 1
    )
    tool = primitive.env.get_direct_action_tool_spec()
    assert "Custom controller" in tool["description"]
    assert "Infinity" not in tool["description"]


def test_tool_action_types_come_from_environment():
    specs = {"velocity": box_action_spec([-2] * 4, [2] * 4, "Wheel speeds")}
    tool = direct_action_tool_spec(specs)
    assert tool["input_schema"]["properties"]["action_type"]["enum"] == ["velocity"]
    assert tool["input_schema"]["properties"]["values"]["minItems"] == 4


@pytest.mark.parametrize("robot", ["libero", "robocasa"])
def test_environment_rpc_reports_native_bounds(robot):
    module = importlib.import_module(f"robots.{robot}.env_server")
    low, high = np.array([-2.0, 0.0, -0.5]), np.array([2.0, 6.0, 0.5])
    if robot == "libero":
        worker = SimpleNamespace(env_call=Mock(return_value=(low, high)))
        facade = module.LiberoEnvFacade(
            SimpleNamespace(env=SimpleNamespace(workers=[worker])), meta={}
        )
    else:
        facade = module.RoboCasaEnvFacade.__new__(module.RoboCasaEnvFacade)
        facade.env = SimpleNamespace(action_spec=(low, high))
    specs = facade.get_action_spec()
    assert specs["default"]["low"] == low.tolist()
    assert specs["default"]["high"] == high.tolist()
    if robot == "libero":
        assert "env.get_action_spec" in facade._readonly_methods
        worker.env_call.assert_called_once_with(
            "__getattribute__", args=["action_spec"], target="robosuite"
        )
