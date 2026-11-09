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

"""Configuration, CLI, registration and Toolkit template contracts."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from robots.behavior.env_client import BehaviorEnvClient, env_runtime_contract
from robots.behavior.env_server import BehaviorEnvFacade
from robots.behavior.robot_spec import get_robot_spec
from robots.behavior.tasks import TASK_LANGUAGE, TASK_NAMES
from robots.behavior.toolkit import BehaviorToolkit
from robots.behavior.vla_client import BehaviorVLAClient
from robots.behavior.vla_server import BehaviorVLAFacade
from rpent.dashboard.events import NullDashboardEventSink
from rpent.memory import MemoryManager
from rpent.robots.base import enumerate_robots
from rpent.robots.base import get_robot_spec as discover_robot
from tests.unit_tests.robots.behavior.conftest import (
    _SAM,
    FakeEnvBackend,
    LocalRpc,
    _args,
    _template_toolkit,
)


def test_registry_and_prompt_use_the_existing_robot_contract(tmp_path: Path) -> None:
    assert "behavior" in enumerate_robots()
    spec = discover_robot("behavior")
    assert spec.name == "behavior"
    assert not spec.supports_exploration
    assert spec.resolve_dashboard is None
    args = _args(tmp_path)
    spec.validate_args(args)
    config = spec.parse_config(args)
    assert config.task_desc["task_id"] == 40
    assert config.task_desc["execution_horizon"] == 2
    assert config.output_dir == tmp_path / "run"
    assert not config.output_dir.exists()
    prompt = spec.prompts.render("system", variables=config.prompt_vars)
    assert "make popcorn" in prompt
    assert "vla_act" in prompt and "view_env_state" in prompt
    assert "at most 2 native actions" in prompt
    user_prompt = spec.prompts.render("user", variables=config.prompt_vars)
    assert "Task 40: make_microwave_popcorn." in user_prompt
    assert args.memory_profile == "local"


@pytest.mark.parametrize("template", ["agent.yaml", "vla_only.yaml"])
def test_generic_templates_require_explicit_task_id(
    tmp_path: Path, template: str
) -> None:
    root = Path(__file__).resolve().parents[4]
    spec = get_robot_spec()
    args = _args(tmp_path)
    args.behavior_config = root / "robots/behavior/configs" / template
    args.task_id = args.task_language = args.execution_horizon = None
    with pytest.raises(ValueError, match="official BEHAVIOR task ID"):
        spec.validate_args(args)
    assert not (tmp_path / "run").exists()


def test_agent_template_still_needs_sam3_endpoint_for_segment_tool(
    tmp_path: Path,
) -> None:
    root = Path(__file__).resolve().parents[4]
    spec = get_robot_spec()
    args = _args(tmp_path)
    args.behavior_config = root / "robots/behavior/configs" / "agent.yaml"
    args.task_id = 7
    args.task_language = None
    with pytest.raises(ValueError, match="sam3_endpoint"):
        spec.validate_args(args)


def test_vla_only_template_runs_with_only_task_id_via_default_language(
    tmp_path: Path,
) -> None:
    root = Path(__file__).resolve().parents[4]
    spec = get_robot_spec()
    args = _args(tmp_path)
    args.behavior_config = root / "robots/behavior/configs" / "vla_only.yaml"
    args.task_id = 7
    args.task_language = None
    spec.validate_args(args)
    config = spec.parse_config(args)
    assert config.prompt_vars["task_language"] == TASK_LANGUAGE[7]


@pytest.mark.parametrize("task_id", [0, 7, 40, 99])
def test_omitted_task_language_defaults_to_official_instruction(
    tmp_path: Path, task_id: int
) -> None:
    spec = get_robot_spec()
    args = _args(tmp_path, "--task-id", str(task_id))
    args.task_language = None
    spec.validate_args(args)
    config = spec.parse_config(args)
    assert config.prompt_vars["task_language"] == TASK_LANGUAGE[task_id]
    assert config.task_desc["task_language"] == TASK_LANGUAGE[task_id]


def test_explicit_task_language_overrides_the_official_default(
    tmp_path: Path,
) -> None:
    spec = get_robot_spec()
    args = _args(tmp_path, "--task-id", "7", "--task-language", "Custom text.")
    spec.validate_args(args)
    config = spec.parse_config(args)
    assert config.prompt_vars["task_language"] == "Custom text."
    assert config.prompt_vars["task_language"] != TASK_LANGUAGE[7]


@pytest.mark.parametrize(
    "task_id, task_name",
    [
        (0, "turning_on_radio"),
        (7, "picking_up_toys"),
        (40, "make_microwave_popcorn"),
        (99, "sorting_books_on_shelf"),
    ],
)
def test_official_task_identity_reaches_prompt_and_run_metadata(
    tmp_path: Path, task_id: int, task_name: str
) -> None:
    spec = get_robot_spec()
    args = _args(tmp_path, "--task-id", str(task_id), "--task-language", "Task text")
    spec.validate_args(args)
    config = spec.parse_config(args)
    assert config.recipe_tag == f"behavior_{task_name}"
    assert config.task_desc["task_id"] == task_id
    assert config.task_desc["task_name"] == task_name
    prompt = spec.prompts.render("user", variables=config.prompt_vars)
    assert f"Task {task_id}: {task_name}." in prompt
    assert config.prompt_vars["task_language"] == "Task text"
    assert not config.output_dir.exists()


def test_official_catalog_has_all_100_unique_task_directory_names() -> None:
    assert set(TASK_NAMES) == set(range(100))
    assert len(set(TASK_NAMES.values())) == 100


@pytest.mark.parametrize(
    "key,value",
    [
        ("execution_horizon", 0),
        ("execution_horizon", True),
        ("task_id", -1),
        ("task_id", 100),
        ("task_id", True),
        ("task_language", " "),
        ("env_endpoint", "ws://localhost:8000"),
        ("vla_endpoint", "http://localhost:99999"),
        ("control_mode", "unknown"),
        ("instruction_mode", "subtask"),
        ("max_chunks_per_call", 0),
        ("max_chunks_per_call", True),
        ("tool_allowlist", ["vla_act"]),
        ("tool_allowlist", ["finish", "view_env_state", "unregistered_tool"]),
        ("env_rpc_timeout_s", 0),
        ("env_rpc_timeout_s", float("nan")),
        ("env_rpc_timeout_s", True),
    ],
)
def test_invalid_settings_fail_before_runtime(tmp_path: Path, key: str, value) -> None:
    args = _args(tmp_path)
    setattr(args, key, value)
    with pytest.raises(ValueError):
        get_robot_spec().validate_args(args)


def test_unknown_yaml_keys_are_not_silently_ignored(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("task_id: 40\ntask_language: popcorn\nexecution_horizn: 2\n")
    args = _args(tmp_path, "--behavior-config", str(path))
    with pytest.raises(ValueError, match="documented task settings"):
        get_robot_spec().parse_config(args)


@pytest.mark.parametrize("contents", [None, "task_id: [40\n"])
def test_missing_or_malformed_yaml_is_a_cli_validation_error(tmp_path: Path, contents):
    path = tmp_path / "invalid.yaml"
    if contents is not None:
        path.write_text(contents)
    args = _args(tmp_path, "--behavior-config", str(path))

    with pytest.raises(ValueError, match="cannot read BEHAVIOR config") as caught:
        get_robot_spec().validate_args(args)

    assert str(path) in str(caught.value)
    assert caught.value.__cause__ is not None
    assert not (tmp_path / "run").exists()


def test_runtime_borrows_facades_without_reset_or_shutdown(
    tmp_path: Path, monkeypatch
) -> None:
    calls = []

    class Transport:
        def call(self, method, args=(), kwargs=None, timeout_s=None):
            calls.append(method)
            if method == "healthz":
                return {"ok": True}
            if method == "env.get_env_meta":
                return env_runtime_contract(40)
            raise AssertionError(f"unexpected runtime call: {method}")

    monkeypatch.setattr("rpent.utils.rpc.make_rpc_client", lambda endpoint: Transport())
    daemons, runtime = get_robot_spec().init_runtime(
        _args(tmp_path), tmp_path, NullDashboardEventSink(), None
    )
    assert daemons == []
    assert set(runtime) == {
        "env",
        "model",
        "task_id",
        "task_language",
        "execution_horizon",
        "max_chunks_per_call",
        "instruction_mode",
        "tool_allowlist",
    }
    assert "env.reset" not in calls and "shutdown" not in calls
    assert calls.count("healthz") == 2


@pytest.mark.parametrize("component", ["env", "vla"])
def test_selected_component_needs_only_its_endpoint(
    tmp_path: Path, monkeypatch, component: str
) -> None:
    endpoints = []

    class Transport:
        def call(self, method, **kwargs):
            if method == "healthz":
                return {"ok": True}
            assert method == "env.get_env_meta"
            return env_runtime_contract(40)

    def make_client(endpoint):
        endpoints.append(endpoint)
        return Transport()

    monkeypatch.setattr("rpent.utils.rpc.make_rpc_client", make_client)
    args = _args(tmp_path)
    unselected = "env" if component == "vla" else "vla"
    setattr(args, f"{unselected}_endpoint", None)
    # A normal full CLI run still requires both endpoints.
    with pytest.raises(ValueError, match=f"{unselected}_endpoint"):
        get_robot_spec().validate_args(args)

    _, runtime = get_robot_spec().init_runtime(
        args, tmp_path, NullDashboardEventSink(), {component}
    )
    assert endpoints == [getattr(args, f"{component}_endpoint")]
    selected_key = "env" if component == "env" else "model"
    unselected_key = "model" if component == "env" else "env"
    assert selected_key in runtime and unselected_key not in runtime


def test_agent_mode_defaults_preserve_the_main_instruction_and_return_control(
    tmp_path: Path,
) -> None:
    spec = get_robot_spec()
    args = _args(tmp_path, "--control-mode", "agent")
    config = spec.parse_config(args)
    assert config.task_desc["max_chunks_per_call"] == 1
    assert config.task_desc["instruction_mode"] == "main"
    prompt = spec.prompts.render("system", variables=config.prompt_vars)
    assert "at most 1 VLA chunks per tool call" in prompt
    assert "subtask text and other instruction overrides are rejected" in prompt
    assert "head, left_wrist, right_wrist" in prompt
    assert "session_ended" in prompt and "not an environment action count" in prompt
    assert "{{" not in prompt


def test_legacy_config_retains_scripted_policy_chain_behavior(tmp_path: Path) -> None:
    config = get_robot_spec().parse_config(_args(tmp_path))
    assert config.task_desc["control_mode"] == "scripted"
    assert config.task_desc["max_chunks_per_call"] is None
    assert config.task_desc["instruction_mode"] == "override"


@pytest.mark.parametrize("template", ["agent.yaml", "vla_only.yaml"])
def test_agent_templates_take_cli_task_and_require_actual_endpoints(
    tmp_path: Path, template: str
) -> None:
    root = Path(__file__).resolve().parents[4]
    spec = get_robot_spec()
    args = _args(tmp_path, "--task-id", "7", "--task-language", "Put away the toys.")
    args.behavior_config = root / "robots/behavior/configs" / template
    args.execution_horizon = None
    args.env_endpoint = args.vla_endpoint = None
    config = spec.parse_config(args)
    assert config.task_desc["task_id"] == 7
    assert config.task_desc["task_name"] == "picking_up_toys"
    assert config.task_desc["task_language"] == "Put away the toys."
    assert config.task_desc["execution_horizon"] == 32
    assert config.task_desc["max_chunks_per_call"] == 1
    assert config.task_desc["env_rpc_timeout_s"] == 1230
    assert config.task_desc["control_mode"] == "agent"
    assert config.task_desc["instruction_mode"] == "main"
    assert {"model", "base_url", "max_turns"}.isdisjoint(config.task_desc)
    with pytest.raises(ValueError, match="env_endpoint"):
        spec.validate_args(args)
    args.env_endpoint = "socket://localhost:24000"
    args.vla_endpoint = "socket://localhost:24002"
    if template == "agent.yaml":
        with pytest.raises(ValueError, match="sam3_endpoint"):
            spec.validate_args(args)
        args.sam3_endpoint = "socket://localhost:24003"
    else:
        assert config.task_desc["tool_allowlist"] == [
            "view_env_state",
            "vla_act",
            "finish",
        ]
    spec.validate_args(args)
    assert not (tmp_path / "run").exists()


def test_disabled_vla_is_not_advertised_in_the_prompt(tmp_path: Path) -> None:
    spec = get_robot_spec()
    args = _args(tmp_path, "--tool-allowlist", "view_env_state", "finish")
    config = spec.parse_config(args)
    prompt = spec.prompts.render("system", variables=config.prompt_vars)
    assert "vla_act" not in prompt


def test_new_motion_tools_are_opt_in(tmp_path):
    spec = get_robot_spec()
    default = spec.parse_config(_args(tmp_path, "--control-mode", "agent"))
    assert "move_eef_delta" not in default.task_desc["tool_allowlist"]
    assert "read_robot_state" not in default.task_desc["tool_allowlist"]
    args = _args(
        tmp_path, "--tool-allowlist", "view_env_state", "finish", "move_eef_delta"
    )
    config = spec.parse_config(args)
    prompt = spec.prompts.render("system", variables=config.prompt_vars)
    assert "move_eef_delta" in prompt
    assert "single-call relative bound" in prompt
    assert "vla_act" not in prompt


def test_robot_state_reader_needs_no_extra_config(tmp_path):
    args = _args(
        tmp_path, "--tool-allowlist", "view_env_state", "read_robot_state", "finish"
    )
    config = get_robot_spec().parse_config(args)
    assert "read_robot_state" in config.task_desc["tool_allowlist"]


def test_geometry_config_selects_pinned_asset_without_changing_vla_recipe(tmp_path):
    args = _args(tmp_path)
    args.behavior_config = tmp_path / "custom_geometry.yaml"
    args.behavior_config.write_text(
        "control_mode: agent\ninstruction_mode: main\nexecution_horizon: 32\n"
        "tool_allowlist: [view_env_state, vla_act, move_eef_delta, finish]\n"
    )
    args.execution_horizon = None
    config = get_robot_spec().parse_config(args)
    assert config.task_desc["execution_horizon"] == 32
    assert config.task_desc["instruction_mode"] == "main"
    assert "move_eef_delta" in config.task_desc["tool_allowlist"]


_CAMERA_KEYS = (
    "robot_r1::robot_r1:zed_link:Camera:0::rgb",
    "robot_r1::robot_r1:left_realsense_link:Camera:0::rgb",
    "robot_r1::robot_r1:right_realsense_link:Camera:0::rgb",
)


_PROPRIO_KEY = "robot_r1::proprio"


def _observation(value: int) -> dict[str, Any]:
    observation = {
        key: np.full((1, 4, 5, 3), value + index, dtype=np.uint8)
        for index, key in enumerate(_CAMERA_KEYS)
    }
    observation[_PROPRIO_KEY] = np.full((1, 256), value, dtype=np.float32)
    observation["task_id"] = np.array([40])
    observation[_CAMERA_KEYS[0].replace("::rgb", "::depth_linear")] = np.full(
        (1, 4, 5, 1), 1.0, dtype=np.float32
    )
    return observation


class _Environment:
    def __init__(self) -> None:
        self.observation = _observation(0)
        self.actions: list[np.ndarray] = []
        self.executed_count: int | None = 2
        self.terminated: bool | None = False
        self.truncated: bool | None = False
        self.closed = False
        self.observation_reads = 0

    def get_observation(self) -> dict[str, Any]:
        self.observation_reads += 1
        return self.observation

    def chunk_step(self, actions: np.ndarray) -> tuple:
        self.actions.append(actions.copy())
        self.observation = _observation(len(self.actions))
        info = {}
        if self.executed_count is not None:
            info["executed_actions"] = self.executed_count
        return self.observation, None, self.terminated, self.truncated, info

    def close(self) -> None:
        self.closed = True


class _Model:
    def __init__(self) -> None:
        self.inputs: list[tuple[dict[str, Any], str]] = []
        self.error: Exception | None = None
        self.closed = False
        self.action_horizon = 4

    def infer(self, observation: dict[str, Any], prompt: str) -> np.ndarray:
        self.inputs.append((observation, prompt))
        if self.error is not None:
            raise self.error
        return np.arange(self.action_horizon * 23, dtype=np.float32).reshape(-1, 23)

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def components(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("rpent.utils.templates.get_output_dir", lambda: tmp_path)
    env, model = _Environment(), _Model()
    toolkit = BehaviorToolkit(
        runtime_kwargs={
            "env": env,
            "model": model,
            "task_id": 40,
            "task_language": "make popcorn",
            "execution_horizon": 2,
        },
        dashboard_events=NullDashboardEventSink(),
        memory=MemoryManager(tmp_path / "memory"),
        state_output_dir=tmp_path / "state",
    )
    yield toolkit, env, model
    toolkit.close()


def test_toolkit_registers_tools_and_records_initial_images(components) -> None:
    toolkit, env, model = components
    declarations = {item.name: item for item in toolkit.list_tools()}
    assert set(declarations) == {
        "read_text_file",
        "write_text_file",
        "list_dir",
        "finish",
        "view_env_state",
        "vla_act",
    }
    assert declarations["view_env_state"].readonly
    assert declarations["finish"].readonly
    assert not declarations["vla_act"].readonly
    initial = toolkit.execute_tool("view_env_state", {"step": 0})
    assert not initial.is_error
    assert initial.data["step_kind"] == "observation_record"
    assert initial.data["log"]["command"] == {"action": "observe"}
    assert len(initial.images) == 3
    assert all(image.startswith(b"\x89PNG") for image in initial.images)
    np.testing.assert_array_equal(
        toolkit.state.load("proprio.npy", step=0), np.zeros(256)
    )
    np.testing.assert_array_equal(
        toolkit.state.load("head_depth.npy", step=0), np.ones((4, 5))
    )
    assert "policy_observation" not in initial.to_text()
    assert "env_step" not in initial.to_text()
    assert toolkit.state.latest_step == 0
    assert env.actions == []
    assert model.inputs == []


def test_vla_tool_refreshes_observations_and_records_actual_execution(
    components,
) -> None:
    toolkit, env, model = components
    result = toolkit.execute_tool("vla_act", {"chunks": 2, "prompt": "open microwave"})
    assert not result.is_error
    assert len(env.actions) == 2
    assert all(actions.shape == (2, 23) for actions in env.actions)
    assert [prompt for _, prompt in model.inputs] == ["open microwave"] * 2
    assert model.inputs[0][0][_CAMERA_KEYS[0]].shape == (4, 5, 3)
    assert model.inputs[0][0][_PROPRIO_KEY].shape == (256,)
    assert model.inputs[0][0][_CAMERA_KEYS[0]][0, 0, 0] == 0
    assert model.inputs[1][0][_CAMERA_KEYS[0]][0, 0, 0] == 1
    execution = result.data["log"]["result"]
    assert execution["requested_actions"] == 4
    assert execution["executed_actions"] == 4
    assert "success" not in execution
    assert len(result.images) == 3
    assert toolkit.state.latest_step == 1
    np.testing.assert_array_equal(
        toolkit.state.load("head_rgb.png", step=1), np.full((4, 5, 3), 2)
    )


def test_missing_execution_feedback_stays_unknown(components) -> None:
    toolkit, env, model = components
    env.executed_count = env.terminated = env.truncated = None
    result = toolkit.execute_tool("vla_act", {})
    execution = result.data["log"]["result"]
    assert execution["executed_actions"] is None
    assert execution["terminated"] is None
    assert execution["truncated"] is None
    assert model.inputs[0][1] == "make popcorn"


@pytest.mark.parametrize("flag", ["terminated", "truncated"])
def test_terminal_feedback_stops_chunks_and_later_action_calls(
    components, flag: str
) -> None:
    toolkit, env, model = components
    setattr(env, flag, True)
    first = toolkit.execute_tool("vla_act", {"chunks": 3})
    second = toolkit.execute_tool("vla_act", {"chunks": 1})
    assert first.data["log"]["result"]["stop_reason"] == flag
    assert second.data["log"]["result"]["stop_reason"] == flag
    assert len(env.actions) == len(model.inputs) == 1
    assert toolkit.solved() is False


def test_partial_execution_does_not_request_another_chunk(components) -> None:
    toolkit, env, model = components
    env.executed_count = 1
    result = toolkit.execute_tool("vla_act", {"chunks": 3})
    assert result.data["log"]["result"]["stop_reason"] == "partial_execution"
    assert result.data["log"]["result"]["executed_actions"] == 1
    assert len(env.actions) == len(model.inputs) == 1


def test_tool_failure_retains_error_and_records_observation(components) -> None:
    toolkit, env, model = components
    model.error = RuntimeError("policy backend unavailable")
    result = toolkit.execute_tool("vla_act", {})
    assert result.is_error
    assert result.data["error"] == "policy backend unavailable"
    assert result.data["log"]["result"]["error"] == "policy backend unavailable"
    assert len(result.images) == 3
    assert env.actions == []
    assert toolkit.state.latest_step == 1


@pytest.mark.parametrize("arguments", [{"chunks": 0}, {"chunks": True}, {"unknown": 1}])
def test_invalid_tool_arguments_do_not_dispatch(components, arguments: dict) -> None:
    toolkit, env, model = components
    assert toolkit.execute_tool("vla_act", arguments).is_error
    assert toolkit.state.latest_step == 0
    assert env.actions == []
    assert model.inputs == []


def test_finish_does_not_claim_official_success_or_export_a_recipe(components) -> None:
    toolkit, env, model = components
    result = toolkit.execute_tool(
        "finish", {"status": "success", "summary": "looks done"}
    )
    assert result.data == {
        "_finish": True,
        "status": "unverified",
        "requested_status": "success",
        "summary": "looks done",
        "success": None,
    }
    assert toolkit.state.latest_step == 0
    assert toolkit.solved() is False
    assert toolkit.write_recipe("behavior_test") is None
    toolkit.close()
    assert env.closed is False
    assert model.closed is False


@pytest.fixture
def agent_components(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("rpent.utils.templates.get_output_dir", lambda: tmp_path)
    env, model = _Environment(), _Model()
    env.executed_count = 32
    model.action_horizon = 64
    toolkit = BehaviorToolkit(
        runtime_kwargs={
            "env": env,
            "model": model,
            "task_id": 40,
            "task_language": "make popcorn",
            "execution_horizon": 32,
            "max_chunks_per_call": 1,
            "instruction_mode": "main",
        },
        dashboard_events=NullDashboardEventSink(),
        memory=MemoryManager(tmp_path / "memory"),
        state_output_dir=tmp_path / "state",
    )
    yield toolkit, env, model
    toolkit.close()


def test_agent_schema_rejects_long_calls_before_dispatch(agent_components) -> None:
    toolkit, env, model = agent_components
    definition = next(item for item in toolkit.list_tools() if item.name == "vla_act")
    assert definition.input_schema["properties"]["chunks"]["maximum"] == 1
    result = toolkit.execute_tool("vla_act", {"chunks": 2})
    assert result.is_error
    assert model.inputs == [] and env.actions == []
    assert toolkit.state.latest_step == 0


def test_agent_returns_control_after_32_actions_and_records_once(
    agent_components,
) -> None:
    toolkit, env, model = agent_components
    result = toolkit.execute_tool("vla_act", {})
    assert not result.is_error
    assert len(model.inputs) == len(env.actions) == 1
    assert env.actions[0].shape == (32, 23)
    assert model.inputs[0][1] == "make popcorn"
    assert result.data["log"]["result"]["executed_actions"] == 32
    assert len(result.images) == 3
    assert toolkit.state.latest_step == 1
    observation_reads = env.observation_reads
    toolkit.execute_tool("view_env_state", {"step": 0})
    toolkit.execute_tool("view_env_state", {"step": -1})
    assert env.observation_reads == observation_reads
    assert len(env.actions) == 1


def test_main_mode_rejects_subtasks_and_accepts_the_exact_main_prompt(
    agent_components,
) -> None:
    toolkit, env, model = agent_components
    result = toolkit.execute_tool("vla_act", {"prompt": "open microwave"})
    assert result.is_error
    assert "instruction_mode=main" in result.data["error"]
    assert not model.inputs and not env.actions
    result = toolkit.execute_tool("vla_act", {"prompt": "make popcorn"})
    assert not result.is_error
    assert model.inputs[0][1] == "make popcorn"


def test_session_ending_before_action_prevents_inference_and_uses_cached_images(
    agent_components,
) -> None:
    toolkit, env, model = agent_components
    env.get_session_status = lambda: {"ended": True, "reason": "connection_closed"}
    observation_reads = env.observation_reads
    result = toolkit.execute_tool("vla_act", {})
    execution = result.data["log"]["result"]
    assert execution["stop_reason"] == "session_ended"
    assert execution["observation_fresh"] is False
    assert execution["executed_actions"] == 0
    assert env.observation_reads == observation_reads
    assert not model.inputs and not env.actions
    assert len(result.images) == 3


def test_session_ending_mid_chunk_preserves_partial_feedback_without_replay(
    components,
) -> None:
    toolkit, env, model = components

    def end_during_chunk(actions):
        env.actions.append(actions.copy())
        return (
            env.observation,
            None,
            None,
            None,
            {
                "stop_reason": "session_ended",
                "executed_actions": 1,
                "issued_actions": 2,
                "confirmed_actions": 1,
                "unconfirmed_actions": 1,
                "observation_after_last_issued_action": False,
            },
        )

    env.chunk_step = end_during_chunk
    first = toolkit.execute_tool("vla_act", {"chunks": 3})
    execution = first.data["log"]["result"]
    assert execution["stop_reason"] == "session_ended"
    assert execution["requested_actions"] == 2
    assert execution["executed_actions"] == execution["confirmed_actions"] == 1
    assert execution["issued_actions"] == 2
    assert execution["unconfirmed_actions"] == 1
    assert execution["terminated"] is execution["truncated"] is None
    assert execution["observation_fresh"] is False
    second = toolkit.execute_tool("vla_act", {})
    assert second.data["log"]["result"]["stop_reason"] == "session_ended"
    assert len(env.actions) == len(model.inputs) == 1
    assert toolkit.solved() is False


def test_tool_allowlist_removes_disabled_action_from_schema_and_dispatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("rpent.utils.templates.get_output_dir", lambda: tmp_path)
    env, model = _Environment(), _Model()
    toolkit = BehaviorToolkit(
        runtime_kwargs={
            "env": env,
            "model": model,
            "task_id": 40,
            "task_language": "make popcorn",
            "execution_horizon": 32,
            "tool_allowlist": ["view_env_state", "finish"],
        },
        dashboard_events=NullDashboardEventSink(),
        memory=MemoryManager(tmp_path / "memory"),
        state_output_dir=tmp_path / "state",
    )
    try:
        names = {item.name for item in toolkit.list_tools()}
        assert "vla_act" not in names
        assert {"view_env_state", "finish", "list_dir"}.issubset(names)
        assert toolkit.execute_tool("vla_act", {}).is_error
        assert not env.actions and not model.inputs
    finally:
        toolkit.close()


_COMMON_TOOLS = {"read_text_file", "write_text_file", "list_dir"}


_READERS = {
    "view_env_state",
    "read_robot_state",
    "segment",
    "pixel_to_robot_base",
    "finish",
}


_INPUTS = {
    "view_env_state": {"step"},
    "read_robot_state": {"step"},
    "segment": {"prompt", "camera", "step", "point", "min_score"},
    "pixel_to_robot_base": {"camera", "pixel", "step", "neighborhood"},
    "vla_act": {"chunks", "prompt"},
    "set_gripper": {"arm", "val", "steps", "settle_steps"},
    "move_to": {"arm", "xyz", "quat", "settle_steps"},
    "move_eef_delta": {"arm", "delta_xyz"},
    "move_base": {"forward", "lateral", "turn", "steps"},
    "rotate_wrist": {"arm", "delta_yaw_deg", "settle_steps"},
    "rotate_wrist_joint": {"arm", "joint", "delta_deg", "steps", "settle_steps"},
    "finish": {"status", "summary"},
}


@pytest.fixture(autouse=True)
def output_directory(tmp_path, monkeypatch):
    monkeypatch.setattr("rpent.utils.templates.get_output_dir", lambda: tmp_path)


@pytest.mark.parametrize(
    ("template", "expected"),
    [
        ("agent.yaml", set(_INPUTS)),
        ("vla_only.yaml", {"view_env_state", "vla_act", "finish"}),
    ],
)
def test_public_templates_publish_declared_tools_and_shared_file_tools(
    tmp_path, template, expected
):
    spec = get_robot_spec()
    args = _args(tmp_path, "--sam3-endpoint", "http://localhost:21002")
    args.behavior_config = (
        Path(__file__).resolve().parents[4] / "robots/behavior/configs" / template
    )
    args.execution_horizon = None
    spec.validate_args(args)
    config = spec.parse_config(args)
    assert set(config.task_desc["tool_allowlist"]) == expected
    toolkit = _template_toolkit(
        tmp_path,
        config.task_desc["tool_allowlist"],
        sam=_SAM() if "segment" in expected else None,
        **{
            key: config.task_desc[key]
            for key in ("execution_horizon", "instruction_mode", "max_chunks_per_call")
        },
    )
    try:
        declarations = {item.name: item for item in toolkit.list_tools()}
        assert set(declarations) == expected | _COMMON_TOOLS
        for name in expected:
            declaration = declarations[name]
            assert set(declaration.input_schema["properties"]) == _INPUTS[name]
            assert declaration.readonly == (name in _READERS)
        vla = declarations["vla_act"]
        assert vla.input_schema["properties"]["chunks"]["maximum"] == 1
        assert "full configured main-task instruction" in vla.description
        prompt = spec.prompts.render("system", variables=config.prompt_vars)
        assert "{{" not in prompt
        assert "complete official episode" in prompt
        assert "actual tool/infrastructure failure" in prompt
        assert "subtask text and other instruction overrides are rejected" in prompt
        assert ("Use move_to" in prompt) == ("move_to" in expected)
        assert ("SAM3" in prompt) == ("segment" in expected)
    finally:
        toolkit.close()


def test_geometry_tool_does_not_require_an_unavailable_segment_tool(tmp_path):
    toolkit = _template_toolkit(
        tmp_path, ["view_env_state", "pixel_to_robot_base", "finish"]
    )
    try:
        declarations = {item.name: item for item in toolkit.list_tools()}
        assert "segment" not in declarations
        description = " ".join(declarations["pixel_to_robot_base"].description.split())
        assert "If segment is available" in description
        result = toolkit.execute_tool(
            "pixel_to_robot_base", {"camera": "head", "pixel": [112, 112]}
        )
        assert not result.is_error
        assert result.data["coordinate_frame_step"] == 0
    finally:
        toolkit.close()


def test_segmentation_failure_only_suggests_geometry_when_available(tmp_path):
    class UnavailableSAM:
        def segment(self, *args, **kwargs):
            raise RuntimeError("service unavailable")

    toolkit = _template_toolkit(
        tmp_path, ["view_env_state", "segment", "finish"], sam=UnavailableSAM()
    )
    try:
        assert "pixel_to_robot_base" not in {item.name for item in toolkit.list_tools()}
        result = toolkit.execute_tool("segment", {"prompt": "visible object"})
        assert result.is_error
        assert "If pixel_to_robot_base is available" in result.data["fallback"]
        assert toolkit.state.latest_step == 0
    finally:
        toolkit.close()


def make_env():
    backend = FakeEnvBackend()
    rpc = LocalRpc(BehaviorEnvFacade(backend, task_id=40))
    return backend, rpc, BehaviorEnvClient(rpc, task_id=40)


def test_connect_and_observe_do_not_reset_or_execute_actions():
    backend, rpc, client = make_env()

    assert rpc.calls == ["env.get_env_meta"]
    assert client.get_observation() is backend.observation
    assert client.last_obs is backend.observation
    assert backend.actions == []

    with pytest.raises(RuntimeError, match="official evaluator"):
        client.reset()
    assert "env.reset" not in rpc.calls
    with pytest.raises(RuntimeError, match="official evaluator"):
        rpc.call("env.reset")
    assert backend.actions == []


def test_client_rejects_an_endpoint_for_a_different_task():
    rpc = LocalRpc(BehaviorEnvFacade(FakeEnvBackend(), task_id=0))
    with pytest.raises(AssertionError, match="env_meta mismatch"):
        BehaviorEnvClient(rpc, task_id=40)
    assert rpc.calls == ["env.get_env_meta"]


def test_step_returns_backend_feedback_without_inventing_success():
    backend, _, client = make_env()
    action = np.arange(23, dtype=np.float32)

    observation, reward, terminated, truncated, info = client.step(action)

    np.testing.assert_array_equal(backend.actions[0], action)
    assert backend.actions[0].dtype == action.dtype
    assert client.last_obs is observation is backend.observation
    assert client.last_info is info is backend.info
    assert reward is None
    assert terminated is None
    assert truncated is None
    assert "success" not in info


@pytest.mark.parametrize("return_all_frames", [False, True])
def test_chunk_returns_real_frames_and_caches_the_final_observation(return_all_frames):
    backend, _, client = make_env()
    actions = np.arange(46, dtype=np.float32).reshape(2, 23)

    observations, _, _, _, info = client.chunk_step(
        actions, return_all_frames=return_all_frames
    )

    np.testing.assert_array_equal(np.asarray(backend.actions), actions)
    assert client.last_obs is backend.observation
    assert client.last_info is info is backend.info
    assert info["executed_actions"] == 2
    if return_all_frames:
        assert observations == [{"fixture_frame": 1}, {"fixture_frame": 2}]
    else:
        assert observations is backend.observation


@pytest.mark.parametrize(
    ("method", "actions"),
    [
        ("step", np.zeros(22)),
        ("step", np.zeros((1, 23))),
        ("step", np.full(23, np.nan)),
        ("chunk_step", np.zeros(23)),
        ("chunk_step", np.zeros((0, 23))),
        ("chunk_step", np.zeros((2, 24))),
        ("chunk_step", np.full((2, 23), np.inf)),
    ],
)
def test_invalid_actions_never_reach_execution_backend(method, actions):
    backend, rpc, client = make_env()

    with pytest.raises(ValueError, match="BEHAVIOR native actions"):
        getattr(client, method)(actions)
    assert rpc.calls == ["env.get_env_meta"]

    # Direct requests must also be rejected at the service boundary.
    with pytest.raises(ValueError, match="BEHAVIOR native actions"):
        rpc.call(f"env.{method}", args=(actions,))
    assert backend.actions == []


def test_invalid_execution_feedback_is_not_cached_as_a_completed_step():
    backend, _, client = make_env()
    original = client.get_observation()
    backend.step = lambda action: (None, None, None, None, {})

    with pytest.raises(TypeError, match="observation and info"):
        client.step(np.zeros(23))
    assert client.last_obs is original


def test_vla_preserves_raw_observation_prompt_and_native_actions():
    calls = []
    actions = np.arange(46, dtype=np.float32).reshape(2, 23)

    def infer_chunk(observation, prompt=None):
        calls.append((observation, prompt))
        return actions

    rpc = LocalRpc(BehaviorVLAFacade(SimpleNamespace(infer_chunk=infer_chunk)))
    client = BehaviorVLAClient(rpc)
    observation = {"fixture_rgb": np.zeros((2, 2, 3), dtype=np.uint8)}

    result = client.infer(observation, prompt="pick up the popcorn bag")

    assert rpc.calls == ["vla.predict"]
    assert calls[0][0] is observation
    assert calls[0][1] == "pick up the popcorn bag"
    assert result is actions


@pytest.mark.parametrize("actions", [np.zeros((2, 16)), np.full((1, 23), np.nan)])
def test_invalid_vla_output_cannot_be_forwarded_as_native_actions(actions):
    backend = SimpleNamespace(infer_chunk=lambda obs, prompt=None: actions)
    client = BehaviorVLAClient(LocalRpc(BehaviorVLAFacade(backend)))

    with pytest.raises(ValueError, match="BEHAVIOR native actions"):
        client.infer({})


@pytest.mark.parametrize(
    ("options", "error"),
    [({"temperature": 0.5}, ValueError), ({"prompt": 7}, TypeError), ([], TypeError)],
)
def test_unsupported_vla_options_do_not_reach_backend(options, error):
    calls = []
    backend = SimpleNamespace(infer_chunk=lambda *args, **kwargs: calls.append(args))
    rpc = LocalRpc(BehaviorVLAFacade(backend))

    with pytest.raises(error):
        rpc.call("vla.predict", args=({}, options))
    assert calls == []


def test_components_require_explicit_backends():
    with pytest.raises(RuntimeError, match="requires an injected environment backend"):
        BehaviorEnvFacade(task_id=40)
    with pytest.raises(RuntimeError, match="VLA policy backend"):
        BehaviorVLAFacade()
