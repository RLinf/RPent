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

"""Task selection, public templates, prompts and borrowed service startup."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from robots.behavior.env_client import env_runtime_contract
from robots.behavior.env_server import BehaviorEnvFacade
from robots.behavior.robot_spec import get_robot_spec
from robots.behavior.tasks import TASK_LANGUAGE, TASK_NAMES
from rpent.dashboard.events import NullDashboardEventSink
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


def test_shared_model_endpoint_overrides_the_behavior_yaml(tmp_path: Path) -> None:
    path = tmp_path / "behavior.yaml"
    path.write_text("vla_endpoint: http://localhost:24002\n")
    spec = get_robot_spec()
    args = _args(
        tmp_path,
        "--behavior-config",
        str(path),
        "--am-endpoint",
        "http://localhost:31001",
    )
    spec.validate_args(args)
    assert args.am_backend == "pi05"
    assert spec.parse_config(args).task_desc["vla_endpoint"] == "http://localhost:31001"
    args.am_endpoint = None
    assert spec.parse_config(args).task_desc["vla_endpoint"] == "http://localhost:24002"


def test_explicit_task_language_overrides_the_official_default(
    tmp_path: Path,
) -> None:
    spec = get_robot_spec()
    args = _args(tmp_path, "--task-id", "7", "--task-language", "Custom text.")
    spec.validate_args(args)
    config = spec.parse_config(args)
    assert config.prompt_vars["task_language"] == "Custom text."
    assert config.prompt_vars["task_language"] != TASK_LANGUAGE[7]


def test_unknown_yaml_keys_are_not_silently_ignored(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("task_id: 40\ntask_language: popcorn\nexecution_horizn: 2\n")
    args = _args(tmp_path, "--behavior-config", str(path))
    with pytest.raises(ValueError, match="documented task settings"):
        get_robot_spec().parse_config(args)


@pytest.mark.parametrize(
    ("mode", "env_language"),
    [("main", "make popcorn"), ("override", "turn on the radio")],
)
def test_runtime_borrows_facades_without_reset_or_shutdown(
    tmp_path: Path, monkeypatch, mode: str, env_language: str
) -> None:
    calls = []

    class Transport:
        def call(self, method, args=(), kwargs=None, timeout_s=None):
            calls.append(method)
            if method == "healthz":
                return {"ok": True}
            if method == "env.get_env_meta":
                return env_runtime_contract(40)
            if method == "env.get_task_language":
                return env_language
            raise AssertionError(f"unexpected runtime call: {method}")

    monkeypatch.setattr("rpent.utils.rpc.make_rpc_client", lambda endpoint: Transport())
    daemons, runtime = get_robot_spec().init_runtime(
        _args(tmp_path, "--instruction-mode", mode),
        tmp_path,
        NullDashboardEventSink(),
        None,
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
    assert calls.count("env.get_task_language") == (1 if mode == "main" else 0)


@pytest.mark.parametrize("env_language", ["turn on the radio", None])
def test_main_mode_rejects_an_env_without_the_planners_task_instruction(
    tmp_path: Path, monkeypatch, env_language: str | None
) -> None:
    args = _args(tmp_path, "--instruction-mode", "main")
    env = LocalRpc(
        BehaviorEnvFacade(FakeEnvBackend(), task_id=40, task_language=env_language)
    )
    monkeypatch.setattr(
        "rpent.utils.rpc.make_rpc_client", {args.env_endpoint: env}.__getitem__
    )

    with pytest.raises(RuntimeError, match="task 40 instruction mismatch") as exc:
        get_robot_spec().init_runtime(args, tmp_path, NullDashboardEventSink(), None)

    assert f"Planner='make popcorn', Env={env_language!r}" in str(exc.value)
    assert env.calls == ["healthz", "env.get_env_meta", "env.get_task_language"]


@pytest.mark.parametrize("task_id", [0, 40, 99])
def test_official_task_and_default_instruction_reach_planner_and_metadata(
    tmp_path, task_id
):
    spec = get_robot_spec()
    args = _args(tmp_path, "--task-id", str(task_id))
    args.task_language = None
    spec.validate_args(args)
    config = spec.parse_config(args)
    language, name = TASK_LANGUAGE[task_id], TASK_NAMES[task_id]
    assert config.task_desc["task_id"] == task_id
    assert config.task_desc["task_name"] == name
    assert config.task_desc["task_language"] == language
    assert config.recipe_tag == f"behavior_{name}"
    assert language in spec.prompts.render("system", variables=config.prompt_vars)
    assert f"Task {task_id}: {name}." in spec.prompts.render(
        "user", variables=config.prompt_vars
    )
    assert set(TASK_NAMES) == set(TASK_LANGUAGE) == set(range(100))
    assert len(set(TASK_NAMES.values())) == 100


@pytest.mark.parametrize("template", ["agent.yaml", "vla_only.yaml"])
def test_public_template_exposes_the_selected_tools_and_bounded_vla(
    tmp_path, monkeypatch, template
):
    monkeypatch.setattr("rpent.utils.templates.get_output_dir", lambda: tmp_path)
    spec = get_robot_spec()
    args = _args(tmp_path, "--sam3-endpoint", "http://localhost:21002")
    args.behavior_config = (
        Path(__file__).resolve().parents[4] / "robots/behavior/configs" / template
    )
    args.execution_horizon = None
    spec.validate_args(args)
    config = spec.parse_config(args)
    expected = {"view_env_state", "vla_act", "finish"}
    if template == "agent.yaml":
        expected |= {
            "read_robot_state",
            "segment",
            "pixel_to_robot_base",
            "set_gripper",
            "move_to",
            "move_eef_delta",
            "move_base",
            "rotate_wrist",
            "rotate_wrist_joint",
        }
    assert set(config.task_desc["tool_allowlist"]) == expected
    with _template_toolkit(
        tmp_path,
        expected,
        sam=_SAM() if "segment" in expected else None,
        **{
            key: config.task_desc[key]
            for key in ("execution_horizon", "instruction_mode", "max_chunks_per_call")
        },
    ) as toolkit:
        declarations = {item.name: item for item in toolkit.list_tools()}
        assert set(declarations) == expected | {
            "read_text_file",
            "write_text_file",
            "list_dir",
        }
        assert (
            declarations["vla_act"].input_schema["properties"]["chunks"]["maximum"] == 1
        )
        prompt = spec.prompts.render("system", variables=config.prompt_vars)
        assert "{{" not in prompt
        assert ("Use move_to" in prompt) == ("move_to" in expected)
        assert ("SAM3" in prompt) == ("segment" in expected)


_OPTIONAL_BEHAVIOR_TOOLS = (
    "vla_act",
    "read_robot_state",
    "pixel_to_robot_base",
    "segment",
    "set_gripper",
    "release",
    "move_to",
    "move_eef_delta",
    "rotate_wrist",
    "rotate_wrist_joint",
    "move_base",
)


@pytest.mark.parametrize(
    "optional_tools",
    [
        (),
        *((name,) for name in _OPTIONAL_BEHAVIOR_TOOLS),
        ("segment", "pixel_to_robot_base"),
        ("move_to", "move_base"),
        ("vla_act", "set_gripper"),
        ("vla_act", "move_to", "rotate_wrist", "move_eef_delta"),
        _OPTIONAL_BEHAVIOR_TOOLS,
    ],
)
def test_allowlist_prompt_only_mentions_declared_action_and_observation_tools(
    tmp_path, monkeypatch, optional_tools
):
    monkeypatch.setattr("rpent.utils.templates.get_output_dir", lambda: tmp_path)
    allowed = {"view_env_state", "finish", *optional_tools}
    spec = get_robot_spec()
    args = _args(
        tmp_path,
        "--sam3-endpoint",
        "http://localhost:21002",
        "--instruction-mode",
        "main",
        "--max-chunks-per-call",
        "1",
        "--tool-allowlist",
        *sorted(allowed),
    )
    spec.validate_args(args)
    config = spec.parse_config(args)
    with _template_toolkit(
        tmp_path, allowed, sam=_SAM() if "segment" in allowed else None
    ) as toolkit:
        declarations = {item.name for item in toolkit.list_tools()}
        prompt = spec.prompts.render("system", variables=config.prompt_vars)
        # Directory names are not tool instructions, even if they contain "release".
        prompt = prompt.replace(str(tmp_path), "<run-dir>").replace(
            config.prompt_vars["memory_dir"], "<memory-dir>"
        )
        mentioned = set(
            re.findall(r"\b(?:" + "|".join(_OPTIONAL_BEHAVIOR_TOOLS) + r")\b", prompt)
        )
        assert mentioned == set(optional_tools)
        assert mentioned <= declarations
        assert "{{" not in prompt
        assert ("Omit prompt in vla_act" in prompt) == ("vla_act" in allowed)
        if set(optional_tools) == set(_OPTIONAL_BEHAVIOR_TOOLS):
            for stage in ("Approach.", "Localize.", "Reach.", "Align.", "Engage."):
                assert stage in prompt


@pytest.mark.parametrize(
    "key,value",
    [
        ("task_id", 100),
        ("task_language", " "),
        ("execution_horizon", 0),
        ("env_endpoint", "ws://localhost:8000"),
        ("tool_allowlist", ["view_env_state", "finish", "unregistered_tool"]),
    ],
)
def test_invalid_run_configuration_is_rejected_before_startup(tmp_path, key, value):
    args = _args(tmp_path)
    setattr(args, key, value)
    with pytest.raises(ValueError):
        get_robot_spec().validate_args(args)
