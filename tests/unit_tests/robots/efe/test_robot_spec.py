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

"""EFE robot registration and configuration contracts."""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from robots.efe.robot_spec import (
    _add_cli_args,
    _parse_config,
    _prepare_prompt_vars,
    _validate_args,
    get_robot_spec,
    get_toolkit,
)
from robots.efe.toolkit import EfeToolkit
from rpent.dashboard.events import NullDashboardEventSink
from rpent.evaluation import RunFinalizationContext
from rpent.planner.base import PlannerResult
from rpent.robots.robot_spec import RobotSpec
from rpent.utils.logging import init_output_dir
from tests.unit_tests.robots.efe.efe_support import SPEC, FakeEfeClient


@pytest.fixture(autouse=True)
def _restore_repo_root(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep ``_parse_config``'s workspace binding out of later tests."""
    if "RPENT_REPO_ROOT" in os.environ:
        monkeypatch.setenv("RPENT_REPO_ROOT", os.environ["RPENT_REPO_ROOT"])
    else:
        monkeypatch.delenv("RPENT_REPO_ROOT", raising=False)


def _args(tmp_path: Path) -> argparse.Namespace:
    instruction = tmp_path / "task" / "instruction.md"
    instruction.parent.mkdir(parents=True)
    instruction.write_text("Pick up the tube.\n", encoding="utf-8")
    return argparse.Namespace(
        instruction_file=instruction,
        task_name=None,
        workdir=tmp_path,
        output_dir=tmp_path / "logs",
        memory_dir=None,
        memory_repo=None,
        explore=False,
        planner="api",
        codex_sandbox="full-access",
        efe_base_url=None,
        efe_connect_timeout_s=30.0,
        efe_trial_timeout_s=None,
        efe_skills=True,
    )


def test_spec_exposes_prompts_and_runtime_hooks():
    spec = get_robot_spec()

    assert isinstance(spec, RobotSpec)
    assert spec.name == "efe"
    assert spec.supports_exploration is False
    assert spec.finalize_run is not None


def test_finalize_run_writes_efe_metrics_to_the_cli_result(tmp_path):
    context = RunFinalizationContext(
        output_dir=tmp_path,
        robot_name="efe",
        task_desc={"task": "pick"},
        environment_success=None,
        agent_error=None,
        elapsed_s=12.5,
        planner="codex",
        model="test-model",
        reasoning_effort="none",
        max_turns=20,
        planner_timeout_s=60,
        finish_result={"status": "completed"},
        stats={"usage_observed": True, "total_input_tokens": 10},
        environment_metrics={
            "efe_submit_calls": 2,
            "done_reason": "finished",
        },
    )
    finalizer = get_robot_spec().finalize_run

    assert finalizer is not None
    result_path = finalizer(context)
    result = json.loads(result_path.read_text(encoding="utf-8"))

    assert result_path.name == "efe-result.json"
    assert result["stats"]["total_input_tokens"] == 10
    assert result["metrics"]["efe_submit_calls"] == 2
    assert "reward" not in result


def test_cli_args_match_the_shared_robot_spec_contract(tmp_path):
    instruction = tmp_path / "instruction.md"
    instruction.write_text("Pick up the tube.\n", encoding="utf-8")
    parser = argparse.ArgumentParser()
    _add_cli_args(parser, use_dashboard=False)

    args = parser.parse_args(["--instruction-file", str(instruction)])

    assert args.instruction_file == instruction
    assert args.efe_skills is True
    assert args.efe_trial_timeout_s is None


def test_parse_config_uses_the_task_instruction(tmp_path):
    config = _parse_config(_args(tmp_path))

    assert config.task_desc["instruction"] == "Pick up the tube.\n"
    assert config.prompt_vars["instruction"] == "Pick up the tube.\n"
    assert config.prompt_vars["workdir"] == str(tmp_path.resolve())
    assert config.recipe_tag.startswith("task_")


def test_parse_config_uses_an_explicit_task_name(tmp_path):
    """An agent harness uploads the instruction away from the task directory."""
    args = _args(tmp_path)
    args.task_name = "pick_the_tube"

    config = _parse_config(args)

    assert config.task_desc["task"] == "pick_the_tube"
    assert config.prompt_vars["task_name"] == "pick_the_tube"
    assert config.recipe_tag.startswith("pick_the_tube_")


def test_parse_config_binds_the_workspace_as_the_repo_root(
    tmp_path, monkeypatch: pytest.MonkeyPatch
):
    """The planner is built before ``init_runtime``, so bind the root here."""
    monkeypatch.delenv("RPENT_REPO_ROOT", raising=False)

    _parse_config(_args(tmp_path))

    assert os.environ["RPENT_REPO_ROOT"] == str(tmp_path.resolve())


def test_parse_config_resolves_default_paths_under_the_workspace(
    tmp_path, monkeypatch: pytest.MonkeyPatch
):
    args = _args(tmp_path)
    args.memory_dir = None
    args.output_dir = None
    monkeypatch.delenv("RPENT_REPO_ROOT", raising=False)

    config = _parse_config(args)

    assert config.prompt_vars["memory_dir"] == str(
        (tmp_path / "memory" / "efe").resolve()
    )
    assert config.output_dir == (tmp_path / "logs" / config.output_dir.name).resolve()


def test_parse_config_keeps_an_explicit_repo_root(
    tmp_path, monkeypatch: pytest.MonkeyPatch
):
    explicit = str(tmp_path / "repo")
    monkeypatch.setenv("RPENT_REPO_ROOT", explicit)

    _parse_config(_args(tmp_path))

    assert os.environ["RPENT_REPO_ROOT"] == explicit


def test_parse_config_uses_the_shared_prompt_variables(tmp_path):
    args = _args(tmp_path)

    config = _parse_config(args)

    assert config.prompt_vars["output_dir"] == str(args.output_dir)
    assert config.prompt_vars["skills"] is True
    assert config.prompt_vars["recipe_tag"] == config.recipe_tag


def test_parse_config_rejects_a_missing_instruction(tmp_path):
    args = _args(tmp_path)
    args.instruction_file = tmp_path / "missing.md"

    with pytest.raises(FileNotFoundError):
        _parse_config(args)


def test_exploration_is_rejected_for_efe_evaluation(tmp_path):
    args = _args(tmp_path)
    args.explore = True

    with pytest.raises(ValueError, match="does not support --explore"):
        _validate_args(args)


def test_memory_repo_is_rejected_because_efe_prepares_its_own_memory(tmp_path):
    args = _args(tmp_path)
    args.memory_repo = "org/dataset"

    with pytest.raises(ValueError, match="--memory-repo is not supported"):
        _validate_args(args)


def test_toolkit_is_built_from_the_configured_memory_and_skills(tmp_path):
    init_output_dir(tmp_path)
    config = _parse_config(_args(tmp_path))
    (tmp_path / "memory").mkdir()
    config.prompt_vars["memory_dir"] = str(tmp_path / "memory")
    config.prompt_vars["skills"] = False

    toolkit = get_toolkit(
        runtime_kwargs={"client": FakeEfeClient()},
        dashboard_events=NullDashboardEventSink(),
        config=config,
    )

    assert isinstance(toolkit, EfeToolkit)
    names = [declaration.name for declaration in toolkit.list_tools()]
    assert "efe_submit" in names
    assert "move_to_pose" not in names


def test_prompt_variables_follow_the_connected_task_capabilities(tmp_path):
    args = _args(tmp_path)
    config = _parse_config(args)

    default = _prepare_prompt_vars(args, config, {"client": FakeEfeClient()})
    assert default == {
        "skills": True,
        "reset": False,
        "end_episode": False,
        "restricted_shell": False,
    }

    spec = {
        **copy.deepcopy(SPEC),
        "budget": {"reset": 2},
        "success_signal": "hidden",
    }
    granted = _prepare_prompt_vars(args, config, {"client": FakeEfeClient(spec)})
    assert granted == {
        "skills": True,
        "reset": True,
        "end_episode": True,
        "restricted_shell": False,
    }

    config.prompt_vars["skills"] = False
    disabled = _prepare_prompt_vars(args, config, {"client": FakeEfeClient()})
    assert disabled["skills"] is False


@pytest.mark.parametrize(
    ("planner", "sandbox", "restricted"),
    [
        ("codex", "read-only", True),
        ("codex", "workspace-write", True),
        ("codex", "full-access", False),
        ("api", "read-only", False),
    ],
)
def test_restricted_shell_follows_the_codex_sandbox(
    tmp_path, planner, sandbox, restricted
):
    args = _args(tmp_path)
    args.planner = planner
    args.codex_sandbox = sandbox
    config = _parse_config(args)

    variables = _prepare_prompt_vars(args, config, {"client": FakeEfeClient()})

    assert variables["restricted_shell"] is restricted


def test_prompt_variables_fall_back_when_the_layout_is_unsupported(tmp_path):
    args = _args(tmp_path)
    config = _parse_config(args)
    spec = {**copy.deepcopy(SPEC), "kinematics": {}}

    variables = _prepare_prompt_vars(args, config, {"client": FakeEfeClient(spec)})

    assert variables["skills"] is False


@pytest.mark.parametrize(
    ("spec_changes", "cli_args", "skills_available", "shell_restricted"),
    [
        ({}, [], True, False),
        (
            {"budget": {"reset": 2}, "success_signal": "hidden"},
            [],
            True,
            False,
        ),
        ({"kinematics": {}}, [], False, False),
        ({}, ["--efe-no-skills"], False, False),
        (
            {},
            ["--planner", "codex", "--codex-sandbox", "read-only"],
            True,
            True,
        ),
    ],
)
def test_generic_cli_renders_the_tools_the_session_registers(
    tmp_path, monkeypatch, spec_changes, cli_args, skills_available, shell_restricted
):
    from rpent.cli import main as cli

    instruction = tmp_path / "task" / "instruction.md"
    instruction.parent.mkdir(parents=True)
    instruction.write_text("Pick up the tube.\n", encoding="utf-8")
    client = FakeEfeClient({**copy.deepcopy(SPEC), **spec_changes})
    captured: dict[str, object] = {}

    class CapturingPlanner:
        def solve(self, *, system_prompt, user_message, toolkit, max_turns, **kwargs):
            captured["system_prompt"] = system_prompt
            captured["tools"] = [
                declaration.name for declaration in toolkit.list_tools()
            ]
            return PlannerResult(stats={})

    robot_spec = replace(
        get_robot_spec(), init_runtime=lambda *args: ([], {"client": client})
    )
    monkeypatch.setattr(cli, "get_robot_spec", lambda name: robot_spec)
    monkeypatch.setattr(
        cli, "build_planner", lambda *args, **kwargs: CapturingPlanner()
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "rpent",
            "--robot",
            "efe",
            "--instruction-file",
            str(instruction),
            "--workdir",
            str(tmp_path),
            "--output-dir",
            str(tmp_path / "logs"),
            "--memory-profile",
            "local",
            "--memory-dir",
            str(tmp_path / "memory"),
            *cli_args,
        ],
    )

    assert cli.main() == 0

    tools = captured["tools"]
    prompt = captured["system_prompt"]
    assert ("efe_submit" in tools) is (not skills_available)
    assert ("move_to_pose" in tools) is skills_available
    assert ("efe_submit" in prompt) is (not skills_available)
    assert ("move_to_pose" in prompt) is skills_available
    assert ("efe_reset" in prompt) is ("efe_reset" in tools)
    assert ("efe_end_episode" in prompt) is ("efe_end_episode" in tools)
    assert ("efe_reset" in prompt) is (client.reset_budget > 0)
    assert ("efe_end_episode" in prompt) is client.success_hidden
    assert ("Shell commands have no network access" in prompt) is shell_restricted
