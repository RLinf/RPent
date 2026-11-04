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
from pathlib import Path

import pytest

from robots.efe import session_vars
from robots.efe.robot_spec import (
    _add_cli_args,
    _parse_config,
    _validate_args,
    get_robot_spec,
    get_toolkit,
)
from robots.efe.toolkit import EfeToolkit
from rpent.dashboard.events import NullDashboardEventSink
from rpent.robots.robot_spec import RobotSpec
from rpent.utils.logging import init_output_dir
from tests.efe_support import FakeEfeClient


def _args(tmp_path: Path) -> argparse.Namespace:
    instruction = tmp_path / "task" / "instruction.md"
    instruction.parent.mkdir(parents=True)
    instruction.write_text("Pick up the tube.\n", encoding="utf-8")
    return argparse.Namespace(
        instruction_file=instruction,
        workdir=tmp_path,
        output_dir=tmp_path / "logs",
        memory_dir=None,
        explore=False,
        efe_connect_timeout_s=30.0,
        efe_skills=True,
        efe_base_url=None,
    )


def test_spec_exposes_prompts_and_runtime_hooks():
    spec = get_robot_spec()

    assert isinstance(spec, RobotSpec)
    assert spec.name == "efe"
    assert spec.supports_exploration is False


def test_cli_args_match_the_shared_robot_spec_contract(tmp_path):
    instruction = tmp_path / "instruction.md"
    instruction.write_text("Pick up the tube.\n", encoding="utf-8")
    parser = argparse.ArgumentParser()
    _add_cli_args(parser, use_dashboard=False)

    args = parser.parse_args(["--instruction-file", str(instruction)])

    assert args.instruction_file == instruction
    assert args.efe_skills is True


def test_parse_config_uses_the_task_instruction(tmp_path):
    config = _parse_config(_args(tmp_path))

    assert config.task_desc["instruction"] == "Pick up the tube.\n"
    assert config.prompt_vars["instruction"] == "Pick up the tube.\n"
    assert config.prompt_vars["workdir"] == str(tmp_path.resolve())
    assert config.recipe_tag.startswith("task_")
    assert config.prompt_vars["session_max"] == 1


def test_parse_config_uses_the_shared_prompt_variables(tmp_path):
    args = _args(tmp_path)

    config = _parse_config(args)

    memory_dir = Path(config.prompt_vars["memory_dir"])
    assert config.prompt_vars["output_dir"] == str(args.output_dir)
    assert config.prompt_vars["memory_inbox"] == str(
        session_vars.inbox_path(memory_dir, config.recipe_tag)
    )
    assert config.prompt_vars["skills"] is True
    assert config.prompt_vars["recipe_tag"] == config.recipe_tag


def test_parse_config_rejects_a_missing_instruction(tmp_path):
    args = _args(tmp_path)
    args.instruction_file = tmp_path / "missing.md"

    with pytest.raises(FileNotFoundError):
        _parse_config(args)


def test_exploration_is_rejected_because_the_runner_starts_its_own_sessions(tmp_path):
    args = _args(tmp_path)
    args.explore = True

    with pytest.raises(ValueError, match="explore-sessions"):
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
    assert "move_to_xyz" not in names
