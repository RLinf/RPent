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

"""No-memory evaluation keeps native tools without exposing historical files."""

from argparse import Namespace
from unittest.mock import Mock

import pytest

from robots.libero import memory, robot_spec, toolkit
from rpent.dashboard.events import NullDashboardEventSink
from rpent.memory import MemoryManager
from rpent.memory.loading import prepare_run_memory
from rpent.robots import RunConfig
from rpent.tools.common import CommonTools


def test_no_memory_skips_corpus_and_removes_solutions(tmp_path, monkeypatch):
    sync = Mock(side_effect=AssertionError("memory download attempted"))
    monkeypatch.setattr(memory, "sync_version", sync)
    spec = robot_spec.get_robot_spec()
    config = RunConfig(
        output_dir=tmp_path,
        recipe_tag="s0",
        task_desc={},
        prompt_vars={
            "memory_profile": "none",
            "output_dir": tmp_path,
            "suite": "libero_spatial",
            "task": 2,
            "seed": 0,
            "recipe_tag": "s0",
        },
    )
    args = Namespace(
        planner="api",
        model="openai:test",
        memory_profile="none",
        memory_version="auto",
        explore=False,
        memory_repo=None,
    )
    prepare_run_memory(args, spec, config)
    sync.assert_not_called()
    system = spec.prompts.render("system", variables=config.prompt_vars)
    user = spec.prompts.render("user", variables=config.prompt_vars)
    for leaked in (
        "PROVEN LEVERS",
        "seed-0",
        "+ 0.045",
        "MEMORY.md",
        "READ THE GUIDES",
    ):
        assert leaked not in system + user
    assert "pi0_pick" in system and "segment" in system
    assert "view_env_state" in user
    assert "./files/s0.json" in user
    assert str(tmp_path) not in system + user


def test_no_memory_file_boundary_blocks_history_and_symlinks(tmp_path):
    corpus = tmp_path / "memory"
    (corpus / "task_only").mkdir(parents=True)
    (corpus / "task_only" / "answer.json").write_text("solution")
    files = tmp_path / "episode" / "files"
    manager = MemoryManager(corpus, memory_access="none", file_root=files)
    manager.check_layout()  # An unrelated legacy corpus must not be consulted.
    tools = CommonTools(memory=manager)
    tools.write_text_file(str(files / "notes.txt"), "current observation")
    assert (
        tools.read_text_file(str(files / "notes.txt")).data["content"]
        == "current observation"
    )
    assert tools.list_dir().data["files"] == ["notes.txt"]
    (files / "escape").symlink_to(corpus, target_is_directory=True)
    for path in (
        corpus / "task_only" / "answer.json",
        files / "escape" / "task_only" / "answer.json",
        files / ".." / "env.log",
    ):
        with pytest.raises(PermissionError):
            tools.read_text_file(str(path))
        with pytest.raises(PermissionError):
            tools.write_text_file(str(path), "overwrite")
    with pytest.raises(PermissionError):
        tools.list_dir(str(corpus))


def test_no_memory_preserves_evaluation_toolset(
    tmp_path, monkeypatch, fake_single_arm_primitives
):
    from rpent.utils import templates

    monkeypatch.setattr(
        templates, "default_variables", lambda: {"output_dir": str(tmp_path)}
    )
    fake = fake_single_arm_primitives.for_robot(toolkit.libero_tools.LiberoPrimitives)
    monkeypatch.setattr(toolkit.libero_tools, "LiberoPrimitives", fake)
    from types import SimpleNamespace

    monkeypatch.setattr(
        toolkit.libero_tools,
        "dump_state",
        lambda *a, **kw: SimpleNamespace(step_idx=0, terminated=False),
    )
    config = RunConfig(
        output_dir=tmp_path,
        recipe_tag="s0",
        task_desc={},
        prompt_vars={"memory_profile": "none"},
    )
    native = robot_spec.get_toolkit(
        runtime_kwargs={"env": object()},
        dashboard_events=NullDashboardEventSink(),
        config=config,
        state_output_dir=tmp_path,
    )
    from tests.unit_tests.robots.libero.test_libero_toolkit_contracts import (
        EVALUATION_TOOLS,
    )

    assert {t.name for t in native.list_tools()} == EVALUATION_TOOLS
    assert native.memory.file_root == tmp_path / "files"
    assert "reset" not in {t.name for t in native.list_tools()}
    for declaration in native.list_tools():
        assert str(tmp_path) not in declaration.description
        assert "{{output_dir}}" not in str(declaration.input_schema)


def _libero_config(tmp_path, memory_profile, memory_dir=None):
    import argparse

    spec = robot_spec.get_robot_spec()
    parser = argparse.ArgumentParser()
    spec.add_cli_args(parser, use_dashboard=True)
    parser.add_argument("--output-dir")
    args = parser.parse_args(
        ["--suite", "libero_goal", "--task", "7", "--seed", "0"]
        + ["--output-dir", str(tmp_path)]
    )
    args.planner, args.memory_profile = "api", memory_profile
    args.memory_dir, args.memory_repo = memory_dir, None
    return spec, spec.parse_config(args)
