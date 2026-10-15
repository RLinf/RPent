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

"""RoboTwin corpus overrides through shared CLI/Dashboard preparation."""

from argparse import Namespace
from types import SimpleNamespace
from unittest.mock import Mock

import huggingface_hub
import pytest

from robots.robotwin import memory
from robots.robotwin.robot_spec import get_robot_spec
from rpent.memory.loading import prepare_run_memory


def args_for(**overrides):
    return Namespace(
        **(
            {
                "planner": "codex",
                "model": "gpt-6-astra",
                "memory_profile": "hf",
                "memory_repo": None,
                "memory_version": "auto",
                "memory_dir": None,
                "explore": False,
            }
            | overrides
        )
    )


def test_default_keeps_existing_corpus(tmp_path, monkeypatch):
    root = tmp_path / "robotwin"
    sync = Mock(return_value=root)
    monkeypatch.setattr(memory.MemoryManager, "sync", sync)
    args = args_for()
    config = SimpleNamespace(prompt_vars={})
    prepare_run_memory(args, get_robot_spec(), config)
    sync.assert_called_once_with(remote_repo="RLinf/RPent-memory")
    assert config.prompt_vars["memory_dir"] == args.memory_dir == str(root)


@pytest.mark.parametrize("version", ["auto", "Astra_reference"])
def test_explicit_source_binds_selected_subtree(tmp_path, monkeypatch, version):
    monkeypatch.setenv("RPENT_REPO_ROOT", str(tmp_path / "cache"))
    monkeypatch.setenv("RPENT_MEMORY_HF_REPO", "ignored/repo")
    snapshot = tmp_path / "hub" / ("a" * 40)
    prefix = "robotwin" if version == "auto" else f"robotwin/{version}"
    corpus = snapshot / prefix
    corpus.mkdir(parents=True)
    (corpus / "MEMORY.md").write_text("selected reference")
    fetch = Mock(return_value=str(snapshot))
    monkeypatch.setattr(huggingface_hub, "snapshot_download", fetch)
    args = args_for(memory_repo="owner/custom@release/v2", memory_version=version)
    config = SimpleNamespace(prompt_vars={})
    prepare_run_memory(args, get_robot_spec(), config)
    from pathlib import Path

    root = Path(config.prompt_vars["memory_dir"])
    assert args.memory_dir == str(root)
    assert (root / "MEMORY.md").read_text() == "selected reference"
    assert fetch.call_args.kwargs["repo_id"] == "owner/custom"
    assert fetch.call_args.kwargs["revision"] == "release/v2"
    assert fetch.call_args.kwargs["allow_patterns"] == [f"{prefix}/**"]


@pytest.mark.parametrize("explore", [False, True])
def test_local_memory_is_unchanged(tmp_path, monkeypatch, explore):
    fetch = Mock()
    monkeypatch.setattr(huggingface_hub, "snapshot_download", fetch)
    args = args_for(memory_profile="local", explore=explore, memory_dir=str(tmp_path))
    config = SimpleNamespace(prompt_vars={"memory_dir": str(tmp_path)})
    prepare_run_memory(args, get_robot_spec(), config)
    fetch.assert_not_called()
    assert config.prompt_vars == {"memory_dir": str(tmp_path)}
    args.memory_version = "Astra_reference"
    with pytest.raises(ValueError, match="requires --memory-profile hf"):
        prepare_run_memory(args, get_robot_spec(), config)


@pytest.mark.parametrize("dashboard", [False, True])
def test_cli_and_dashboard_accept_explicit_version(dashboard):
    from rpent.cli.main import _build_argparser

    parser = _build_argparser()
    get_robot_spec().add_cli_args(parser, use_dashboard=dashboard)
    task = (
        ["--dashboard"]
        if dashboard
        else ["--task-name", "beat_block_hammer", "--seed", "1"]
    )
    args = parser.parse_args(
        ["--robot", "robotwin", *task, "--memory-version", "Astra_reference"]
    )
    assert args.memory_version == "Astra_reference"
