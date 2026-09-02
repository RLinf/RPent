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

"""Contracts for RoboCasa configuration, prompts, and resources."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

from robots.robocasa.prompt_bundle import system_prompt
from robots.robocasa.robot_spec import _parse_config, get_robot_spec
from rpent.memory import MemoryManager
from rpent.prompt.utils import format_prompt
from rpent.utils.resources import ensure_resources


def _args(
    tmp_path: Path,
    *,
    memory_dir: Path | None,
) -> argparse.Namespace:
    return argparse.Namespace(
        task_name="OpenDrawer",
        split="target",
        seed=1,
        output_dir=tmp_path / "run",
        memory_dir=memory_dir,
    )


def test_parse_config_uses_default_results_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("RPENT_REPO_ROOT", str(tmp_path))

    config = _parse_config(
        _args(tmp_path, memory_dir=None),
    )

    assert config.prompt_vars["memory_dir"] == str(
        tmp_path / "resources" / "robocasa" / "results"
    )


def test_default_resources_sync_uses_robocasa_subtree(monkeypatch, tmp_path):
    calls = {}

    def fake_snapshot_download(**kwargs):
        calls.update(kwargs)

    monkeypatch.setenv("RPENT_REPO_ROOT", str(tmp_path))
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("RPENT_RESOURCES_HF_REPO", raising=False)
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(snapshot_download=fake_snapshot_download),
    )

    resources_dir = ensure_resources(get_robot_spec())

    assert resources_dir == tmp_path / "resources" / "robocasa"
    assert calls == {
        "repo_id": "RLinf/RPent-memory",
        "repo_type": "dataset",
        "local_dir": str(tmp_path / "resources"),
        "allow_patterns": ["robocasa/**"],
    }


def test_results_corpus_is_readable_through_memory_tool(monkeypatch, tmp_path):
    monkeypatch.setenv("RPENT_REPO_ROOT", str(tmp_path))
    robot_resources = tmp_path / "resources" / "robocasa"
    results_dir = robot_resources / "results"
    results_dir.mkdir(parents=True)
    audit = results_dir / "OpenDrawer_s0.json"
    audit.write_text('{"success": true}\n')

    manager = MemoryManager(root=robot_resources / "memory")
    read_text_file = manager.get_common_tool_bindings()["read_text_file"][1]

    assert read_text_file(path=str(audit))["content"] == '{"success": true}\n'


def test_parse_config_resolves_local_results_dir(tmp_path):
    memory_dir = tmp_path / "local-memory"

    config = _parse_config(
        _args(tmp_path, memory_dir=memory_dir),
    )

    assert config.prompt_vars["memory_dir"] == str(memory_dir.resolve())


def test_prompt_names_only_current_task_memory(tmp_path):
    memory_dir = tmp_path / "results"
    rendered = format_prompt(
        system_prompt(),
        variables={
            "task_name": "OpenDrawer",
            "memory_dir": str(memory_dir),
        },
    )

    assert str(memory_dir / "OpenDrawer_s0.json") in rendered
    assert str(memory_dir / "recipe_OpenDrawer_s0.jsonl") in rendered
    assert str(memory_dir / "OpenDrawer.md") in rendered
    assert "read every existing file" in rendered
    assert "ArrangeTea_s0" not in rendered
    assert "GLOBAL_MEMORY" not in rendered
    assert "{{" not in rendered
