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

import pytest

from robots.robocasa.prompt_bundle import system_prompt
from robots.robocasa.robot_spec import _parse_config
from rpent.memory import MemoryManager
from rpent.prompt.utils import format_prompt


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


def test_parse_config_uses_default_memory_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("RPENT_REPO_ROOT", str(tmp_path))

    config = _parse_config(
        _args(tmp_path, memory_dir=None),
    )

    assert config.prompt_vars["memory_dir"] == str(tmp_path / "memory" / "robocasa")


def test_default_memory_sync_uses_robocasa_subtree(monkeypatch, tmp_path):
    calls = {}

    def fake_snapshot_download(**kwargs):
        calls.update(kwargs)

    monkeypatch.setenv("RPENT_REPO_ROOT", str(tmp_path))
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("RPENT_MEMORY_HF_REPO", raising=False)
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(snapshot_download=fake_snapshot_download),
    )

    root = MemoryManager(tmp_path / "memory" / "robocasa").sync(
        remote_repo="RLinf/RPent-memory"
    )

    assert root == (tmp_path / "memory" / "robocasa")
    assert calls == {
        "repo_id": "RLinf/RPent-memory",
        "repo_type": "dataset",
        "local_dir": str(tmp_path / "memory"),
        "allow_patterns": ["robocasa/**"],
    }


def test_results_corpus_is_readable_through_memory_tool(monkeypatch, tmp_path):
    monkeypatch.setenv("RPENT_REPO_ROOT", str(tmp_path))
    memory_root = tmp_path / "memory" / "robocasa"
    results = memory_root / "results"
    results.mkdir(parents=True)
    audit = results / "OpenDrawer_s0.json"
    audit.write_text('{"success": true}\n')

    manager = MemoryManager(root=memory_root)
    bindings = manager.get_common_tool_bindings()
    read_text_file = bindings["read_text_file"][1]
    write_text_file = bindings["write_text_file"][1]

    assert read_text_file(path=str(audit))["content"] == '{"success": true}\n'
    with pytest.raises(PermissionError, match="writing to memory is denied"):
        write_text_file(path=str(audit), content="{}\n")


def test_parse_config_resolves_local_memory_dir(tmp_path):
    memory_dir = tmp_path / "local-memory"

    config = _parse_config(
        _args(tmp_path, memory_dir=memory_dir),
    )

    assert config.prompt_vars["memory_dir"] == str(memory_dir.resolve())


def test_prompt_names_only_current_task_memory(tmp_path):
    memory_dir = tmp_path / "memory"
    rendered = format_prompt(
        system_prompt(),
        variables={
            "task_name": "OpenDrawer",
            "memory_dir": str(memory_dir),
        },
    )

    assert str(memory_dir / "results" / "OpenDrawer_s0.json") in rendered
    assert str(memory_dir / "results" / "recipe_OpenDrawer_s0.jsonl") in rendered
    assert str(memory_dir / "results" / "OpenDrawer.md") in rendered
    assert "read every existing file" in rendered
    assert "ArrangeTea_s0" not in rendered
    assert "GLOBAL_MEMORY" not in rendered
    assert "{{" not in rendered
