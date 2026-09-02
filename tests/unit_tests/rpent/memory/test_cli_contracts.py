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

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from rpent.cli import memory as memory_cli


def test_memory_cli_merge_forwards_paths_and_prints_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    memory_dir = tmp_path / "memory"
    output_dir = tmp_path / "run"
    captured: dict[str, object] = {}
    result = {"suite": 1, "skipped": [], "message": "已合并"}

    def fake_merge_memory(
        *, cell_tag: str, run_state_dir: Path, solved: bool
    ) -> dict[str, object]:
        captured.update(
            cell_tag=cell_tag,
            run_state_dir=run_state_dir,
            solved=solved,
        )
        return result

    def fake_manager(root: Path) -> SimpleNamespace:
        captured["memory_dir"] = root
        return SimpleNamespace(merge_memory=fake_merge_memory)

    monkeypatch.setattr(memory_cli, "MemoryManager", fake_manager)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "rpent-memory",
            "--memory-dir",
            str(memory_dir),
            "merge",
            "--cell",
            "10_task_t2_s0",
            "--output-dir",
            str(output_dir),
            "--solved",
        ],
    )

    assert memory_cli.main() == 0
    assert captured == {
        "memory_dir": memory_dir,
        "cell_tag": "10_task_t2_s0",
        "run_state_dir": output_dir,
        "solved": True,
    }
    stdout = capsys.readouterr().out
    assert json.loads(stdout) == result
    assert "已合并" in stdout


def test_memory_cli_merge_rejects_missing_required_output_dir(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    manager_created = False

    def fake_manager(root: Path) -> SimpleNamespace:
        nonlocal manager_created
        manager_created = True
        return SimpleNamespace()

    monkeypatch.setattr(memory_cli, "MemoryManager", fake_manager)
    monkeypatch.setattr(
        sys,
        "argv",
        ["rpent-memory", "merge", "--cell", "10_task_t2_s0"],
    )

    with pytest.raises(SystemExit) as exc_info:
        memory_cli.main()

    assert exc_info.value.code == 2
    assert "--output-dir" in capsys.readouterr().err
    assert manager_created is False


def test_memory_cli_validate_reports_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    memory_dir = tmp_path / "memory"
    validated: list[Path] = []

    def fake_manager(root: Path) -> SimpleNamespace:
        return SimpleNamespace(validate=lambda: validated.append(root) or [])

    monkeypatch.setattr(memory_cli, "MemoryManager", fake_manager)
    monkeypatch.setattr(
        sys,
        "argv",
        ["rpent-memory", "--memory-dir", str(memory_dir), "validate"],
    )

    assert memory_cli.main() == 0
    assert validated == [memory_dir]
    assert capsys.readouterr().out == "local memory is valid\n"


def test_memory_cli_validate_prints_problems_and_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    memory_dir = tmp_path / "memory"
    problems = [
        "global/broken.md: unterminated YAML frontmatter",
        "suite/wrong.md: id does not match filename",
    ]
    monkeypatch.setattr(
        memory_cli,
        "MemoryManager",
        lambda root: SimpleNamespace(validate=lambda: problems),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["rpent-memory", "--memory-dir", str(memory_dir), "validate"],
    )

    assert memory_cli.main() == 1
    assert capsys.readouterr().out == "\n".join(problems) + "\n"


def test_memory_cli_build_index_prints_created_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    memory_dir = tmp_path / "memory"
    index_path = memory_dir / "MEMORY.md"
    built: list[Path] = []

    def fake_manager(root: Path) -> SimpleNamespace:
        return SimpleNamespace(
            rebuild_index=lambda: built.append(root) or index_path,
        )

    monkeypatch.setattr(memory_cli, "MemoryManager", fake_manager)
    monkeypatch.setattr(
        sys,
        "argv",
        ["rpent-memory", "--memory-dir", str(memory_dir), "build-index"],
    )

    assert memory_cli.main() == 0
    assert built == [memory_dir]
    assert capsys.readouterr().out == f"{index_path}\n"
