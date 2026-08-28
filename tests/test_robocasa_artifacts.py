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

from pathlib import Path

import pytest

from rpent.reproduce.robocasa.artifacts import (
    CellResult,
    Completion,
    Integrity,
    Outcome,
    publish_completed_cell,
    secure_artifact_subdirectory,
)
from rpent.reproduce.robocasa.protocol import cell_for


def test_cell_result_rejects_contradictory_axes():
    with pytest.raises(ValueError):
        CellResult(Completion.INCOMPLETE, Outcome.SUCCESS, Integrity.UNKNOWN)


def _minimal_publication(cell):
    return (
        CellResult(Completion.COMPLETED, Outcome.SUCCESS, Integrity.VALID),
        {
            "task": cell.task,
            "seed": cell.seed,
            "success": True,
            "run_config_sha256": "a" * 64,
            "task_language": "open the drawer",
        },
        [{"action": "release"}],
    )


def test_publisher_securely_creates_split_and_task_directories(tmp_path: Path):
    root = tmp_path / "results"
    root.mkdir(mode=0o700)
    cell = cell_for("atomic", "OpenDrawer", 1)
    result, audit, commands = _minimal_publication(cell)

    publish_completed_cell(root, cell, result, audit, commands)

    split = root / "atomic"
    task = split / cell.task
    for directory in (split, task):
        metadata = directory.lstat()
        assert directory.is_dir() and not directory.is_symlink()
        assert metadata.st_uid == __import__("os").geteuid()
        assert metadata.st_mode & 0o022 == 0
    assert (task / f"{cell.tag}.completed.json").is_file()
    assert not list(task.glob(f".{cell.tag}.*"))


@pytest.mark.parametrize("component", ["root", "split", "task"])
def test_publisher_rejects_symlink_directory_components(tmp_path: Path, component: str):
    cell = cell_for("atomic", "OpenDrawer", 1)
    outside = tmp_path / "outside"
    outside.mkdir(mode=0o700)
    root = tmp_path / "results"
    if component == "root":
        root.symlink_to(outside, target_is_directory=True)
    else:
        root.mkdir(mode=0o700)
        split = root / "atomic"
        if component == "split":
            split.symlink_to(outside, target_is_directory=True)
        else:
            split.mkdir(mode=0o700)
            (split / cell.task).symlink_to(outside, target_is_directory=True)
    result, audit, commands = _minimal_publication(cell)

    with pytest.raises(ValueError, match="artifact"):
        publish_completed_cell(root, cell, result, audit, commands)

    assert not list(outside.rglob("*.completed.json"))


@pytest.mark.parametrize("component", ["root", "split", "task"])
def test_publisher_rejects_non_directory_components(tmp_path: Path, component: str):
    cell = cell_for("atomic", "OpenDrawer", 1)
    root = tmp_path / "results"
    if component == "root":
        root.write_text("not a directory\n", encoding="utf-8")
    else:
        root.mkdir(mode=0o700)
        split = root / "atomic"
        if component == "split":
            split.write_text("not a directory\n", encoding="utf-8")
        else:
            split.mkdir(mode=0o700)
            (split / cell.task).write_text("not a directory\n", encoding="utf-8")
    result, audit, commands = _minimal_publication(cell)

    with pytest.raises(ValueError, match="artifact"):
        publish_completed_cell(root, cell, result, audit, commands)


@pytest.mark.parametrize("component", ["root", "split", "task"])
@pytest.mark.parametrize("unsafe_mode", [0o720, 0o702])
def test_publisher_rejects_group_or_world_writable_components(
    tmp_path: Path, component: str, unsafe_mode: int
):
    cell = cell_for("atomic", "OpenDrawer", 1)
    root = tmp_path / "results"
    split = root / "atomic"
    task = split / cell.task
    task.mkdir(parents=True, mode=0o700)
    for directory in (root, split, task):
        directory.chmod(0o700)
    {"root": root, "split": split, "task": task}[component].chmod(unsafe_mode)
    result, audit, commands = _minimal_publication(cell)

    with pytest.raises(ValueError, match="writable by group or other"):
        publish_completed_cell(root, cell, result, audit, commands)


@pytest.mark.parametrize(
    ("component", "mismatched_call"), [("root", 1), ("split", 2), ("task", 3)]
)
def test_publisher_rejects_directory_not_owned_by_current_uid(
    tmp_path: Path, monkeypatch, component: str, mismatched_call: int
):
    from rpent.reproduce.robocasa import artifacts as artifact_module

    cell = cell_for("atomic", "OpenDrawer", 1)
    root = tmp_path / "results"
    (root / "atomic" / cell.task).mkdir(parents=True, mode=0o700)
    actual_uid = artifact_module.os.geteuid()
    calls = 0

    def observed_uid():
        nonlocal calls
        calls += 1
        return actual_uid + int(calls == mismatched_call)

    monkeypatch.setattr(artifact_module.os, "geteuid", observed_uid)
    result, audit, commands = _minimal_publication(cell)

    with pytest.raises(ValueError, match=f"artifact {component} must be owned"):
        publish_completed_cell(root, cell, result, audit, commands)


def test_log_archive_rejects_symlink_descendant(tmp_path: Path):
    cell = cell_for("atomic", "OpenDrawer", 1)
    root = tmp_path / "results"
    task = root / "atomic" / cell.task
    task.mkdir(parents=True, mode=0o700)
    root.chmod(0o700)
    (root / "atomic").chmod(0o700)
    outside = tmp_path / "outside-logs"
    outside.mkdir(mode=0o700)
    (task / "run_logs").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="artifact subdirectory"):
        with secure_artifact_subdirectory(root, cell, "run_logs", cell.tag, "run-1"):
            pass
