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


import pytest

from rpent.memory import MemoryManager


@pytest.fixture
def memory(tmp_path, monkeypatch):
    monkeypatch.setenv("RPENT_REPO_ROOT", str(tmp_path))
    return MemoryManager(
        tmp_path / "memory/libero",
        memory_access="inbox_write",
        inbox_cell_tag="current",
    )


@pytest.mark.parametrize(
    "relative",
    [
        "",
        "MEMORY.md",
        "notes.md",
        "global/strategy.md",
        "suite/task.md",
        "task_only/audit.json",
        "results/recipe.jsonl",
    ],
)
def test_published_scopes_are_readable_and_never_writable(memory, relative):
    path = memory.root / relative
    assert memory.authorize_read(path) == path
    with pytest.raises(PermissionError):
        memory.authorize_write(path)


@pytest.mark.parametrize(
    "relative",
    [
        "_internal/inbox/other/draft.md",
        "_internal/merged/current/draft.md",
        "private/file.md",
    ],
)
def test_other_private_paths_are_denied(memory, relative):
    for authorize in (memory.authorize_read, memory.authorize_write):
        with pytest.raises(PermissionError):
            authorize(memory.root / relative)


def test_current_inbox_is_readable_and_writable_only_in_exploration(memory):
    path = memory.root / "_internal/inbox/current/draft.md"
    assert memory.authorize_read(path) == path
    assert memory.authorize_write(path) == path
    evaluation = MemoryManager(memory.root)
    for authorize in (evaluation.authorize_read, evaluation.authorize_write):
        with pytest.raises(PermissionError):
            authorize(path)


def test_nonmemory_paths_and_relative_paths_resolve(memory, tmp_path):
    assert memory.authorize_write("output/note.txt") == tmp_path / "output/note.txt"
    assert memory.authorize_read("memory/libero/MEMORY.md") == memory.root / "MEMORY.md"
    assert memory.authorize_read("") == tmp_path


def test_foreign_memory_and_symlink_escapes_are_denied(memory, tmp_path):
    foreign = tmp_path / "memory/robotwin/global/note.md"
    foreign.parent.mkdir(parents=True)
    foreign.write_text("foreign")
    link = tmp_path / "link.md"
    link.symlink_to(foreign)
    for authorize in (memory.authorize_read, memory.authorize_write):
        for path in (foreign, link):
            with pytest.raises(PermissionError, match="another robot"):
                authorize(path)
    alias = tmp_path / "memory_alias"
    alias.symlink_to(tmp_path / "memory", target_is_directory=True)
    with pytest.raises(PermissionError, match="another robot"):
        memory.authorize_read(alias / "robotwin/global/note.md")


def test_authorization_returns_resolved_target_for_io(memory, tmp_path):
    destination = tmp_path / "files/note.txt"
    destination.parent.mkdir()
    destination.write_text("note")
    link = tmp_path / "note-link.txt"
    link.symlink_to(destination)
    assert memory.authorize_read(link) == destination
    assert memory.authorize_write(link) == destination
