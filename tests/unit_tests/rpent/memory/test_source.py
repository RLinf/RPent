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

"""Shared source flags and robot subtree selection."""

from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import huggingface_hub
import pytest

from rpent.memory.loading import prepare_run_memory
from rpent.memory.source import parse_repo, prepare_explicit_source


@pytest.fixture(autouse=True)
def isolated_memory_root(tmp_path, monkeypatch):
    monkeypatch.setenv("RPENT_REPO_ROOT", str(tmp_path / "rpent"))


@pytest.mark.parametrize(
    "value,expected",
    [
        ("team/memory", ("team/memory", "main")),
        ("team/memory@release/v1", ("team/memory", "release/v1")),
        (
            "https://huggingface.co/datasets/team/memory@" + "a" * 40,
            ("team/memory", "a" * 40),
        ),
    ],
)
def test_repo_source_parsing(value, expected):
    assert parse_repo(value) == expected


@pytest.mark.parametrize(
    "value",
    [
        "",
        "team",
        "team/memory@",
        "https://github.com/team/memory",
        "https://huggingface.co/team/memory",
        "https://user:secret@huggingface.co/datasets/team/memory",
        "../memory",
        "team/memory@ main",
    ],
)
def test_invalid_source_urls(value):
    with pytest.raises(ValueError):
        parse_repo(value)


@pytest.mark.parametrize("robot", ["robocasa", "robotwin", "franka"])
def test_common_preparation_selects_requested_robot_at_pinned_commit(
    tmp_path, monkeypatch, robot
):
    commit = "a" * 40
    snapshot = tmp_path / "snapshots" / commit
    corpus = snapshot / robot
    corpus.mkdir(parents=True)
    (corpus / "MEMORY.md").write_text("source")
    fetch = Mock(return_value=str(snapshot))
    monkeypatch.setattr(huggingface_hub, "snapshot_download", fetch)
    spec = SimpleNamespace(
        name=robot, prepare_memory=None, memory_repo_id="default/memory"
    )
    args = Namespace(
        planner="api",
        model="test",
        memory_repo=f"team/memory@{commit}",
        memory_profile="hf",
    )
    config = SimpleNamespace(output_dir=tmp_path / "run", prompt_vars={})
    prepare_run_memory(args, spec, config)
    selected = Path(config.prompt_vars["memory_dir"])
    assert selected != corpus
    assert (selected / "MEMORY.md").read_text() == "source"
    assert args.memory_dir == str(selected)
    assert fetch.call_args.kwargs["allow_patterns"] == [f"{robot}/**"]
    assert fetch.call_args.kwargs["revision"] == commit
    assert not config.output_dir.exists()


@pytest.mark.parametrize("robot", ["robocasa", "robotwin", "franka"])
def test_hub_symlinks_become_read_only_memory_files(tmp_path, monkeypatch, robot):
    from robots.robocasa.memory import TaskMemory
    from rpent.memory import MemoryManager
    from rpent.tools.common import CommonTools

    snapshot = tmp_path / "hub" / "snapshots" / ("a" * 40)
    blob = tmp_path / "hub" / "blobs" / "content"
    blob.parent.mkdir(parents=True)
    blob.write_text("published memory")
    name = "global/GLOBAL_MEMORY.md" if robot == "robocasa" else "MEMORY.md"
    link = snapshot / robot / name
    link.parent.mkdir(parents=True)
    link.symlink_to(blob)
    monkeypatch.setattr(
        huggingface_hub, "snapshot_download", lambda **kwargs: str(snapshot)
    )

    root = prepare_explicit_source("team/memory@" + "a" * 40, robot)
    path = root / name
    assert not path.is_symlink()
    assert path.resolve().is_relative_to(root.resolve())
    if robot == "robocasa":
        memory = TaskMemory.load(root, None, profile="hf")
        assert memory.contents[name] == "published memory"
    tools = CommonTools(memory=MemoryManager(root))
    assert tools.read_text_file(str(path)).data["content"] == "published memory"
    with pytest.raises(PermissionError, match="writing to memory is denied"):
        tools.write_text_file(str(path), "changed")
    assert blob.read_text() == "published memory"


def test_copies_are_reused_and_isolated_by_repository_and_commit(tmp_path, monkeypatch):
    snapshots = {}
    for commit in ("a" * 40, "b" * 40):
        snapshot = tmp_path / "snapshots" / commit
        corpus = snapshot / "robotwin"
        corpus.mkdir(parents=True)
        (corpus / "MEMORY.md").write_text(commit)
        snapshots[commit] = snapshot
    monkeypatch.setattr(
        huggingface_hub,
        "snapshot_download",
        lambda **kwargs: str(snapshots[kwargs["revision"]]),
    )
    first = prepare_explicit_source("team/memory@" + "a" * 40, "robotwin")
    modified = (first / "MEMORY.md").stat().st_mtime_ns
    assert prepare_explicit_source("team/memory@" + "a" * 40, "robotwin") == first
    assert (first / "MEMORY.md").stat().st_mtime_ns == modified
    other_commit = prepare_explicit_source("team/memory@" + "b" * 40, "robotwin")
    other_repo = prepare_explicit_source("other/memory@" + "a" * 40, "robotwin")
    assert len({first, other_commit, other_repo}) == 3
    assert (first / "MEMORY.md").read_text() == "a" * 40
    assert (other_commit / "MEMORY.md").read_text() == "b" * 40


def test_failed_copy_is_not_reused(tmp_path, monkeypatch):
    snapshot = tmp_path / "snapshots" / ("a" * 40)
    corpus = snapshot / "robotwin"
    corpus.mkdir(parents=True)
    blob = tmp_path / "missing-blob"
    (corpus / "MEMORY.md").symlink_to(blob)
    monkeypatch.setattr(
        huggingface_hub, "snapshot_download", lambda **kwargs: str(snapshot)
    )
    import shutil

    with pytest.raises(shutil.Error):
        prepare_explicit_source("team/memory@" + "a" * 40, "robotwin")
    blob.write_text("complete")
    root = prepare_explicit_source("team/memory@" + "a" * 40, "robotwin")
    assert (root / "MEMORY.md").read_text() == "complete"


def test_missing_robot_subtree_does_not_fall_back_to_default(tmp_path, monkeypatch):
    snapshot = tmp_path / ("a" * 40)
    snapshot.mkdir()
    monkeypatch.setattr(
        huggingface_hub, "snapshot_download", lambda **kwargs: str(snapshot)
    )
    args = Namespace(planner="api", model="test", memory_repo="team/memory")
    spec = SimpleNamespace(name="robotwin", prepare_memory=None)
    config = SimpleNamespace(output_dir=tmp_path / "run", prompt_vars={})
    with pytest.raises(ValueError, match="no robotwin"):
        prepare_run_memory(args, spec, config)
    assert not config.output_dir.exists()


@pytest.mark.parametrize("dashboard", [False, True])
@pytest.mark.parametrize("option", ["--memory-revision", "--memory-source"])
def test_removed_source_flags_are_rejected(dashboard, option, capsys):
    from robots.libero.robot_spec import get_robot_spec
    from rpent.cli.main import _build_argparser

    parser = _build_argparser()
    get_robot_spec().add_cli_args(parser, use_dashboard=dashboard)
    task = ["--dashboard"] if dashboard else ["--suite", "libero_object", "--task", "0"]
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["--robot", "libero", *task, option, "unused"])
    assert exc.value.code == 2
    assert f"unrecognized arguments: {option}" in capsys.readouterr().err


@pytest.mark.parametrize("version", ["../other", "/absolute", "a/b", "..", ""])
def test_invalid_corpus_directory_rejected_before_download(monkeypatch, version):
    fetch = Mock()
    monkeypatch.setattr(huggingface_hub, "snapshot_download", fetch)
    with pytest.raises(ValueError, match="single corpus directory"):
        prepare_explicit_source("team/memory", "robotwin", version=version)
    fetch.assert_not_called()


def test_versioned_copies_do_not_reuse_unversioned_copy(tmp_path, monkeypatch):
    snapshot = tmp_path / "snapshots" / ("a" * 40)
    root = snapshot / "robotwin"
    root.mkdir(parents=True)
    (root / "MEMORY.md").write_text("default")
    monkeypatch.setattr(
        huggingface_hub, "snapshot_download", lambda **kw: str(snapshot)
    )
    default = prepare_explicit_source("team/memory", "robotwin")
    corpus = root / "Astra_reference"
    corpus.mkdir()
    (corpus / "MEMORY.md").write_text("astra")
    selected = prepare_explicit_source(
        "team/memory", "robotwin", version="Astra_reference"
    )
    assert (default / "MEMORY.md").read_text() == "default"
    assert (selected / "MEMORY.md").read_text() == "astra"


def test_unversioned_copy_after_versioned_download(tmp_path, monkeypatch):
    snapshot = tmp_path / "snapshots" / ("a" * 40)
    root = snapshot / "robotwin"
    corpus = root / "Astra_reference"
    corpus.mkdir(parents=True)
    (corpus / "MEMORY.md").write_text("astra")
    (root / "MEMORY.md").write_text("default")
    monkeypatch.setattr(
        huggingface_hub, "snapshot_download", lambda **kw: str(snapshot)
    )
    selected = prepare_explicit_source(
        "team/memory", "robotwin", version="Astra_reference"
    )
    default = prepare_explicit_source("team/memory", "robotwin")
    assert (default / "MEMORY.md").read_text() == "default"
    assert (selected / "MEMORY.md").read_text() == "astra"
