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

"""Shared source flags, portable records and robot subtree selection."""

import json
from argparse import Namespace
from types import SimpleNamespace
from unittest.mock import Mock

import huggingface_hub
import pytest

from rpent.memory.loading import prepare_run_memory
from rpent.memory.source import (
    parse_repo,
    read_source,
    source_record,
    validate_source_options,
)


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


@pytest.mark.parametrize(
    "args",
    [
        Namespace(memory_repo=""),
        Namespace(memory_source=""),
        Namespace(memory_repo="team/memory", memory_source="source.json"),
        Namespace(memory_repo="team/memory", memory_profile="local"),
        Namespace(memory_repo="team/memory", explore=True),
        Namespace(memory_repo="team/memory", memory_revision="abc"),
    ],
)
def test_conflicts_fail_before_download(args):
    with pytest.raises(ValueError):
        validate_source_options(args, "robotwin")


@pytest.mark.parametrize("robot", ["robocasa", "robotwin", "franka"])
def test_common_preparation_selects_requested_robot_and_replays_pin(
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
        memory_repo="team/memory@release",
        memory_profile="hf",
    )
    config = SimpleNamespace(output_dir=tmp_path / "run", prompt_vars={})
    prepare_run_memory(args, spec, config)
    assert config.prompt_vars["memory_dir"] == str(corpus)
    assert args.memory_dir == str(corpus)
    assert fetch.call_args.kwargs["allow_patterns"] == [f"{robot}/**"]
    record = config.output_dir / "memory_source.json"
    assert json.loads(record.read_text()) == source_record("team/memory", commit, robot)
    args.memory_repo = None
    args.memory_source = str(record)
    config.output_dir = tmp_path / "replay"
    prepare_run_memory(args, spec, config)
    assert fetch.call_args.kwargs["revision"] == commit
    assert json.loads(
        (config.output_dir / "memory_source.json").read_text()
    ) == json.loads(record.read_text())


@pytest.mark.parametrize(
    "change",
    [
        {"robot": "libero"},
        {"profile": "local"},
        {"schema_version": 1},
        {"source": "team/memory@main"},
    ],
)
def test_replay_rejects_incompatible_or_mutable_records(tmp_path, change):
    p = tmp_path / "source.json"
    p.write_text(
        json.dumps({**source_record("team/memory", "a" * 40, "robotwin"), **change})
    )
    with pytest.raises(ValueError):
        read_source(p, "robotwin")


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
    assert not (config.output_dir / "memory_source.json").exists()
