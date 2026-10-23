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

"""BEHAVIOR task evidence publication through the canonical memory merge."""

from __future__ import annotations

import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest

from robots.behavior import memory as behavior_memory
from robots.behavior.dino_v2.index import _source_set
from robots.behavior.memory import BehaviorMemoryManager


def _manager(root: Path, prepared: Path) -> BehaviorMemoryManager:
    manager = BehaviorMemoryManager(root)
    manager.task_artifacts = prepared
    return manager


def test_publishes_audit_last_for_concurrent_dino_readers(
    prepared_corpus: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = prepared_corpus.root
    run = prepared_corpus.run
    cell = prepared_corpus.cell
    prepared = prepared_corpus.prepared
    published: list[tuple[str, int]] = []
    original_replace = os.replace

    def observe_replace(source: str | Path, target: str | Path) -> None:
        original_replace(source, target)
        if Path(target).name in {cell, f"{cell}.json", f"{cell}_recipe.jsonl"}:
            published.append((Path(target).name, len(_source_set(root)["sources"])))

    monkeypatch.setattr(behavior_memory.os, "replace", observe_replace)
    result = _manager(root, prepared).merge_memory(
        cell_tag=cell, run_state_dir=run, solved=True
    )

    assert result["task"] == 1
    assert published == [
        (cell, 0),
        (f"{cell}_recipe.jsonl", 0),
        (f"{cell}.json", 1),
    ]


def test_failed_audit_commit_rolls_back_and_retry_publishes(
    prepared_corpus: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = prepared_corpus.root
    run = prepared_corpus.run
    cell = prepared_corpus.cell
    prepared = prepared_corpus.prepared
    manager = _manager(root, prepared)
    task_dir = root / "task-specific"
    audit_target = task_dir / f"{cell}.json"
    orphan = task_dir / "artifacts" / cell
    orphan.parent.mkdir(parents=True)
    shutil.copytree(prepared, orphan)
    (orphan / "manifest.json").write_text("{}")
    original_replace = os.replace

    def fail_audit(source: str | Path, target: str | Path) -> None:
        if Path(target) == audit_target:
            raise OSError("simulated audit publication failure")
        original_replace(source, target)

    with monkeypatch.context() as patch:
        patch.setattr(behavior_memory.os, "replace", fail_audit)
        with pytest.raises(OSError, match="simulated audit publication failure"):
            manager.merge_memory(cell_tag=cell, run_state_dir=run, solved=True)

    assert not audit_target.exists()
    assert not (task_dir / f"{cell}_recipe.jsonl").exists()
    assert (orphan / "manifest.json").read_text() == "{}"
    assert (
        manager.merge_memory(cell_tag=cell, run_state_dir=run, solved=True)["task"] == 1
    )
    assert len(_source_set(root)["sources"]) == 1


def test_concurrent_merges_keep_one_canonical_task_pair(
    prepared_corpus: SimpleNamespace,
) -> None:
    root = prepared_corpus.root
    run = prepared_corpus.run
    cell = prepared_corpus.cell
    prepared = prepared_corpus.prepared

    def merge() -> dict:
        return _manager(root, prepared).merge_memory(
            cell_tag=cell, run_state_dir=run, solved=True
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: merge(), range(2)))

    assert sorted(item["task"] for item in results) == [0, 1]
    assert len(_source_set(root)["sources"]) == 1
