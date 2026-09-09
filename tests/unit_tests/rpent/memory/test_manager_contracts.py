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
from pathlib import Path
from typing import Any

import yaml

from rpent.memory import MemoryManager


def _write_memory_leaf(
    path: Path,
    *,
    memory_id: str,
    scope: str,
    cells: list[str],
    attempts: int = 1,
    body: str = "Contract body.\n",
) -> None:
    metadata: dict[str, Any] = {
        "id": memory_id,
        "scope": scope,
        "evidence": {
            "cells": cells,
            "attempts": attempts,
            "solved_seeds": [0],
            "failed_seeds": [],
        },
        "confidence": "single-shot",
        "related": [],
    }
    if scope == "suite":
        metadata.update(
            suite="libero10",
            regime="task",
            task_id=2,
            task_language="turn on the stove and put the pan on it",
        )
    else:
        metadata.update(
            kind="strategy",
            title="Reliable strategy",
            applies_when="the scene matches",
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "---\n"
        + yaml.safe_dump(metadata, sort_keys=False, allow_unicode=True)
        + "---\n"
        + body
    )


def _write_task_pair(output_dir: Path, cell: str, *, solved: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"{cell}.json").write_text(json.dumps({"libero_terminated": solved}))
    (output_dir / f"{cell}_recipe.jsonl").write_text('{"action":"move_to"}\n')


def test_memory_manager_index_lists_valid_leaves_by_scope(tmp_path: Path) -> None:
    memory_dir = tmp_path / "memory"
    _write_memory_leaf(
        memory_dir / "global" / "global_strategy.md",
        memory_id="global_strategy",
        scope="global",
        cells=["10_task_t2_s0"],
    )
    _write_memory_leaf(
        memory_dir / "suite" / "suite_libero10_task_t2.md",
        memory_id="suite_libero10_task_t2",
        scope="suite",
        cells=["10_task_t2_s0"],
    )

    index = MemoryManager(memory_dir).rebuild_index()
    text = index.read_text()

    assert index == memory_dir / "MEMORY.md"
    assert text.index("## Global") < text.index("## Suite")
    assert "[Reliable strategy](global/global_strategy.md) — the scene matches" in text
    assert "[suite_libero10_task_t2](suite/suite_libero10_task_t2.md)" in text


def test_rebuild_index_regenerates_from_valid_leaves_skipping_plain_ones(
    tmp_path: Path,
) -> None:
    memory_dir = tmp_path / "memory"
    _write_memory_leaf(
        memory_dir / "global" / "global_strategy.md",
        memory_id="global_strategy",
        scope="global",
        cells=["10_task_t2_s0"],
    )
    # A hand-curated published note without frontmatter is skipped, not
    # indexed, and does not block regeneration from valid leaves.
    (memory_dir / "global" / "hand_note.md").write_text("# Hand-curated note\n")
    hand_index = memory_dir / "MEMORY.md"
    hand_index.write_text("# Hand-maintained index\n\n- [note](global/hand_note.md)\n")

    index = MemoryManager(memory_dir).rebuild_index()

    assert index == hand_index
    text = index.read_text()
    # The hand-maintained index is replaced by the auto-generated one built
    # from frontmatter-bearing leaves; the plain leaf is not listed.
    assert text.startswith("# Layered memory index")
    assert "[Reliable strategy](global/global_strategy.md)" in text
    assert "hand_note.md" not in text


def test_rebuild_index_noops_when_no_frontmatter_leaves_exist(
    tmp_path: Path,
) -> None:
    memory_dir = tmp_path / "memory"
    (memory_dir / "global").mkdir(parents=True)
    (memory_dir / "global" / "hand_note.md").write_text("# Hand-curated note\n")
    hand_index = memory_dir / "MEMORY.md"
    hand_index.write_text("# Hand-maintained index\n")

    index = MemoryManager(memory_dir).rebuild_index()

    assert index is None
    # Nothing written, so a hand-maintained index (if any) is left alone.
    assert hand_index.read_text() == "# Hand-maintained index\n"


def test_memory_manager_validation_reports_schema_filename_and_duplicate_errors(
    tmp_path: Path,
) -> None:
    memory_dir = tmp_path / "memory"
    _write_memory_leaf(
        memory_dir / "global" / "shared_id.md",
        memory_id="shared_id",
        scope="global",
        cells=["10_task_t2_s0"],
    )
    _write_memory_leaf(
        memory_dir / "suite" / "wrong_filename.md",
        memory_id="shared_id",
        scope="suite",
        cells=["10_task_t2_s1"],
    )
    (memory_dir / "global" / "broken.md").write_text("---\nscope: global\n")

    problems = MemoryManager(memory_dir).validate()

    assert any("broken.md: unterminated YAML frontmatter" in item for item in problems)
    assert any("id 'shared_id' does not match filename" in item for item in problems)
    assert any("duplicate id also in global/shared_id.md" in item for item in problems)


def test_memory_manager_publishes_draft_and_solved_task_pair(tmp_path: Path) -> None:
    memory_dir = tmp_path / "memory"
    output_dir = tmp_path / "run"
    cell = "10_task_t2_s0"
    _write_memory_leaf(
        memory_dir / "_internal" / "inbox" / cell / "suite_draft.md",
        memory_id="draft_id_is_replaced",
        scope="suite",
        cells=[cell],
    )
    _write_task_pair(output_dir, cell, solved=True)

    result = MemoryManager(memory_dir).merge_memory(
        cell_tag=cell,
        run_state_dir=output_dir,
        solved=True,
    )

    assert result["suite"] == 1
    assert result["task"] == 1
    assert result["skipped"] == []
    assert (memory_dir / "suite" / "suite_libero10_task_t2.md").is_file()
    assert (memory_dir / "task_only" / f"{cell}.json").is_file()
    assert (memory_dir / "task_only" / f"{cell}_recipe.jsonl").is_file()
    assert (memory_dir / "_internal" / "merged" / cell / "suite_draft.md").is_file()
    assert "suite_libero10_task_t2.md" in (memory_dir / "MEMORY.md").read_text()


def test_memory_manager_does_not_publish_unsolved_task_artifacts(
    tmp_path: Path,
) -> None:
    memory_dir = tmp_path / "memory"
    output_dir = tmp_path / "run"
    cell = "10_task_t9_s0"
    _write_task_pair(output_dir, cell, solved=False)

    result = MemoryManager(memory_dir).merge_memory(
        cell_tag=cell,
        run_state_dir=output_dir,
        solved=False,
    )

    assert result["task"] == 0
    assert not (memory_dir / "task_only" / f"{cell}.json").exists()
    assert not (memory_dir / "task_only" / f"{cell}_recipe.jsonl").exists()


def test_memory_manager_skips_invalid_draft_without_archiving_its_inbox(
    tmp_path: Path,
) -> None:
    memory_dir = tmp_path / "memory"
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    cell = "10_task_t0_s0"
    inbox = memory_dir / "_internal" / "inbox" / cell
    inbox.mkdir(parents=True)
    (inbox / "new_global_strategy.md").write_text(
        """---
scope: global
kind: strategy
title: Test strategy
applies_when: Testing malformed evidence
evidence: [10_task_t0_s0]
confidence: single-shot
---
Test body.
"""
    )

    result = MemoryManager(memory_dir).merge_memory(
        cell_tag=cell,
        run_state_dir=output_dir,
        solved=False,
    )

    assert result["global"] == 0
    assert result["skipped"] == ["new_global_strategy.md: evidence must be a mapping"]
    assert inbox.is_dir()
    assert not (memory_dir / "_internal" / "merged" / cell).exists()


def test_memory_manager_accumulates_evidence_and_preserves_conflicting_prose(
    tmp_path: Path,
) -> None:
    memory_dir = tmp_path / "memory"
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    first_cell = "10_task_t2_s0"
    second_cell = "10_task_t2_s1"
    first_inbox = memory_dir / "_internal" / "inbox" / first_cell
    second_inbox = memory_dir / "_internal" / "inbox" / second_cell
    _write_memory_leaf(
        first_inbox / "suite_draft.md",
        memory_id="ignored",
        scope="suite",
        cells=[first_cell],
        attempts=1,
        body="First published prose.\n",
    )
    manager = MemoryManager(memory_dir)
    manager.merge_memory(
        cell_tag=first_cell,
        run_state_dir=output_dir,
        solved=False,
    )
    _write_memory_leaf(
        second_inbox / "suite_draft.md",
        memory_id="ignored",
        scope="suite",
        cells=[second_cell],
        attempts=2,
        body="Conflicting new prose.\n",
    )

    result = manager.merge_memory(
        cell_tag=second_cell,
        run_state_dir=output_dir,
        solved=False,
    )

    published = memory_dir / "suite" / "suite_libero10_task_t2.md"
    published_metadata = yaml.safe_load(published.read_text().split("---", 2)[1])
    assert result["suite"] == 0
    assert result["evidence"] == 1
    assert result["conflicts"] == 1
    assert published_metadata["evidence"]["cells"] == [first_cell, second_cell]
    assert published_metadata["evidence"]["attempts"] == 3
    assert published_metadata["confidence"] == "probable"
    assert "First published prose." in published.read_text()
    conflict = (
        memory_dir
        / "_internal"
        / "conflicts"
        / f"suite_libero10_task_t2__from_{second_cell}.md"
    )
    assert "Conflicting new prose." in conflict.read_text()


def test_merge_memory_leaves_existing_plain_markdown_untouched(
    tmp_path: Path,
) -> None:
    memory_dir = tmp_path / "memory"
    output_dir = tmp_path / "run"
    cell = "10_task_t2_s0"
    existing = memory_dir / "global" / "grasp_strategy.md"
    existing.parent.mkdir(parents=True)
    existing.write_text("# Hand-curated grasp note\nno frontmatter\n")
    _write_memory_leaf(
        memory_dir / "_internal" / "inbox" / cell / "grasp_strategy_draft.md",
        memory_id="grasp_strategy",
        scope="global",
        cells=[cell],
        body="Conflicting structured draft.\n",
    )

    result = MemoryManager(memory_dir).merge_memory(
        cell_tag=cell,
        run_state_dir=output_dir,
        solved=False,
    )

    assert result["global"] == 0
    assert result["conflicts"] == 1
    assert any("grasp_strategy" in item for item in result["skipped"])
    # The hand-curated plain note is left untouched.
    assert existing.read_text() == "# Hand-curated grasp note\nno frontmatter\n"
    # The incoming structured draft is archived as a conflict.
    conflict = (
        memory_dir / "_internal" / "conflicts" / f"grasp_strategy__from_{cell}.md"
    )
    assert "Conflicting structured draft." in conflict.read_text()


def test_memory_manager_is_idempotent_for_evidence_from_the_same_cell(
    tmp_path: Path,
) -> None:
    memory_dir = tmp_path / "memory"
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    cell = "10_task_t2_s0"
    draft = memory_dir / "_internal" / "inbox" / cell / "suite_draft.md"
    _write_memory_leaf(
        draft,
        memory_id="ignored",
        scope="suite",
        cells=[cell],
    )
    manager = MemoryManager(memory_dir)
    first = manager.merge_memory(
        cell_tag=cell,
        run_state_dir=output_dir,
        solved=False,
    )
    _write_memory_leaf(
        draft,
        memory_id="ignored",
        scope="suite",
        cells=[cell],
    )

    second = manager.merge_memory(
        cell_tag=cell,
        run_state_dir=output_dir,
        solved=False,
    )

    assert first["suite"] == 1
    assert second["suite"] == 0
    assert second["evidence"] == 0
    assert second["conflicts"] == 0


def test_memory_manager_refuses_to_complete_an_existing_partial_task_pair(
    tmp_path: Path,
) -> None:
    memory_dir = tmp_path / "memory"
    output_dir = tmp_path / "run"
    cell = "10_task_t2_s0"
    _write_task_pair(output_dir, cell, solved=True)
    task_dir = memory_dir / "task_only"
    task_dir.mkdir(parents=True)
    (task_dir / f"{cell}.json").write_text("{}")

    result = MemoryManager(memory_dir).merge_memory(
        cell_tag=cell,
        run_state_dir=output_dir,
        solved=True,
    )

    assert result["task"] == 0
    assert result["skipped"] == ["incomplete existing task audit/recipe pair"]
    assert not (task_dir / f"{cell}_recipe.jsonl").exists()
