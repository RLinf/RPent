from __future__ import annotations

import json
from pathlib import Path

from rpent.memory.layered import merge_cell


def _write_suite_draft(inbox: Path, cell: str) -> None:
    inbox.mkdir(parents=True)
    (inbox / f"suite_{cell}_draft.md").write_text(
        f"""---
id: suite_libero10_task_t2
scope: suite
suite: libero10
regime: task
task_id: 2
task_language: turn on the stove and put the pan on it
evidence:
  cells: [{cell}]
  attempts: 3
  solved_seeds: [0]
  failed_seeds: []
confidence: single-shot
related: []
---
## Applicable pattern

Test body.
"""
    )


def test_merge_cell_publishes_drafts_and_ground_truth_task_pair(tmp_path: Path):
    memory_dir = tmp_path / "memory"
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    cell = "10_task_t2_s0"
    _write_suite_draft(memory_dir / "_inbox" / cell, cell)
    (output_dir / f"{cell}.json").write_text(
        json.dumps({"libero_terminated": True})
    )
    (output_dir / f"recipe_{cell}.jsonl").write_text('{"action":"move_to"}\n')

    result = merge_cell(
        memory_dir=memory_dir, cell_tag=cell, output_dir=output_dir
    )

    assert result["suite"] == 1
    assert result["task"] == 1
    assert (memory_dir / "suite" / "suite_libero10_task_t2.md").exists()
    assert (memory_dir / "task" / f"{cell}.json").exists()
    assert (memory_dir / "task" / f"recipe_{cell}.jsonl").exists()
    assert (memory_dir / "_merged" / cell).is_dir()
    assert "suite_libero10_task_t2.md" in (memory_dir / "MEMORY.md").read_text()


def test_merge_cell_never_publishes_unsolved_task_recipe(tmp_path: Path):
    memory_dir = tmp_path / "memory"
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    cell = "10_swap_t9_s0"
    (output_dir / f"{cell}.json").write_text(
        json.dumps({"libero_terminated": False})
    )
    (output_dir / f"recipe_{cell}.jsonl").write_text('{"action":"move_to"}\n')

    result = merge_cell(
        memory_dir=memory_dir, cell_tag=cell, output_dir=output_dir
    )

    assert result["task"] == 0
    assert not (memory_dir / "task" / f"{cell}.json").exists()
    assert not (memory_dir / "task" / f"recipe_{cell}.jsonl").exists()


def test_merge_cell_skips_draft_with_non_mapping_evidence(tmp_path: Path):
    memory_dir = tmp_path / "memory"
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    cell = "10_task_t0_s0"
    inbox = memory_dir / "_inbox" / cell
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

    result = merge_cell(
        memory_dir=memory_dir, cell_tag=cell, output_dir=output_dir
    )

    assert result["global"] == 0
    assert result["skipped"] == [
        "new_global_strategy.md: evidence must be a mapping"
    ]


def test_merge_cell_is_idempotent_for_same_cell(tmp_path: Path):
    memory_dir = tmp_path / "memory"
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    cell = "10_task_t2_s0"
    _write_suite_draft(memory_dir / "_inbox" / cell, cell)

    first = merge_cell(memory_dir=memory_dir, cell_tag=cell, output_dir=output_dir)
    # Simulate a repeated distillation of the same cell.
    _write_suite_draft(memory_dir / "_inbox" / cell, cell)
    second = merge_cell(memory_dir=memory_dir, cell_tag=cell, output_dir=output_dir)

    assert first["suite"] == 1
    assert second["suite"] == 0
    assert second["evidence"] == 0
