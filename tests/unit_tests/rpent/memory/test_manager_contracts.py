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

import numpy as np
import pytest
import yaml

from robots.behavior.dino_v2.encoder import DINOV2_DIMENSION
from robots.behavior.memory import BehaviorMemoryManager
from robots.behavior.tools import BehaviorPrimitives
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
    if scope == "task-family":
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
        memory_dir / "task-family" / "task-family_libero10_task_t2.md",
        memory_id="task-family_libero10_task_t2",
        scope="task-family",
        cells=["10_task_t2_s0"],
    )

    index = MemoryManager(memory_dir).rebuild_index()
    text = index.read_text()

    assert index == memory_dir / "MEMORY.md"
    assert text.index("## Global") < text.index("## Task-family")
    assert "[Reliable strategy](global/global_strategy.md) — the scene matches" in text
    assert (
        "[task-family_libero10_task_t2](task-family/task-family_libero10_task_t2.md)"
        in text
    )


@pytest.mark.parametrize("legacy", ["task_only", "suite"])
def test_unmigrated_layers_fail_before_reading_or_publishing(tmp_path, legacy):
    root = tmp_path / "memory"
    (root / legacy).mkdir(parents=True)
    (root / legacy / "note.md").write_text("legacy note")
    manager = MemoryManager(root)
    with pytest.raises(ValueError, match="legacy memory directory"):
        manager.get_common_tool_bindings()
    with pytest.raises(ValueError, match="legacy memory directory"):
        manager.merge_memory(cell_tag="cell", run_state_dir=tmp_path, solved=True)
    assert not (root / "_internal").exists()


def test_renamed_layers_are_readable_and_stale_cache_paths_are_denied(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("RPENT_REPO_ROOT", str(tmp_path))
    root = tmp_path / "memory" / "libero"
    for name in ("task_only", "suite", "task-specific", "task-family", "global"):
        (root / name).mkdir(parents=True)
        (root / name / "note.md").write_text(name)
    read = MemoryManager(root).get_common_tool_bindings()["read_text_file"][1]
    for name in ("task-specific", "task-family", "global"):
        assert read(path=str(root / name / "note.md"))["content"] == name
    for name in ("task_only", "suite"):
        with pytest.raises(PermissionError, match="reading this memory path is denied"):
            read(path=str(root / name / "note.md"))


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
    (memory_dir / "global" / "malformed.md").write_text("---\nscope: [\n---\n")
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
    assert "malformed.md" not in text


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
        memory_dir / "task-family" / "wrong_filename.md",
        memory_id="shared_id",
        scope="task-family",
        cells=["10_task_t2_s1"],
    )
    (memory_dir / "global" / "broken.md").write_text("---\nscope: global\n")
    (memory_dir / "global" / "malformed.md").write_text("---\nscope: [\n---\n")

    problems = MemoryManager(memory_dir).validate()

    assert any("broken.md: unterminated YAML frontmatter" in item for item in problems)
    assert any("malformed.md: invalid YAML frontmatter" in item for item in problems)
    assert any("id 'shared_id' does not match filename" in item for item in problems)
    assert any("duplicate id also in global/shared_id.md" in item for item in problems)


def test_memory_manager_publishes_draft_and_solved_task_pair(tmp_path: Path) -> None:
    memory_dir = tmp_path / "memory"
    output_dir = tmp_path / "run"
    cell = "10_task_t2_s0"
    _write_memory_leaf(
        memory_dir / "_internal" / "inbox" / cell / "task-family_draft.md",
        memory_id="draft_id_is_replaced",
        scope="task-family",
        cells=[cell],
    )
    _write_task_pair(output_dir, cell, solved=True)
    (memory_dir / "_internal" / "inbox" / cell / "malformed.md").write_text(
        "---\nscope: [\n---\n"
    )

    result = MemoryManager(memory_dir).merge_memory(
        cell_tag=cell,
        run_state_dir=output_dir,
        solved=True,
    )

    assert result["task-family"] == 1
    assert result["task"] == 1
    assert len(result["skipped"]) == 1
    assert result["skipped"][0].startswith("malformed.md: invalid YAML frontmatter")
    assert (memory_dir / "task-family" / "task-family_libero10_task_t2.md").is_file()
    assert (memory_dir / "task-specific" / f"{cell}.json").is_file()
    assert (memory_dir / "task-specific" / f"{cell}_recipe.jsonl").is_file()
    assert (
        memory_dir / "_internal" / "merged" / cell / "task-family_draft.md"
    ).is_file()
    assert "task-family_libero10_task_t2.md" in (memory_dir / "MEMORY.md").read_text()


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
    assert not (memory_dir / "task-specific" / f"{cell}.json").exists()
    assert not (memory_dir / "task-specific" / f"{cell}_recipe.jsonl").exists()


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
        first_inbox / "task-family_draft.md",
        memory_id="ignored",
        scope="task-family",
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
        second_inbox / "task-family_draft.md",
        memory_id="ignored",
        scope="task-family",
        cells=[second_cell],
        attempts=2,
        body="Conflicting new prose.\n",
    )

    result = manager.merge_memory(
        cell_tag=second_cell,
        run_state_dir=output_dir,
        solved=False,
    )

    published = memory_dir / "task-family" / "task-family_libero10_task_t2.md"
    published_metadata = yaml.safe_load(published.read_text().split("---", 2)[1])
    assert result["task-family"] == 0
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
        / f"task-family_libero10_task_t2__from_{second_cell}.md"
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
    draft = memory_dir / "_internal" / "inbox" / cell / "task-family_draft.md"
    _write_memory_leaf(
        draft,
        memory_id="ignored",
        scope="task-family",
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
        scope="task-family",
        cells=[cell],
    )

    second = manager.merge_memory(
        cell_tag=cell,
        run_state_dir=output_dir,
        solved=False,
    )

    assert first["task-family"] == 1
    assert second["task-family"] == 0
    assert second["evidence"] == 0
    assert second["conflicts"] == 0


def test_memory_manager_refuses_to_complete_an_existing_partial_task_pair(
    tmp_path: Path,
) -> None:
    memory_dir = tmp_path / "memory"
    output_dir = tmp_path / "run"
    cell = "10_task_t2_s0"
    _write_task_pair(output_dir, cell, solved=True)
    task_dir = memory_dir / "task-specific"
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


@pytest.fixture
def dino_encoder():
    class Encoder:
        calls = 0
        image_count = 0
        revision = "test-encoder"

        def revision_metadata(self):
            return {"dimension": DINOV2_DIMENSION, "model_revision": self.revision}

        def encode_batch(self, images):
            self.calls += 1
            self.image_count += len(images)
            vector = np.zeros(DINOV2_DIMENSION, dtype=np.float32)
            vector[0] = 1
            return [vector.copy() for image in images]

    return Encoder()


def test_official_memory_matches_are_structured_in_public_results(
    tmp_path, dino_encoder
):

    class Env:
        def observe(self, **kwargs):
            return {"status": "ok", "info": {"done": {"success": False}}}

    primitives = BehaviorPrimitives(
        env=Env(),
        initial_observation={"main_images": np.zeros((8, 8, 3), dtype=np.uint8)},
        memory_dir=tmp_path,
        dino_component=dino_encoder,
    )
    decision = primitives.snapshot()["memory_matches"]
    assert isinstance(decision, dict)
    assert decision["status"] == "empty"
    assert decision["matches"] == []
    assert dino_encoder.calls == 1
    assert primitives.observe(camera="head")["memory_matches"] == decision
    assert json.loads(json.dumps(decision)) == decision


@pytest.fixture
def official_corpus(tmp_path):
    from robots.behavior.dino_v2.index import prepare_evidence_pack
    from robots.behavior.terminal_success import make_raw_success_receipt
    from rpent.session import EnvState

    cell = "turning_on_radio_s0"
    root = tmp_path / "memory"
    run = tmp_path / "run"
    state = EnvState(run / "sessions" / "session_001")
    receipt = make_raw_success_receipt({"done": {"success": True}}, env_step=32)
    terminal = {
        "_finish": True,
        "task_success": True,
        "official_success_receipt": receipt,
    }
    image = np.zeros((16, 16, 3), dtype=np.uint8)
    with state.record_step(
        state={"total_env_steps": 32},
        terminated=True,
        command={"action": "pi0_nav_pick", "chunks": 1},
        result={"official_success": True},
    ):
        for name in ("head_rgb.png", "left_wrist_rgb.png", "right_wrist_rgb.png"):
            state.save(name, image)
    state.save("terminal_receipt.json", terminal, step=None)
    run_state = EnvState(run)
    run_state.save(
        f"{cell}.json",
        {
            "task_name": "turning_on_radio",
            "public_seed": 0,
            "official_success_receipt": receipt,
        },
        step=None,
    )
    run_state.save(
        f"{cell}_recipe.jsonl",
        [{"action": "pi0_nav_pick", "instruction": "turn on radio", "chunks": 1}],
        step=None,
    )
    prepared = prepare_evidence_pack(run, cell, state, terminal)
    assert prepared is not None
    manager = BehaviorMemoryManager(root)
    manager.task_artifacts = prepared
    result = manager.merge_memory(cell_tag=cell, run_state_dir=run, solved=True)
    assert result["task"] == 1
    return root, cell, image, state, terminal


def test_dino_rebuild_query_and_delete_rebuild_match(official_corpus, dino_encoder):
    import shutil

    from robots.behavior.dino_v2.index import query, rebuild_index

    root, cell, image, _, _ = official_corpus
    first = dict(rebuild_index(root, dino_encoder))
    match = dict(query(root, dino_encoder, image, "turning_on_radio"))
    assert match["matches"][0]["cell_tag"] == cell
    assert match["matches"][0]["distance"] == pytest.approx(0.0)
    assert match["matches"][0]["matched_frame"]["env_step"] == 32
    assert (
        match["matches"][0]["task_memory"]["recipe_preview"][0]["action"]
        == "pi0_nav_pick"
    )
    assert not query(root, dino_encoder, image, "picking_up_trash")["matches"]
    manifest = json.loads((root / "dino_v2" / "manifest.json").read_text())
    before = {
        p.name: p.read_bytes()
        for p in (root / "dino_v2").iterdir()
        if p.suffix in {".jsonl", ".npz"}
    }
    shutil.rmtree(root / "dino_v2")
    second = dict(rebuild_index(root, dino_encoder))
    assert first["build_key"] == second["build_key"] == manifest["build_key"]
    assert before == {
        p.name: p.read_bytes()
        for p in (root / "dino_v2").iterdir()
        if p.suffix in {".jsonl", ".npz"}
    }


@pytest.mark.parametrize("changed", ["audit", "recipe", "encoder"])
def test_dino_source_or_encoder_changes_rebuild_cache(
    official_corpus, dino_encoder, changed
):
    from robots.behavior.dino_v2.index import query

    root, cell, image, _, _ = official_corpus
    before = query(root, dino_encoder, image, "turning_on_radio")["build_key"]
    if changed == "encoder":
        dino_encoder.revision = "new-encoder"
    else:
        file = (
            root
            / "task-specific"
            / (f"{cell}.json" if changed == "audit" else f"{cell}_recipe.jsonl")
        )
        file.write_text(file.read_text() + "\n")
    after = query(root, dino_encoder, image, "turning_on_radio")["build_key"]
    assert before != after


@pytest.mark.parametrize("changed", ["records", "embeddings", "image"])
def test_dino_tampered_cache_or_evidence_is_rejected(
    official_corpus, dino_encoder, changed
):
    from robots.behavior.dino_v2.index import query

    root, cell, image, _, _ = official_corpus
    query(root, dino_encoder, image, "turning_on_radio")
    if changed == "image":
        frame_dir = root / "task-specific" / "artifacts" / cell / "frames" / "head"
        file = next(frame_dir.glob("*.png"))
    else:
        file = next((root / "dino_v2").glob(f"{changed}.*"))
    file.write_bytes(file.read_bytes() + b"tampered")
    with pytest.raises(ValueError):
        query(root, dino_encoder, image, "turning_on_radio")


@pytest.mark.parametrize("missing", ["audit", "recipe", "receipt"])
def test_dino_requires_complete_verified_official_pair(
    official_corpus, dino_encoder, missing
):
    from robots.behavior.dino_v2.index import query

    root, cell, image, _, _ = official_corpus
    audit = root / "task-specific" / f"{cell}.json"
    if missing == "receipt":
        payload = json.loads(audit.read_text())
        payload["official_success_receipt"]["raw_done"]["success"] = False
        audit.write_text(json.dumps(payload))
    elif missing == "audit":
        audit.unlink()
    else:
        (root / "task-specific" / f"{cell}_recipe.jsonl").unlink()
    assert query(root, dino_encoder, image, "turning_on_radio")["matches"] == []


def test_dino_published_cell_evidence_is_not_overwritten(official_corpus, tmp_path):
    from robots.behavior.dino_v2.index import prepare_evidence_pack
    from robots.behavior.terminal_success import make_raw_success_receipt
    from rpent.session import EnvState

    root, cell, _, state, terminal = official_corpus
    path = root / "task-specific" / "artifacts" / cell / "manifest.json"
    before = path.read_bytes()
    run = tmp_path / "second_run"
    second_state = EnvState(run / "sessions" / "session_002")
    second_receipt = make_raw_success_receipt({"done": {"success": True}}, env_step=64)
    second_terminal = {
        "_finish": True,
        "task_success": True,
        "official_success_receipt": second_receipt,
    }
    with second_state.record_step(state={"total_env_steps": 64}):
        second_state.save("head_rgb.png", np.ones((16, 16, 3), dtype=np.uint8))
    run_state = EnvState(run)
    run_state.save(
        f"{cell}.json",
        {
            "task_name": "turning_on_radio",
            "public_seed": 0,
            "official_success_receipt": second_receipt,
        },
        step=None,
    )
    run_state.save(f"{cell}_recipe.jsonl", [{"action": "retry"}], step=None)
    prepared = prepare_evidence_pack(run, cell, second_state, second_terminal)
    assert prepared is not None
    manager = BehaviorMemoryManager(root)
    manager.task_artifacts = prepared
    result = manager.merge_memory(cell_tag=cell, run_state_dir=run, solved=True)
    assert result["task"] == 0
    assert path.read_bytes() == before


def test_dino_orphan_evidence_is_replaced_by_matching_merge_artifact(
    tmp_path, dino_encoder
):
    import shutil

    from robots.behavior.dino_v2.index import prepare_evidence_pack, query
    from robots.behavior.terminal_success import make_raw_success_receipt
    from rpent.session import EnvState

    def prepare_run(name: str, env_step: int, value: int) -> tuple[Path, np.ndarray]:
        run = tmp_path / name
        state = EnvState(run / "sessions" / name)
        receipt = make_raw_success_receipt(
            {"done": {"success": True}}, env_step=env_step
        )
        terminal = {
            "_finish": True,
            "task_success": True,
            "official_success_receipt": receipt,
        }
        image = np.full((16, 16, 3), value, dtype=np.uint8)
        with state.record_step(state={"total_env_steps": env_step}):
            state.save("head_rgb.png", image)
        run_state = EnvState(run)
        run_state.save(
            "turning_on_radio_s0.json",
            {
                "task_name": "turning_on_radio",
                "public_seed": 0,
                "official_success_receipt": receipt,
            },
            step=None,
        )
        run_state.save(
            "turning_on_radio_s0_recipe.jsonl", [{"action": name}], step=None
        )
        prepared = prepare_evidence_pack(run, "turning_on_radio_s0", state, terminal)
        assert prepared is not None
        return prepared, image

    root = tmp_path / "memory"
    orphan, _ = prepare_run("orphan", 32, 1)
    orphan_target = root / "task-specific" / "artifacts" / "turning_on_radio_s0"
    orphan_target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(orphan, orphan_target)

    prepared, image = prepare_run("winner", 64, 2)
    manager = BehaviorMemoryManager(root)
    manager.task_artifacts = prepared
    result = manager.merge_memory(
        cell_tag="turning_on_radio_s0",
        run_state_dir=tmp_path / "winner",
        solved=True,
    )
    assert result["task"] == 1
    match = query(root, dino_encoder, image, "turning_on_radio")
    assert match["matches"][0]["matched_frame"]["env_step"] == 64


def test_dino_prepare_rejects_solved_audit_terminal_receipt_mismatch(tmp_path):
    from robots.behavior.dino_v2.index import prepare_evidence_pack
    from robots.behavior.terminal_success import make_raw_success_receipt
    from rpent.session import EnvState

    cell = "turning_on_radio_s0"
    run = tmp_path / "run"
    state = EnvState(run / "sessions" / "session_001")
    terminal_receipt = make_raw_success_receipt(
        {"done": {"success": True}}, env_step=32
    )
    audit_receipt = make_raw_success_receipt({"done": {"success": True}}, env_step=33)
    terminal = {
        "_finish": True,
        "task_success": True,
        "official_success_receipt": terminal_receipt,
    }
    with state.record_step(state={"total_env_steps": 32}):
        state.save("head_rgb.png", np.zeros((16, 16, 3), dtype=np.uint8))
    run_state = EnvState(run)
    run_state.save(
        f"{cell}.json",
        {
            "task_name": "turning_on_radio",
            "public_seed": 0,
            "official_success_receipt": audit_receipt,
        },
        step=None,
    )
    with pytest.raises(ValueError):
        prepare_evidence_pack(run, cell, state, terminal)


def test_dino_prepare_rejects_existing_run_pack_for_different_receipt(tmp_path):
    from robots.behavior.dino_v2.index import prepare_evidence_pack
    from robots.behavior.terminal_success import make_raw_success_receipt
    from rpent.session import EnvState

    cell = "turning_on_radio_s0"
    run = tmp_path / "run"
    state = EnvState(run / "sessions" / "session_001")
    image = np.zeros((16, 16, 3), dtype=np.uint8)
    with state.record_step(state={"total_env_steps": 32}):
        state.save("head_rgb.png", image)
    run_state = EnvState(run)
    receipt1 = make_raw_success_receipt({"done": {"success": True}}, env_step=32)
    terminal1 = {
        "_finish": True,
        "task_success": True,
        "official_success_receipt": receipt1,
    }
    run_state.save(
        f"{cell}.json",
        {
            "task_name": "turning_on_radio",
            "public_seed": 0,
            "official_success_receipt": receipt1,
        },
        step=None,
    )
    assert prepare_evidence_pack(run, cell, state, terminal1) is not None

    receipt2 = make_raw_success_receipt({"done": {"success": True}}, env_step=33)
    terminal2 = {
        "_finish": True,
        "task_success": True,
        "official_success_receipt": receipt2,
    }
    run_state.save(
        f"{cell}.json",
        {
            "task_name": "turning_on_radio",
            "public_seed": 0,
            "official_success_receipt": receipt2,
        },
        step=None,
    )
    with pytest.raises(ValueError):
        prepare_evidence_pack(run, cell, state, terminal2)


def test_dino_parallel_builders_publish_one_consistent_cache(
    official_corpus, dino_encoder
):
    from concurrent.futures import ThreadPoolExecutor

    from robots.behavior.dino_v2.index import query, rebuild_index

    root, _, image, _, _ = official_corpus
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(
            executor.map(lambda _: dict(rebuild_index(root, dino_encoder)), range(2))
        )
    assert results[0]["build_key"] == results[1]["build_key"]
    assert query(root, dino_encoder, image, "turning_on_radio")["matches"]


@pytest.mark.parametrize("bad", ["norm", "nan", "dtype", "shape"])
def test_dino_rejects_invalid_vectors_even_with_matching_file_digest(
    official_corpus, dino_encoder, bad
):
    import hashlib

    from robots.behavior.dino_v2.index import query, rebuild_index

    root, _, image, _, _ = official_corpus
    rebuild_index(root, dino_encoder)
    manifest_path = root / "dino_v2" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    path = manifest_path.parent / manifest["embeddings"]["path"]
    matrix = np.zeros((1, DINOV2_DIMENSION), dtype=np.float32)
    matrix[0, 0] = 2 if bad == "norm" else 1
    if bad == "nan":
        matrix[0, 1] = np.nan
    elif bad == "dtype":
        matrix = matrix.astype(np.float64)
    elif bad == "shape":
        matrix = matrix[:, :-1]
    np.savez(path, head=matrix)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    renamed = path.with_name(f"embeddings.{digest}.npz")
    path.rename(renamed)
    manifest["embeddings"] = {"sha256": digest, "path": renamed.name}
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        query(root, dino_encoder, image, "turning_on_radio")


def test_dino_rejects_false_terminal_evidence_with_matching_file_digest(
    official_corpus, dino_encoder
):
    import hashlib

    from robots.behavior.dino_v2.index import rebuild_index

    root, cell, _, _, _ = official_corpus
    directory = root / "task-specific" / "artifacts" / cell
    terminal_path = directory / "terminal_receipt.json"
    terminal = json.loads(terminal_path.read_text())
    terminal["task_success"] = False
    terminal_path.write_text(json.dumps(terminal))
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["terminal_receipt"]["sha256"] = hashlib.sha256(
        terminal_path.read_bytes()
    ).hexdigest()
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        rebuild_index(root, dino_encoder)


def test_dino_reuses_embedding_for_identical_evidence_images(
    official_corpus, dino_encoder
):
    from robots.behavior.dino_v2.index import rebuild_index

    root, cell, _, _, _ = official_corpus
    path = root / "task-specific" / "artifacts" / cell / "manifest.json"
    manifest = json.loads(path.read_text())
    frame = dict(
        next(frame for frame in manifest["frames"] if frame["camera"] == "head")
    )
    frame.update(step_idx=1, env_step=33)
    manifest["frames"].append(frame)
    path.write_text(json.dumps(manifest))
    result = rebuild_index(root, dino_encoder)
    assert result["records"] == 2
    assert dino_encoder.image_count == 1


def test_dino_failed_manifest_publication_keeps_previous_cache(
    official_corpus, dino_encoder, monkeypatch
):
    from robots.behavior.dino_v2 import index

    root, _, image, _, _ = official_corpus
    index.rebuild_index(root, dino_encoder)
    path = root / "dino_v2" / "manifest.json"
    before = path.read_bytes()
    replace = index.os.replace

    def fail_manifest(source, destination):
        if Path(destination) == path:
            raise OSError("simulated interrupted manifest publication")
        return replace(source, destination)

    with monkeypatch.context() as patch:
        patch.setattr(index.os, "replace", fail_manifest)
        with pytest.raises(OSError, match="interrupted"):
            index.rebuild_index(root, dino_encoder)
    assert path.read_bytes() == before
    assert index.query(root, dino_encoder, image, "turning_on_radio")["matches"]


@pytest.mark.parametrize(
    "field,value", [("task_name", "picking_up_trash"), ("public_seed", 1)]
)
def test_dino_rejects_task_identity_mismatching_cell(
    official_corpus, dino_encoder, field, value
):
    from robots.behavior.dino_v2.index import query

    root, cell, image, _, _ = official_corpus
    audit_path = root / "task-specific" / f"{cell}.json"
    audit = json.loads(audit_path.read_text())
    audit[field] = value
    audit_path.write_text(json.dumps(audit))
    with pytest.raises(ValueError):
        query(root, dino_encoder, image, "picking_up_trash")
