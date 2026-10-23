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

import numpy as np
import pytest

from robots.behavior.dino_v2.encoder import DINOV2_DIMENSION
from robots.behavior.memory import BehaviorMemoryManager
from robots.behavior.tools import BehaviorPrimitives


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
