from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
BEHAVIOR_ROOT = REPO_ROOT / "robots" / "behavior"
pytestmark = pytest.mark.skipif(
    not BEHAVIOR_ROOT.is_dir(),
    reason="BEHAVIOR robot plugin has not landed in this worktree yet",
)


def _memory_module():
    return importlib.import_module("robots.behavior.episode_memory_index")


def _unit(row: int) -> np.ndarray:
    vec = np.zeros(384, dtype=np.float32)
    vec[row] = 1.0
    return vec


def _experience(module, *, episode: str, task: str, row: int):
    frame = module.EpisodeFrameKey(
        frame_id=f"{episode}:head:{row}",
        episode_id=episode,
        experience_id=f"exp:{episode}",
        task_name=task,
        frame_index=row,
        embedding_row=row,
        keyframe_kind="head",
        source_record_id=f"record:{row}",
        frame_identity={"camera": "head"},
    )
    return module.EpisodeExperience(
        episode_id=episode,
        experience_id=f"exp:{episode}",
        logical_experience_id=f"logical:{episode}",
        task_name=task,
        usage={"phase": "explore"},
        outcome={"task_success": True},
        frame_keys=(frame,),
        canonical_trajectory_ref={"path": f"{episode}.jsonl"},
        trajectory_refs=(),
        reproduction_evidence=(),
        source={"kind": "unit"},
        metadata={},
    )


def _index():
    module = _memory_module()
    return module.EpisodeMemoryIndex(
        experiences=(
            _experience(module, episode="episode:radio", task="turning_on_radio", row=0),
            _experience(module, episode="episode:trash", task="picking_up_trash", row=1),
        ),
        head_embeddings=np.stack([_unit(0), _unit(1)]),
        wrist_shadow_embeddings={
            "left_wrist": np.stack([_unit(2), _unit(3)]),
            "right_wrist": np.stack([_unit(4), _unit(5)]),
        },
        revision={"schema_id": module.REVISION_SCHEMA_ID},
    )


def test_episode_query_filters_by_task_before_similarity_and_returns_whole_hit():
    module = _memory_module()
    index = _index()

    radio_hits = index.search(task_name="turning_on_radio", head_embedding=_unit(1))
    trash_hits = index.search(task_name="picking_up_trash", head_embedding=_unit(1))

    assert [hit.experience.episode_id for hit in radio_hits] == ["episode:radio"]
    assert radio_hits[0].distance > module.HEAD_ACTIVE_DISTANCE_MAX
    assert [hit.experience.episode_id for hit in trash_hits] == ["episode:trash"]
    assert trash_hits[0].distance <= module.HEAD_ACTIVE_DISTANCE_MAX
    assert trash_hits[0].to_dict()["returned_scope"] == "whole_experience"
    assert trash_hits[0].to_dict()["stage_inference"] is None


def test_head_threshold_decides_use_while_wrist_is_shadow_only() -> None:
    index = _index()

    result = index.retrieve(
        task_name="picking_up_trash",
        head_embedding=_unit(1),
        wrist_shadow_embeddings={
            "left_wrist": _unit(3),
            "right_wrist": _unit(5),
        },
    )

    assert result["decision"] == "use_experience"
    assert result["task_filter_applied_before_vision"] is True
    assert result["active_channel"] == "head"
    assert result["wrist_shadow_only"] is True
    assert result["stage_inference"] is None
    assert result["hit"]["shadow_distances"] == {
        "left_wrist": 0.0,
        "right_wrist": 0.0,
    }


def test_cross_task_head_match_records_new_without_stage_inference() -> None:
    index = _index()

    result = index.retrieve(task_name="turning_on_radio", head_embedding=_unit(1))
    unknown = index.retrieve(task_name="unsupported_task", head_embedding=_unit(1))

    assert result["decision"] == "record_new"
    assert result["hit"] is None
    assert result["stage_inference"] is None
    assert result["candidate_count_after_task_filter"] == 1
    assert unknown["decision"] == "record_new"
    assert unknown["candidate_count_after_task_filter"] == 0
    assert unknown["stage_inference"] is None


def test_bidirectional_95pct_merge_appends_evidence_without_overwriting() -> None:
    module = _memory_module()
    existing = _experience(
        module,
        episode="episode:existing",
        task="picking_up_trash",
        row=0,
    )
    candidate = _experience(
        module,
        episode="episode:candidate",
        task="picking_up_trash",
        row=0,
    )

    decision = module.merge_same_task_experience(
        existing=existing,
        candidate=candidate,
        existing_head_embeddings=np.stack([_unit(0), _unit(1)]),
        candidate_head_embeddings=np.stack([_unit(0), _unit(1)]),
        evidence={"attempt": 2},
    )

    assert decision["decision"] == "append_reproduction_evidence"
    assert decision["reason"] == "same_task_bidirectional_95pct_keyframe_coverage"
    assert decision["coverage_required"] == 0.95
    assert decision["forward_coverage"] == 1.0
    assert decision["backward_coverage"] == 1.0
    assert decision["canonical_trajectory_overwritten"] is False
    assert decision["reproduction_evidence_to_append"] == {"attempt": 2}


def test_memory_catalog_is_empty_only_when_implicit_and_explicit_missing_fails(tmp_path):
    module = _memory_module()

    empty = module.load_current_catalog(None)
    assert empty.episode_count == 0
    assert empty.revision["empty_catalog_reason"] == "memory_dir_omitted"

    with pytest.raises(module.MemoryValidationError) as excinfo:
        module.load_current_catalog(tmp_path / "missing")
    assert excinfo.value.code == "MEMORY_EPISODE_CATALOG_MISSING"


def test_candidate_revision_is_content_addressed_and_atomically_readable(tmp_path):
    module = _memory_module()
    experience = _experience(
        module,
        episode="episode:trash",
        task="picking_up_trash",
        row=0,
    )

    result = module.write_candidate_revision(
        memory_dir=tmp_path,
        experiences=[experience],
        head_embeddings=np.stack([_unit(0)]),
        wrist_shadow_embeddings={"left_wrist": np.stack([_unit(1)])},
        encoder_identity={"model": "dinov2"},
    )

    revision_dir = Path(result["revision_dir"])
    assert result["revision_document_sha256"] == revision_dir.name
    assert (tmp_path / "current.json").is_file()
    loaded = module.load_current_catalog(tmp_path)
    assert loaded.episode_count == 1
    assert loaded.frame_count == 1
    assert loaded.retrieve(task_name="picking_up_trash", head_embedding=_unit(0))[
        "decision"
    ] == "use_experience"
