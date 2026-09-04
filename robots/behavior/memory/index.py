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

"""Production episode-level BEHAVIOR memory index.

Only head DINOv2 CLS384 keyframes are active.  Wrist embeddings may be carried
for audit and shadow distances, but they never decide use vs record.
"""

from __future__ import annotations

import io
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np

from robots.behavior.dino_v2.encoder import (
    DINOV2_DIMENSION,
    DISTANCE_METRIC,
    l2_matrix,
    l2_normalize_row,
)
from robots.behavior.memory.schema import (
    MemoryValidationError,
    canonical_json_file_bytes,
    fail,
    require_exact_keys,
    require_sha256,
    sha256_bytes,
)

SCHEMA_ID = "rpent_behavior_episode_memory_index_v1"
REVISION_SCHEMA_ID = "rpent_behavior_episode_memory_revision_v1"
CURRENT_POINTER_SCHEMA_ID = "rpent_behavior_episode_memory_current_v1"
MANIFEST_SCHEMA_ID = "rpent_behavior_episode_memory_manifest_v1"
HEAD_ACTIVE_DISTANCE_MAX = 0.05367707759141922
MERGE_COVERAGE = 0.95
ACTIVE_CHANNEL = "head"
SHADOW_CHANNELS = ("left_wrist", "right_wrist")


def _nonempty_string(value: Any, *, path: str) -> str:
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        fail("MEMORY_EPISODE_SCHEMA_INVALID", path, "expected non-empty string")
    return value.strip()


@dataclass(frozen=True, slots=True)
class EpisodeFrameKey:
    frame_id: str
    episode_id: str
    experience_id: str
    task_name: str
    frame_index: int
    embedding_row: int
    keyframe_kind: str
    source_record_id: str
    frame_identity: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for field_name in (
            "frame_id",
            "episode_id",
            "experience_id",
            "task_name",
            "keyframe_kind",
            "source_record_id",
        ):
            _nonempty_string(getattr(self, field_name), path=f"frame.{field_name}")
        if isinstance(self.frame_index, bool) or self.frame_index < 0:
            fail(
                "MEMORY_EPISODE_SCHEMA_INVALID",
                "frame.frame_index",
                "expected non-negative int",
            )
        if isinstance(self.embedding_row, bool) or self.embedding_row < 0:
            fail(
                "MEMORY_EPISODE_SCHEMA_INVALID",
                "frame.embedding_row",
                "expected non-negative int",
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_id": self.frame_id,
            "episode_id": self.episode_id,
            "experience_id": self.experience_id,
            "task_name": self.task_name,
            "frame_index": self.frame_index,
            "embedding_row": self.embedding_row,
            "keyframe_kind": self.keyframe_kind,
            "source_record_id": self.source_record_id,
            "frame_identity": dict(self.frame_identity),
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "EpisodeFrameKey":
        require_exact_keys(
            value,
            {
                "frame_id",
                "episode_id",
                "experience_id",
                "task_name",
                "frame_index",
                "embedding_row",
                "keyframe_kind",
                "source_record_id",
                "frame_identity",
            },
            path="frame",
        )
        return cls(
            frame_id=str(value["frame_id"]),
            episode_id=str(value["episode_id"]),
            experience_id=str(value["experience_id"]),
            task_name=str(value["task_name"]),
            frame_index=int(value["frame_index"]),
            embedding_row=int(value["embedding_row"]),
            keyframe_kind=str(value["keyframe_kind"]),
            source_record_id=str(value["source_record_id"]),
            frame_identity=dict(value["frame_identity"]),
        )


@dataclass(frozen=True, slots=True)
class EpisodeExperience:
    episode_id: str
    experience_id: str
    logical_experience_id: str
    task_name: str
    usage: Mapping[str, Any]
    outcome: Mapping[str, Any]
    frame_keys: tuple[EpisodeFrameKey, ...]
    canonical_trajectory_ref: Mapping[str, Any] | None = None
    trajectory_refs: tuple[Mapping[str, Any], ...] = ()
    reproduction_evidence: tuple[Mapping[str, Any], ...] = ()
    source: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for field_name in (
            "episode_id",
            "experience_id",
            "logical_experience_id",
            "task_name",
        ):
            _nonempty_string(getattr(self, field_name), path=f"experience.{field_name}")
        if not self.frame_keys:
            fail(
                "MEMORY_EPISODE_SCHEMA_INVALID",
                "experience.frame_keys",
                "at least one head keyframe required",
            )
        for frame in self.frame_keys:
            if (
                frame.episode_id != self.episode_id
                or frame.experience_id != self.experience_id
                or frame.task_name != self.task_name
            ):
                fail(
                    "MEMORY_EPISODE_SCHEMA_INVALID",
                    "experience.frame_keys",
                    "frame identity does not match experience",
                )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_id": SCHEMA_ID,
            "episode_id": self.episode_id,
            "experience_id": self.experience_id,
            "logical_experience_id": self.logical_experience_id,
            "task_name": self.task_name,
            "usage": dict(self.usage),
            "outcome": dict(self.outcome),
            "canonical_trajectory_ref": None
            if self.canonical_trajectory_ref is None
            else dict(self.canonical_trajectory_ref),
            "trajectory_refs": [dict(item) for item in self.trajectory_refs],
            "reproduction_evidence": [
                dict(item) for item in self.reproduction_evidence
            ],
            "source": dict(self.source),
            "metadata": dict(self.metadata),
            "frame_keys": [frame.to_dict() for frame in self.frame_keys],
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "EpisodeExperience":
        require_exact_keys(
            value,
            {
                "schema_id",
                "episode_id",
                "experience_id",
                "logical_experience_id",
                "task_name",
                "usage",
                "outcome",
                "canonical_trajectory_ref",
                "trajectory_refs",
                "reproduction_evidence",
                "source",
                "metadata",
                "frame_keys",
            },
            path="experience",
        )
        if value["schema_id"] != SCHEMA_ID:
            fail(
                "MEMORY_EPISODE_SCHEMA_INVALID",
                "experience.schema_id",
                "schema mismatch",
            )
        frame_values = value["frame_keys"]
        if not isinstance(frame_values, list):
            fail(
                "MEMORY_EPISODE_SCHEMA_INVALID",
                "experience.frame_keys",
                "expected list",
            )
        return cls(
            episode_id=str(value["episode_id"]),
            experience_id=str(value["experience_id"]),
            logical_experience_id=str(value["logical_experience_id"]),
            task_name=str(value["task_name"]),
            usage=dict(value["usage"]),
            outcome=dict(value["outcome"]),
            canonical_trajectory_ref=None
            if value["canonical_trajectory_ref"] is None
            else dict(value["canonical_trajectory_ref"]),
            trajectory_refs=tuple(dict(item) for item in value["trajectory_refs"]),
            reproduction_evidence=tuple(
                dict(item) for item in value["reproduction_evidence"]
            ),
            source=dict(value["source"]),
            metadata=dict(value["metadata"]),
            frame_keys=tuple(
                EpisodeFrameKey.from_mapping(item) for item in frame_values
            ),
        )


@dataclass(frozen=True, slots=True)
class EpisodeMemoryHit:
    rank: int
    distance: float
    matched_frame: EpisodeFrameKey
    experience: EpisodeExperience
    shadow_distances: Mapping[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_id": "rpent_behavior_episode_memory_hit_v1",
            "rank": self.rank,
            "distance": self.distance,
            "distance_metric": DISTANCE_METRIC,
            "threshold": HEAD_ACTIVE_DISTANCE_MAX,
            "episode_id": self.experience.episode_id,
            "experience_id": self.experience.experience_id,
            "logical_experience_id": self.experience.logical_experience_id,
            "task_name": self.experience.task_name,
            "usage": dict(self.experience.usage),
            "outcome": dict(self.experience.outcome),
            "matched_frame": self.matched_frame.to_dict(),
            "experience": self.experience.to_dict(),
            "returned_scope": "whole_experience",
            "stage_inference": None,
            "wrist_shadow_only": True,
            "shadow_distances": dict(self.shadow_distances),
        }


class EpisodeMemoryIndex:
    def __init__(
        self,
        *,
        experiences: Sequence[EpisodeExperience],
        head_embeddings: np.ndarray,
        wrist_shadow_embeddings: Mapping[str, np.ndarray] | None = None,
        revision: Mapping[str, Any] | None = None,
    ) -> None:
        self._experiences = tuple(experiences)
        self._frames = tuple(
            frame for exp in self._experiences for frame in exp.frame_keys
        )
        self._head = l2_matrix(head_embeddings, path="head_embeddings")
        if self._head.shape[0] != len(self._frames):
            fail(
                "MEMORY_EPISODE_INDEX_INVALID",
                "head_embeddings",
                "row count must equal head keyframes",
            )
        self._experience_by_id = {
            item.experience_id: item for item in self._experiences
        }
        self._experience_by_episode = {
            item.episode_id: item for item in self._experiences
        }
        if len(self._experience_by_id) != len(self._experiences) or len(
            self._experience_by_episode
        ) != len(self._experiences):
            fail(
                "MEMORY_EPISODE_INDEX_INVALID",
                "experiences",
                "experience and episode IDs must be unique",
            )
        by_task: dict[str, list[int]] = {}
        for index, frame in enumerate(self._frames):
            if frame.embedding_row != index:
                fail(
                    "MEMORY_EPISODE_INDEX_INVALID",
                    "frames",
                    "embedding rows must be contiguous",
                )
            by_task.setdefault(frame.task_name, []).append(index)
        self._by_task = {task: tuple(indices) for task, indices in by_task.items()}
        shadow: dict[str, np.ndarray] = {}
        for channel, values in (wrist_shadow_embeddings or {}).items():
            name = str(channel)
            if name not in SHADOW_CHANNELS:
                fail(
                    "MEMORY_EPISODE_INDEX_INVALID",
                    f"shadow.{name}",
                    "only wrist shadow channels are accepted",
                )
            matrix = l2_matrix(values, path=f"shadow.{name}")
            if matrix.shape[0] != len(self._frames):
                fail(
                    "MEMORY_EPISODE_INDEX_INVALID",
                    f"shadow.{name}",
                    "row count mismatch",
                )
            shadow[name] = matrix
        self._shadow = MappingProxyType(shadow)
        self._revision = MappingProxyType(dict(revision or {}))

    @property
    def episode_count(self) -> int:
        return len(self._experiences)

    @property
    def frame_count(self) -> int:
        return len(self._frames)

    @property
    def experiences(self) -> tuple[EpisodeExperience, ...]:
        return self._experiences

    @property
    def revision(self) -> Mapping[str, Any]:
        return self._revision

    def search(
        self,
        *,
        task_name: str,
        head_embedding: np.ndarray,
        k: int = 1,
        wrist_shadow_embeddings: Mapping[str, np.ndarray] | None = None,
    ) -> tuple[EpisodeMemoryHit, ...]:
        task = _nonempty_string(task_name, path="query.task_name")
        if isinstance(k, bool) or k < 1:
            fail("MEMORY_EPISODE_QUERY_INVALID", "query.k", "expected positive int")
        candidates = self._by_task.get(task, ())
        if not candidates:
            return ()
        query = l2_normalize_row(head_embedding, path="query.head_embedding")[None, :]
        distances = np.asarray(
            1.0 - np.clip(query @ self._head[list(candidates)].T, -1.0, 1.0),
            dtype=np.float64,
        )[0]
        best_by_experience: dict[
            str, tuple[float, EpisodeFrameKey, dict[str, float]]
        ] = {}
        for offset, frame_index in enumerate(candidates):
            frame = self._frames[frame_index]
            shadow_distances = self._shadow_distances(
                frame_index, wrist_shadow_embeddings
            )
            candidate = (float(distances[offset]), frame, shadow_distances)
            current = best_by_experience.get(frame.experience_id)
            if current is None or (candidate[0], frame.frame_id) < (
                current[0],
                current[1].frame_id,
            ):
                best_by_experience[frame.experience_id] = candidate
        hits = [
            EpisodeMemoryHit(
                rank=0,
                distance=distance,
                matched_frame=frame,
                experience=self._experience_by_id[frame.experience_id],
                shadow_distances=MappingProxyType(shadow),
            )
            for distance, frame, shadow in best_by_experience.values()
        ]
        ordered = sorted(
            hits,
            key=lambda hit: (
                hit.distance,
                hit.experience.experience_id,
                hit.matched_frame.frame_id,
            ),
        )
        return tuple(
            EpisodeMemoryHit(
                rank=index,
                distance=hit.distance,
                matched_frame=hit.matched_frame,
                experience=hit.experience,
                shadow_distances=hit.shadow_distances,
            )
            for index, hit in enumerate(ordered[:k], start=1)
        )

    def retrieve(
        self,
        *,
        task_name: str,
        head_embedding: np.ndarray,
        wrist_shadow_embeddings: Mapping[str, np.ndarray] | None = None,
    ) -> Mapping[str, Any]:
        hits = self.search(
            task_name=task_name,
            head_embedding=head_embedding,
            k=max(1, self.episode_count),
            wrist_shadow_embeddings=wrist_shadow_embeddings,
        )
        selected = next(
            (hit for hit in hits if hit.distance <= HEAD_ACTIVE_DISTANCE_MAX), None
        )
        return MappingProxyType(
            {
                "schema_id": "rpent_behavior_episode_memory_retrieval_v1",
                "decision": "use_experience" if selected is not None else "record_new",
                "reason": "head_keyframe_under_active_threshold"
                if selected is not None
                else "no_same_task_head_keyframe_under_active_threshold",
                "task_filter_applied_before_vision": True,
                "active_channel": ACTIVE_CHANNEL,
                "head_active_distance_max": HEAD_ACTIVE_DISTANCE_MAX,
                "wrist_shadow_only": True,
                "hit": None if selected is None else selected.to_dict(),
                "stage_inference": None,
                "candidate_count_after_task_filter": len(
                    self._by_task.get(str(task_name).strip(), ())
                ),
            }
        )

    def _shadow_distances(
        self,
        frame_index: int,
        queries: Mapping[str, np.ndarray] | None,
    ) -> dict[str, float]:
        result: dict[str, float] = {}
        for channel, query in (queries or {}).items():
            name = str(channel)
            if name not in self._shadow or query is None:
                continue
            row = l2_normalize_row(query, path=f"query.{name}")[None, :]
            result[name] = float(
                1.0
                - np.clip(
                    row @ self._shadow[name][frame_index : frame_index + 1].T, -1.0, 1.0
                )[0, 0]
            )
        return result


def empty_episode_memory_index() -> EpisodeMemoryIndex:
    return EpisodeMemoryIndex(
        experiences=(),
        head_embeddings=np.zeros((0, DINOV2_DIMENSION), dtype=np.float32),
        revision={
            "schema_id": REVISION_SCHEMA_ID,
            "empty_catalog_reason": "memory_dir_omitted",
            "activation_allowed": False,
        },
    )


def load_current_catalog(memory_dir: Path | None) -> EpisodeMemoryIndex:
    """Load the current catalog; omitted memory_dir is the only legal empty catalog."""

    if memory_dir is None:
        return empty_episode_memory_index()
    root = Path(memory_dir)
    if not root.is_dir():
        fail(
            "MEMORY_EPISODE_CATALOG_MISSING",
            str(root),
            "explicit memory-dir is missing",
        )
    pointer_path = root / "current.json"
    pointer = _read_json(pointer_path)
    require_exact_keys(
        pointer, {"schema_id", "revision_document_sha256"}, path="current.json"
    )
    if pointer["schema_id"] != CURRENT_POINTER_SCHEMA_ID:
        fail("MEMORY_EPISODE_POINTER_INVALID", "current.json", "schema mismatch")
    revision_sha = require_sha256(
        pointer["revision_document_sha256"], path="current.revision_document_sha256"
    )
    revision_dir = root / "revisions" / revision_sha
    return load_revision_dir(revision_dir, expected_revision_sha256=revision_sha)


def load_revision_dir(
    revision_dir: Path, *, expected_revision_sha256: str | None = None
) -> EpisodeMemoryIndex:
    if not revision_dir.is_dir():
        fail(
            "MEMORY_EPISODE_REVISION_MISSING",
            str(revision_dir),
            "revision directory missing",
        )
    manifest = _read_json(revision_dir / "manifest.json")
    require_exact_keys(
        manifest,
        {
            "schema_id",
            "revision_document_sha256",
            "catalog_sha256",
            "embeddings_npz_sha256",
            "experience_count",
            "frame_count",
        },
        path="manifest.json",
    )
    if manifest["schema_id"] != MANIFEST_SCHEMA_ID:
        fail("MEMORY_EPISODE_MANIFEST_INVALID", "manifest.schema_id", "schema mismatch")
    revision_sha = require_sha256(
        manifest["revision_document_sha256"], path="manifest.revision_document_sha256"
    )
    if (
        expected_revision_sha256 is not None
        and revision_sha != expected_revision_sha256
    ):
        fail(
            "MEMORY_EPISODE_HASH_MISMATCH",
            "manifest.revision_document_sha256",
            "current pointer mismatch",
        )
    revision_bytes = _read_regular(revision_dir / "revision.json")
    if sha256_bytes(revision_bytes) != revision_sha:
        fail(
            "MEMORY_EPISODE_HASH_MISMATCH", "revision.json", "document digest mismatch"
        )
    catalog_bytes = _read_regular(revision_dir / "catalog.jsonl")
    if sha256_bytes(catalog_bytes) != require_sha256(
        manifest["catalog_sha256"], path="manifest.catalog_sha256"
    ):
        fail("MEMORY_EPISODE_HASH_MISMATCH", "catalog.jsonl", "catalog digest mismatch")
    embeddings_bytes = _read_regular(revision_dir / "embeddings.npz")
    if sha256_bytes(embeddings_bytes) != require_sha256(
        manifest["embeddings_npz_sha256"], path="manifest.embeddings_npz_sha256"
    ):
        fail(
            "MEMORY_EPISODE_HASH_MISMATCH",
            "embeddings.npz",
            "embedding digest mismatch",
        )
    revision = json.loads(revision_bytes.decode("utf-8"))
    experiences = tuple(
        EpisodeExperience.from_mapping(json.loads(line.decode("utf-8")))
        for line in catalog_bytes.splitlines()
        if line
    )
    with np.load(io.BytesIO(embeddings_bytes), allow_pickle=False) as data:
        head = np.asarray(data["head"], dtype=np.float32)
        shadow = {
            name: np.asarray(data[name], dtype=np.float32)
            for name in SHADOW_CHANNELS
            if name in data.files
        }
    index = EpisodeMemoryIndex(
        experiences=experiences,
        head_embeddings=head,
        wrist_shadow_embeddings=shadow,
        revision=revision,
    )
    if index.episode_count != int(
        manifest["experience_count"]
    ) or index.frame_count != int(manifest["frame_count"]):
        fail("MEMORY_EPISODE_MANIFEST_INVALID", "manifest.counts", "count mismatch")
    return index


def write_candidate_revision(
    *,
    memory_dir: Path,
    experiences: Sequence[EpisodeExperience],
    head_embeddings: np.ndarray,
    wrist_shadow_embeddings: Mapping[str, np.ndarray] | None = None,
    encoder_identity: Mapping[str, Any] | None = None,
    parent_revision_document_sha256: str | None = None,
    activate_current: bool = True,
) -> Mapping[str, Any]:
    """Validate, write content-addressed revision, then atomically advance current."""

    root = Path(memory_dir)
    root.mkdir(parents=True, exist_ok=True)
    candidate_index = EpisodeMemoryIndex(
        experiences=experiences,
        head_embeddings=head_embeddings,
        wrist_shadow_embeddings=wrist_shadow_embeddings,
    )
    catalog_bytes = b"".join(
        canonical_json_file_bytes(exp.to_dict(), path=f"experience[{index}]")
        for index, exp in enumerate(candidate_index.experiences)
    )
    embedding_payload = _npz_bytes(
        {"head": candidate_index._head, **dict(candidate_index._shadow)}
    )
    catalog_sha = sha256_bytes(catalog_bytes)
    embeddings_sha = sha256_bytes(embedding_payload)
    revision = {
        "schema_id": REVISION_SCHEMA_ID,
        "format_version": 1,
        "preliminary": True,
        "activation_allowed": False,
        "active_thresholds": {"head_distance_max": HEAD_ACTIVE_DISTANCE_MAX},
        "distance_metric": DISTANCE_METRIC,
        "active_channel": ACTIVE_CHANNEL,
        "wrist_policy": "shadow_only",
        "encoder_identity": dict(encoder_identity or {}),
        "parent_revision_document_sha256": parent_revision_document_sha256,
        "catalog_sha256": catalog_sha,
        "embeddings_npz_sha256": embeddings_sha,
        "experience_count": candidate_index.episode_count,
        "frame_count": candidate_index.frame_count,
    }
    revision_bytes = canonical_json_file_bytes(revision, path="revision")
    revision_sha = sha256_bytes(revision_bytes)
    revision_dir = root / "revisions" / revision_sha
    _write_revision_dir(
        revision_dir,
        revision_bytes=revision_bytes,
        catalog_bytes=catalog_bytes,
        embedding_bytes=embedding_payload,
        manifest={
            "schema_id": MANIFEST_SCHEMA_ID,
            "revision_document_sha256": revision_sha,
            "catalog_sha256": catalog_sha,
            "embeddings_npz_sha256": embeddings_sha,
            "experience_count": candidate_index.episode_count,
            "frame_count": candidate_index.frame_count,
        },
    )
    load_revision_dir(revision_dir, expected_revision_sha256=revision_sha)
    pointer = {
        "schema_id": CURRENT_POINTER_SCHEMA_ID,
        "revision_document_sha256": revision_sha,
    }
    if activate_current:
        _atomic_write(
            root / "current.json", canonical_json_file_bytes(pointer, path="current")
        )
    return MappingProxyType(
        {
            "revision_document_sha256": revision_sha,
            "revision_dir": str(revision_dir),
            "current": bool(activate_current),
        }
    )


def merge_same_task_experience(
    *,
    existing: EpisodeExperience,
    candidate: EpisodeExperience,
    existing_head_embeddings: np.ndarray,
    candidate_head_embeddings: np.ndarray,
    evidence: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Return a same-layout merge proposal without overwriting the canonical trajectory."""

    if existing.task_name != candidate.task_name:
        fail("MEMORY_EPISODE_MERGE_REJECTED", "task_name", "same-task merge required")
    forward = keyframe_coverage(candidate_head_embeddings, existing_head_embeddings)
    backward = keyframe_coverage(existing_head_embeddings, candidate_head_embeddings)
    accepted = forward >= MERGE_COVERAGE and backward >= MERGE_COVERAGE
    return MappingProxyType(
        {
            "schema_id": "rpent_behavior_episode_memory_merge_v1",
            "decision": "append_reproduction_evidence"
            if accepted
            else "record_new_experience",
            "reason": "same_task_bidirectional_95pct_keyframe_coverage"
            if accepted
            else "coverage_below_threshold",
            "head_distance_max": HEAD_ACTIVE_DISTANCE_MAX,
            "coverage_required": MERGE_COVERAGE,
            "forward_coverage": forward,
            "backward_coverage": backward,
            "same_layout_success_failure_can_share_logical_experience": accepted,
            "logical_experience_id": existing.logical_experience_id
            if accepted
            else candidate.logical_experience_id,
            "canonical_trajectory_ref": None
            if existing.canonical_trajectory_ref is None
            else dict(existing.canonical_trajectory_ref),
            "canonical_trajectory_overwritten": False,
            "reproduction_evidence_to_append": dict(evidence) if accepted else None,
            "existing_outcome": dict(existing.outcome),
            "candidate_outcome": dict(candidate.outcome),
        }
    )


def keyframe_coverage(
    query_embeddings: np.ndarray, catalog_embeddings: np.ndarray
) -> float:
    query = l2_matrix(query_embeddings, path="merge.query")
    catalog = l2_matrix(catalog_embeddings, path="merge.catalog")
    if query.shape[0] == 0 or catalog.shape[0] == 0:
        return 0.0
    distances = 1.0 - np.clip(query @ catalog.T, -1.0, 1.0)
    return float(np.mean(np.min(distances, axis=1) <= HEAD_ACTIVE_DISTANCE_MAX))


def _npz_bytes(arrays: Mapping[str, np.ndarray]) -> bytes:
    with io.BytesIO() as buffer:
        np.savez(
            buffer,
            **{
                name: np.asarray(value, dtype=np.float32)
                for name, value in arrays.items()
            },
        )
        return buffer.getvalue()


def _read_regular(path: Path) -> bytes:
    if path.is_symlink() or not path.is_file():
        fail("MEMORY_EPISODE_SOURCE_INVALID", str(path), "expected regular file")
    return path.read_bytes()


def _read_json(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(_read_regular(path).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        fail("MEMORY_EPISODE_SOURCE_INVALID", str(path), str(exc))
    if not isinstance(value, Mapping):
        fail("MEMORY_EPISODE_SOURCE_INVALID", str(path), "expected JSON object")
    return value


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb", prefix=f".{path.name}.", dir=path.parent, delete=False
    ) as handle:
        tmp = Path(handle.name)
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def _write_new(path: Path, payload: bytes) -> None:
    if path.exists():
        if path.is_file() and not path.is_symlink() and path.read_bytes() == payload:
            return
        fail("MEMORY_EPISODE_OUTPUT_COLLISION", str(path), "existing bytes differ")
    _atomic_write(path, payload)


def _write_revision_dir(
    revision_dir: Path,
    *,
    revision_bytes: bytes,
    catalog_bytes: bytes,
    embedding_bytes: bytes,
    manifest: Mapping[str, Any],
) -> None:
    revision_dir.mkdir(parents=True, exist_ok=True)
    _write_new(revision_dir / "revision.json", revision_bytes)
    _write_new(revision_dir / "catalog.jsonl", catalog_bytes)
    _write_new(revision_dir / "embeddings.npz", embedding_bytes)
    _write_new(
        revision_dir / "manifest.json",
        canonical_json_file_bytes(dict(manifest), path="manifest"),
    )


__all__ = [
    "ACTIVE_CHANNEL",
    "HEAD_ACTIVE_DISTANCE_MAX",
    "EpisodeExperience",
    "EpisodeFrameKey",
    "EpisodeMemoryHit",
    "EpisodeMemoryIndex",
    "MemoryValidationError",
    "empty_episode_memory_index",
    "keyframe_coverage",
    "load_current_catalog",
    "load_revision_dir",
    "merge_same_task_experience",
    "write_candidate_revision",
]
