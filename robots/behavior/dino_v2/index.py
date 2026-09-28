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

"""Derived DINOv2 cache over the official BEHAVIOR MemoryManager corpus.

The authoritative corpus is ``<memory_root>/task-specific``:

* ``<cell>.json`` is the task audit.
* ``<cell>_recipe.jsonl`` is the replay recipe.
* ``artifacts/<cell>/manifest.json`` is the success-session evidence pack.

This module never promotes, edits, or copies authoritative task memory.  It
prepares immutable run-local evidence packs for the MemoryManager merge and
maintains a lazy, rebuildable DINO cache under ``<memory_root>/dino_v2``.
"""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import io
import json
import os
import shutil
import tempfile
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np

from robots.behavior.dino_v2.encoder import (
    DINOV2_DIMENSION,
    DISTANCE_METRIC,
    MAX_BATCH_SIZE,
    _sha256_file,
    l2_normalize_row,
    require_sha256,
)
from robots.behavior.task_specs import get_task_spec
from robots.behavior.terminal_success import (
    _canonical_json_bytes,
    validate_official_success_receipt,
)

EVIDENCE_SCHEMA_ID = "rpent.behavior.dino_v2.evidence_pack.v1"
SOURCE_SET_SCHEMA_ID = "rpent.behavior.dino_v2.source_set.v1"
CACHE_SCHEMA_ID = "rpent.behavior.dino_v2.cache.v1"
RECORD_SCHEMA_ID = "rpent.behavior.dino_v2.record.v1"
QUERY_SCHEMA_ID = "rpent.behavior.dino_v2.query.v1"
KEYFRAME_POLICY = "all_head_frames_with_optional_wrists_v1"
TASK_MEMORY_SOURCE = "official_memory_manager_task_specific"
ACTIVE_CHANNEL = "head"
EVIDENCE_CHANNELS = ("head", "left_wrist", "right_wrist")
_CAMERA_ARTIFACTS = {
    "head": "head_rgb.png",
    "left_wrist": "left_wrist_rgb.png",
    "right_wrist": "right_wrist_rgb.png",
}
_CACHE_DIRNAME = "dino_v2"
_LOCK_NAME = ".build.lock"
_MANIFEST_NAME = "manifest.json"


class DinoIndexError(ValueError):
    """Validation error for the derived DINO cache or evidence layer."""

    def __init__(self, code: str, path: str, detail: str) -> None:
        super().__init__(f"{code}: {path}: {detail}")
        self.code = code
        self.path = path
        self.detail = detail


def prepare_evidence_pack(
    output_dir: str | Path,
    cell_tag: str,
    session_state: Any,
    terminal_result: Mapping[str, Any],
) -> Path | None:
    """Prepare immutable run-local success evidence for ``cell_tag``.

    The returned directory is a robot-neutral artifact for MemoryManager to copy
    into the official corpus under its merge lock. This function never writes
    into ``memory_root`` and never manufactures the run audit.
    """

    root = Path(output_dir).expanduser().resolve()
    cell = _nonempty_string(cell_tag, "cell_tag")
    if not isinstance(terminal_result, Mapping):
        return None
    receipt = validate_official_success_receipt(
        terminal_result.get("official_success_receipt")
    )
    if (
        terminal_result.get("_finish") is not True
        or terminal_result.get("task_success") is not True
        or receipt is None
    ):
        return None
    audit_path = root / f"{cell}.json"
    if not audit_path.is_file():
        return None
    audit = _read_json(audit_path)
    if not isinstance(audit, Mapping):
        raise DinoIndexError("invalid", str(audit_path), "expected object")
    audit_receipt = validate_official_success_receipt(
        audit.get("official_success_receipt")
    )
    if audit_receipt is None:
        raise DinoIndexError(
            "invalid",
            f"{audit_path}.official_success_receipt",
            "missing valid official success receipt",
        )
    if audit_receipt["receipt_sha256"] != receipt["receipt_sha256"]:
        raise DinoIndexError(
            "invalid",
            f"{audit_path}.official_success_receipt",
            "audit receipt does not match terminal receipt",
        )
    task_name = _nonempty_string(audit.get("task_name"), f"{audit_path}.task_name")
    _validate_audit_cell(cell, task_name, audit.get("public_seed"), audit_path)

    frames = _session_frame_sources(session_state)
    if not any(frame["camera"] == ACTIVE_CHANNEL for frame in frames):
        raise DinoIndexError("invalid", "session_state", "missing head evidence")

    artifacts_root = root / "task_artifacts"
    evidence_dir = artifacts_root / cell
    manifest_path = evidence_dir / _MANIFEST_NAME
    with _writer_lock(artifacts_root):
        if manifest_path.is_file():
            _read_evidence(root, manifest_path, receipt["receipt_sha256"])
            return evidence_dir
        if evidence_dir.exists():
            raise DinoIndexError(
                "invalid", str(evidence_dir), "existing run evidence pack is incomplete"
            )

        artifacts_root.mkdir(parents=True, exist_ok=True)
        temporary: Path | None = Path(
            tempfile.mkdtemp(
                prefix=f".{cell}.", suffix=".tmp", dir=os.fspath(artifacts_root)
            )
        )
        try:
            copied: list[dict[str, Any]] = []
            for frame in frames:
                source = Path(frame["path"])
                rel = (
                    Path("frames")
                    / frame["camera"]
                    / f"{cell}_step_{frame['step_idx']}.png"
                )
                target = temporary / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
                shape = _image_shape(target)
                copied.append(
                    {
                        "camera": frame["camera"],
                        "kind": frame["kind"],
                        "step_idx": frame["step_idx"],
                        "env_step": frame["env_step"],
                        "path": rel.as_posix(),
                        "shape": shape,
                        "sha256": _sha256_file(target, label=str(target)),
                    }
                )

            terminal_bytes = _canonical_json_bytes(dict(terminal_result))
            terminal_rel = "terminal_receipt.json"
            _write_bytes_atomic(temporary / terminal_rel, terminal_bytes)
            manifest = {
                "schema_id": EVIDENCE_SCHEMA_ID,
                "cell_tag": cell,
                "receipt_sha256": receipt["receipt_sha256"],
                "terminal_receipt": {
                    "path": terminal_rel,
                    "sha256": _sha256_bytes(terminal_bytes),
                },
                "keyframe_policy": KEYFRAME_POLICY,
                "channels": list(EVIDENCE_CHANNELS),
                "frames": sorted(
                    copied, key=lambda item: (item["step_idx"], item["camera"])
                ),
            }
            _write_bytes_atomic(
                temporary / _MANIFEST_NAME, _canonical_json_bytes(manifest)
            )
            try:
                os.replace(temporary, evidence_dir)
            except FileExistsError:
                _read_evidence(root, manifest_path, receipt["receipt_sha256"])
                return evidence_dir
            temporary = None
            return evidence_dir
        finally:
            if temporary is not None and temporary.exists():
                shutil.rmtree(temporary)


def rebuild_index(memory_root: str | Path, encoder: Any) -> dict[str, Any]:
    """Rebuild the derived DINO cache from ``task-specific`` only."""

    root = Path(memory_root).expanduser().resolve()
    cache_dir = root / _CACHE_DIRNAME
    with _writer_lock(cache_dir):
        source_set = _source_set(root)
        encoder_meta = _encoder_revision_metadata(encoder)
        build_key = _build_key(source_set, encoder_meta)
        rows = source_set["sources"]
        records = [
            {"schema_id": RECORD_SCHEMA_ID, "row": index, **row}
            for index, row in enumerate(rows)
        ]
        vectors = _encode_sources(root, encoder, rows)
        records_bytes = _jsonl_bytes(records)
        embeddings_bytes = _npz_bytes(ACTIVE_CHANNEL, vectors)
        records_sha = _sha256_bytes(records_bytes)
        embeddings_sha = _sha256_bytes(embeddings_bytes)
        records_name = f"records.{records_sha}.jsonl"
        embeddings_name = f"embeddings.{embeddings_sha}.npz"
        _write_bytes_once(cache_dir / records_name, records_bytes)
        _write_bytes_once(cache_dir / embeddings_name, embeddings_bytes)

        manifest = {
            "schema_id": CACHE_SCHEMA_ID,
            "source": TASK_MEMORY_SOURCE,
            "source_set_sha256": source_set["source_set_sha256"],
            "encoder_sha256": _sha256_json(encoder_meta),
            "build_key": build_key,
            "encoder": encoder_meta,
            "count": len(rows),
            "dimension": DINOV2_DIMENSION,
            "metric": DISTANCE_METRIC,
            "active_channel": ACTIVE_CHANNEL,
            "records": {"path": records_name, "sha256": records_sha},
            "embeddings": {"path": embeddings_name, "sha256": embeddings_sha},
        }
        _write_bytes_atomic(cache_dir / _MANIFEST_NAME, _canonical_json_bytes(manifest))
        loaded = _load_index(root, encoder_meta, source_set)
        return {
            "status": "ok",
            "source": TASK_MEMORY_SOURCE,
            "cache_dir": str(cache_dir),
            "manifest": str(cache_dir / _MANIFEST_NAME),
            "source_set_sha256": source_set["source_set_sha256"],
            "encoder_sha256": manifest["encoder_sha256"],
            "build_key": build_key,
            "records": len(loaded["records"]),
        }


def query(
    memory_root: str | Path,
    encoder: Any,
    image: Any,
    task_name: str,
    limit: int = 3,
) -> dict[str, Any]:
    """Return canonical task-memory references for the closest head frames."""

    root = Path(memory_root).expanduser().resolve()
    query_vector = _encode_query(encoder, image)
    task = _nonempty_string(task_name, "task_name")
    count = _positive_int(limit, "limit")
    source_set = _source_set(root)
    encoder_meta = _encoder_revision_metadata(encoder)
    build_key = _build_key(source_set, encoder_meta)
    if not source_set["sources"]:
        return {
            "schema_id": QUERY_SCHEMA_ID,
            "status": "empty",
            "source": TASK_MEMORY_SOURCE,
            "task_name": task,
            "source_set_sha256": source_set["source_set_sha256"],
            "build_key": build_key,
            "matches": [],
        }
    try:
        loaded = _load_index(root, encoder_meta, source_set)
    except DinoIndexError as exc:
        if exc.code not in {"cache_missing", "cache_stale"}:
            raise
        rebuild_index(root, encoder)
        loaded = _load_index(root, encoder_meta, source_set)

    candidates = [
        (index, record)
        for index, record in enumerate(loaded["records"])
        if record.get("task_name") == task
    ]
    if not candidates:
        return {
            "schema_id": QUERY_SCHEMA_ID,
            "status": "no_task_matches",
            "source": TASK_MEMORY_SOURCE,
            "task_name": task,
            "source_set_sha256": source_set["source_set_sha256"],
            "build_key": build_key,
            "matches": [],
        }
    matrix = loaded["embeddings"]
    scored = sorted(
        (
            (float(matrix[index].dot(query_vector)), index, record)
            for index, record in candidates
        ),
        key=lambda item: (-item[0], item[2]["cell_tag"], item[2]["env_step"], item[1]),
    )[:count]
    return {
        "schema_id": QUERY_SCHEMA_ID,
        "status": "ok" if scored else "empty",
        "source": TASK_MEMORY_SOURCE,
        "task_name": task,
        "source_set_sha256": source_set["source_set_sha256"],
        "encoder_sha256": loaded["manifest"]["encoder_sha256"],
        "build_key": build_key,
        "matches": [
            _match_payload(root, rank, score, row_index, record)
            for rank, (score, row_index, record) in enumerate(scored, start=1)
        ],
    }


def _session_frame_sources(session_state: Any) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for record in session_state.records():
        step = _nonnegative_int(record.step_idx, "record.step_idx")
        env_step = _env_step_from_state(record.state)
        for camera, name in _CAMERA_ARTIFACTS.items():
            if name not in record.artifacts:
                continue
            path = Path(session_state.artifact_path(name, step=step))
            if path.is_file():
                result.append(
                    {
                        "camera": camera,
                        "kind": name,
                        "step_idx": step,
                        "env_step": env_step,
                        "path": path,
                    }
                )
    return sorted(result, key=lambda item: (item["step_idx"], item["camera"]))


def _env_step_from_state(state: Any) -> int:
    if not isinstance(state, Mapping):
        raise DinoIndexError("invalid", "state.total_env_steps", "missing state")
    value = state.get("total_env_steps")
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise DinoIndexError(
            "invalid", "state.total_env_steps", "expected non-negative int"
        )
    return value


def _validate_audit_cell(
    cell: str, task_name: str, public_seed: Any, audit_path: Path
) -> None:
    if isinstance(public_seed, bool) or not isinstance(public_seed, int):
        raise DinoIndexError(
            "invalid", f"{audit_path}.public_seed", "expected integer public seed"
        )
    try:
        expected = get_task_spec(task_name).tag(public_seed)
    except ValueError as exc:
        raise DinoIndexError(
            "invalid", f"{audit_path}.task_identity", str(exc)
        ) from exc
    if expected != cell:
        raise DinoIndexError(
            "invalid",
            f"{audit_path}.task_identity",
            f"expected cell {expected!r}, actual {cell!r}",
        )


def _source_set(root: Path) -> dict[str, Any]:
    task_dir = root / "task-specific"
    sources: list[dict[str, Any]] = []
    if task_dir.is_dir():
        for audit_path in sorted(task_dir.glob("*.json")):
            cell = audit_path.stem
            recipe_path = task_dir / f"{cell}_recipe.jsonl"
            evidence_path = task_dir / "artifacts" / cell / _MANIFEST_NAME
            if not recipe_path.is_file() or not evidence_path.is_file():
                continue
            audit = _read_json(audit_path)
            if not isinstance(audit, Mapping):
                raise DinoIndexError("invalid", str(audit_path), "expected object")
            receipt = validate_official_success_receipt(
                audit.get("official_success_receipt")
            )
            if receipt is None:
                continue
            task_name = _nonempty_string(
                audit.get("task_name"), f"{audit_path}.task_name"
            )
            public_seed = audit.get("public_seed")
            _validate_audit_cell(cell, task_name, public_seed, audit_path)
            evidence = _read_evidence(root, evidence_path, receipt["receipt_sha256"])
            head_frames = [
                frame
                for frame in evidence["frames"]
                if frame["camera"] == ACTIVE_CHANNEL
            ]
            wrists_by_step: dict[int, list[dict[str, Any]]] = {}
            for frame in evidence["frames"]:
                if frame["camera"] == ACTIVE_CHANNEL:
                    continue
                wrists_by_step.setdefault(frame["step_idx"], []).append(frame)
            audit_sha = _sha256_file(audit_path, label=str(audit_path))
            recipe_sha = _sha256_file(recipe_path, label=str(recipe_path))
            evidence_sha = _sha256_file(evidence_path, label=str(evidence_path))
            for frame in head_frames:
                row = {
                    "source": TASK_MEMORY_SOURCE,
                    "cell_tag": cell,
                    "task_name": task_name,
                    "public_seed": audit.get("public_seed"),
                    "receipt_sha256": receipt["receipt_sha256"],
                    "audit_path": _relative(root, audit_path),
                    "audit_sha256": audit_sha,
                    "recipe_path": _relative(root, recipe_path),
                    "recipe_sha256": recipe_sha,
                    "evidence_manifest": _relative(root, evidence_path),
                    "evidence_manifest_sha256": evidence_sha,
                    "step_idx": frame["step_idx"],
                    "env_step": frame["env_step"],
                    "image_kind": frame["kind"],
                    "image_path": frame["path"],
                    "image_shape": frame["shape"],
                    "image_sha256": frame["sha256"],
                    "wrist_evidence": [
                        {
                            "camera": wrist["camera"],
                            "kind": wrist["kind"],
                            "path": wrist["path"],
                            "shape": wrist["shape"],
                            "sha256": wrist["sha256"],
                            "env_step": wrist["env_step"],
                            "step_idx": wrist["step_idx"],
                        }
                        for wrist in sorted(
                            wrists_by_step.get(frame["step_idx"], ()),
                            key=lambda item: item["camera"],
                        )
                    ],
                }
                row["source_id"] = _sha256_json(row)
                sources.append(row)
    sources.sort(
        key=lambda item: (
            item["task_name"],
            item["cell_tag"],
            item["env_step"],
            item["step_idx"],
            item["image_sha256"],
        )
    )
    payload = {"schema_id": SOURCE_SET_SCHEMA_ID, "sources": sources}
    return {**payload, "source_set_sha256": _sha256_json(payload)}


def _read_evidence(root: Path, path: Path, expected_receipt_sha: str) -> dict[str, Any]:
    manifest = _read_json(path)
    if not isinstance(manifest, Mapping):
        raise DinoIndexError("invalid", str(path), "expected object")
    _require_keys(
        manifest,
        {
            "schema_id",
            "cell_tag",
            "receipt_sha256",
            "terminal_receipt",
            "keyframe_policy",
            "channels",
            "frames",
        },
        str(path),
    )
    if manifest["schema_id"] != EVIDENCE_SCHEMA_ID:
        raise DinoIndexError("invalid", f"{path}.schema_id", "schema mismatch")
    if manifest["receipt_sha256"] != expected_receipt_sha:
        raise DinoIndexError("invalid", f"{path}.receipt_sha256", "receipt mismatch")
    if manifest["keyframe_policy"] != KEYFRAME_POLICY:
        raise DinoIndexError("invalid", f"{path}.keyframe_policy", "policy mismatch")
    terminal = manifest["terminal_receipt"]
    if not isinstance(terminal, Mapping):
        raise DinoIndexError("invalid", f"{path}.terminal_receipt", "expected object")
    terminal_path = (
        path.parent / _nonempty_string(terminal.get("path"), "terminal.path")
    ).resolve()
    _require_under(path.parent.resolve(), terminal_path)
    if _sha256_file(terminal_path, label=str(terminal_path)) != require_sha256(
        terminal.get("sha256"), path="terminal.sha256"
    ):
        raise DinoIndexError("invalid", str(terminal_path), "SHA-256 mismatch")
    terminal_payload = _read_json(terminal_path)
    if not isinstance(terminal_payload, Mapping):
        raise DinoIndexError("invalid", str(terminal_path), "expected object")
    terminal_receipt = validate_official_success_receipt(
        terminal_payload.get("official_success_receipt")
    )
    if (
        terminal_payload.get("_finish") is not True
        or terminal_payload.get("task_success") is not True
        or terminal_receipt is None
        or terminal_receipt["receipt_sha256"] != expected_receipt_sha
    ):
        raise DinoIndexError(
            "invalid",
            str(terminal_path),
            "terminal receipt does not match official audit success",
        )

    frames_raw = manifest["frames"]
    if not isinstance(frames_raw, list) or not frames_raw:
        raise DinoIndexError("invalid", f"{path}.frames", "expected non-empty list")
    frames = [
        _normalize_frame(root, path, frame, index)
        for index, frame in enumerate(frames_raw)
    ]
    if not any(frame["camera"] == ACTIVE_CHANNEL for frame in frames):
        raise DinoIndexError("invalid", f"{path}.frames", "missing head frame")
    return {"manifest": dict(manifest), "frames": frames}


def _normalize_frame(
    root: Path, manifest_path: Path, frame: Any, index: int
) -> dict[str, Any]:
    if not isinstance(frame, Mapping):
        raise DinoIndexError(
            "invalid", f"{manifest_path}.frames[{index}]", "expected object"
        )
    _require_keys(
        frame,
        {"camera", "kind", "step_idx", "env_step", "path", "shape", "sha256"},
        f"{manifest_path}.frames[{index}]",
    )
    camera = frame["camera"]
    if camera not in EVIDENCE_CHANNELS:
        raise DinoIndexError(
            "invalid", f"{manifest_path}.frames[{index}].camera", "unknown camera"
        )
    kind = _nonempty_string(frame["kind"], f"{manifest_path}.frames[{index}].kind")
    if kind != _CAMERA_ARTIFACTS[camera]:
        raise DinoIndexError(
            "invalid", f"{manifest_path}.frames[{index}].kind", "camera/kind mismatch"
        )
    rel_path = _nonempty_string(frame["path"], f"{manifest_path}.frames[{index}].path")
    image_path = (manifest_path.parent / rel_path).resolve()
    _require_under(root, image_path)
    shape = _shape_list(frame["shape"], f"{manifest_path}.frames[{index}].shape")
    if _image_shape(image_path) != shape:
        raise DinoIndexError("invalid", str(image_path), "shape mismatch")
    expected_sha = require_sha256(
        frame["sha256"], path=f"{manifest_path}.frames[{index}].sha256"
    )
    if _sha256_file(image_path, label=str(image_path)) != expected_sha:
        raise DinoIndexError("invalid", str(image_path), "SHA-256 mismatch")
    return {
        "camera": camera,
        "kind": kind,
        "step_idx": _nonnegative_int(
            frame["step_idx"], f"{manifest_path}.frames[{index}].step_idx"
        ),
        "env_step": _nonnegative_int(
            frame["env_step"], f"{manifest_path}.frames[{index}].env_step"
        ),
        "path": _relative(root, image_path),
        "shape": shape,
        "sha256": expected_sha,
    }


def _load_index(
    root: Path, encoder_meta: dict[str, Any], source_set: dict[str, Any]
) -> dict[str, Any]:
    cache_dir = root / _CACHE_DIRNAME
    manifest_path = cache_dir / _MANIFEST_NAME
    if not manifest_path.is_file():
        raise DinoIndexError(
            "cache_missing", str(manifest_path), "run rebuild_index first"
        )
    manifest = _read_json(manifest_path)
    if not isinstance(manifest, Mapping):
        raise DinoIndexError("invalid", str(manifest_path), "expected object")
    _require_keys(
        manifest,
        {
            "schema_id",
            "source",
            "source_set_sha256",
            "encoder_sha256",
            "build_key",
            "encoder",
            "count",
            "dimension",
            "metric",
            "active_channel",
            "records",
            "embeddings",
        },
        str(manifest_path),
    )
    if (
        manifest["schema_id"] != CACHE_SCHEMA_ID
        or manifest["source"] != TASK_MEMORY_SOURCE
    ):
        raise DinoIndexError("invalid", "manifest.schema_id", "cache schema mismatch")
    if manifest["source_set_sha256"] != source_set["source_set_sha256"]:
        raise DinoIndexError("cache_stale", "source_set", "official corpus changed")
    if (
        manifest["encoder_sha256"] != _sha256_json(encoder_meta)
        or manifest["encoder"] != encoder_meta
    ):
        raise DinoIndexError("cache_stale", "encoder", "encoder revision changed")
    if manifest["build_key"] != _build_key(source_set, encoder_meta):
        raise DinoIndexError("cache_stale", "build_key", "derived identity changed")
    if (
        manifest["dimension"] != DINOV2_DIMENSION
        or manifest["active_channel"] != ACTIVE_CHANNEL
    ):
        raise DinoIndexError("invalid", "manifest.dimension", "DINO contract mismatch")
    if manifest["metric"] != DISTANCE_METRIC:
        raise DinoIndexError("invalid", "manifest.metric", "DINO metric mismatch")

    records_info = manifest["records"]
    embeddings_info = manifest["embeddings"]
    if not isinstance(records_info, Mapping) or not isinstance(
        embeddings_info, Mapping
    ):
        raise DinoIndexError("invalid", "manifest.artifacts", "expected objects")
    records_path = _cache_child(cache_dir, records_info.get("path"))
    embeddings_path = _cache_child(cache_dir, embeddings_info.get("path"))
    records_sha = require_sha256(records_info.get("sha256"), path="records.sha256")
    embeddings_sha = require_sha256(
        embeddings_info.get("sha256"), path="embeddings.sha256"
    )
    _require_content_address(records_path.name, "records", ".jsonl", records_sha)
    _require_content_address(embeddings_path.name, "embeddings", ".npz", embeddings_sha)
    if _sha256_file(records_path, label=str(records_path)) != records_sha:
        raise DinoIndexError("invalid", str(records_path), "SHA-256 mismatch")
    if _sha256_file(embeddings_path, label=str(embeddings_path)) != embeddings_sha:
        raise DinoIndexError("invalid", str(embeddings_path), "SHA-256 mismatch")

    records = _read_jsonl(records_path)
    sources = source_set["sources"]
    if len(records) != len(sources) or manifest["count"] != len(sources):
        raise DinoIndexError("invalid", "records", "row count mismatch")
    for row, (record, source) in enumerate(zip(records, sources, strict=True)):
        expected = {"schema_id": RECORD_SCHEMA_ID, "row": row, **source}
        if record != expected:
            raise DinoIndexError("invalid", f"records[{row}]", "record/source mismatch")
        _assert_source_files_current(root, record)
    matrix = _load_l2_cache_matrix(embeddings_path, len(records))
    return {"manifest": dict(manifest), "records": records, "embeddings": matrix}


def _assert_source_files_current(root: Path, record: Mapping[str, Any]) -> None:
    for path_key, sha_key in (
        ("audit_path", "audit_sha256"),
        ("recipe_path", "recipe_sha256"),
        ("evidence_manifest", "evidence_manifest_sha256"),
        ("image_path", "image_sha256"),
    ):
        path = (root / _nonempty_string(record.get(path_key), path_key)).resolve()
        _require_under(root, path)
        if _sha256_file(path, label=str(path)) != require_sha256(
            record.get(sha_key), path=sha_key
        ):
            raise DinoIndexError("cache_stale", str(path), f"{path_key} changed")


def _load_l2_cache_matrix(path: Path, rows: int) -> np.ndarray:
    try:
        with np.load(path) as archive:
            if ACTIVE_CHANNEL not in archive.files:
                raise DinoIndexError("invalid", str(path), "missing head embeddings")
            matrix = archive[ACTIVE_CHANNEL]
    except DinoIndexError:
        raise
    except Exception as exc:
        raise DinoIndexError(
            "invalid", str(path), f"{type(exc).__name__}: {exc}"
        ) from exc
    if matrix.dtype != np.float32:
        raise DinoIndexError("invalid", str(path), "expected float32 embeddings")
    if matrix.shape != (rows, DINOV2_DIMENSION):
        raise DinoIndexError("invalid", str(path), "embedding shape mismatch")
    if not np.isfinite(matrix).all():
        raise DinoIndexError("invalid", str(path), "embeddings must be finite")
    if rows:
        norms = np.linalg.norm(matrix.astype(np.float64), axis=1)
        if not np.allclose(norms, 1.0, rtol=1e-5, atol=1e-5):
            raise DinoIndexError(
                "invalid", str(path), "embeddings must already be L2-normalized"
            )
    return np.ascontiguousarray(matrix)


def _encode_sources(
    root: Path, encoder: Any, rows: Sequence[Mapping[str, Any]]
) -> np.ndarray:
    if not rows:
        return np.zeros((0, DINOV2_DIMENSION), dtype=np.float32)
    unique_rows: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        unique_rows.setdefault(str(row["image_sha256"]), row)
    encoded_by_sha: dict[str, np.ndarray] = {}
    ordered = list(unique_rows.items())
    for start in range(0, len(ordered), MAX_BATCH_SIZE):
        chunk = ordered[start : start + MAX_BATCH_SIZE]
        images = [
            _read_image((root / str(row["image_path"])).resolve()) for _, row in chunk
        ]
        encoded = encoder.encode_batch(images)
        if len(encoded) != len(images):
            raise DinoIndexError("invalid", "encoder.output", "batch length mismatch")
        for offset, (image_sha, _) in enumerate(chunk):
            vector = encoded[offset]
            if vector is None:
                raise DinoIndexError(
                    "invalid", f"encoder.output[{start + offset}]", "missing embedding"
                )
            encoded_by_sha[image_sha] = l2_normalize_row(
                vector, path=f"encoder.output[{start + offset}]"
            )
    return np.stack(
        [encoded_by_sha[str(row["image_sha256"])] for row in rows],
        axis=0,
    ).astype(np.float32, copy=False)


def _encode_query(encoder: Any, image: Any) -> np.ndarray:
    encoded = encoder.encode_batch([np.asarray(image, dtype=np.uint8)])
    if len(encoded) != 1 or encoded[0] is None:
        raise DinoIndexError("invalid", "encoder.output[0]", "missing query embedding")
    return l2_normalize_row(encoded[0], path="encoder.output[0]")


def _encoder_revision_metadata(encoder: Any) -> dict[str, Any]:
    method = getattr(encoder, "revision_metadata", None)
    if not callable(method):
        raise DinoIndexError("invalid", "encoder.revision_metadata", "missing method")
    meta = method()
    if not isinstance(meta, Mapping):
        raise DinoIndexError("invalid", "encoder.revision_metadata", "expected object")
    result = dict(meta)
    if result.get("dimension") != DINOV2_DIMENSION:
        raise DinoIndexError("invalid", "encoder.dimension", "expected 384")
    return result


def _match_payload(
    root: Path,
    rank: int,
    score: float,
    row_index: int,
    record: Mapping[str, Any],
) -> dict[str, Any]:
    audit_path = root / record["audit_path"]
    recipe_path = root / record["recipe_path"]
    audit_excerpt = _excerpt(
        _canonical_json_bytes(_read_json(audit_path)).decode("utf-8")
    )
    recipe_preview = _recipe_preview(recipe_path)
    return {
        "rank": rank,
        "row": row_index,
        "score": round(score, 8),
        "distance": round(1.0 - score, 8),
        "task_name": record["task_name"],
        "cell_tag": record["cell_tag"],
        "matched_frame": {
            "path": record["image_path"],
            "sha256": record["image_sha256"],
            "kind": record["image_kind"],
            "shape": record["image_shape"],
            "step_idx": record["step_idx"],
            "env_step": record["env_step"],
        },
        "source_reference": {
            "audit_path": record["audit_path"],
            "recipe_path": record["recipe_path"],
            "evidence_manifest": record["evidence_manifest"],
            "head_image": {
                "path": record["image_path"],
                "sha256": record["image_sha256"],
                "kind": record["image_kind"],
                "shape": record["image_shape"],
                "step_idx": record["step_idx"],
                "env_step": record["env_step"],
            },
            "wrist_evidence": record["wrist_evidence"],
        },
        "task_memory": {
            "source": TASK_MEMORY_SOURCE,
            "audit_excerpt": audit_excerpt,
            "recipe_preview": recipe_preview,
        },
    }


def _recipe_preview(path: Path) -> list[Any]:
    return _read_jsonl(path)[:10]


def _excerpt(text: str, limit: int = 2000) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _read_image(path: Path) -> np.ndarray:
    image = np.asarray(imageio.imread(path), dtype=np.uint8)
    if image.ndim == 2:
        image = np.repeat(image[:, :, None], 3, axis=2)
    if image.ndim != 3 or image.shape[2] < 3:
        raise DinoIndexError("invalid", str(path), "expected RGB image")
    return np.ascontiguousarray(image[:, :, :3])


def _image_shape(path: Path) -> list[int]:
    return [int(value) for value in _read_image(path).shape]


def _shape_list(value: Any, path: str) -> list[int]:
    if (
        not isinstance(value, list)
        or len(value) != 3
        or any(
            isinstance(item, bool) or not isinstance(item, int) or item <= 0
            for item in value
        )
    ):
        raise DinoIndexError("invalid", path, "expected positive image shape [H,W,3]")
    if value[2] != 3:
        raise DinoIndexError("invalid", path, "expected RGB shape [H,W,3]")
    return [int(item) for item in value]


@contextlib.contextmanager
def _writer_lock(cache_dir: Path) -> Any:
    cache_dir.mkdir(parents=True, exist_ok=True)
    lock_path = cache_dir / _LOCK_NAME
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        with contextlib.suppress(OSError):
            fcntl.flock(fd, fcntl.LOCK_UN)
        with contextlib.suppress(OSError):
            os.close(fd)


def _cache_child(cache_dir: Path, raw: Any) -> Path:
    value = _nonempty_string(raw, "cache.path")
    path = (cache_dir / value).resolve()
    _require_under(cache_dir.resolve(), path)
    if not path.is_file():
        raise DinoIndexError("cache_missing", str(path), "missing cache artifact")
    return path


def _require_content_address(name: str, prefix: str, suffix: str, digest: str) -> None:
    if name != f"{prefix}.{digest}{suffix}":
        raise DinoIndexError(
            "invalid",
            name,
            f"expected content-addressed name {prefix}.{digest}{suffix}",
        )


def _require_under(root: Path, path: Path) -> None:
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise DinoIndexError("invalid", str(path), f"outside {root}") from exc


def _relative(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise DinoIndexError(
            "invalid", str(path), f"{type(exc).__name__}: {exc}"
        ) from exc


def _read_jsonl(path: Path) -> list[Any]:
    rows: list[Any] = []
    try:
        with path.open() as stream:
            for line in stream:
                if line.strip():
                    rows.append(json.loads(line))
    except (OSError, json.JSONDecodeError) as exc:
        raise DinoIndexError(
            "invalid", str(path), f"{type(exc).__name__}: {exc}"
        ) from exc
    return rows


def _jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(_canonical_json_bytes(row) + b"\n" for row in rows)


def _npz_bytes(name: str, value: np.ndarray) -> bytes:
    npy = io.BytesIO()
    np.save(npy, np.asarray(value, dtype=np.float32), allow_pickle=False)
    buffer = io.BytesIO()
    info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o644 << 16
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(info, npy.getvalue())
    return buffer.getvalue()


def _build_key(source_set: Mapping[str, Any], encoder_meta: Mapping[str, Any]) -> str:
    return _sha256_json(
        {
            "schema_id": CACHE_SCHEMA_ID,
            "source_set_sha256": source_set["source_set_sha256"],
            "encoder_sha256": _sha256_json(encoder_meta),
        }
    )


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_json(value: Any) -> str:
    return _sha256_bytes(_canonical_json_bytes(value))


def _write_bytes_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=os.fspath(path.parent)
    )
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _write_bytes_once(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise DinoIndexError("invalid", str(path), "existing bytes differ")
        return
    _write_bytes_atomic(path, payload)


def _require_keys(value: Mapping[str, Any], keys: set[str], path: str) -> None:
    actual = set(value)
    if actual != keys:
        raise DinoIndexError(
            "invalid",
            path,
            f"expected keys {sorted(keys)!r}, actual {sorted(actual)!r}",
        )


def _nonempty_string(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise DinoIndexError("invalid", path, "expected non-empty stable string")
    return value


def _nonnegative_int(value: Any, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise DinoIndexError("invalid", path, "expected non-negative int")
    return value


def _positive_int(value: Any, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise DinoIndexError("invalid", path, "expected positive int")
    return value


__all__ = [
    "DinoIndexError",
    "prepare_evidence_pack",
    "query",
    "rebuild_index",
]
