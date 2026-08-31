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

"""Offline SFT selection rollup into a non-activating episode-memory artifact."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import MappingProxyType
from typing import Any

from robots.behavior.memory_schema import (
    canonical_json_file_bytes,
    fail,
    require_exact_keys,
    require_sha256,
    sha256_bytes,
)

SELECTION_SCHEMA_ID = "rpent_behavior_sft_expert_selection_v1"
ROLLED_ARTIFACT_SCHEMA_ID = "rpent_behavior_sft_offline_rollup_v1"
EXPECTED_TASK_IDS = ("task-0000", "task-0001", "task-0010", "task-0034", "task-0040")
ACTIVE_VIEW_TASKS = {"turning_on_radio", "picking_up_trash"}
EXPECTED_EPISODES = 10
EXPECTED_SEGMENTS = 91


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_selection(path: Path) -> Mapping[str, Any]:
    if not path.is_file() or path.is_symlink():
        fail(
            "MEMORY_SFT_SELECTION_MISSING",
            str(path),
            "selection manifest must be a regular file",
        )
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        fail("MEMORY_SFT_SELECTION_INVALID", str(path), f"{type(exc).__name__}: {exc}")
    if not isinstance(value, Mapping):
        fail("MEMORY_SFT_SELECTION_INVALID", str(path), "expected JSON object")
    validate_selection(value)
    return MappingProxyType(dict(value))


def validate_selection(document: Mapping[str, Any]) -> None:
    require_exact_keys(
        document,
        {
            "schema_id",
            "created_at",
            "preliminary",
            "activation_allowed",
            "active",
            "formal_compiler_admission",
            "contract_status",
            "source_release",
            "coverage",
            "evidence_boundaries",
            "episodes",
        },
        path="$",
    )
    if (
        document["schema_id"] != SELECTION_SCHEMA_ID
        or document["preliminary"] is not True
        or document["activation_allowed"] is not False
        or document["active"] is not False
        or document["formal_compiler_admission"] is not False
    ):
        fail("MEMORY_SFT_SELECTION_INVALID", "$", "non-activation identity mismatch")
    coverage = document["coverage"]
    if not isinstance(coverage, Mapping):
        fail("MEMORY_SFT_SELECTION_INVALID", "coverage", "expected object")
    expected_coverage = {
        "selected_episode_count": EXPECTED_EPISODES,
        "selected_segment_count": EXPECTED_SEGMENTS,
        "catalog_episode_count": 5,
        "query_episode_count": 5,
    }
    for key, expected in expected_coverage.items():
        if coverage.get(key) != expected:
            fail(
                "MEMORY_SFT_SELECTION_INVALID",
                f"coverage.{key}",
                f"expected {expected}",
            )
    episodes = document["episodes"]
    if not isinstance(episodes, list) or len(episodes) != EXPECTED_EPISODES:
        fail("MEMORY_SFT_SELECTION_INVALID", "episodes", "expected 10 episodes")
    task_ids = {str(row.get("task_id")) for row in episodes if isinstance(row, Mapping)}
    if task_ids != set(EXPECTED_TASK_IDS):
        fail(
            "MEMORY_SFT_SELECTION_INVALID",
            "episodes.task_id",
            "expected exact five-task coverage",
        )
    segment_count = 0
    for index, episode in enumerate(episodes):
        if not isinstance(episode, Mapping):
            fail(
                "MEMORY_SFT_SELECTION_INVALID", f"episodes[{index}]", "expected object"
            )
        for file_key in ("annotation", "metadata", "parquet"):
            entry = episode.get(file_key)
            if not isinstance(entry, Mapping):
                fail(
                    "MEMORY_SFT_SELECTION_INVALID",
                    f"episodes[{index}].{file_key}",
                    "expected object",
                )
            require_sha256(
                entry.get("sha256"), path=f"episodes[{index}].{file_key}.sha256"
            )
        videos = episode.get("videos")
        if not isinstance(videos, Mapping) or set(videos) != {
            "head",
            "left_wrist",
            "right_wrist",
        }:
            fail(
                "MEMORY_SFT_SELECTION_INVALID",
                f"episodes[{index}].videos",
                "expected three camera pins",
            )
        for camera, entry in videos.items():
            if not isinstance(entry, Mapping):
                fail(
                    "MEMORY_SFT_SELECTION_INVALID",
                    f"episodes[{index}].videos.{camera}",
                    "expected object",
                )
            require_sha256(
                entry.get("sha256"), path=f"episodes[{index}].videos.{camera}.sha256"
            )
        segments = episode.get("segments")
        if not isinstance(segments, list) or not segments:
            fail(
                "MEMORY_SFT_SELECTION_INVALID",
                f"episodes[{index}].segments",
                "expected non-empty list",
            )
        segment_count += len(segments)
    if segment_count != EXPECTED_SEGMENTS:
        fail(
            "MEMORY_SFT_SELECTION_INVALID", "segments", "expected 91 selected segments"
        )


def keyframes_for_episode(episode: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    frames: dict[int, dict[str, Any]] = {}
    segments = episode["segments"]
    for segment in segments:
        start = int(segment["start_frame"])
        end_exclusive = int(segment["end_frame_exclusive"])
        end = max(start, end_exclusive - 1)
        _add_frame(frames, start, "segment_start", segment)
        _add_frame(frames, end, "segment_end", segment)
        if end_exclusive - start >= 96:
            _add_frame(
                frames,
                start + (end_exclusive - start) // 2,
                "long_segment_midpoint",
                segment,
            )
    first_start = min(int(segment["start_frame"]) for segment in segments)
    last_end = max(int(segment["end_frame_exclusive"]) - 1 for segment in segments)
    _add_frame(frames, first_start, "episode_first", segments[0])
    _add_frame(frames, last_end, "episode_last", segments[-1])
    return tuple(frames[index] for index in sorted(frames))


def build_rollup(
    selection: Mapping[str, Any], *, selection_sha256: str
) -> Mapping[str, Any]:
    active: list[Mapping[str, Any]] = []
    sealed: list[Mapping[str, Any]] = []
    for episode in selection["episodes"]:
        row = {
            "episode_id": episode["episode_id"],
            "task_id": episode["task_id"],
            "task_name": episode["task_name"],
            "split": episode["split"],
            "segments": episode["segments"],
            "keyframes": list(keyframes_for_episode(episode)),
            "source_refs": {
                "annotation": episode["annotation"],
                "metadata": episode["metadata"],
                "parquet": episode["parquet"],
                "videos": episode["videos"],
            },
            "usage": {
                "source": "official_sft_offline",
                "active_view": episode["task_name"] in ACTIVE_VIEW_TASKS,
                "wrist_policy": "shadow_only",
            },
            "outcome": {
                "success": None,
                "authority": "official_sft_demonstration_without_runtime_success_receipt",
            },
        }
        if episode["task_name"] in ACTIVE_VIEW_TASKS:
            active.append(row)
        else:
            sealed.append(row)
    if len(active) != 4 or len(sealed) != 6:
        fail(
            "MEMORY_SFT_ROLLUP_INVALID",
            "active_view",
            "expected Radio/Trash 4 active-view episodes and 6 sealed episodes",
        )
    return {
        "schema_id": ROLLED_ARTIFACT_SCHEMA_ID,
        "preliminary": True,
        "activation_allowed": False,
        "selection_manifest_sha256": selection_sha256,
        "task_count": 5,
        "episode_count": 10,
        "segment_count": 91,
        "keyframe_policy": "episode first/last, segment start/end, long midpoint, dedupe by frame index",
        "active_view_policy": "turning_on_radio and picking_up_trash only",
        "active_view": active,
        "sealed_archive": sealed,
    }


def write_content_addressed_rollup(
    *, selection_manifest: Path, output_dir: Path
) -> Mapping[str, Any]:
    raw = selection_manifest.read_bytes()
    selection_sha = sha256_bytes(raw)
    selection = load_selection(selection_manifest)
    artifact = build_rollup(selection, selection_sha256=selection_sha)
    payload = canonical_json_file_bytes(artifact, path="rollup")
    digest = sha256_bytes(payload)
    object_dir = output_dir / "objects"
    object_path = object_dir / f"{digest}.json"
    _write_once(object_path, payload)
    pointer = {
        "schema_id": "rpent_behavior_sft_offline_rollup_pointer_v1",
        "artifact_sha256": digest,
        "artifact_path": str(object_path),
        "preliminary": True,
        "activation_allowed": False,
    }
    _atomic_write(
        output_dir / "latest.json", canonical_json_file_bytes(pointer, path="pointer")
    )
    return MappingProxyType(pointer)


def _resolve_source_file(
    relative_path: str, roots: Sequence[Path], *, expected_sha256: str
) -> Path:
    matches = [
        root / relative_path for root in roots if (root / relative_path).is_file()
    ]
    if len(matches) != 1:
        fail(
            "MEMORY_SFT_SOURCE_RESOLUTION_INVALID",
            relative_path,
            f"expected one source under configured roots, found {len(matches)}",
        )
    path = matches[0].resolve()
    actual = _sha256_file(path)
    if actual != expected_sha256:
        fail(
            "MEMORY_SFT_SOURCE_HASH_MISMATCH",
            relative_path,
            f"expected {expected_sha256}, actual {actual}",
        )
    return path


def _decode_video_frames(path: Path, frame_indices: Sequence[int]) -> list[Any]:
    # imageio-ffmpeg is already part of the Behavior optional extra and avoids
    # adding OpenCV to the source-plugin contract.
    import imageio.v2 as imageio

    try:
        reader = imageio.get_reader(str(path), format="ffmpeg")
    except Exception as exc:
        fail("MEMORY_SFT_VIDEO_INVALID", str(path), f"reader open failed: {exc}")
    decoded: list[Any] = []
    try:
        for frame_index in frame_indices:
            try:
                frame = reader.get_data(int(frame_index))
            except Exception as exc:
                fail(
                    "MEMORY_SFT_VIDEO_INVALID",
                    str(path),
                    f"cannot decode frame {frame_index}: {type(exc).__name__}: {exc}",
                )
            decoded.append(frame)
    finally:
        reader.close()
    return decoded


def _encode_in_batches(encoder: Any, images: Sequence[Any], *, batch_size: int) -> Any:
    import numpy as np

    rows: list[Any] = []
    for offset in range(0, len(images), batch_size):
        batch = encoder.encode_batch(list(images[offset : offset + batch_size]))
        rows.extend(item for item in batch if item is not None)
    if len(rows) != len(images):
        fail("MEMORY_SFT_EMBEDDING_INVALID", "encoder", "missing embedding row")
    return np.stack(rows, axis=0).astype(np.float32, copy=False)


def _load_episode_rollups(rollups_dir: Path) -> Mapping[str, tuple[Path, str]]:
    result: dict[str, tuple[Path, str]] = {}
    pattern = re.compile(r"^Episode id: `([^`]+)`\.$", re.MULTILINE)
    for path in sorted(rollups_dir.glob("*.memory.md")):
        text = path.read_text(encoding="utf-8")
        match = pattern.search(text)
        if match:
            result[match.group(1)] = (path, text)
    if len(result) != EXPECTED_EPISODES:
        fail(
            "MEMORY_SFT_ROLLUP_INVALID",
            str(rollups_dir),
            "expected 10 episode memory.md rollups",
        )
    return MappingProxyType(result)


def compile_runtime_catalog(
    *,
    selection_manifest: Path,
    output_dir: Path,
    video_roots: Sequence[Path],
    rollups_dir: Path,
    source_archive: Path,
    weights: Path,
    cache_dir: Path | None,
    batch_size: int,
) -> Mapping[str, Any]:
    """Compile all ten official SFT episodes and a four-episode runtime view."""

    if output_dir.exists():
        fail(
            "MEMORY_SFT_OUTPUT_COLLISION",
            str(output_dir),
            "output directory already exists",
        )
    if batch_size < 1 or batch_size > 32:
        fail("MEMORY_SFT_BATCH_INVALID", "batch_size", "expected 1..32")
    selection_raw = selection_manifest.read_bytes()
    selection = load_selection(selection_manifest)
    rollups = _load_episode_rollups(rollups_dir)

    # CUDA visibility is set by main() before these imports.
    import numpy as np
    import torch
    import torchvision

    from robots.behavior.episode_memory_index import (
        EpisodeExperience,
        EpisodeFrameKey,
        write_candidate_revision,
    )
    from robots.behavior.memory_embeddings_dinov2 import (
        EXPECTED_SOURCE_COMMIT,
        MODEL_ID,
        MODEL_REVISION,
        Dinov2DeploymentPaths,
        Dinov2Encoder,
        Dinov2RevisionIdentity,
    )

    if not torch.cuda.is_available():
        fail(
            "MEMORY_SFT_CUDA_UNAVAILABLE",
            "cuda",
            "compiler requires one visible CUDA device",
        )
    identity = Dinov2RevisionIdentity(
        model_id=MODEL_ID,
        model_revision=MODEL_REVISION,
        source_commit=EXPECTED_SOURCE_COMMIT,
        source_archive_sha256=_sha256_file(source_archive),
        weights_sha256=_sha256_file(weights),
        torch_version=str(torch.__version__),
        torchvision_version=str(torchvision.__version__),
        device="cuda",
    )
    encoder = Dinov2Encoder(
        identity,
        Dinov2DeploymentPaths(
            source_archive_path=source_archive.resolve(),
            weights_path=weights.resolve(),
            cache_dir=None if cache_dir is None else cache_dir.resolve(),
        ),
    )
    active_experiences: list[Any] = []
    active_head: list[Any] = []
    active_left: list[Any] = []
    active_right: list[Any] = []
    all_inventory: list[dict[str, Any]] = []
    all_head: list[Any] = []
    all_left: list[Any] = []
    all_right: list[Any] = []
    try:
        for episode in selection["episodes"]:
            episode_id = str(episode["episode_id"])
            keyframes = keyframes_for_episode(episode)
            frame_indices = [int(item["frame_index"]) for item in keyframes]
            encoded_channels: dict[str, Any] = {}
            source_videos: dict[str, dict[str, Any]] = {}
            for channel in ("head", "left_wrist", "right_wrist"):
                video = episode["videos"][channel]
                path = _resolve_source_file(
                    str(video["relative_path"]),
                    video_roots,
                    expected_sha256=str(video["sha256"]),
                )
                images = _decode_video_frames(path, frame_indices)
                encoded_channels[channel] = _encode_in_batches(
                    encoder, images, batch_size=batch_size
                )
                source_videos[channel] = {
                    "relative_path": str(video["relative_path"]),
                    "sha256": str(video["sha256"]),
                }
            all_offset = sum(array.shape[0] for array in all_head)
            all_head.append(encoded_channels["head"])
            all_left.append(encoded_channels["left_wrist"])
            all_right.append(encoded_channels["right_wrist"])
            rollup_source, memory_markdown = rollups[episode_id]
            active = str(episode["task_name"]) in ACTIVE_VIEW_TASKS
            all_inventory.append(
                {
                    "episode_id": episode_id,
                    "task_name": episode["task_name"],
                    "active_view": active,
                    "sealed": not active,
                    "frame_count": len(keyframes),
                    "all_embedding_rows": [all_offset, all_offset + len(keyframes)],
                    "memory_markdown": f"episode_rollups/{rollup_source.name}",
                    "source_videos": source_videos,
                }
            )
            if not active:
                continue
            active_offset = sum(array.shape[0] for array in active_head)
            frames = tuple(
                EpisodeFrameKey(
                    frame_id=f"{episode_id}:head:{item['frame_index']}",
                    episode_id=episode_id,
                    experience_id=f"episode:{episode_id}",
                    task_name=str(episode["task_name"]),
                    frame_index=int(item["frame_index"]),
                    embedding_row=active_offset + index,
                    keyframe_kind="+".join(item["keyframe_kinds"]),
                    source_record_id="+".join(item["source_segment_ids"]),
                    frame_identity={
                        "camera": "head",
                        "keyframe_kinds": list(item["keyframe_kinds"]),
                        "source_segment_ids": list(item["source_segment_ids"]),
                    },
                )
                for index, item in enumerate(keyframes)
            )
            active_experiences.append(
                EpisodeExperience(
                    episode_id=episode_id,
                    experience_id=f"episode:{episode_id}",
                    logical_experience_id=f"official-sft:{episode_id}",
                    task_name=str(episode["task_name"]),
                    usage={
                        "returned_scope": "whole_experience",
                        "episode_memory_markdown": memory_markdown,
                        "stage_inference": None,
                        "summary_status": "builder_generated_pending_phase6_review",
                    },
                    outcome={
                        "success": None,
                        "authority": "user_authorized_official_sft_expert_demonstration",
                        "raw_done_success": None,
                    },
                    frame_keys=frames,
                    canonical_trajectory_ref={
                        "kind": "official_sft_parquet",
                        **dict(episode["parquet"]),
                    },
                    trajectory_refs=tuple(
                        {"kind": f"official_sft_{channel}_video", **video}
                        for channel, video in source_videos.items()
                    ),
                    source={
                        "selection_manifest_sha256": sha256_bytes(selection_raw),
                        "episode_split": episode["split"],
                        "layout_fingerprint_sha256": episode[
                            "layout_fingerprint_sha256"
                        ],
                    },
                    metadata={
                        "preliminary": True,
                        "activation_allowed": False,
                        "segments": episode["segments"],
                        "wrist_policy": "shadow_only",
                    },
                )
            )
            active_head.append(encoded_channels["head"])
            active_left.append(encoded_channels["left_wrist"])
            active_right.append(encoded_channels["right_wrist"])
    finally:
        encoder.close()

    output_dir.mkdir(parents=True, exist_ok=False)
    episode_rollup_output = output_dir / "episode_rollups"
    episode_rollup_output.mkdir()
    for _, (source_path, text) in sorted(rollups.items()):
        _write_once(episode_rollup_output / source_path.name, text.encode("utf-8"))
    all_embedding_bytes = io.BytesIO()
    np.savez(
        all_embedding_bytes,
        head=np.concatenate(all_head, axis=0),
        left_wrist=np.concatenate(all_left, axis=0),
        right_wrist=np.concatenate(all_right, axis=0),
    )
    all_embedding_payload = all_embedding_bytes.getvalue()
    _write_once(output_dir / "all_episode_embeddings.npz", all_embedding_payload)
    _write_once(
        output_dir / "all_episode_inventory.json",
        canonical_json_file_bytes(
            {"episodes": all_inventory}, path="all_episode_inventory"
        ),
    )
    candidate = write_candidate_revision(
        memory_dir=output_dir / "active_catalog",
        experiences=active_experiences,
        head_embeddings=np.concatenate(active_head, axis=0),
        wrist_shadow_embeddings={
            "left_wrist": np.concatenate(active_left, axis=0),
            "right_wrist": np.concatenate(active_right, axis=0),
        },
        encoder_identity=identity.as_dict(),
        activate_current=True,
    )
    manifest = {
        "schema_id": "rpent_behavior_sft_episode_catalog_artifact_v1",
        "preliminary": True,
        "activation_allowed": False,
        "connected_to_active_runtime": False,
        "source_kind": "user_authorized_official_behavior_sft_training_data",
        "selection_manifest_sha256": sha256_bytes(selection_raw),
        "task_count": 5,
        "episode_count": 10,
        "segment_count": 91,
        "active_view_episode_count": 4,
        "sealed_episode_count": 6,
        "keyframe_policy": "episode first/last, segment start/end, long-segment midpoint, deduplicated",
        "active_channel": "head",
        "wrist_policy": "shadow_only_pending_fresh_policy_query_review",
        "stage_evidence_boundary": "SFT expert annotations are preserved as content; runtime retrieval makes no stage inference",
        "held_out_observed": False,
        "batch_size": batch_size,
        "cuda_visible_device_count": int(torch.cuda.device_count()),
        "encoder_identity": identity.as_dict(),
        "all_episode_embeddings_sha256": sha256_bytes(all_embedding_payload),
        "active_catalog_revision_document_sha256": candidate[
            "revision_document_sha256"
        ],
        "active_catalog_path": "active_catalog",
        "sealed_tasks": sorted(
            {row["task_name"] for row in all_inventory if row["sealed"]}
        ),
    }
    manifest_payload = canonical_json_file_bytes(manifest, path="artifact_manifest")
    _write_once(output_dir / "manifest.json", manifest_payload)
    return MappingProxyType(
        {
            "artifact_dir": str(output_dir),
            "manifest_sha256": sha256_bytes(manifest_payload),
            "active_catalog_revision_document_sha256": candidate[
                "revision_document_sha256"
            ],
            "preliminary": True,
            "activation_allowed": False,
        }
    )


def _add_frame(
    frames: dict[int, dict[str, Any]],
    frame_index: int,
    kind: str,
    segment: Mapping[str, Any],
) -> None:
    frames.setdefault(
        frame_index,
        {
            "frame_index": frame_index,
            "keyframe_kinds": [],
            "source_segment_ids": [],
        },
    )
    row = frames[frame_index]
    if kind not in row["keyframe_kinds"]:
        row["keyframe_kinds"].append(kind)
    segment_id = str(segment["segment_id"])
    if segment_id not in row["source_segment_ids"]:
        row["source_segment_ids"].append(segment_id)


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


def _write_once(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.is_file() and not path.is_symlink() and path.read_bytes() == payload:
            return
        fail("MEMORY_SFT_OUTPUT_COLLISION", str(path), "existing bytes differ")
    _atomic_write(path, payload)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="behavior-sft-offline-rollup")
    sub = parser.add_subparsers(dest="command", required=True)
    rollup = sub.add_parser("rollup")
    rollup.add_argument("--selection-manifest", required=True, type=Path)
    rollup.add_argument("--output-dir", required=True, type=Path)
    compile_catalog = sub.add_parser("compile-runtime-catalog")
    compile_catalog.add_argument("--selection-manifest", required=True, type=Path)
    compile_catalog.add_argument("--output-dir", required=True, type=Path)
    compile_catalog.add_argument(
        "--video-root", required=True, type=Path, action="append"
    )
    compile_catalog.add_argument("--rollups-dir", required=True, type=Path)
    compile_catalog.add_argument("--source-archive", required=True, type=Path)
    compile_catalog.add_argument("--weights", required=True, type=Path)
    compile_catalog.add_argument("--cache-dir", type=Path, default=None)
    compile_catalog.add_argument("--cuda-device", choices=("2", "7"), required=True)
    compile_catalog.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args(argv)
    if args.command == "rollup":
        result = write_content_addressed_rollup(
            selection_manifest=args.selection_manifest.resolve(),
            output_dir=args.output_dir.resolve(),
        )
        print(json.dumps(dict(result), sort_keys=True))
        return 0
    if args.command == "compile-runtime-catalog":
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device
        result = compile_runtime_catalog(
            selection_manifest=args.selection_manifest.resolve(),
            output_dir=args.output_dir.resolve(),
            video_roots=tuple(path.resolve() for path in args.video_root),
            rollups_dir=args.rollups_dir.resolve(),
            source_archive=args.source_archive.resolve(),
            weights=args.weights.resolve(),
            cache_dir=None if args.cache_dir is None else args.cache_dir.resolve(),
            batch_size=args.batch_size,
        )
        print(json.dumps(dict(result), sort_keys=True))
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
