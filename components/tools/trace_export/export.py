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

"""Validate recorded episodes and export audited, model-agnostic dataset rows.

An episode is exported whole or not at all: one bad turn quarantines the
episode, because a later cell can depend on names an earlier cell defined.
"""

from __future__ import annotations

import json
from collections.abc import Collection, Sequence
from pathlib import Path
from typing import Any

from components.tools.trace_export.cells import namespace_check, project_code
from components.tools.trace_export.projection import PROJECTION
from components.tools.trace_export.trace import (
    IMAGE_DIR,
    TRACE_SCHEMA,
    TraceError,
    discover_episodes,
    load_episode,
    read_image,
    require,
    sha256_hex,
    validate_robo_methods,
)

SCHEMA = "rpent.trace_export.v1"
SPLIT_POLICY = "episode-level only; no independent heldout implied"


def _user_content(turn: dict[str, Any], refs: list[dict[str, Any]]) -> list[dict]:
    content = []
    for part in turn["user"]:
        if part["type"] == "text":
            content.append(part)
        else:
            content.append({"type": "image", "path": refs[part["image"]]["path"]})
    return content


def validate_episode(
    root: Path | str,
    methods: Collection[str] | None = None,
    *,
    allow_unsuccessful: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, bytes]]:
    """Validate one episode and build its rows without writing anything.

    Raises ``TraceError`` if any part of the episode fails validation.

    Args:
        root: Episode directory in trace v1 format.
        methods: Fallback public ``robo`` method names used when
            ``episode.json`` omits ``robo_methods``.
        allow_unsuccessful: Export an episode without an environment-confirmed
            success (with ``environment_success`` set to ``False``) instead of
            rejecting it.

    Returns:
        The episode's rows and the PNG bytes they reference, keyed by the
        content-addressed path under ``images/``.
    """
    episode, turns = load_episode(root)
    if "robo_methods" in episode:
        allowed_methods = episode["robo_methods"]
    else:
        require(
            methods is not None,
            "episode has no robo_methods; record robo_methods in "
            "episode.json or pass --robo-method",
        )
        allowed_methods = validate_robo_methods(methods, "fallback robo_methods")
    verdict = episode["verdict"]
    success = verdict["accepted"] is True and verdict["truth_source"] == "environment"
    require(
        success or allow_unsuccessful,
        "native verdict is not an environment-confirmed success; opt-in required",
    )
    episode_id = episode["episode_id"]
    rows: list[dict[str, Any]] = []
    images: dict[str, bytes] = {}
    defined: set[str] = set()
    for turn in turns:
        index = turn["turn"]
        refs = []
        for ref in turn["images"]:
            path = f"{IMAGE_DIR}/{ref['png_sha256']}.png"
            images[path] = read_image(root, ref)
            refs.append({**ref, "path": path})
        projected, audit = project_code(turn["code"], allowed_methods)
        defined = namespace_check(projected, defined)
        rows.append(
            {
                "schema": SCHEMA,
                "id": f"{episode_id}:turn_{index:06d}",
                "episode_id": episode_id,
                "split_group": episode_id,
                "environment_success": success,
                "messages": [
                    {"role": "system", "content": turn["system"]},
                    {"role": "user", "content": _user_content(turn, refs)},
                ],
                "target": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "type": "function",
                            "function": {
                                "name": "python",
                                "arguments": {"code": projected},
                            },
                        }
                    ],
                },
                "images": refs,
                "source": {
                    "episode_dir": str(Path(root).resolve()),
                    "robot": episode["robot"],
                    "model": episode["model"],
                    "turn": index,
                    "original_code": turn["code"],
                },
                "audit": {
                    **audit,
                    "namespace_prefix_cells": index - 1,
                    "native_verdict": verdict,
                },
            }
        )
    return rows, images


def _write_images(output: Path, images: dict[str, bytes]) -> None:
    """Write content-addressed images; refuse a path holding different bytes."""
    for path, data in images.items():
        target = output / path
        require(
            not target.exists() or target.read_bytes() == data,
            "output image collision",
        )
    for path, data in images.items():
        target = output / path
        target.parent.mkdir(exist_ok=True)
        target.write_bytes(data)


def export(
    sources: Sequence[Path | str] | Path | str,
    output: Path | str,
    methods: Collection[str] | None = None,
    *,
    allow_unsuccessful: bool = False,
) -> dict[str, Any]:
    """Export validated episodes to ``rows.jsonl`` plus ``export-audit.json``.

    Raises ``TraceError`` if there are no sources, discovery finds no episodes,
    ``methods`` is invalid, or the output overlaps a source, and
    ``FileExistsError`` if ``output`` already exists.

    Args:
        sources: Episode directory, parent directory to search recursively, or
            sequence of such paths.
        output: New directory, outside every source, for the dataset.
        methods: Fallback public ``robo`` method names used when an episode's
            ``episode.json`` omits ``robo_methods``.
        allow_unsuccessful: Keep episodes without an environment-confirmed
            success (marked with ``environment_success`` set to ``False``).

    Returns:
        The audit record that was written to ``export-audit.json``.
    """
    fallback_methods = (
        validate_robo_methods(methods, "fallback robo_methods")
        if methods is not None
        else None
    )
    raw_sources = [sources] if isinstance(sources, (Path, str)) else list(sources)
    require(raw_sources, "no source episodes")
    input_roots = [Path(s).expanduser().resolve() for s in raw_sources]
    output = Path(output).expanduser().resolve()
    require(
        not any(output.is_relative_to(root) for root in input_roots),
        "output must be outside source traces",
    )
    roots = discover_episodes(input_roots)
    output.mkdir(parents=True, exist_ok=False)
    rows: list[dict[str, Any]] = []
    quarantine: list[dict[str, str]] = []
    seen: set[str] = set()
    for root in roots:
        try:
            episode_rows, images = validate_episode(
                root, fallback_methods, allow_unsuccessful=allow_unsuccessful
            )
            identity = episode_rows[0]["episode_id"]
            require(
                identity not in seen,
                "duplicate episode; refusing to inflate dataset",
            )
            _write_images(output, images)
        except (TraceError, OSError) as exc:
            quarantine.append({"episode": str(root), "reason": str(exc)})
            continue
        seen.add(identity)
        rows.extend(episode_rows)
    encoded = "".join(
        json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows
    ).encode("utf-8")
    (output / "rows.jsonl").write_bytes(encoded)
    audit = {
        "schema": SCHEMA,
        "trace_schema": TRACE_SCHEMA,
        "projection": PROJECTION,
        "accepted_episodes": len(seen),
        "rows": len(rows),
        "successful_rows": sum(r["environment_success"] is True for r in rows),
        "unsuccessful_rows": sum(r["environment_success"] is False for r in rows),
        "quarantined_episodes": quarantine,
        "rows_sha256": sha256_hex(encoded),
        "split_policy": SPLIT_POLICY,
    }
    (output / "export-audit.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )
    return audit
