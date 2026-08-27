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

"""Publish one exploration inbox into a layered memory corpus.

Exploration agents write drafts under ``_inbox/<cell>/``.  This module is the
trusted publication boundary: it validates frontmatter, serializes concurrent
publishers, merges evidence without overwriting conflicting prose, copies only
ground-truth-backed task audit/recipe pairs, and rebuilds ``MEMORY.md``.
Generated memory remains external data under ``resources/``; this code is part
of the installable RPent package.
"""

from __future__ import annotations

import fcntl
import json
import re
import shutil
from pathlib import Path
from typing import Any

import yaml

SCOPES = {"global", "suite"}
KINDS = {"primitive", "perception", "strategy", "failure", "infra"}
CONFIDENCE = {"single-shot", "probable", "verified"}
_PREFIXES = ("new_global_", "new_suite_", "new_", "suite_")
_KINDS = tuple(f"{kind}_" for kind in sorted(KINDS))


def _split_frontmatter(path: Path) -> tuple[dict[str, Any], str]:
    text = path.read_text(errors="replace")
    if not text.startswith("---"):
        raise ValueError("missing YAML frontmatter")
    end = text.find("\n---", 3)
    if end < 0:
        raise ValueError("unterminated YAML frontmatter")
    metadata = yaml.safe_load(text[3:end])
    if not isinstance(metadata, dict):
        raise ValueError("frontmatter must be a mapping")
    return metadata, text[end + 4 :]


def _validate(metadata: dict[str, Any]) -> None:
    scope = metadata.get("scope")
    if scope not in SCOPES:
        raise ValueError(f"scope must be one of {sorted(SCOPES)}, got {scope!r}")
    if scope == "suite":
        for field in ("suite", "regime", "task_id", "task_language"):
            if metadata.get(field) in (None, ""):
                raise ValueError(f"suite memory requires {field!r}")
    else:
        if metadata.get("kind") not in KINDS:
            raise ValueError(f"kind must be one of {sorted(KINDS)}")
        for field in ("title", "applies_when"):
            if not str(metadata.get(field, "")).strip():
                raise ValueError(f"global memory requires {field!r}")
    if metadata.get("confidence") not in CONFIDENCE:
        raise ValueError(f"confidence must be one of {sorted(CONFIDENCE)}")
    evidence = metadata.get("evidence") or {}
    if not isinstance(evidence, dict):
        raise ValueError("evidence must be a mapping")
    if not isinstance(evidence.get("cells"), list) or not evidence["cells"]:
        raise ValueError("evidence.cells must be a non-empty list")


def _canonical_id(path: Path, metadata: dict[str, Any]) -> str:
    if metadata["scope"] == "suite":
        return f"suite_{metadata['suite']}_{metadata['regime']}_t{metadata['task_id']}"
    stem = re.sub(r"_draft$", "", path.stem)
    for prefix in (*_PREFIXES, *_KINDS):
        if stem.startswith(prefix):
            stem = stem[len(prefix) :]
            break
    return str(metadata.get("id") or stem).strip()


def _render(metadata: dict[str, Any], body: str) -> str:
    return (
        "---\n"
        + yaml.safe_dump(metadata, sort_keys=False, allow_unicode=True)
        + "---\n"
        + body
    )


def _merge_evidence(old: dict[str, Any], new: dict[str, Any]) -> dict[str, Any]:
    old_evidence = old.get("evidence") or {}
    new_evidence = new.get("evidence") or {}
    cells = sorted({*old_evidence.get("cells", []), *new_evidence.get("cells", [])})
    evidence = {
        **old_evidence,
        "cells": cells,
        "attempts": int(old_evidence.get("attempts") or 0)
        + int(new_evidence.get("attempts") or 0),
    }
    for key in ("solved_seeds", "failed_seeds", "contradicted_by"):
        if key in old_evidence or key in new_evidence:
            evidence[key] = sorted(
                {*old_evidence.get(key, []), *new_evidence.get(key, [])}
            )
    tasks = {str(cell).rsplit("_s", 1)[0] for cell in cells}
    confidence = (
        "verified"
        if len(cells) >= 3 and len(tasks) >= 2
        else "probable"
        if len(cells) >= 2
        else "single-shot"
    )
    return {**old, "evidence": evidence, "confidence": confidence}


def build_index(memory_dir: str | Path) -> Path:
    """Rebuild and return the generated ``MEMORY.md`` path."""
    memory_dir = Path(memory_dir).resolve()
    groups: dict[str, list[tuple[str, dict[str, Any]]]] = {
        "global": [],
        "suite": [],
    }
    for scope in groups:
        for path in sorted((memory_dir / scope).glob("*.md")):
            try:
                metadata, _ = _split_frontmatter(path)
            except ValueError:
                continue
            groups[scope].append((path.name, metadata))
    lines = ["# Layered memory index", "", "Generated from memory leaf frontmatter."]
    for scope, title in (("global", "Global"), ("suite", "Suite")):
        lines.extend(("", f"## {title}", ""))
        if not groups[scope]:
            lines.append("_(none)_")
            continue
        for filename, metadata in groups[scope]:
            label = metadata.get("title") or metadata.get("id") or filename
            applies = metadata.get("applies_when") or ""
            suffix = f" — {applies}" if applies else ""
            lines.append(f"- [{label}]({scope}/{filename}){suffix}")
    index = memory_dir / "MEMORY.md"
    index.parent.mkdir(parents=True, exist_ok=True)
    index.write_text("\n".join(lines) + "\n")
    return index


def validate_corpus(memory_dir: str | Path) -> list[str]:
    """Return validation problems for all global/suite leaves."""
    root = Path(memory_dir).resolve()
    problems: list[str] = []
    ids: dict[str, Path] = {}
    for scope in ("global", "suite"):
        for path in sorted((root / scope).glob("*.md")):
            try:
                metadata, _ = _split_frontmatter(path)
                _validate(metadata)
            except ValueError as exc:
                problems.append(f"{path.relative_to(root)}: {exc}")
                continue
            memory_id = str(metadata.get("id") or "")
            if memory_id != path.stem:
                problems.append(
                    f"{path.relative_to(root)}: id {memory_id!r} does not match filename"
                )
            if memory_id in ids:
                problems.append(
                    f"{path.relative_to(root)}: duplicate id also in "
                    f"{ids[memory_id].relative_to(root)}"
                )
            ids[memory_id] = path
    return problems


def merge_cell(
    *, memory_dir: str | Path, cell_tag: str, output_dir: str | Path
) -> dict[str, Any]:
    """Validate and publish the completed exploration artifacts for one cell."""
    root = Path(memory_dir).resolve()
    run_dir = Path(output_dir).resolve()
    inbox = root / "_inbox" / cell_tag
    conflicts = root / "_conflicts"
    merged = root / "_merged"
    tiers = {scope: root / scope for scope in ("global", "suite", "task")}
    for directory in (*tiers.values(), conflicts, merged, inbox.parent):
        directory.mkdir(parents=True, exist_ok=True)

    result: dict[str, Any] = {
        "cell": cell_tag,
        "global": 0,
        "suite": 0,
        "task": 0,
        "evidence": 0,
        "conflicts": 0,
        "skipped": [],
    }
    lock_path = root / ".merge.lock"
    lock_path.touch(exist_ok=True)
    with lock_path.open("r+") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        published_draft = False
        if inbox.is_dir():
            for source in sorted(inbox.glob("*.md")):
                try:
                    metadata, body = _split_frontmatter(source)
                    _validate(metadata)
                except ValueError as exc:
                    result["skipped"].append(f"{source.name}: {exc}")
                    continue
                memory_id = _canonical_id(source, metadata)
                metadata["id"] = memory_id
                destination = tiers[metadata["scope"]] / f"{memory_id}.md"
                if not destination.exists():
                    destination.write_text(_render(metadata, body))
                    result[metadata["scope"]] += 1
                else:
                    old_metadata, old_body = _split_frontmatter(destination)
                    credited = cell_tag in (
                        (old_metadata.get("evidence") or {}).get("cells") or []
                    )
                    if not credited:
                        merged_metadata = _merge_evidence(old_metadata, metadata)
                        destination.write_text(_render(merged_metadata, old_body))
                        result["evidence"] += 1
                        if body.strip() != old_body.strip():
                            conflict = conflicts / f"{memory_id}__from_{cell_tag}.md"
                            conflict.write_text(_render(metadata, body))
                            result["conflicts"] += 1
                published_draft = True

            if published_draft:
                archive = merged / cell_tag
                if archive.exists():
                    shutil.rmtree(archive)
                shutil.move(str(inbox), str(archive))

        audit = run_dir / f"{cell_tag}.json"
        recipe = run_dir / f"recipe_{cell_tag}.jsonl"
        try:
            solved = json.loads(audit.read_text()).get("libero_terminated") is True
        except (OSError, ValueError):
            solved = False
        if solved and recipe.exists():
            audit_target = tiers["task"] / audit.name
            recipe_target = tiers["task"] / recipe.name
            if not audit_target.exists() and not recipe_target.exists():
                shutil.copy2(audit, audit_target)
                shutil.copy2(recipe, recipe_target)
                result["task"] = 1
            elif audit_target.exists() != recipe_target.exists():
                result["skipped"].append("incomplete existing task audit/recipe pair")

        build_index(root)
    return result
