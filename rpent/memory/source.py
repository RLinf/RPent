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

"""Shared Hugging Face memory source selection and replayable source records."""

from __future__ import annotations

import json
import re
from argparse import Namespace
from pathlib import Path
from urllib.parse import urlsplit

from rpent.evaluation import write_json_atomic


def parse_repo(value: str) -> tuple[str, str]:
    """Parse owner/repo or https://huggingface.co/datasets/owner/repo@revision."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError("memory repository must not be empty")
    if value.startswith("https://"):
        url = urlsplit(value)
        if url.netloc != "huggingface.co" or url.query or url.fragment:
            raise ValueError("memory repository must be a Hugging Face dataset URL")
        if not url.path.startswith("/datasets/"):
            raise ValueError("memory repository URL must start with /datasets/")
        value = url.path.removeprefix("/datasets/")
    repo, separator, revision = value.partition("@")
    if not re.fullmatch(r"[\w.-]+/[\w.-]+", repo) or any(
        p in (".", "..") for p in repo.split("/")
    ):
        raise ValueError(
            "expected owner/repo or a Hugging Face dataset URL with optional @revision"
        )
    if separator and (not revision.strip() or revision != revision.strip()):
        raise ValueError(
            "memory revision must not be empty or contain surrounding whitespace"
        )
    return repo, revision if separator else "main"


def read_source(path: str | Path, robot: str) -> dict:
    """Read a compact pinned HF source; reject local directories and mutable refs."""
    try:
        data = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
        if (
            not isinstance(data, dict)
            or data.get("schema_version") != 2
            or data.get("profile") != "hf"
        ):
            raise ValueError("expected a schema-version-2 HF memory source record")
        if data.get("robot") != robot:
            raise ValueError("memory source belongs to another robot")
        repo, commit = parse_repo(data["source"])
        if not re.fullmatch(r"[0-9a-f]{40}", commit):
            raise ValueError("memory source must contain a full immutable commit SHA")
        return {**data, "repository": repo, "resolved_commit": commit}
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ValueError(f"Cannot read memory source record: {exc}") from exc


def validate_source_options(args: Namespace, robot: str) -> None:
    """Validate explicit source flags before CLI or Dashboard services start."""
    repo = getattr(args, "memory_repo", None)
    source = getattr(args, "memory_source", None)
    if repo is None and source is None:
        return
    if (
        getattr(args, "explore", False)
        or getattr(args, "memory_profile", None) == "local"
    ):
        raise ValueError("--memory-repo and --memory-source require HF evaluation")
    if repo is not None and source is not None:
        raise ValueError("choose either --memory-repo or --memory-source")
    if getattr(args, "memory_revision", None):
        raise ValueError(
            "use @revision in --memory-repo or the commit from --memory-source"
        )
    if source is not None:
        data = read_source(source, robot)
        if getattr(args, "memory_version", "auto") != "auto":
            raise ValueError("--memory-source already selects the corpus version")
        if robot == "libero" and not isinstance(data.get("memory_version"), str):
            raise ValueError("LIBERO source record must specify memory_version")
    else:
        parse_repo(repo)


def selected_source(args: Namespace, robot: str) -> tuple[str, str, str | None] | None:
    """Return explicit repository, revision and optional corpus version."""
    validate_source_options(args, robot)
    if getattr(args, "memory_source", None):
        data = read_source(args.memory_source, robot)
        return data["repository"], data["resolved_commit"], data.get("memory_version")
    if getattr(args, "memory_repo", None):
        repo, revision = parse_repo(args.memory_repo)
        return repo, revision, None
    return None


def source_record(
    repo: str, commit: str, robot: str, *, memory_version: str | None = None
) -> dict:
    """Build a small portable identity, without duplicating cache file manifests."""
    record = {
        "schema_version": 2,
        "profile": "hf",
        "robot": robot,
        "source": f"https://huggingface.co/datasets/{repo}@{commit}",
    }
    if memory_version is not None:
        record["memory_version"] = memory_version
    return record


def prepare_explicit_source(args: Namespace, robot: str, output_dir: Path) -> Path:
    """Select a robot subtree from an exact Hub snapshot; default sync is unchanged."""
    from huggingface_hub import snapshot_download

    selection = selected_source(args, robot)
    if selection is None:
        raise ValueError("No explicit memory source selected")
    repo, revision, _ = selection
    snapshot = Path(
        snapshot_download(
            repo_id=repo,
            repo_type="dataset",
            revision=revision,
            allow_patterns=[f"{robot}/**"],
        )
    )
    commit = snapshot.name
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("Hub memory snapshot did not resolve to a commit directory")
    if re.fullmatch(r"[0-9a-fA-F]{40}", revision) and revision.lower() != commit:
        raise ValueError("Hub memory snapshot does not match the requested commit")
    root = snapshot / robot
    if not root.is_dir() or not any(root.rglob("*")):
        raise ValueError(f"Memory source has no {robot}/ corpus")
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(
        output_dir / "memory_source.json", source_record(repo, commit, robot)
    )
    return root
