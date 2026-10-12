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

"""Shared Hugging Face memory source selection."""

from __future__ import annotations

import re
from argparse import Namespace
from pathlib import Path
from urllib.parse import urlsplit


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


def validate_source_options(args: Namespace) -> None:
    """Validate an explicit source before CLI or Dashboard services start."""
    repo = getattr(args, "memory_repo", None)
    if repo is None:
        return
    if (
        getattr(args, "explore", False)
        or getattr(args, "memory_profile", None) == "local"
    ):
        raise ValueError("--memory-repo requires HF evaluation")
    if getattr(args, "memory_revision", None) is not None:
        raise ValueError("use @revision in --memory-repo instead of --memory-revision")
    parse_repo(repo)


def selected_source(args: Namespace) -> tuple[str, str] | None:
    """Return an explicit repository and revision, if supplied."""
    validate_source_options(args)
    repo = getattr(args, "memory_repo", None)
    return parse_repo(repo) if repo is not None else None


def prepare_explicit_source(args: Namespace, robot: str) -> Path:
    """Select a robot subtree from an exact Hub snapshot; default sync is unchanged."""
    from huggingface_hub import snapshot_download

    selection = selected_source(args)
    if selection is None:
        raise ValueError("No explicit memory source selected")
    repo, revision = selection
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
    return root
