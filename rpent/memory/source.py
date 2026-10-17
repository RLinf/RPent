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

import fcntl
import hashlib
import re
import shutil
import tempfile
from pathlib import Path
from urllib.parse import urlsplit

from rpent.utils.config import get_memory_dir


def parse_repo(value: str) -> tuple[str, str]:
    """Parse owner/repo or https://huggingface.co/datasets/owner/repo@revision."""
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


def prepare_explicit_source(
    source: str,
    robot: str,
    *,
    version: str | None = None,
    cache_dir: Path | None = None,
) -> Path:
    """Materialize a robot subtree without exposing Hub cache symlinks.

    Copies are shared across runs of the same repository and resolved commit.
    Robot-specific layout validation remains with the caller.
    """
    from huggingface_hub import snapshot_download

    if version is not None and (
        not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", version)
        or version in (".", "..")
    ):
        raise ValueError("memory version must be a single corpus directory name")
    prefix = f"{robot}/{version}" if version else robot
    repo, revision = parse_repo(source)
    snapshot = Path(
        snapshot_download(
            repo_id=repo,
            repo_type="dataset",
            revision=revision,
            allow_patterns=[f"{prefix}/**"],
        )
    )
    root = snapshot / prefix
    if not root.is_dir() or not any(root.rglob("*")):
        raise ValueError(f"Memory source has no {prefix}/ corpus")
    key = hashlib.sha256(repo.encode()).hexdigest()[:20]
    cache = (cache_dir or get_memory_dir(robot) / ".sources") / key
    cache.mkdir(parents=True, exist_ok=True)
    relative = Path("versioned") / prefix if version else Path(robot)
    destination = cache / snapshot.name / relative
    with (cache / "sync.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not destination.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(dir=destination.parent) as staging:
                corpus = Path(staging) / robot
                # Dereference Hub's snapshot-to-blob links so memory permissions
                # apply to the actual files and writes cannot alter shared blobs.
                shutil.copytree(root, corpus, symlinks=False)
                corpus.replace(destination)
    return destination
