# Copyright 2026 The RPent Authors.
"""Deterministic source fingerprints for scripted releases."""

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path


def sha256_file(path: str | Path) -> str:
    """Hash file bytes without loading the whole file into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_sources(paths: Iterable[str | Path]) -> str:
    """Hash a sorted mapping of absolute source paths to content digests."""
    sources = {str(Path(p).resolve()): sha256_file(p) for p in paths}
    return hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest()
