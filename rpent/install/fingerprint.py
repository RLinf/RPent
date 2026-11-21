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

"""Read, write, and hash the installation fingerprint stored in a venv."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

FINGERPRINT_NAME = "rpent-install.json"
#: The installed package versions recorded at install time, next to the
#: fingerprint. The fingerprint stores their hash; this file stores the listing
#: it hashes, so the versions an environment actually got stay readable after
#: the environment itself changes.
FREEZE_NAME = "rpent-install.freeze.txt"
#: The fingerprint key recording which installation inputs were hashed. It is
#: named apart from ``schema``, which versions the manifest layout.
FINGERPRINT_SCHEMA_FIELD = "fingerprint_schema"


def fingerprint_path(venv: Path) -> Path:
    """Return the fingerprint path for ``venv``."""
    return Path(venv) / FINGERPRINT_NAME


def freeze_path(venv: Path) -> Path:
    """Return the recorded package listing path for ``venv``.

    The listing is a readable record of what the install resolved; the
    fingerprint's ``freeze_sha256`` is what ``doctor`` compares, so editing the
    listing does not make an environment count as unchanged.
    """
    return Path(venv) / FREEZE_NAME


def _write_atomic(destination: Path, text: str) -> None:
    """Write ``text`` to ``destination`` without leaving a partial file."""
    temporary = destination.with_name(f".{destination.name}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as file:
            file.write(text)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def read_fingerprint(venv: Path) -> dict[str, Any] | None:
    """Return the venv's fingerprint, or ``None`` when it has none."""
    path = fingerprint_path(venv)
    try:
        with path.open(encoding="utf-8") as file:
            data = json.load(file)
    except FileNotFoundError:
        return None
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def write_fingerprint(venv: Path, fingerprint: dict[str, Any]) -> Path:
    """Write ``fingerprint`` atomically into ``venv`` and return its path."""
    destination = fingerprint_path(venv)
    _write_atomic(destination, json.dumps(fingerprint, indent=2, sort_keys=True) + "\n")
    return destination


def write_freeze(venv: Path, frozen: str) -> Path:
    """Record ``uv pip freeze``'s listing beside the fingerprint and return it."""
    destination = freeze_path(venv)
    _write_atomic(destination, frozen if frozen.endswith("\n") else frozen + "\n")
    return destination


def fingerprint_sha256(fingerprint: dict[str, Any]) -> str:
    """Return the canonical hash referenced by run artifacts."""
    payload = json.dumps(fingerprint, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def freeze_sha256(frozen: str) -> str:
    """Hash a ``uv pip freeze`` listing, ignoring its ordering."""
    lines = sorted(line.strip() for line in frozen.splitlines() if line.strip())
    payload = "\n".join(lines) + "\n"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def source_state(root: Path) -> tuple[str | None, bool]:
    """Return the checkout's commit and whether tracked files are modified."""
    try:
        commit = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return None, False
    return commit or None, bool(status.strip())


def runtime_platform() -> str:
    """Return this interpreter's platform in manifest notation."""
    system = platform.system().lower()
    machine = platform.machine().lower()
    if system == "linux":
        return "linux-" + ("aarch64" if machine in {"aarch64", "arm64"} else "x86_64")
    if system == "darwin":
        return "darwin-" + ("arm64" if machine in {"aarch64", "arm64"} else "x86_64")
    return f"{system}-{machine}"


def current_venv() -> Path | None:
    """Return the active virtual environment, or ``None``."""
    prefix = Path(sys.prefix)
    if (prefix / "pyvenv.cfg").is_file():
        return prefix
    return None
