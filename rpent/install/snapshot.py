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

"""The per-run manifest: what ran, in which environment, and what it wrote.

``<output_dir>/run.json`` is the entry point of a run directory. It names the
run, keeps the installation identity that produced it, lists the artifacts
written beside it, and carries the run verdict. One directory holds one run, so
a second start against a used directory is refused instead of silently mixing
two transcripts.
"""

from __future__ import annotations

import json
import platform
import sys
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rpent.evaluation.result import write_json_atomic

RUN_MANIFEST_NAME = "run.json"
RUN_MANIFEST_SCHEMA = 1


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _runtime_platform() -> str:
    """Return this interpreter's platform in manifest notation."""
    system = platform.system().lower()
    machine = platform.machine().lower()
    if system == "linux":
        return "linux-" + ("aarch64" if machine in {"aarch64", "arm64"} else "x86_64")
    if system == "darwin":
        return "darwin-" + ("arm64" if machine in {"aarch64", "arm64"} else "x86_64")
    return f"{system}-{machine}"


def _current_venv() -> Path | None:
    """Return the active virtual environment, or ``None``."""
    prefix = Path(sys.prefix)
    if (prefix / "pyvenv.cfg").is_file():
        return prefix
    return None


def installation_snapshot(venv: Path | None = None) -> dict[str, Any]:
    """Return the current environment's installation identity.

    A run started from a source checkout records the interpreter and host that
    produced it; the components it holds are recorded once the installer writes
    them, so an unmanaged environment reports none.
    """
    active = Path(venv).resolve() if venv is not None else _current_venv()
    return {
        "targets": None,
        "install_sha256": None,
        "python": platform.python_version(),
        "platform": _runtime_platform(),
        "venv": str(active) if active is not None else None,
        "components": [],
    }


def start_run_manifest(
    output_dir: Path | str,
    *,
    robot: str,
    planner: str,
    model: str | None,
    task: Mapping[str, Any],
    venv: Path | None = None,
) -> Path:
    """Claim ``output_dir`` for one run and record what the run is.

    Raises:
        FileExistsError: The directory already records a run.
    """
    destination = Path(output_dir) / RUN_MANIFEST_NAME
    if destination.exists():
        raise FileExistsError(
            f"{destination} already records a run; choose an empty output directory"
        )
    return write_json_atomic(
        destination,
        {
            "schema": RUN_MANIFEST_SCHEMA,
            "started_at": _utc_now(),
            "robot": robot,
            "planner": planner,
            "model": model,
            "task": dict(task),
            "environment": installation_snapshot(venv),
            "artifacts": {},
        },
    )


def finish_run_manifest(
    output_dir: Path | str,
    *,
    artifacts: Mapping[str, Path | str | None],
    verdict: Mapping[str, Any],
    error: str | None = None,
    artifact_error: str | None = None,
) -> Path:
    """Record the closed run's artifacts, verdict, and failures, if any.

    ``error`` is the failure the verdict explains. ``artifact_error`` is a
    command-level failure to write the run's own artifacts: the verdict still
    reports the run, and a reader sees separately why its record is incomplete.
    """
    destination = Path(output_dir) / RUN_MANIFEST_NAME
    record = json.loads(destination.read_text(encoding="utf-8"))
    record["finished_at"] = _utc_now()
    record["artifacts"] = {name: str(path) for name, path in artifacts.items() if path}
    record.update(verdict)
    record["error"] = error
    record["artifact_error"] = artifact_error
    return write_json_atomic(destination, record)
