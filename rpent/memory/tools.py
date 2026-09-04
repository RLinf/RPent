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

"""Memory-aware file-tool handlers for the common tools.

These wrap the shared IO in :mod:`rpent.tools.common` with two access
checks: robot isolation and run-mode permission. Each handler
only validates the path and checks access, then delegates the actual IO.
"""

from __future__ import annotations

from pathlib import Path

from rpent.tools.toolkit import readonly
from rpent.utils.config import get_repo_root

# Published memory subtrees an eval run may read. ``results`` retains
# compatibility with the immutable RoboCasa task-only corpus.
_READABLE_SCOPES = {"global", "suite", "task_only", "results"}

# Suffix appended to shared file-tool descriptions when registered through
# the memory access boundary.
MEMORY_BOUNDARY_NOTE = (
    " Published memory is read-only. During exploration, you may write only "
    "to your current memory inbox. Memory for other robots is unavailable."
)


def _resolve_memory_path(path: str) -> Path:
    """Resolve a file-tool path before memory boundary checks."""
    p = Path(path)
    if not p.is_absolute():
        p = get_repo_root() / p
    return p.resolve()


def _classify_memory_path(resolved: Path, *, memory_root: Path) -> str:
    """Classify a resolved path as current, foreign, or non_memory."""
    if resolved.is_relative_to(memory_root):
        return "current"
    memory_namespace = get_repo_root() / "memory"
    if resolved.is_relative_to(memory_namespace):
        return "foreign"
    return "non_memory"


def _check_memory_access(
    path: str,
    *,
    memory_root: Path,
    access: str,
    memory_access: str,
    cell_tag: str | None,
) -> None:
    """Enforce current-memory read/write permissions.

    Other robots' memory is always rejected. Within the current robot's memory,
    read_only exposes published subtrees read-only and rejects writes;
    inbox_write also allows the current cell's inbox. Empty and non-memory
    paths are skipped.
    """
    if not path:
        return
    resolved = _resolve_memory_path(path)
    bucket = _classify_memory_path(resolved, memory_root=memory_root)
    if bucket == "non_memory":
        return
    if bucket == "foreign":
        raise PermissionError(f"access to another robot's memory is denied: {path}")
    parts = resolved.relative_to(memory_root).parts
    if not parts:
        if access == "read":
            return
        raise PermissionError(f"writing to memory is denied in this mode: {path}")

    top = parts[0]
    own_inbox = (
        len(parts) >= 3
        and parts[0] == "_internal"
        and parts[1] == "inbox"
        and parts[2] == cell_tag
    )

    if access == "write":
        if memory_access == "inbox_write" and own_inbox:
            return
        raise PermissionError(f"writing to memory is denied in this mode: {path}")

    if top in _READABLE_SCOPES or (len(parts) == 1 and top.endswith(".md")):
        return
    if memory_access == "inbox_write" and own_inbox:
        return
    raise PermissionError(f"reading this memory path is denied: {path}")


@readonly
def read_text_file(
    path: str,
    *,
    memory_root: Path,
    memory_access: str,
    cell_tag: str | None,
    max_chars: int = 40000,
) -> dict:
    _check_memory_access(
        path,
        memory_root=memory_root,
        access="read",
        memory_access=memory_access,
        cell_tag=cell_tag,
    )
    from rpent.tools import common

    return common.read_text_file(path, max_chars)


@readonly
def write_text_file(
    path: str,
    content: str,
    *,
    memory_root: Path,
    memory_access: str,
    cell_tag: str | None,
) -> dict:
    _check_memory_access(
        path,
        memory_root=memory_root,
        access="write",
        memory_access=memory_access,
        cell_tag=cell_tag,
    )
    from rpent.tools import common

    return common.write_text_file(path, content)


@readonly
def list_dir(
    path: str = "",
    *,
    memory_root: Path,
    memory_access: str,
    cell_tag: str | None,
) -> dict:
    _check_memory_access(
        path,
        memory_root=memory_root,
        access="read",
        memory_access=memory_access,
        cell_tag=cell_tag,
    )
    from rpent.tools import common

    return common.list_dir(path)
