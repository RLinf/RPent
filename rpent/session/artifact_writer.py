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

"""Bounded background writes for session observations and recordings."""

from __future__ import annotations

import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from rpent.utils.logging import get_logger

logger = get_logger("artifact_writer")

ArtifactValue = np.ndarray | bytes | tuple[np.ndarray, ...]


def _snapshot(value: ArtifactValue) -> ArtifactValue:
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, tuple):
        return tuple(frame.copy() for frame in value)
    return value


@dataclass
class PendingArtifact:
    """Own a detached payload and share PNG encoding between readers and writer."""

    value: ArtifactValue
    derived: bool = False
    _encoded: bytes | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def png_bytes(self) -> bytes:
        """Encode once even if planner feedback races with the worker."""
        with self._lock:
            if self._encoded is None:
                self._encoded = imageio.imwrite(
                    "<bytes>", np.asarray(self.value, dtype=np.uint8), format="PNG"
                )
            return self._encoded


class ArtifactWriter:
    """Own one writer and bound the queued snapshots by bytes and item count."""

    def __init__(self, max_bytes: int, *, max_items: int = 64) -> None:
        self.max_bytes = max_bytes
        self._max_items = max_items
        self._condition = threading.Condition()
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="artifact"
        )
        self._pending: dict[Path, PendingArtifact] = {}
        self._failures: dict[Path, str] = {}
        self._buffered_bytes = 0

    def submit(
        self,
        path: Path,
        value: ArtifactValue,
        size: int,
        write: Callable[[Path, PendingArtifact], None],
        *,
        derived: bool = False,
    ) -> None:
        """Reserve space before copying and dispatching a payload."""
        with self._condition:
            self._condition.wait_for(
                lambda: (
                    len(self._pending) < self._max_items
                    and self._buffered_bytes + size <= self.max_bytes
                )
            )
            pending = PendingArtifact(_snapshot(value), derived=derived)
            self._executor.submit(self._write, path, pending, size, write)
            self._pending[path] = pending
            self._buffered_bytes += size
            self._failures.pop(path, None)

    def _write(
        self,
        path: Path,
        pending: PendingArtifact,
        size: int,
        write: Callable[[Path, PendingArtifact], None],
    ) -> None:
        error = None
        try:
            write(path, pending)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            logger.warning("failed to write artifact %s: %s", path, error)
        finally:
            with self._condition:
                if error is not None:
                    self._failures[path] = error
                del self._pending[path]
                self._buffered_bytes -= size
                self._condition.notify_all()

    def pending(self, path: Path) -> PendingArtifact | None:
        """Read a pending snapshot, or validate fallback to the file."""
        with self._condition:
            self._check(path)
            return self._pending.get(path)

    def wait(self, path: Path) -> None:
        """Wait for publication of one file and surface any write failure."""
        with self._condition:
            self._condition.wait_for(lambda: path not in self._pending)
            self._check(path)

    def discard(self, path: Path) -> None:
        """Drain a write before deleting or overwriting its file."""
        with self._condition:
            self._condition.wait_for(lambda: path not in self._pending)
            self._failures.pop(path, None)

    def _check(self, path: Path) -> None:
        if path in self._failures:
            raise FileNotFoundError(
                f"artifact write failed: {path}: {self._failures[path]}"
            )

    def flush(self) -> None:
        """Wait for all submitted writes."""
        with self._condition:
            self._condition.wait_for(lambda: not self._pending)

    def close(self) -> None:
        """Drain the queue and join the writer thread."""
        self._executor.shutdown(wait=True)

    def failures(self) -> dict[str, str]:
        """Return write failures by path for the run report."""
        with self._condition:
            return {str(path): error for path, error in self._failures.items()}
