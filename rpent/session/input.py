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

"""Input queue contracts and cancellation scopes for attended sessions."""

from __future__ import annotations

import queue
import threading
import time
from typing import Protocol


class InputQueue(Protocol):
    """Input operations shared by readers and planner consumers."""

    def get(self, block: bool = True, timeout: float | None = None) -> str | None:
        """Read a submitted line or the source's EOF signal."""
        ...

    def put(self, item: str | None) -> None:
        """Forward a line or the source's EOF signal."""
        ...


class SessionInputQueue:
    """Cancel one input consumer without putting EOF in the shared queue.

    A scope may wrap another scope, allowing a planner reply to end without
    cancelling its attended attempt or consuming the next attempt's input.
    """

    def __init__(self, source: InputQueue) -> None:
        """Wrap a source queue without taking ownership of its lifetime."""
        self._source = source
        self._cancelled = threading.Event()
        self._lock = threading.Lock()

    @property
    def cancelled(self) -> bool:
        """Return whether this scope has been cancelled."""
        return self._cancelled.is_set()

    def cancel(self) -> None:
        """Wake this scope's readers without closing or changing its source."""
        with self._lock:
            self._cancelled.set()

    def get(self, block: bool = True, timeout: float | None = None) -> str | None:
        """Read input, returning None when this scope or its source closes."""
        if block and timeout is not None and timeout < 0:
            raise ValueError("timeout must be a non-negative number")
        deadline = time.monotonic() + timeout if timeout is not None else None
        while True:
            with self._lock:
                if self.cancelled:
                    return None
                try:
                    return self._source.get(block=False)
                except queue.Empty:
                    if not block:
                        raise
            wait_time = 0.1
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise queue.Empty
                wait_time = min(wait_time, remaining)
            self._cancelled.wait(wait_time)

    def put(self, item: str | None) -> None:
        """Forward real input to the source, including returned unread lines."""
        self._source.put(item)
