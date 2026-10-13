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


class InputCancelled(Exception):
    """This consumer closed; the shared terminal did not reach EOF."""


class SessionInputQueue:
    """Cancel one input consumer without putting EOF in the shared queue.

    A scope may wrap another scope, allowing a planner reply to end without
    cancelling its attended attempt or consuming the next attempt's input.
    Terminal adapters retain reads until backend acceptance. Cancel stops new
    reads; close returns unaccepted input after the adapter settles any in-flight
    submission, including a read still returning from a worker thread.
    """

    def __init__(self, source: InputQueue, *, retain_until_ack: bool = False) -> None:
        """Wrap a source queue without taking ownership of its lifetime."""
        self._source = source
        self._cancelled = threading.Event()
        self._lock = threading.Lock()
        self._retain_until_ack = retain_until_ack
        self._pending: list[str | None] = []

    @property
    def cancelled(self) -> bool:
        """Return whether this scope has been cancelled."""
        return self._cancelled.is_set()

    def cancel(self) -> None:
        """Stop new reads without releasing input whose acceptance is unresolved."""
        self._cancelled.set()

    def close(self) -> None:
        """Return unaccepted input after the consumer has stopped submitting."""
        with self._lock:
            self._cancelled.set()
            for item in reversed(self._pending):
                self._return_unread(item)
            self._pending.clear()

    def acknowledge(self) -> None:
        """Release input after the backend successfully accepts it."""
        with self._lock:
            self._pending.clear()

    def _return_unread(self, item: str | None) -> None:
        """Return unaccepted input ahead of newer lines, through nested scopes."""
        if isinstance(self._source, SessionInputQueue):
            self._source._return_unread(item)
        else:
            with self._source.mutex:
                self._source.queue.appendleft(item)
                self._source.not_empty.notify()

    def get(self, block: bool = True) -> str | None:
        """Read input; raise InputCancelled on scope cancellation, preserving real EOF."""
        while True:
            with self._lock:
                if self.cancelled:
                    raise InputCancelled
                try:
                    item = self._source.get(block=False)
                    if self._retain_until_ack:
                        self._pending.append(item)
                    return item
                except queue.Empty:
                    if not block:
                        raise
            self._cancelled.wait(0.1)

    def put(self, item: str | None) -> None:
        """Forward real input to the shared source."""
        self._source.put(item)


InputQueue = queue.Queue[str | None] | SessionInputQueue
