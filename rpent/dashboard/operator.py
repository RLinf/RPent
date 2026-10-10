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

"""Cancelable, request-scoped operator confirmations for physical tasks."""

from __future__ import annotations

import threading
import uuid
from collections.abc import Callable
from typing import Any

from rpent.tools.toolkit import ToolCancelled


class DashboardOperator:
    """Own one pending confirmation for the active Dashboard task."""

    def __init__(self, on_change: Callable[[], None] = lambda: None) -> None:
        """Notify on_change when a confirmation opens or closes."""
        self._condition = threading.Condition()
        self._pending: dict[str, Any] | None = None
        self._answer: str | None = None
        self._closed = False
        self._on_change = on_change

    def snapshot(self) -> dict[str, Any] | None:
        """Return the pending request, or None when no confirmation is pending."""
        with self._condition:
            return dict(self._pending) if self._pending else None

    def request(
        self,
        prompt: str,
        check_cancelled: Callable[[], None],
        *,
        kind: str | None = None,
    ) -> str:
        """Wait for an operator reply while checking cancellation.

        Args:
            prompt: Instructions shown to the operator.
            check_cancelled: Callback that raises when the operation is cancelled.
            kind: ``reset`` permits done/abort; other kinds permit task verdicts.

        Returns:
            The accepted answer followed by any operator notes.

        Raises:
            ToolCancelled: The channel closes or the callback reports cancellation.
            RuntimeError: Another confirmation is already pending.
        """
        choices = (
            ["done", "abort"]
            if kind == "reset"
            else ["success", "failure", "continue", "abort"]
        )
        with self._condition:
            if self._closed:
                raise ToolCancelled("operator channel closed")
            if self._pending:
                raise RuntimeError("another operator confirmation is pending")
            self._pending = {
                "id": uuid.uuid4().hex,
                "kind": kind,
                "prompt": prompt,
                "choices": choices,
            }
            self._answer = None
        self._on_change()
        try:
            with self._condition:
                while self._answer is None:
                    check_cancelled()
                    if self._closed:
                        raise ToolCancelled("operator channel closed")
                    self._condition.wait(0.1)
                check_cancelled()
                return self._answer
        finally:
            with self._condition:
                self._pending = None
                self._answer = None
            self._on_change()

    def reply(self, request_id: str, answer: str, notes: str = "") -> None:
        """Answer the matching pending request once.

        Args:
            request_id: Identifier from the pending request snapshot.
            answer: One of the choices exposed by that request.
            notes: Optional instructions or evidence accompanying the answer.

        Raises:
            ValueError: The request is expired, answered, or the reply is invalid.
        """
        with self._condition:
            if (
                self._closed
                or not self._pending
                or request_id != self._pending["id"]
                or self._answer is not None
            ):
                raise ValueError("expired or already answered operator request")
            if answer not in self._pending["choices"] or not isinstance(notes, str):
                raise ValueError("invalid operator response")
            self._answer = answer + (" " + notes.strip() if notes.strip() else "")
            self._condition.notify_all()

    def close(self) -> None:
        """Close the channel and wake waiting callers without confirming anything."""
        with self._condition:
            self._closed = True
            self._condition.notify_all()
