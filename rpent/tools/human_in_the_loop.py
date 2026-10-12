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

"""Human feedback for attended real-robot exploration, separate from steering."""

from __future__ import annotations

import queue
import select
import sys
import threading
import uuid
from collections.abc import Callable

from rpent.utils.logging import get_logger

logger = get_logger("human_in_the_loop")


class HumanInTheLoopInput:
    """Use the existing interactive reader, or read an otherwise unowned TTY.

    Interactive replies include a request ID so old steering messages and late
    confirmations cannot authorize a different reset. No robot code reads stdin.
    """

    help_text = """Human-interactive exploration commands:
    /done             Confirm the pending scene reset.
    /continue         Continue from a pending operator verdict.
    /success          Finish successfully and save exploration memory.
    /failure          Finish with a failure record.
    /abort            Abort exploration without publishing success memory.
    Words without / remain normal messages to the agent.

"""

    def __init__(
        self, *, interactive: bool, on_change: Callable[[], None] | None = None
    ) -> None:
        self._on_change = on_change
        self._pending_prompt = ""
        self.interactive = interactive
        self._lock = threading.Lock()
        self._pending: tuple[str, queue.Queue[str | None]] | None = None
        self._pending_kind: str | None = None
        self._closed = False
        self._verdict_handler: Callable[[str], bool] | None = None

    def bind_verdict(self, handler: Callable[[str], bool] | None) -> None:
        """Bind program-level verdict control for the active exploration session."""
        with self._lock:
            self._verdict_handler = handler

    def route_line(self, line: str) -> bool:
        """Consume operator commands; return False for ordinary planner input."""
        command = line.strip().lower()
        if command in {"/done", "/continue"}:
            expected = "reset" if command == "/done" else "verdict"
            with self._lock:
                if self._pending is None or self._pending_kind != expected:
                    logger.warning(
                        "%s refused: no pending %s request.", command, expected
                    )
                else:
                    self._submit_pending_locked(command[1:])
                    logger.info("%s accepted.", command)
            return True
        if command in {"/success", "/failure", "/abort"}:
            verdict = command[1:]
            with self._lock:
                handler = self._verdict_handler
            if handler is None or not handler(verdict):
                logger.warning(
                    "/%s refused: no eligible exploration attempt, or a verdict is already closing the run.",
                    verdict,
                )
            return True
        if line.split(maxsplit=1)[:1] != ["/operator"]:
            return False
        parts = line.split(maxsplit=2)
        with self._lock:
            pending = self._pending
            if pending is None or len(parts) != 3 or parts[1] != pending[0]:
                logger.warning(
                    "No matching operator request; use /operator <request-id> <answer>."
                )
            else:
                self._submit_pending_locked(parts[2])
        return True

    def _submit_pending_locked(self, answer: str) -> None:
        """Submit to an existing pending request; caller must hold ``_lock``."""
        self._pending[1].put(answer)
        self._pending = None
        self._pending_kind = None

    def snapshot(self) -> dict | None:
        """Return the pending request for the Web transport."""
        with self._lock:
            if self._pending is None:
                return None
            return {
                "id": self._pending[0],
                "kind": self._pending_kind,
                "prompt": self._pending_prompt,
            }

    def reply(self, request_id: str, answer: str) -> None:
        """Accept a single Web reply to the matching pending request."""
        with self._lock:
            if self._closed or self._pending is None or request_id != self._pending[0]:
                raise ValueError("expired or already answered operator request")
            choices = (
                {"done", "abort"}
                if self._pending_kind == "reset"
                else {"success", "failure", "continue", "abort"}
            )
            if answer not in choices:
                raise ValueError("invalid operator response")
            self._submit_pending_locked(answer)
        self._notify_change()

    def _notify_change(self) -> None:
        if self._on_change is not None:
            self._on_change()

    def close(self) -> None:
        """Release pending input and detach the active verdict handler."""
        with self._lock:
            self._closed = True
            self._verdict_handler = None
            self._pending_kind = None
            if self._pending is not None:
                self._pending[1].put(None)
                self._pending = None

        self._notify_change()

    def __call__(self, prompt: str, check_cancelled: Callable[[], None]) -> str | None:
        return self.request(prompt, check_cancelled)

    def request(
        self,
        prompt: str,
        check_cancelled: Callable[[], None],
        *,
        kind: str | None = None,
    ) -> str | None:
        """Wait for typed operator feedback; shortcuts apply only to this request."""
        if not self.interactive:
            if sys.stdin is None or not sys.stdin.isatty():
                return None
            print(prompt, flush=True)
            while True:
                check_cancelled()
                if self._closed:
                    return None
                readable, _, _ = select.select([sys.stdin], [], [], 0.1)
                if readable:
                    line = sys.stdin.readline()
                    return line.strip() if line else None
        request_id = uuid.uuid4().hex[:12]
        replies: queue.Queue[str | None] = queue.Queue()
        with self._lock:
            if self._closed:
                return None
            if self._pending is not None:
                raise RuntimeError("another operator request is pending")
            self._pending = (request_id, replies)
            self._pending_kind = kind
            self._pending_prompt = prompt
        shortcut = (
            "/done" if kind == "reset" else "/continue" if kind == "verdict" else None
        )
        hint = (
            f"\nShortcut: {shortcut}; /success, /failure or /abort ends exploration."
            if shortcut
            else ""
        )
        if self._on_change is None:
            print(
                f"\n{prompt}\nReply: /operator {request_id} <answer>{hint}", flush=True
            )
        try:
            self._notify_change()
            while True:
                check_cancelled()
                try:
                    return replies.get(timeout=0.1)
                except queue.Empty:
                    pass
        finally:
            with self._lock:
                if self._pending is not None and self._pending[0] == request_id:
                    self._pending = None
                    self._pending_kind = None
            self._notify_change()
