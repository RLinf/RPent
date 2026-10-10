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

import json
import queue
import select
import sys
import threading
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from rpent.utils.logging import get_logger

logger = get_logger("human_in_the_loop")


@dataclass(frozen=True)
class OperatorCommand:
    """Parsed operator input, without granting any scene or motion permission."""

    name: str
    notes: str = ""
    request_id: str | None = None
    answer: str = ""


def parse_operator_command(line: str) -> OperatorCommand | None:
    """Parse the shared terminal/Web commands; ordinary chat returns None."""
    parts = line.strip().split(maxsplit=1)
    command = parts[0].lower() if parts else ""
    if command == "/succes":
        command = "/success"
    if command not in {
        "/done",
        "/continue",
        "/success",
        "/failure",
        "/abort",
        "/operator",
    }:
        return None
    notes = parts[1] if len(parts) > 1 else ""
    if command == "/operator":
        tokens = notes.split(maxsplit=2)
        if len(tokens) < 2:
            raise ValueError("Usage: /operator <request-id> <answer> [notes]")
        return OperatorCommand(
            name="operator",
            request_id=tokens[0],
            answer=tokens[1],
            notes=tokens[2] if len(tokens) > 2 else "",
        )
    return OperatorCommand(name=command[1:], notes=notes)


class HumanInTheLoopInput:
    """Use the existing interactive reader, or read an otherwise unowned TTY.

    Interactive replies include a request ID so old steering messages and late
    confirmations cannot authorize a different reset. No robot code reads stdin.
    """

    help_text = """Human-interactive exploration commands:
    /done [notes]     Confirm the pending scene reset, optionally describing the scene.
    /continue [notes] Continue from a pending verdict with optional instructions.
    /success [notes]  Record this successful attempt and continue in a new session.
    /failure [notes]  Record this failed attempt and continue in a new session.
    /abort [notes]    Abort exploration without publishing success memory.
    Words without / remain normal messages to the agent.

"""

    def __init__(
        self,
        *,
        interactive: bool,
        feedback_path: Path | None = None,
        on_change: Callable[[], None] | None = None,
    ) -> None:
        self.interactive = interactive
        self._feedback_path = feedback_path
        self._feedback: list[str] = []
        self._lock = threading.Lock()
        self._pending: tuple[str, queue.Queue[str | None]] | None = None
        self._pending_kind: str | None = None
        self._pending_prompt = ""
        self._on_change = on_change
        self._closed = False
        self._verdict_handler: Callable[[str, str], bool] | None = None

    def bind_verdict(self, handler: Callable[[str, str], bool] | None) -> None:
        """Bind program-level verdict control for the active exploration session."""
        with self._lock:
            self._verdict_handler = handler

    def route_line(self, line: str) -> bool:
        """Consume operator commands; return False for ordinary planner input."""
        try:
            parsed = parse_operator_command(line)
        except ValueError as exc:
            logger.warning("%s", exc)
            return True
        if parsed is None:
            if line.strip() and not line.startswith("/"):
                with self._lock:
                    self._record_feedback(line.strip())
            return False
        command = f"/{parsed.name}"
        notes = parsed.notes
        if command in {"/done", "/continue"}:
            expected = "reset" if command == "/done" else "verdict"
            with self._lock:
                if command == "/continue" and self._pending is None:
                    if notes:
                        self._record_feedback(notes)
                    return False
                request_id = (
                    self._pending[0]
                    if self._pending is not None and self._pending_kind == expected
                    else ""
                )
            try:
                self.reply(request_id, command[1:], notes)
            except ValueError as exc:
                logger.warning("%s refused: %s", command, exc)
            return True
        if command in {"/success", "/failure", "/abort"}:
            verdict = command[1:]
            with self._lock:
                handler = self._verdict_handler
            if handler is None or not handler(verdict, notes):
                logger.warning(
                    "/%s refused: no eligible exploration attempt, or a verdict is already closing the run.",
                    verdict,
                )
            elif notes:
                with self._lock:
                    self._record_feedback(notes)
            return True
        try:
            self.reply(parsed.request_id, parsed.answer, notes)
        except ValueError as exc:
            logger.warning("%s", exc)
        return True

    def snapshot(self) -> dict | None:
        """Return the current request for the terminal or Dashboard transport."""
        with self._lock:
            if self._pending is None:
                return None
            return {
                "id": self._pending[0],
                "kind": self._pending_kind,
                "prompt": self._pending_prompt,
                "choices": self._choices(),
            }

    def _choices(self) -> list[str]:
        return (
            ["done", "abort"]
            if self._pending_kind == "reset"
            else ["success", "failure", "continue", "abort"]
        )

    def reply(self, request_id: str, answer: str, notes: str = "") -> None:
        """Accept one matching confirmation; reject expired or invalid replies."""
        with self._lock:
            if self._closed or self._pending is None or request_id != self._pending[0]:
                raise ValueError("expired or already answered operator request")
            if (
                self._pending_kind is not None and answer not in self._choices()
            ) or not isinstance(notes, str):
                raise ValueError("invalid operator response")
            reply = answer + (" " + notes.strip() if notes.strip() else "")
            self._pending[1].put(reply)
            if notes:
                self._record_feedback(notes.strip())
            self._pending = None
            self._pending_kind = None
        self._notify_change()

    def _notify_change(self) -> None:
        if self._on_change is not None:
            self._on_change()

    def _record_feedback(self, text: str) -> None:
        self._feedback.append(text)
        if self._feedback_path is not None:
            self._feedback_path.parent.mkdir(parents=True, exist_ok=True)
            with self._feedback_path.open("a") as f:
                f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
        logger.info("[feedback] recorded for exploration handoff: %s", text)

    def feedback_prompt(self) -> str:
        with self._lock:
            if not self._feedback:
                return ""
            return (
                "\nOperator feedback, chronological (later task corrections supersede "
                "earlier task assumptions, not hardware safety requirements):\n"
                + "\n".join(self._feedback)
                + "\nAcknowledge the latest exploration focus and adapt your plan.\n"
            )

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

    @property
    def pending_kind(self) -> str | None:
        with self._lock:
            return self._pending_kind

    @property
    def closed(self) -> bool:
        with self._lock:
            return self._closed

    def cancel_pending(self) -> None:
        """Retire the old request, never confirm it or close the input reader."""
        with self._lock:
            if self._pending is not None:
                self._pending[1].put(None)
                self._pending = None
            self._pending_kind = None
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
            f"\nShortcut: {shortcut}; /success or /failure starts a new session; "
            "/abort ends exploration."
            if shortcut
            else ""
        )
        try:
            if self._on_change is None:
                print(
                    f"\n{prompt}\nReply: /operator {request_id} <answer>{hint}",
                    flush=True,
                )
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
