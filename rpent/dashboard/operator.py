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

import json
import threading
import uuid
from pathlib import Path

from rpent.tools.toolkit import ToolCancelled


class DashboardOperator:
    def __init__(self, output_dir, on_change=lambda: None):
        self._condition = threading.Condition()
        self._pending = None
        self._answer = None
        self._closed = False
        self._path = Path(output_dir) / "dashboard_operator.jsonl"
        self._on_change = on_change

    def _record(self, event):
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a") as file:
            file.write(json.dumps(event, ensure_ascii=False) + "\n")

    def snapshot(self):
        with self._condition:
            return dict(self._pending) if self._pending else None

    def request(self, prompt, check_cancelled, *, kind=None):
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
            self._record({"event": "request", **self._pending})
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

    def reply(self, request_id, answer, notes=""):
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
            self._record(
                {"event": "reply", "id": request_id, "answer": answer, "notes": notes}
            )
            self._condition.notify_all()

    def close(self):
        with self._condition:
            self._closed = True
            self._condition.notify_all()
