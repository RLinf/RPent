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

from __future__ import annotations

import queue
import threading

import pytest

from rpent.cli.tui import initial_user_message, next_user_line
from rpent.session.input import SessionInputQueue


@pytest.mark.parametrize("read_line", [initial_user_message, next_user_line])
@pytest.mark.parametrize("end_of_input", [None, "/quit", "/exit", "/q"])
def test_scoped_input_preserves_feedback_and_real_exit(read_line, end_of_input):
    source = queue.Queue()
    scope = SessionInputQueue(source)
    scope.put("  ")
    scope.put("  inspect the other bowl  ")
    scope.put(end_of_input)

    assert read_line(scope) == "inspect the other bowl"
    assert read_line(scope) is None
    assert source.empty()


def test_cancelled_scope_leaves_feedback_and_real_eof_for_next_scope():
    source = queue.Queue()
    previous = SessionInputQueue(source)
    previous.cancel()
    source.put("next attempt feedback")
    source.put(None)

    assert previous.cancelled
    assert previous.get(block=False) is None
    assert previous.get(timeout=0) is None
    current = SessionInputQueue(source)
    assert not current.cancelled
    assert current.get(block=False) == "next attempt feedback"
    assert current.get(block=False) is None
    assert source.empty()


def test_nested_scope_cleanup_does_not_cancel_attended_attempt():
    source = queue.Queue()
    attempt = SessionInputQueue(source)
    previous_reply = SessionInputQueue(attempt)
    previous_reply.cancel()
    attempt.put("continue after that reply")

    assert previous_reply.get() is None
    assert not attempt.cancelled
    current_reply = SessionInputQueue(attempt)
    assert next_user_line(current_reply) == "continue after that reply"
    attempt.cancel()
    assert current_reply.get() is None
    assert source.empty()


def test_scoped_input_retains_queue_nonblocking_and_timeout_contract():
    scope = SessionInputQueue(queue.Queue())
    with pytest.raises(queue.Empty):
        scope.get(block=False)
    with pytest.raises(queue.Empty):
        scope.get(timeout=0)
    with pytest.raises(queue.Empty):
        scope.get(timeout=0.01)
    with pytest.raises(ValueError, match="non-negative"):
        scope.get(timeout=-1)
    scope.put("available immediately")
    assert scope.get(timeout=0) == "available immediately"


def test_cancel_retires_waiting_reader_without_consuming_later_input():
    reading = threading.Event()

    class ObservedQueue(queue.Queue):
        def get(self, block=True, timeout=None):
            reading.set()
            return super().get(block=block, timeout=timeout)

    source = ObservedQueue()
    scope = SessionInputQueue(source)
    received = []
    reader = threading.Thread(target=lambda: received.append(scope.get()), daemon=True)
    reader.start()
    try:
        assert reading.wait(1)
        scope.cancel()
        source.put("new attempt")
        reader.join(timeout=1)
        assert not reader.is_alive()
        assert received == [None]
        assert source.get(block=False) == "new attempt"
        assert source.empty()
    finally:
        scope.cancel()
        reader.join(timeout=1)
