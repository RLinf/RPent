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

"""Session cancellation must never become shared terminal EOF."""

import queue
from concurrent.futures import ThreadPoolExecutor
from threading import Event

import pytest

from rpent.session.input import InputCancelled, SessionInputQueue


@pytest.mark.parametrize("nested", [False, True])
def test_cancel_waiting_reader_preserves_next_session(nested):
    source = queue.Queue()
    scope = SessionInputQueue(source)
    consumer = SessionInputQueue(scope, retain_until_ack=True) if nested else scope
    started = Event()

    def read():
        started.set()
        with pytest.raises(InputCancelled):
            consumer.get()

    with ThreadPoolExecutor() as pool:
        future = pool.submit(read)
        assert started.wait(2)
        scope.cancel()
        future.result(2)
    consumer.close()
    source.put("next session")
    assert source.get_nowait() == "next session"
    with pytest.raises(queue.Empty):
        source.get_nowait()


@pytest.mark.parametrize("item", ["unaccepted", None])
def test_unaccepted_input_returns_before_later_lines_and_real_eof_is_consumed(item):
    source = queue.Queue()
    source.put(item)
    reader = SessionInputQueue(SessionInputQueue(source), retain_until_ack=True)
    assert reader.get() == item
    source.put("later")
    reader.cancel()
    assert list(source.queue) == ["later"]
    reader.close()
    next_reader = SessionInputQueue(source, retain_until_ack=True)
    assert next_reader.get() == item
    next_reader.acknowledge()
    next_reader.close()
    assert source.get_nowait() == "later"
    assert source.empty()
