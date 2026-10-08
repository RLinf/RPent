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

import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from rpent.dashboard.events import NullDashboardEventSink
from rpent.session import EnvState
from rpent.tools import Toolkit, ToolResult, tool
from rpent.tools.toolkit import ToolCancelled


class ScheduledToolkit(Toolkit):
    def __init__(self, tmp_path):
        super().__init__(
            dashboard_events=NullDashboardEventSink(),
            memory=SimpleNamespace(),
            state=EnvState(tmp_path),
        )
        self.entered = {
            name: threading.Event() for name in ("read1", "read2", "write1", "write2")
        }
        self.release = {name: threading.Event() for name in self.entered}
        self.captured = threading.Event()
        self.release_capture = threading.Event()
        self.order = []
        self.add_tools((self.read, self.write))

    def _register_common_tools(self):
        pass

    @tool(readonly=True)
    def read(self, label: str) -> ToolResult:
        return self.block(label)

    @tool
    def write(self, label: str) -> ToolResult:
        return self.block(label)

    def block(self, label):
        self.order.append(label)
        self.entered[label].set()
        assert self.release[label].wait(5)
        self.raise_if_cancelled()
        return ToolResult(data={"label": label})

    def get_env_state(self, *, command, result, elapsed_s):
        self.captured.set()
        assert self.release_capture.wait(5)
        return ToolResult(data=result)

    def release_all(self):
        for event in self.release.values():
            event.set()
        self.release_capture.set()


@pytest.fixture
def scheduled(tmp_path):
    toolkit = ScheduledToolkit(tmp_path)
    with ThreadPoolExecutor(max_workers=6) as pool:
        try:
            yield toolkit, pool
        finally:
            toolkit.release_all()
    toolkit.close()


def submit(pool, toolkit, label):
    return pool.submit(toolkit.execute_tool, label.rstrip("12"), {"label": label})


def wait_pending(toolkit, count):
    # Observe admission so ordering assertions never rely on thread timing.
    with toolkit._scheduler.condition:
        assert toolkit._scheduler.condition.wait_for(
            lambda: len(toolkit._scheduler.pending) == count, timeout=3
        )


def test_readonly_calls_overlap(scheduled):
    toolkit, pool = scheduled
    first = submit(pool, toolkit, "read1")
    second = submit(pool, toolkit, "read2")
    assert toolkit.entered["read1"].wait(3)
    assert toolkit.entered["read2"].wait(3)
    assert not toolkit.captured.is_set()
    toolkit.release_all()
    assert not first.result(timeout=3).is_error
    assert not second.result(timeout=3).is_error


def test_exclusive_calls_keep_order_and_block_new_readers_through_capture(scheduled):
    toolkit, pool = scheduled
    first = submit(pool, toolkit, "read1")
    assert toolkit.entered["read1"].wait(3)
    writer1 = submit(pool, toolkit, "write1")
    wait_pending(toolkit, 1)
    reader = submit(pool, toolkit, "read2")
    wait_pending(toolkit, 2)
    writer2 = submit(pool, toolkit, "write2")
    wait_pending(toolkit, 3)
    toolkit.release["read1"].set()
    assert not first.result(timeout=3).is_error
    assert toolkit.entered["write1"].wait(3)
    assert not toolkit.entered["read2"].is_set()
    assert not toolkit.entered["write2"].is_set()
    toolkit.release["write1"].set()
    assert toolkit.captured.wait(3)
    assert not toolkit.entered["read2"].is_set()
    assert not toolkit.entered["write2"].is_set()
    toolkit.release_capture.set()
    assert not writer1.result(timeout=3).is_error
    assert toolkit.entered["write2"].wait(3)
    assert not toolkit.entered["read2"].is_set()
    toolkit.release_all()
    assert not writer2.result(timeout=3).is_error
    assert not reader.result(timeout=3).is_error
    assert toolkit.order == ["read1", "write1", "write2", "read2"]


@pytest.mark.parametrize("close", [False, True])
def test_cancel_drains_shared_and_queued_calls_then_requires_resume(scheduled, close):
    toolkit, pool = scheduled
    active = [submit(pool, toolkit, label) for label in ("read1", "read2")]
    for label in ("read1", "read2"):
        assert toolkit.entered[label].wait(3)
    waiting = submit(pool, toolkit, "write1")
    wait_pending(toolkit, 1)
    cancel = pool.submit(toolkit.close if close else toolkit.cancel_active_and_wait)
    assert waiting.result(timeout=3).data["code"] == "tool_cancelled"
    assert not cancel.done()
    with pytest.raises(RuntimeError, match="cleanup"):
        toolkit.resume_calls()
    assert toolkit.execute_tool("read", {"label": "read1"}).is_error
    toolkit.release_all()
    for result in active:
        assert result.result(timeout=3).data["code"] == "tool_cancelled"
    cancel.result(timeout=3)
    assert not toolkit.entered["write1"].is_set()
    toolkit.resume_calls()
    assert toolkit.execute_tool("read", {"label": "read1"}).is_error is close


def test_close_waits_for_final_capture(scheduled):
    toolkit, pool = scheduled
    active = submit(pool, toolkit, "write1")
    assert toolkit.entered["write1"].wait(3)
    toolkit.release["write1"].set()
    assert toolkit.captured.wait(3)
    closing = pool.submit(toolkit.close)
    with toolkit._scheduler.condition:
        assert toolkit._scheduler.condition.wait_for(
            lambda: toolkit._scheduler.closed, timeout=3
        )
    assert not closing.done()
    toolkit.release_capture.set()
    active.result(timeout=3)
    closing.result(timeout=3)


def test_overlapping_cancellations_wait_for_the_same_call(scheduled):
    toolkit, pool = scheduled
    active = submit(pool, toolkit, "read1")
    assert toolkit.entered["read1"].wait(3)
    cancellations = [pool.submit(toolkit.cancel_active_and_wait) for _ in range(2)]
    with toolkit._scheduler.condition:
        assert toolkit._scheduler.condition.wait_for(
            lambda: toolkit._scheduler.paused, timeout=3
        )
    with pytest.raises(ToolCancelled):
        toolkit.raise_if_cancelled()
    assert not any(future.done() for future in cancellations)
    toolkit.release_all()
    assert active.result(timeout=3).is_error
    for future in cancellations:
        future.result(timeout=3)
    toolkit.resume_calls()
    assert not toolkit.execute_tool("read", {"label": "read1"}).is_error
