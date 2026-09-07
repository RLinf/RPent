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

"""Deterministic CPU scheduling checks using events and queue registration."""

from concurrent.futures import ThreadPoolExecutor
from threading import Event, get_ident
from types import SimpleNamespace

import pytest

from rpent.memory import MemoryManager
from rpent.session import EnvState
from rpent.tools import (
    ToolContext,
    Toolkit,
    ToolResult,
    parallel,
    readonly,
    tool,
)


@tool
@readonly
@parallel
def read(key: str, *, ctx: ToolContext) -> ToolResult:
    """Read through the test's controlled callback."""
    return ctx.robot.invoke(key, ctx)


@tool
def write(key: str, *, ctx: ToolContext) -> ToolResult:
    """Act through the test's controlled callback."""
    return ctx.robot.invoke(key, ctx)


@tool
@readonly
def finish(status: str, summary: str, *, ctx: ToolContext) -> ToolResult:
    """Finish with this robot's test outcome."""
    if ctx.robot.finish_hook:
        return ctx.robot.finish_hook()
    return ToolResult(data={"_finish": True, "status": status, "summary": summary})


class ScheduledToolkit(Toolkit):
    def __init__(self, root, invoke):
        self.captures = 0
        self.capture_hook = None
        super().__init__(
            state=EnvState(root),
            memory=MemoryManager(root / "memory"),
            robot=SimpleNamespace(invoke=invoke, finish_hook=None),
            output_dir=root,
            tools=(
                finish,
                read,
                write,
            ),
        )

    def _capture_observation(self, *, result, **kwargs):
        if self.capture_hook:
            self.capture_hook()
        step = self.captures
        self.captures += 1
        data = {"step": step}
        return data, []


class Harness:
    def __init__(self, root):
        self.started = {}
        self.hooks = {}
        self.order = []
        self.gates = []
        self.pool = ThreadPoolExecutor(max_workers=12)
        self.toolkit = ScheduledToolkit(root, self.invoke)
        self.queue_changed = Event()
        scheduler = self.toolkit._scheduler
        original_can_start = scheduler._can_start

        def observe_queue(call):
            allowed = original_can_start(call)
            if not allowed:
                self.queue_changed.set()
            return allowed

        scheduler._can_start = observe_queue

    def event(self, key):
        return self.started.setdefault(key, Event())

    def gate(self):
        gate = Event()
        self.gates.append(gate)
        return gate

    def invoke(self, key, ctx):
        self.order.append(key)
        self.event(key).set()
        if key in self.hooks:
            self.hooks[key](ctx)
        return ToolResult(data={"key": key})

    def submit(self, name, key):
        self.event(key)
        return self.pool.submit(self.toolkit.execute_tool, name, {"key": key})

    def wait_for_queued(self, count=1):
        """Wait for submissions to queue while active calls are held by test gates."""
        scheduler = self.toolkit._scheduler
        while True:
            with scheduler._condition:
                if len(scheduler._pending) == count:
                    return
                self.queue_changed.clear()
            assert self.queue_changed.wait(2), f"Expected {count} queued calls."

    def wait(self, key):
        assert self.event(key).wait(2), f"Call {key} never started."

    def block(self, key, gate):
        def wait(ctx):
            assert gate.wait(4), f"Gate for {key} not released."

        self.hooks[key] = wait


@pytest.fixture
def harness(tmp_path):
    value = Harness(tmp_path)
    yield value
    for gate in value.gates:
        gate.set()
    if value.toolkit._scheduler._state != "closed":
        value.toolkit.close()
    value.pool.shutdown(wait=True)


def test_shared_calls_overlap_and_waiting_writer_blocks_new_readers(harness):
    gate = harness.gate()
    for key in ("a", "b"):
        harness.block(key, gate)
    a = harness.submit("read", "a")
    b = harness.submit("read", "b")
    harness.wait("a")
    harness.wait("b")
    writer = harness.submit("write", "w")
    harness.wait_for_queued()
    c = harness.submit("read", "c")
    harness.wait_for_queued(2)
    assert not harness.event("w").is_set()
    assert not harness.event("c").is_set()
    gate.set()
    assert all(not f.result(3).is_error for f in (a, b, writer, c))
    assert harness.order[-2:] == ["w", "c"]


def test_writers_keep_registration_order_and_can_pass_waiting_reads(harness):
    gate = harness.gate()
    harness.block("active", gate)
    active = harness.submit("write", "active")
    harness.wait("active")
    futures = []
    for name, key in (("read", "a"), ("write", "b"), ("read", "c"), ("write", "d")):
        futures.append(harness.submit(name, key))
        harness.wait_for_queued(len(futures))
    gate.set()
    for future in [active, *futures]:
        assert not future.result(3).is_error
    assert harness.order[:3] == ["active", "b", "d"]
    assert set(harness.order[3:]) == {"a", "c"}


@pytest.mark.parametrize("outcome", ["accepted", "refused", "exception"])
def test_finish_uses_normal_exclusive_order_and_keeps_admission_open(harness, outcome):
    gate = harness.gate()
    finish_gate = harness.gate()
    harness.block("active", gate)
    active = harness.submit("write", "active")
    harness.wait("active")
    early = harness.submit("read", "early")
    harness.wait_for_queued()

    def finish():
        harness.order.append("finish")
        harness.event("finish").set()
        assert finish_gate.wait(4)
        if outcome == "accepted":
            return ToolResult(
                data={
                    "_finish": True,
                    "status": "failure",
                    "summary": "Verified outcome.",
                }
            )
        if outcome == "exception":
            raise RuntimeError("Verification failed.")
        return ToolResult(error="More attempts.")

    harness.toolkit._robot.finish_hook = finish
    finishing = harness.pool.submit(
        harness.toolkit.execute_tool,
        "finish",
        {"status": "success", "summary": "requested"},
    )
    harness.wait_for_queued(2)
    late_read = harness.submit("read", "late_read")
    harness.wait_for_queued(3)
    late_write = harness.submit("write", "late_write")
    harness.wait_for_queued(4)
    assert not harness.event("finish").is_set()
    gate.set()
    harness.wait("finish")
    assert not harness.event("early").is_set()
    assert not harness.event("late_write").is_set()
    assert not harness.event("late_read").is_set()
    finish_gate.set()
    for future in (active, early, late_read, late_write):
        assert not future.result(3).is_error
    result = finishing.result(3)
    assert result.is_error is (outcome != "accepted")
    assert harness.order[:3] == ["active", "finish", "late_write"]
    assert set(harness.order[3:]) == {"early", "late_read"}
    if outcome == "accepted":
        expected = {"status": "failure", "summary": "Verified outcome."}
        assert harness.toolkit.finish_result == expected
        result.data["status"] = "changed"
        saved = harness.toolkit.finish_result
        saved["summary"] = "changed"
        assert harness.toolkit.finish_result == expected
    else:
        assert harness.toolkit.finish_result is None
    assert not harness.toolkit.execute_tool("read", {"key": "new"}).is_error
    assert not harness.toolkit.execute_tool("write", {"key": "new_write"}).is_error


def test_other_tools_cannot_record_a_finish_result(harness):
    harness.toolkit._robot.invoke = lambda key, ctx: ToolResult(
        data={"_finish": True, "status": "success", "summary": "ordinary result"}
    )
    assert not harness.toolkit.execute_tool("read", {"key": "probe"}).is_error
    assert harness.toolkit.finish_result is None


@pytest.mark.parametrize("name", ["read", "write"])
def test_cancel_stops_waiting_calls_and_waits_for_active_cleanup(harness, name):
    cancelled = harness.gate()
    cleanup_gate = harness.gate()

    def loop(ctx):
        try:
            assert ctx._cancel_event.wait(3)
            ctx.check_cancelled()
        finally:
            cancelled.set()
            assert cleanup_gate.wait(4)

    harness.hooks["active"] = loop
    active = harness.submit(name, "active")
    harness.wait("active")
    waiting = harness.submit("write", "waiting")
    harness.wait_for_queued()
    stop = harness.pool.submit(harness.toolkit.cancel_active_and_wait)
    assert cancelled.wait(2)
    assert waiting.result(2).error == "Tool call cancelled."
    assert not harness.event("waiting").is_set()
    assert not stop.done()
    assert (
        harness.toolkit.execute_tool("read", {"key": "paused"}).error
        == "Tool calls are paused."
    )
    with pytest.raises(RuntimeError, match="cleanup"):
        harness.toolkit.resume_calls()
    cleanup_gate.set()
    assert active.result(3).error == "Tool call cancelled."
    stop.result(3)
    assert harness.toolkit.captures == (1 if name == "write" else 0)
    harness.toolkit.resume_calls()
    assert not harness.toolkit.execute_tool("read", {"key": "new"}).is_error


def test_all_active_shared_calls_receive_cancellation(harness):
    def loop(ctx):
        assert ctx._cancel_event.wait(3)
        ctx.check_cancelled()

    harness.hooks.update(a=loop, b=loop)
    futures = [harness.submit("read", key) for key in ("a", "b")]
    harness.wait("a")
    harness.wait("b")
    harness.toolkit.cancel_active_and_wait()
    assert [f.result(2).error for f in futures] == ["Tool call cancelled."] * 2


def test_exclusive_admission_is_retained_through_final_capture(harness):
    capture_started = harness.gate()
    capture_gate = harness.gate()

    def capture():
        capture_started.set()
        assert capture_gate.wait(4)

    harness.toolkit.capture_hook = capture
    active = harness.submit("write", "write")
    assert capture_started.wait(2)
    reader = harness.submit("read", "reader")
    harness.wait_for_queued()
    assert not harness.event("reader").is_set()
    capture_gate.set()
    assert not active.result(3).is_error
    assert not reader.result(3).is_error


def test_close_waits_for_final_capture_before_saving_video(harness, monkeypatch):
    capture_started = harness.gate()
    capture_gate = harness.gate()
    cleaned = Event()

    def capture():
        capture_started.set()
        assert capture_gate.wait(4)

    harness.toolkit.capture_hook = capture
    harness.toolkit.record_frame([[[0, 0, 0]]])
    monkeypatch.setattr(
        harness.toolkit.state, "save", lambda *args, **kwargs: cleaned.set()
    )
    active = harness.submit("write", "write")
    assert capture_started.wait(2)
    closing = harness.pool.submit(harness.toolkit.close)
    scheduler = harness.toolkit._scheduler
    with scheduler._condition:
        assert scheduler._condition.wait_for(
            lambda: scheduler._state == "closed", timeout=2
        )
    assert not cleaned.is_set()
    assert not closing.done()
    assert (
        harness.toolkit.execute_tool("read", {"key": "late"}).error
        == "Toolkit is closed."
    )
    capture_gate.set()
    active.result(3)
    closing.result(3)
    assert cleaned.is_set()


def test_toolkits_do_not_share_execution_locks(harness, tmp_path):
    gate = harness.gate()
    harness.block("blocked", gate)
    active = harness.submit("write", "blocked")
    harness.wait("blocked")
    other = ScheduledToolkit(tmp_path / "other", lambda key, ctx: ToolResult())
    try:
        assert not other.execute_tool("write", {"key": "independent"}).is_error
    finally:
        gate.set()
        other.close()
    assert not active.result(3).is_error


def test_cancel_between_admission_and_handler_skips_execution_and_capture(
    harness, monkeypatch
):
    admitted = harness.gate()
    start_gate = harness.gate()
    scheduler = harness.toolkit._scheduler
    original = scheduler.acquire

    def acquire(tool):
        call = original(tool)
        admitted.set()
        assert start_gate.wait(4)
        return call

    monkeypatch.setattr(scheduler, "acquire", acquire)
    future = harness.submit("write", "never_start")
    assert admitted.wait(2)
    stopping = harness.pool.submit(harness.toolkit.cancel_active_and_wait)
    with scheduler._condition:
        assert scheduler._condition.wait_for(
            lambda: scheduler._state == "paused", timeout=2
        )
    start_gate.set()
    assert future.result(3).error == "Tool call cancelled."
    stopping.result(3)
    assert harness.toolkit.captures == 0
    assert not harness.event("never_start").is_set()


def test_cancellation_can_start_with_tool_workers_saturated(harness):
    def loop(ctx):
        assert ctx._cancel_event.wait(3)
        ctx.check_cancelled()

    harness.hooks["active"] = loop
    with ThreadPoolExecutor(max_workers=2) as pool:
        active = pool.submit(harness.toolkit.execute_tool, "write", {"key": "active"})
        harness.wait("active")
        waiting = pool.submit(harness.toolkit.execute_tool, "write", {"key": "queued"})
        harness.wait_for_queued()
        harness.toolkit.cancel_active_and_wait()
        assert active.result(3).error == "Tool call cancelled."
        assert waiting.result(3).error == "Tool call cancelled."
    assert not harness.event("queued").is_set()


def test_overlapping_cancellation_requests_wait_for_active_cleanup(
    harness, monkeypatch
):
    cancelled = harness.gate()
    cleanup_gate = harness.gate()

    def loop(ctx):
        try:
            assert ctx._cancel_event.wait(3)
            ctx.check_cancelled()
        finally:
            cancelled.set()
            assert cleanup_gate.wait(4)

    harness.hooks["active"] = loop
    active = harness.submit("write", "active")
    harness.wait("active")
    scheduler = harness.toolkit._scheduler
    original_wait = scheduler._condition.wait
    waiting = set()
    both_waiting = Event()

    def wait_done(timeout=None):
        waiting.add(get_ident())
        if len(waiting) == 2:
            both_waiting.set()
        return original_wait(timeout)

    monkeypatch.setattr(scheduler._condition, "wait", wait_done)
    first = harness.pool.submit(harness.toolkit.cancel_active_and_wait)
    assert cancelled.wait(2)
    second = harness.pool.submit(harness.toolkit.cancel_active_and_wait)
    assert both_waiting.wait(2)
    with pytest.raises(RuntimeError, match="cleanup"):
        harness.toolkit.resume_calls()
    assert not first.done() and not second.done()
    cleanup_gate.set()
    assert active.result(3).error == "Tool call cancelled."
    first.result(3)
    second.result(3)
    harness.toolkit.resume_calls()
    assert not harness.toolkit.execute_tool("read", {"key": "resumed"}).is_error


def test_resume_does_not_revive_cancelled_waiters_or_extend_old_cancellation(
    harness, monkeypatch
):
    scheduler = harness.toolkit._scheduler
    active_gate = harness.gate()
    rejection_gate = harness.gate()
    new_gate = harness.gate()
    old_woken = Event()
    old_thread = None
    original_wait = scheduler._condition.wait

    def delay_old_waiter(timeout=None):
        nonlocal old_thread
        if old_thread is None:
            old_thread = get_ident()
        result = original_wait(timeout)
        if get_ident() != old_thread:
            return result
        scheduler._condition.release()
        try:
            old_woken.set()
            assert rejection_gate.wait(4)
        finally:
            scheduler._condition.acquire()
        return result

    monkeypatch.setattr(scheduler._condition, "wait", delay_old_waiter)
    harness.block("active", active_gate)
    active = harness.submit("write", "active")
    harness.wait("active")
    old = harness.submit("write", "old")
    harness.wait_for_queued()
    stopping = harness.pool.submit(harness.toolkit.cancel_active_and_wait)
    assert old_woken.wait(2)
    active_gate.set()
    active.result(3)
    assert not stopping.done()
    harness.toolkit.resume_calls()

    def new_call(ctx):
        assert new_gate.wait(4)
        ctx.check_cancelled()

    harness.hooks["new"] = new_call
    new = harness.submit("read", "new")
    harness.wait("new")
    rejection_gate.set()
    assert old.result(3).error == "Tool call cancelled."
    assert not harness.event("old").is_set()
    stopping.result(3)
    assert not new.done()
    new_gate.set()
    assert not new.result(3).is_error


def test_cancel_and_resume_do_not_reopen_closed_toolkit(harness):
    harness.toolkit.close()
    harness.toolkit.cancel_active_and_wait()
    harness.toolkit.resume_calls()
    assert (
        harness.toolkit.execute_tool("read", {"key": "closed"}).error
        == "Toolkit is closed."
    )
    assert not harness.event("closed").is_set()
